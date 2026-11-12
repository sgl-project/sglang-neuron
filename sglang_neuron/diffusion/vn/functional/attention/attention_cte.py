# SPDX-License-Identifier: Apache-2.0
import torch
import nki

from typing import Optional, Tuple, Union
from torch import Tensor

from nkilib.core.attention.attention_cte import attention_cte
from sglang_neuron.diffusion.neuron_api import nki_wrap_function
from sglang_neuron.diffusion.vn.utils.hardware_config import get_logical_nc_config
from sglang_neuron.diffusion.vn.utils.neuron_utils import can_run_kernel


def wrap_nki(kernel):
    """The runtime's ``wrap_nki``, resolved at call time: importing a runtime at module import
    would start it before the worker claims its NeuronCore."""
    return nki_wrap_function()(kernel)


jitted_attention_cte = nki.jit()(attention_cte)

# attention_cte kernel limits; larger shapes take the torch fallback.
MAX_BS = 512
MAX_SEQLEN = 131072
MAX_HEAD_DIM = 128


# TODO: move this check into nkilib.
def _can_use_flash_attention_kernel(
    v: Tensor,
) -> bool:
    """Whether the attention_cte kernel supports ``v`` ``[B, S, D]`` on this device."""

    if not can_run_kernel(v):
        return False

    B, S, D = v.shape

    if B > MAX_BS:
        return False

    if S > MAX_SEQLEN:
        return False

    if D > MAX_HEAD_DIM:
        return False

    return True


def _torch_attention_impl(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: float,
    causal_mask: bool = True,
    sliding_window: int = 0,
    sink: Optional[Tensor] = None,
    tp_q: bool = True,
    tp_k: bool = False,
    tp_out: bool = False,
    k_prior: Optional[Tensor] = None,
    v_prior: Optional[Tensor] = None,
    prior_used_len: Optional[Tensor] = None,
    bound_min: Optional[Tensor] = None,
    bound_max: Optional[Tensor] = None,
    cp_offset: int = 0,  # Q position offset for CP causal mask
) -> Tensor:
    """Torch reference of the attention_cte kernel, with the layouts of :func:`flash_attention`."""
    bs = q.shape[0]
    bs_kv = k.shape[0]

    # Q as [B, S_q, D].
    if tp_q:
        q_for_matmul = q
        seqlen_q, d = q.shape[1], q.shape[2]
    else:
        q_for_matmul = q.transpose(-2, -1)  # [B, D, S_q] -> [B, S_q, D]
        seqlen_q = q.shape[2]

    # K as [B, D, S_k].
    if tp_k:
        k_for_matmul = k.transpose(-2, -1)  # [B, S_k, D] -> [B, D, S_k]
        seqlen_k = k.shape[1]
    else:
        k_for_matmul = k
        seqlen_k = k.shape[2]

    # Prefix caching: prepend the used part of the prior K/V.
    if k_prior is not None:
        prior_len = int(prior_used_len.item()) if prior_used_len is not None else 0

        if tp_k:
            k_prior_slice = k_prior[:, :prior_len, :].transpose(
                -2, -1
            )  # [B, prior_len, D] -> [B, D, prior_len]
        else:
            k_prior_slice = k_prior[:, :, :prior_len]  # [B, D, prior_len]

        v_prior_slice = v_prior[:, :prior_len, :]  # [B, prior_len, D]

        k_for_matmul = torch.cat(
            [k_prior_slice, k_for_matmul], dim=-1
        )  # [B, D, prior_len + S_k]
        v = torch.cat([v_prior_slice, v], dim=1)  # [B, prior_len + S_k, D]
        seqlen_k_total = prior_len + seqlen_k
    else:
        seqlen_k_total = seqlen_k
        prior_len = 0

    # GQA: repeat each K/V head over its query group.
    if bs_kv < bs:
        repeat_factor = bs // bs_kv
        k_for_matmul = k_for_matmul.repeat_interleave(repeat_factor, dim=0)
        v = v.repeat_interleave(repeat_factor, dim=0)

    # [B, S_q, D] @ [B, D, S_k] -> [B, S_q, S_k]
    scores = torch.matmul(q_for_matmul, k_for_matmul) * scale

    if causal_mask:
        # Query positions are offset by the prefix and by this CP rank's offset.
        q_pos = (
            torch.arange(seqlen_q, device=scores.device).unsqueeze(1)
            + prior_len
            + cp_offset
        )
        k_pos = torch.arange(seqlen_k_total, device=scores.device).unsqueeze(0)
        mask = q_pos < k_pos

        if sliding_window > 0:
            window_mask = q_pos >= k_pos + sliding_window
            mask = mask | window_mask

        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

    # Per-query KV range [bound_min, bound_max), each [B, S_q, 1].
    if bound_min is not None or bound_max is not None:
        assert (bound_min is None) == (bound_max is None), (
            "bound_min and bound_max must both be set or both be None"
        )
        k_pos_all = (
            torch.arange(seqlen_k_total, device=scores.device).unsqueeze(0).unsqueeze(1)
        )  # [1, 1, S_k]
        scores = scores.masked_fill(
            k_pos_all < bound_min, torch.finfo(scores.dtype).min
        )
        scores = scores.masked_fill(
            k_pos_all >= bound_max, torch.finfo(scores.dtype).min
        )

    # The sink is an extra logit column, dropped after the softmax.
    if sink is not None:
        sink_expanded = sink.unsqueeze(1).expand(-1, seqlen_q, -1)  # [B, S_q, 1]
        scores = torch.cat([scores, sink_expanded], dim=-1)

    attn_weights = torch.softmax(scores.float(), dim=-1).type_as(q)

    if sink is not None:
        attn_weights = attn_weights[..., :-1]

    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    # [B, S_q, S_k] @ [B, S_k, D] -> [B, S_q, D]
    output = torch.matmul(attn_weights, v)

    if tp_out:
        output = output.transpose(-2, -1)  # [B, S_q, D] -> [B, D, S_q]

    return output


def flash_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Optional[float] = None,
    causal_mask: bool = True,
    sink: Optional[Tensor] = None,
    sliding_window: Optional[int] = None,
    tp_q: bool = True,
    tp_k: bool = False,
    tp_out: bool = False,
    cp_offset: Optional[Tensor] = None,
    global_cp_deg: Optional[int] = None,
    cp_strided_q_slicing: bool = False,
    k_prior: Optional[Tensor] = None,
    v_prior: Optional[Tensor] = None,
    prior_used_len: Optional[Tensor] = None,
    bound_min: Optional[Tensor] = None,
    bound_max: Optional[Tensor] = None,
    allow_vector_exp: bool = True,
) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
    """``softmax(scale * Q @ K^T + mask) @ V`` on the nkilib attention_cte kernel, else torch.

    ``q`` is ``[B, S_q, D]`` with ``tp_q`` (default), else ``[B, D, S_q]``; ``k`` is
    ``[B, S_k, D]`` with ``tp_k``, else ``[B, D, S_k]`` (default); ``v`` is ``[B, S_k, D]``.
    The output is ``[B, D, S_q]`` with ``tp_out``, else ``[B, S_q, D]``. ``B`` folds the heads,
    and a K/V batch that divides the Q batch is GQA. ``scale`` defaults to ``1 / sqrt(D)``; the
    kernel path folds it into ``q`` and runs the kernel at scale 1.0.

    Args:
        sink: ``[B, 1]`` attention-sink logit.
        sliding_window: Local window. None or 0 is full attention.
        cp_offset: ``[1, 1]`` query position offset under context parallel (``global_cp_deg``).
        k_prior: Prefix-cache K, in ``k``'s layout.
        v_prior: Prefix-cache V, ``[B_kv, S_prior, D]``.
        prior_used_len: ``[1]`` used length of the prefix cache.
        bound_min: ``[B, S_q, 1]`` inclusive lower KV index per query.
        bound_max: ``[B, S_q, 1]`` exclusive upper KV index per query.
        allow_vector_exp: Allow the Vector Engine exp on trn3.
    """
    if sliding_window is None:
        sliding_window = 0

    if scale is None:
        d_head = v.shape[2]
        scale = 1.0 / (d_head**0.5)

    can_use_kernel = _can_use_flash_attention_kernel(v=v)

    if can_use_kernel:
        q = q * scale

        wrapped_attention_cte = wrap_nki(jitted_attention_cte)

        return wrapped_attention_cte[get_logical_nc_config()](
            q=q,
            k=k,
            v=v,
            scale=1.0,
            causal_mask=causal_mask,
            k_prior=k_prior,
            v_prior=v_prior,
            prior_used_len=prior_used_len,
            sink=sink,
            sliding_window=sliding_window,
            tp_q=tp_q,
            tp_k=tp_k,
            tp_out=tp_out,
            cache_softmax=False,
            softmax_dtype=torch.float32,
            cp_offset=cp_offset,
            global_cp_deg=global_cp_deg,
            cp_strided_q_slicing=cp_strided_q_slicing,
            bound_min=bound_min,
            bound_max=bound_max,
            allow_vector_exp=allow_vector_exp,
        )
    else:
        return _torch_attention_impl(
            q=q,
            k=k,
            v=v,
            scale=scale,
            causal_mask=causal_mask,
            sliding_window=sliding_window,
            sink=sink,
            tp_q=tp_q,
            tp_k=tp_k,
            tp_out=tp_out,
            k_prior=k_prior,
            v_prior=v_prior,
            prior_used_len=prior_used_len,
            bound_min=bound_min,
            bound_max=bound_max,
            cp_offset=cp_offset[0, 0] if cp_offset is not None else 0,
        )
