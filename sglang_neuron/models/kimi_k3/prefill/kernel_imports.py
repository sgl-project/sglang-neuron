# SPDX-License-Identifier: Apache-2.0
"""The one place the Kimi-K3 prefill imports its NKI kernels from.

The kernels are vendored under ``KERNEL_ROOT`` (``models/kimi_k3/kernels/prefill/``, its MANIFEST.json);
``KERNELS`` maps each kernel to its module under the root and its entry. Importing this module never imports ``nki``.
The first accessor call outside a traced graph (or ``prepare()``) imports every kernel and builds the FFN trace shims,
and raises ``KernelUnavailable`` naming the module when one does not import, so a host without the kernels fails at
the backend build. Inside a traced forward the accessors only read the loaded kernels from a dict, which dynamo reads
as a constant; a kernel that was not loaded before the trace raises ``KernelUnavailable`` there. The FFN kernels are
called through ``backend.stack.wrap_nki`` at one logical core with the static options of the serving form; a CPU-mode
process gets the host stand-in of ``wrap_nki``, which raises at the call.
"""

from __future__ import annotations

import importlib
from typing import Any

import torch
from torch import Tensor

from sglang_neuron.backend.stack import wrap_nki

KERNEL_ROOT = "sglang_neuron.models.kimi_k3.kernels.prefill"

KERNELS = {
    "attn_res_cte": ("attn_res.attn_res_cte", "attn_res_cte"),
    "attn_res_bank_scores": ("attn_res.attn_res_bank_scores", "attn_res_bank_scores"),
    "kda_block": ("kda.kda_block_prefill", "kimi_k3_kda_block_prefill"),
    "kda_front_mx_quantize": ("kda.kda_front_mx_torch", "mx_quantize"),
    "kda_front_mx_pack": ("kda.kda_front_mx_torch", "pack_from_mx"),
    "mx_linear": ("linear.mx_linear", "prefill_mx_linear"),
    "mx_head_linear": ("linear.mx_head_linear", "prefill_mx_head_linear"),
    "bwmm": ("moe.bwmm_shard_on_block_mx", "bwmm_shard_on_block_mx"),
    "bwmm_act_types": ("moe.utils.common_types", "ActFnType"),
    "bwmm_skip_mode": ("moe.routed_utils", "SkipMode"),
    "mla_flash_latent": ("attention.mla_flash_latent", "mla_flash_latent"),
    "mla_dcp_merge": ("attention.mla_dcp_merge", "kimi_k3_mla_dcp_merge"),
    "rows_write_after": ("page_io", "kimi_k3_rows_write_after"),
    "page_read": ("page_io", "kimi_k3_page_read"),
    "page_write": ("page_io", "kimi_k3_page_write"),
    "moe_token_runs": ("moe.token_map", "kimi_k3_moe_token_runs"),
}
"""key -> (module under ``KERNEL_ROOT``, entry) of each kernel."""

LNC = 1
"""The logical core count of each kernel call: the prefill stage runs at one core per rank."""

TOKEN_RUNS_MAX_EXPERTS = 32
"""The most local experts of one token-run kernel call."""
TOKEN_RUNS_MAX_TOKENS = 128 * 128
"""The most tokens of one token-run kernel call."""


class KernelUnavailable(ImportError):
    """A Kimi-K3 prefill kernel does not import on this host."""


_LOADED: dict[str, Any] = {}
"""key -> kernel entry, filled by ``prepare``. Plain dicts, not function caches: dynamo traces through the body of a
cached function (and its imports), but reads a dict entry as a constant."""
_SHIMS: dict[str, Any] = {}
"""name -> FFN trace shim, filled by ``prepare``."""


def _load(key: str) -> Any:
    module, name = KERNELS[key]
    path = f"{KERNEL_ROOT}.{module}"
    try:
        return getattr(importlib.import_module(path), name)
    except (ImportError, AttributeError) as err:
        raise KernelUnavailable(f"The Kimi-K3 prefill kernel {path}.{name} is not available ({err!r}). The kernels "
                                f"are vendored under {KERNEL_ROOT}; run from a checkout that has them, or use the "
                                "torch backend.") from err


def prepare() -> None:
    """Import every kernel of ``KERNELS`` and build the FFN trace shims; call outside any traced forward.

    Raises:
        KernelUnavailable: A kernel module does not import, or has no such entry.
    """
    if len(_LOADED) == len(KERNELS) and _SHIMS:
        return
    for key in KERNELS:
        if key not in _LOADED:
            _LOADED[key] = _load(key)
    _SHIMS.update(_build_ffn_shims())


def import_kernel(key: str) -> Any:
    """The entry of kernel ``key`` (``KERNELS``); outside a trace, the first call runs ``prepare``.

    Raises:
        KernelUnavailable: A kernel does not import, or a traced forward asks for one before ``prepare`` ran.
    """
    kernel = _LOADED.get(key)
    if kernel is not None:
        return kernel
    if torch.compiler.is_compiling():
        raise KernelUnavailable(f"The Kimi-K3 prefill kernel {key} was not loaded before the trace: call "
                                "kernel_imports.prepare() when the kernel backend is built.")
    prepare()
    return _LOADED[key]


def kernels_available(key: str = "attn_res_cte") -> bool:
    """True when the module of kernel ``key`` imports."""
    try:
        importlib.import_module(f"{KERNEL_ROOT}.{KERNELS[key][0]}")
    except ImportError:
        return False
    return True


# attention and AttnRes kernels (the callers build their own shims)


def attn_res_cte() -> Any:
    """``attn_res_cte(bank [T, C, H], stream [T, H], score_w [H] fp32, eps, *, bank_scores [T, S] fp32,
    ocp_fp8_sinks)`` -> ``[T, H]``: the AttnRes mixture of a prefill row block."""
    return import_kernel("attn_res_cte")


def attn_res_bank_scores() -> Any:
    """``attn_res_bank_scores(stream [T, H], score_ws [L, H] fp32, eps, bank=[T, C, H])`` -> ``[L, T, C + 1]`` fp32:
    the bank-score table of a stage (the stage input scored as the last column)."""
    return import_kernel("attn_res_bank_scores")


def kda_block_prefill() -> Any:
    """The KDA block kernel: gather, projections, short convs, scan, output norm and gate, o_proj, reduce-scatter."""
    return import_kernel("kda_block")


def kda_front_mx_packer() -> tuple[Any, Any]:
    """The host MXFP8 quantizer and packer of the KDA block front weights: ``(mx_quantize, pack_from_mx)``."""
    return import_kernel("kda_front_mx_quantize"), import_kernel("kda_front_mx_pack")


def mla_flash_latent() -> Any:
    """The shared-key MLA flash kernel of one query tile: cache stream plus own keys, one softmax."""
    return import_kernel("mla_flash_latent")


def mla_dcp_merge() -> Any:
    """The cross-rank merge of feature-major MLA flash partials (global row max and sum, weighted partial)."""
    return import_kernel("mla_dcp_merge")


def rows_write_after() -> Any:
    """The indirect row write of a flat cache pool, ordered after an ``after`` operand."""
    return import_kernel("rows_write_after")


def page_read() -> Any:
    """The read of one run of a flat cache pool's page at a runtime page index (one DMA)."""
    return import_kernel("page_read")


def page_write() -> Any:
    """The in-place write of one run of a flat cache pool's page at a runtime page index (one DMA)."""
    return import_kernel("page_write")


# FFN kernels: MX linears, token map, blockwise routed experts


def _build_ffn_shims() -> dict[str, Any]:
    """The trace shims of the MX linears, the token map and the blockwise MoE (without and with the two order
    operands).

    Static options of the blockwise MoE in the serving form: the dynamic block loop (``n_static_blocks =
    n_dynamic_blocks = -1``), accumulating output, expert weight reuse across blocks of one expert, no sink row (pad
    slots hold -1; the kernel skips their affinity and output DMAs), and the down-input quantize issued before the
    next block's hidden quantize.
    """
    import nki
    import nki.language as nl

    mx_kernel, head_kernel, runs_kernel = _LOADED["mx_linear"], _LOADED["mx_head_linear"], _LOADED["moe_token_runs"]
    bwmm, act, skip_mode = _LOADED["bwmm"], _LOADED["bwmm_act_types"], _LOADED["bwmm_skip_mode"]

    @nki.jit(mode="trace")
    def k3_prefill_mx_linear(x, w_mx, w_scale, order: str = "hT_quad", feature_major_x: bool = False):
        return mx_kernel(x, w_mx, w_scale, order=order, feature_major_x=feature_major_x)

    @nki.jit(mode="trace")
    def k3_prefill_mx_head_linear(x, w_mx, w_scale):
        return head_kernel(x, w_mx, w_scale)

    @nki.jit(mode="trace")
    def k3_moe_token_runs(sel, start, num_slots: int):
        return runs_kernel(sel, start, num_slots)

    def call(hidden, affinities, token_position_to_id, block_to_expert, conditions, gate_up_w, down_w, gate_up_s,
             down_s, block_size, beta, linear_beta, top_k, ep_degree):
        return bwmm(hidden, affinities, gate_up_w.view(nl.float4_e2m1fn_x4), down_w.view(nl.float4_e2m1fn_x4),
                    token_position_to_id, block_to_expert, conditions=conditions, gate_up_proj_scale=gate_up_s,
                    down_proj_scale=down_s, block_size=block_size, n_static_blocks=-1, n_dynamic_blocks=-1,
                    top_k=top_k, ep_degree=ep_degree, activation_function=act.SiTU, situ_beta=beta,
                    situ_linear_beta=linear_beta, is_tensor_update_accumulating=True, reuse_expert_weights=True,
                    skip_dma=skip_mode(True, False), remap_padded_slots=True, vector_order_down_first=True)

    @nki.jit(mode="trace")
    def k3_bwmm(hidden, affinities, token_position_to_id, block_to_expert, conditions, gate_up_w, down_w, gate_up_s,
                down_s, block_size: int, beta: float, linear_beta: float, top_k: int, ep_degree: int):
        return call(hidden, affinities, token_position_to_id, block_to_expert, conditions, gate_up_w, down_w,
                    gate_up_s, down_s, block_size, beta, linear_beta, top_k, ep_degree)

    # order_first and order_after are operands the kernel does not read: the compiler emits the producers of a call's
    # operands first, in operand order, so order_first's producer runs before hidden's and order_after's after the
    # routing inputs'.
    @nki.jit(mode="trace")
    def k3_bwmm_ordered(order_first, hidden, affinities, token_position_to_id, block_to_expert, conditions, gate_up_w,
                        down_w, gate_up_s, down_s, order_after, block_size: int, beta: float, linear_beta: float,
                        top_k: int, ep_degree: int):
        return call(hidden, affinities, token_position_to_id, block_to_expert, conditions, gate_up_w, down_w,
                    gate_up_s, down_s, block_size, beta, linear_beta, top_k, ep_degree)

    return {"mx_linear": k3_prefill_mx_linear, "mx_head_linear": k3_prefill_mx_head_linear,
            "moe_token_runs": k3_moe_token_runs, "bwmm": k3_bwmm, "bwmm_ordered": k3_bwmm_ordered}


def _shim(name: str) -> Any:
    if not _SHIMS:
        import_kernel("mx_linear")
    return _SHIMS[name]


def mx_linear(x: Tensor, layer, feature_major: bool = False) -> Tensor:
    """The ``MxLinear.linear_fn`` of the device: ``x [M, in]`` (or ``[in, M]`` feature-major) -> ``[M, out]`` bf16.

    The kernel quantizes ``x`` to MXFP8 in the groups of ``layer.order`` and multiplies it by ``layer.w_mx`` /
    ``layer.w_scale`` with an fp32 sum. It takes 1 to 8192 rows.
    """
    from sglang_neuron.models.kimi_k3.prefill.mx_linear import MX_LINEAR_MAX_ROWS

    width, rows = (x.shape[0], x.shape[1]) if feature_major else (x.shape[-1], x.numel() // x.shape[-1])
    if x.dim() != 2 or width != layer.in_features or not 1 <= rows <= MX_LINEAR_MAX_ROWS:
        raise ValueError(f"{layer.name}: the MX linear input has shape {tuple(x.shape)}; the kernel takes "
                         f"{'[in, M]' if feature_major else '[M, in]'} with in = {layer.in_features} and "
                         f"1 <= M <= {MX_LINEAR_MAX_ROWS}.")
    return wrap_nki(_shim("mx_linear"))[LNC](x=x.to(torch.bfloat16).contiguous(), w_mx=layer.w_mx,
                                              w_scale=layer.w_scale, order=layer.order,
                                              feature_major_x=bool(feature_major))


def mx_head_linear(x_t: Tensor, layer) -> Tensor:
    """The ``MxHeadLinear.linear_fn`` of the device: ``x_t [G, K, M]`` -> ``[G, N, M]`` bf16, ``W[h]^T Q(x_t[h])``."""
    if x_t.dim() != 3 or tuple(x_t.shape[:2]) != (layer.heads, layer.in_features) or layer.in_features > 128 \
            or layer.out_features % 128:
        raise ValueError(f"{layer.name}: the MX head linear input has shape {tuple(x_t.shape)}; the kernel takes "
                         f"[{layer.heads}, {layer.in_features}, M] with K <= 128 and an output width that is a "
                         "multiple of 128.")
    return wrap_nki(_shim("mx_head_linear"))[LNC](x=x_t.to(torch.bfloat16).contiguous(), w_mx=layer.w_mx,
                                                   w_scale=layer.w_scale)


def moe_token_runs(sel: Tensor, start: Tensor, num_slots: int) -> Tensor:
    """Write the tokens that pick local expert ``e`` to the slots from ``start[e]``, in token order.

    Args:
        sel: ``[T, E]`` 1.0 where token ``t`` picks local expert ``e`` (``T % 128 == 0``, ``T <= 16384``,
            ``E <= 32``).
        start: ``[E]`` the first slot of each expert, a multiple of 128.
        num_slots: The slot count.

    Returns:
        ``[num_slots]`` int32; the slots outside the runs are not defined.
    """
    tokens, experts = sel.shape
    if tokens % 128 or tokens > TOKEN_RUNS_MAX_TOKENS or experts > TOKEN_RUNS_MAX_EXPERTS:
        raise ValueError(f"The pick mask has shape {tuple(sel.shape)}; the token-run kernel needs a multiple of 128 "
                         f"rows (at most {TOKEN_RUNS_MAX_TOKENS}) and at most {TOKEN_RUNS_MAX_EXPERTS} experts.")
    out = wrap_nki(_shim("moe_token_runs"))[LNC](sel=sel.to(torch.bfloat16).contiguous(),
                                                 start=start.reshape(1, -1).to(torch.float32),
                                                 num_slots=int(num_slots))
    return out.reshape(num_slots)


def bwmm_mx(hidden: Tensor, affinities: Tensor, token_position_to_id: Tensor, block_to_expert: Tensor,
            conditions: Tensor, gate_up_w: Tensor, down_w: Tensor, gate_up_s: Tensor, down_s: Tensor, *,
            block_size: int, beta: float, linear_beta: float, top_k: int, ep_degree: int,
            order_first: Tensor | None = None, order_after: Tensor | None = None) -> Tensor:
    """Run the blockwise MXFP4 routed experts; return the accumulating output ``[2, T, D]`` (plane 0 at one core).

    Args:
        hidden: ``[T, D]`` bf16 latent in the kernel column order (``fp4_pack.permute_hidden_bwmm``).
        affinities: ``[T E_L, 1]`` fp32 routing weight of token ``t`` for local expert ``e`` at row ``t E_L + e``.
        token_position_to_id: ``[B N]`` int32 token of each block slot, -1 for a pad slot.
        block_to_expert: ``[B, 1]`` int32 local expert of each block.
        conditions: ``[B + 2]`` int32, 1 for the active blocks (a prefix), 0 after.
        gate_up_w, down_w, gate_up_s, down_s: The packed experts (``fp4_pack``), expert-major.
        order_first, order_after: Operands that order the producers (both or neither).
    """
    static = dict(block_size=int(block_size), beta=float(beta), linear_beta=float(linear_beta), top_k=int(top_k),
                  ep_degree=int(ep_degree))
    operands = dict(hidden=hidden, affinities=affinities, token_position_to_id=token_position_to_id,
                    block_to_expert=block_to_expert, conditions=conditions, gate_up_w=gate_up_w, down_w=down_w,
                    gate_up_s=gate_up_s, down_s=down_s)
    if order_first is None and order_after is None:
        return wrap_nki(_shim("bwmm"))[LNC](**operands, **static)
    if order_first is None or order_after is None:
        raise ValueError("The blockwise experts take both order operands or neither.")
    return wrap_nki(_shim("bwmm_ordered"))[LNC](order_first=order_first, order_after=order_after, **operands,
                                                **static)
