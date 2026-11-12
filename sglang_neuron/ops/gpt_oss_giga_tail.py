# SPDX-License-Identifier: Apache-2.0
"""The GPT-OSS giga-kernel decode's fused sampling tail: final RMSNorm, LM head, exchange and greedy pick in one NKI
kernel call.

``gpt_oss_tail_chunked_megakernel`` (nkilib ``experimental.gpt_oss_tail``, from the image's kernel library) takes the
post-stack rows of the whole global batch on every rank, ``[gbs, H]``, and this rank's vocabulary shard of the LM head
as an ``[H, V_s]`` bf16 C-contiguous matrix. It norms the rows (the MXFP4 convention: variance ``sum(x^2) / H_u`` over
the interleaved hidden layout, padding lanes zeroed through a ``[4, H / 4]`` view), multiplies them by the shard, and
exchanges the bf16 logits with all-to-alls that turn the vocabulary-parallel ``[gbs, V_s]`` into the batch-parallel
``[gbs / world, V]``. It takes the top 256 of each of its rows in one sorted, chunked top-k and samples from them with
those rows' ``[top_k, top_p, temperature]``.

Exchange. An all-to-all replica group lies within one device or holds exactly one core per device: a group of several
cores on each of several devices, such as a single hop over all 128 ranks, runs on the runtime's relayed
latency-optimized algorithm, which corrupts under rank skew on runtime 2.35 (``sglang_neuron/nki/ep_all_to_all.py``).
The exchange therefore runs as two hops, ``HOPS = (8, 16)``: hop 0 over
the 8 contiguous ranks of one device (LNC1), hop 1 over the 16 ranks one per device that share a core index
(``hop_meshes``, the kernel module's ``compose_hops``). Each group is rank-ordered, so a column of an exchanged row is a
global token id.

Rows. After the two hops rank ``r = q * 8 + s`` holds global rows ``[s * 16 R + q * R, ... + R)``, ``R = gbs / world``
(the kernel module's ``owned_row_block``), not its own block. The tokens are gathered over the world in rank order,
which places that block at position ``r``, and ``batch_order`` undoes the mixed-radix map with a static permute of the
hop axes (no index table). The result is every row's token in the gathered rows' order, from which the graph keeps
its DP group's rows (``groups.shard_select`` over the DP group). No index depends on the rank, so one NEFF serves every
rank.

Greedy. The device sampler here is greedy only, so every row takes ``[top_k=1, top_p=1, temperature=0]``: a temperature
below the kernel's 1e-5 marks a row greedy, and a greedy row takes slot 0 of the descending top-k, the row's maximum
over the full vocabulary. The values compared are the bf16 logits the exchange carries. They can differ from the
per-layer decode's bf16 LM-head output in the last place (the kernel applies the norm's scale after its GEMM), so a row
whose two highest logits are that close can pick another token than the device sampler
(``sampling.greedy_sharded_logits``); among bf16-equal maxima the kernel returns the one the hardware sort puts first,
where the device sampler returns the smallest token id.

Toolchain. The sorted top-k needs both the nki ``nisa.topk(sorted=...)`` signature and a BIR field of neuronx-cc, which
move independently (the kernel library's ``chunked_topk``); ``refusal`` checks both, the compiler by its installed
version against the release the kernel is validated with.
"""
from importlib import metadata

import torch
from torch import Tensor

from sglang_neuron.backend.stack import wrap_nki

try:
    import nki
    from nkilib.experimental.gpt_oss_tail import gpt_oss_tail_chunked_megakernel as _tail_kernel
    from nkilib.experimental.topk.chunked_topk import _HW_SORT_AVAILABLE
except Exception as _err:  # noqa: BLE001 -- an image without the tail kernel; the giga decode refuses the launch
    _tail_kernel = None
    _HW_SORT_AVAILABLE = False
    _IMPORT_ERROR = _err
else:
    _IMPORT_ERROR = None

# The one geometry the kernel serves here: GPT-OSS-120B's vocabulary over 128 LNC1 ranks (V_s = 1571), the padded MXFP4
# hidden size with its unpadded width.
VOCAB = 201088
WORLD = 128
HIDDEN = 3072
HIDDEN_ACTUAL = 2880
# candidates each row keeps (the kernel's chunked top-k width, the value it is built and validated at)
TOP_K = 256
# per-row sampling parameters of a greedy row: top_k, top_p, temperature (below the kernel's 1e-5: greedy, slot 0)
GREEDY_PARAMS = (1.0, 1.0, 0.0)
# the exchange's hops: the 8 cores of one device at LNC1, then one core of each of the 16 devices
HOPS = (8, 16)
# the oldest neuronx-cc release the sorted top-k is validated with: the setup guide's. The kernel library names the BIR
# field's first build by a pre-release version (2.0.280935), which public releases (2.N.build.0) all sort above.
SORTED_TOPK_MIN_COMPILER = "2.28.4568.0"
# the kernel's outputs with emit_logits=False and sampling parameters: (topk_values, topk_indices, tokens)
KERNEL_OUTPUTS = 3


def available() -> bool:
    """Whether the image's kernel library ships the tail kernel with the hardware-sorted top-k it requires."""
    return _tail_kernel is not None and bool(_HW_SORT_AVAILABLE)


def compiler_version() -> "str | None":
    """The installed neuronx-cc's version, or None where it is not installed."""
    try:
        return metadata.version("neuronx-cc")
    except metadata.PackageNotFoundError:
        return None


def compiler_refusal(version: "str | None") -> "str | None":
    """Why this neuronx-cc cannot compile the kernel's sorted top-k, or None (PEP 440 order against
    SORTED_TOPK_MIN_COMPILER; an unreadable version is refused)."""
    from packaging.version import InvalidVersion, Version

    try:
        if version is not None and Version(version) >= Version(SORTED_TOPK_MIN_COMPILER):
            return None
    except InvalidVersion:
        pass
    return (f"the fused tail's sorted top-k is validated with neuronx-cc {SORTED_TOPK_MIN_COMPILER} or later (the "
            f"release whose BIR carries its sorted-top-k field); the installed neuronx-cc is {version!r}")


def world_mesh(world: int) -> tuple:
    """A world partition of one cell of every rank, in rank order (the LM-head group the hops are composed over)."""
    return (tuple(range(world)),)


def hop_meshes(world: int, hops: tuple = HOPS) -> tuple:
    """One replica-group mesh per all-to-all hop (the kernel module's ``compose_hops``): hop ``h`` groups the ranks
    that agree on every other hop's coordinate (rank ``r = sum_h c_h * prod(hops[:h])``), each group rank-ordered."""
    size = 1
    for hop in hops:
        size *= hop
    if size != world:
        raise ValueError(f"the hop sizes {tuple(hops)} multiply to {size}, not the world's {world} ranks")
    meshes, stride = [], 1
    for hop in hops:
        cells = [tuple(base + i * stride for i in range(hop)) for base in range(world) if (base // stride) % hop == 0]
        meshes.append(tuple(sorted(cells)))
        stride *= hop
    return tuple(meshes)


def batch_order(gathered: Tensor, hops: tuple) -> Tensor:
    """A rank-ordered world gather of the kernel's per-rank rows, ``[world * R, ...]``, in batch order. Rank ``r``'s
    digits ``c_h`` (``r = sum_h c_h * prod(hops[:h])``) place its block at batch row ``sum_h c_h * gbs / prod(hops[:h
    + 1])`` (``owned_row_block``): the gathered axis read as ``[hops[-1], ..., hops[0], R]`` is batch order with its
    hop axes reversed. A static permute, the same on every rank; the identity for one hop."""
    n = len(hops)
    if n == 1:
        return gathered
    rest = gathered.shape[1:]
    view = gathered.reshape(*reversed(hops), -1, *rest)
    order = tuple(reversed(range(n))) + tuple(range(n, view.ndim))
    return view.permute(order).reshape(gathered.shape)


def refusal(*, vocab: int, world: int, lnc: int, hidden: int, hidden_actual: int, per_rank: int, mesh: tuple,
            compiler: "str | None") -> "str | None":
    """Why the fused tail cannot serve this launch, or None. Every input is a structural fact of the launch, the same
    on every rank."""
    if not available():
        if _tail_kernel is None:
            return f"the image's kernel library has no nkilib.experimental.gpt_oss_tail: {_IMPORT_ERROR!r}"
        return ("this image's nki.isa.topk has no sorted= parameter, which the fused tail's sorted top-k needs "
                "(nkilib.experimental.topk.chunked_topk._HW_SORT_AVAILABLE)")
    reason = compiler_refusal(compiler)
    if reason is not None:
        return reason
    if lnc != 1:
        return f"the tail kernel runs at one logical core per physical core (NEURON_LOGICAL_NC_CONFIG=1), not {lnc}"
    if vocab != VOCAB or world != WORLD or per_rank * world != vocab:
        return (f"the tail kernel serves a vocabulary of {VOCAB} over {WORLD} ranks; got {vocab} over {world} "
                f"({per_rank} per rank)")
    if (hidden, hidden_actual) != (HIDDEN, HIDDEN_ACTUAL):
        return f"the tail kernel serves hidden {HIDDEN} ({HIDDEN_ACTUAL} unpadded); got {hidden} ({hidden_actual})"
    if tuple(mesh) != world_mesh(world):
        return (f"the LM-head group must be the ranks 0..{world - 1} in rank order (the exchange's hop groups are "
                f"composed over it); its partition is {mesh}")
    return None


def require(**launch) -> None:
    """Refuse the launch where the fused tail cannot serve it (``refusal``)."""
    reason = refusal(**launch)
    if reason is not None:
        raise ValueError(f"the GPT-OSS giga-kernel decode's fused tail: {reason}")


def lm_head_mirror(weight: Tensor) -> Tensor:
    """This rank's LM-head shard ``[V_s, H]`` as the kernel reads it: ``[H, V_s]`` bf16, C-contiguous, copied on the
    host (a transposed view keeps the shard's strides, and the kernel's weight DMA is built from strides). The hidden
    axis keeps the shard's shuffled order, the order of the rows and of the norm weight."""
    if weight.ndim != 2:
        raise ValueError(f"the LM-head shard is [V_s, H]; got {tuple(weight.shape)}")
    mirror = weight.detach().to("cpu").t().contiguous().to(torch.bfloat16)
    if not mirror.is_contiguous() or mirror.data_ptr() == weight.data_ptr():
        raise RuntimeError("the [H, V_s] LM-head mirror must be its own C-contiguous copy")
    return mirror


def greedy_params(rows: int, device) -> Tensor:
    """``[rows, 3]`` float32 sampling parameters of greedy rows (GREEDY_PARAMS), built in the graph (a factory op per
    column, no host constant)."""
    top_k, top_p, temperature = GREEDY_PARAMS
    return torch.cat([torch.full((rows, 1), value, dtype=torch.float32, device=device)
                      for value in (top_k, top_p, temperature)], dim=-1)


if _tail_kernel is not None:

    @nki.jit
    def _tail_jit(hidden_states, gamma, lm_head_weight_t, sampling_params, hop_meshes: tuple = (), eps: float = 1e-5,
                  hidden_actual: int = 0, k: int = TOP_K):
        """Rebuild the hops' replica groups inside the trace (as the giga-kernel's wrapper does; NKI's tracer rejects
        comprehensions, hence the loops) and run the tail with no logits output."""
        meshes = []
        for mesh in hop_meshes:
            cells = []
            for cell in mesh:
                cells.append(list(cell))
            meshes.append(cells)
        return _tail_kernel(
            hidden_states,
            gamma,
            lm_head_weight_t,
            eps=eps,
            hidden_actual=hidden_actual,
            zero_interleaved_pad=True,
            k=k,
            sampling_params=sampling_params,
            hop_meshes=meshes,
            emit_logits=False,
            deferred_rms=True,
            skip_nan_fold=True,
        )
else:
    _tail_jit = None


def owned_tokens(rows: Tensor, *, gamma: Tensor, lm_head_weight_t: Tensor, meshes: tuple, eps: float,
                 hidden_actual: int) -> Tensor:
    """This rank's greedy tokens, int32 ``[gbs / world]``: the rows of the ``[gbs, H]`` pre-norm ``rows`` the exchange
    over ``meshes`` gives this rank (``owned_row_block``)."""
    if _tail_jit is None:
        raise RuntimeError(f"the GPT-OSS fused tail kernel does not import here: {_IMPORT_ERROR!r}")
    world = 1
    for mesh in meshes:
        world *= len(mesh[0])
    gbs = rows.shape[0]
    if gbs % world:
        raise ValueError(f"the fused tail shards the global batch's rows over the world: {gbs} rows, {world} ranks")
    owned = gbs // world
    out = wrap_nki(_tail_jit)[1](
        rows.to(torch.bfloat16),
        gamma,
        lm_head_weight_t,
        greedy_params(owned, rows.device),
        hop_meshes=meshes,
        eps=eps,
        hidden_actual=hidden_actual,
        k=TOP_K,
    )
    # checked at trace time: another arity (debug taps, a logits output) or dtype would turn another output into ids
    if not isinstance(out, (tuple, list)) or len(out) != KERNEL_OUTPUTS:
        raise RuntimeError(f"gpt_oss_tail_chunked_megakernel returned {type(out).__name__} of "
                           f"{len(out) if isinstance(out, (tuple, list)) else 1} outputs, not its {KERNEL_OUTPUTS} "
                           "(topk_values, topk_indices, tokens)")
    tokens = out[-1]
    if tokens.dtype != torch.uint32 or tuple(tokens.shape) != (owned, 1):
        raise RuntimeError(f"gpt_oss_tail_chunked_megakernel returned tokens {tokens.dtype} "
                           f"{tuple(tokens.shape)}, not uint32 ({owned}, 1)")
    return tokens[:, 0].to(torch.int32)


def batch_tokens(rows: Tensor, *, group, gamma: Tensor, lm_head_weight_t: Tensor, hops: tuple, eps: float,
                 hidden_actual: int) -> Tensor:
    """Every row's greedy token, int32 ``[gbs]`` in the order of ``rows``: each rank's own rows (``owned_tokens``)
    gathered over the world ``group`` in rank order and put back in batch order (``batch_order``). The ids travel as
    float32, exact below 2**24, the form the device sampler's gathers carry them in."""
    from sglang_neuron.parallel import collectives as neuron_collectives

    own = owned_tokens(rows, gamma=gamma, lm_head_weight_t=lm_head_weight_t, meshes=hop_meshes(group.world_size, hops),
                       eps=eps, hidden_actual=hidden_actual)
    gathered = neuron_collectives.all_gather(group, own.to(torch.float32), dim=0)
    return batch_order(gathered, hops).to(torch.int32)
