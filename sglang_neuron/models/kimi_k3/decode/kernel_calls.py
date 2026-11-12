# SPDX-License-Identifier: Apache-2.0
"""The NKI wrappers and calls of the Kimi-K3 decode programs: the verify, the drafter, the full-vocab sampler and the
KDA ``exp(A_log)`` init.

A collective NKI kernel takes its groups as ``ReplicaGroup`` objects, which torch cannot pass across the custom-call
boundary. Each kernel therefore runs inside an ``@nki.jit`` wrapper that takes the groups as rank tuples
(``layout.DecodeLayout.rank_groups``), builds the ``ReplicaGroup`` of each, and calls the kernel with the arguments
it was given. An option that is off (None, False, 0 or an empty tuple) gives the kernel no argument, so the kernel
default runs; the dict options travel as sorted ``(key, value)`` pairs. A traced graph calls a wrapper through
``backend.stack.wrap_nki`` at one logical core (``[1]``).

``strict_wrap_nki`` adds a bind check to the launch: ``wrap_nki`` reads only the parameter names of the wrapper and
drops a keyword the wrapper does not take, so the check refuses a call that does not bind to the wrapper signature.
Under ``torch.compile`` the check runs at trace time only. Each wrapper takes a ``kernel_digest`` keyword that the
trace does not read: it puts a hash of the kernel sources (``kernel_digests``) into the NKI cache key.

``prepare()`` imports the kernels (``kernel_imports.prepare``), builds the wrappers and computes the digests; call it
outside any traced forward. The ``VERIFY_*`` and ``DRAFT_*`` tables hold the kernel keywords of the two served forms
of the 2-server decode (``k3``: Q 4, B 4; ``k7b2``: Q 8, B 2; 16 verify rows a replica each).
"""

from __future__ import annotations

import ast
import hashlib
import inspect
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor

from sglang_neuron.backend import stack
from sglang_neuron.models.kimi_k3.decode import kernel_imports
from sglang_neuron.models.kimi_k3.decode.layout import DP8_LAYOUT, DecodeLayout

# --- shapes and argument tables of the entries

MAX_POOLS = 24
"""Page pool arguments of the verify entry: the 93 layers use 24 pools."""

WEIGHT_ARGUMENTS = (
    "norm_weights",
    "attn_res_weights",
    "kda_w_in",
    "kda_w_out",
    "kda_w_small",
    "kda_params",
    "mla_in_proj",
    "mla_q_a_norm",
    "mla_q_b_proj",
    "mla_kv_a_norm",
    "mla_kv_b_proj",
    "mla_o_proj",
    "moe_in_proj",
    "moe_out_proj",
    "moe_vectors",
    "expert_gate_up_weights",
    "expert_gate_up_weights_scale",
    "expert_down_weights",
    "expert_down_weights_scale",
    "dense_gate_up",
    "dense_down",
)
"""The weight arguments of the verify entry, in its positional order (after ``stream``)."""

TAIL_ARGUMENTS = ("final_attn_res_weight", "final_norm_weight", "lm_head_weight")
"""The tail weights of the verify entry (final AttnRes, final norm, lm_head shard)."""

MX_SCALE_ARGUMENTS = (
    "kda_w_in_scale",
    "mla_in_proj_scale",
    "mla_q_b_proj_scale",
    "mla_kv_b_proj_scale",
    "mla_o_proj_scale",
    "moe_in_proj_scale",
    "kda_w_out_scale",
    "moe_out_proj_scale",
)
"""The E8M0 scale stacks of the MXFP8 weights; each goes by keyword, only when its stack is given."""

ROUTER_STACK = "moe_router_weight"
"""The FP16 router stack; with it the verify takes ``router_dtype="fp16"``."""

VERIFY_PAIR_ARGUMENTS = (
    "mla_k1_pack_options",
    "moe_glue",
    "moe_expert_prefetch",
    "moe_expert_prefetch_start",
)
"""The dict options of the verify; the wrapper takes each as sorted ``(key, value)`` pairs."""

VERIFY_MAX_ROWS = 32
"""The most verify rows (Q x B) of a replica in one verify call."""

MOE_FORMS = ("moe_2srv",)
"""The MoE forms of the 2-server verify."""

DRAFT_WEIGHT_ARGUMENTS = (
    "context_proj",
    "context_norm",
    "kv_a_proj",
    "kv_a_norm",
    "mla_in_proj",
    "mla_q_a_norm",
    "mla_q_b_proj",
    "mla_kv_a_norm",
    "mla_kv_b_proj",
    "mla_o_proj",
    "input_norm",
    "post_norm",
    "dense_gate_up",
    "dense_down",
    "final_norm",
    "markov_w1",
    "markov_w2",
)
"""The per-rank drafter weights, in the positional order of the drafter wrapper."""

DRAFT_LAYERS = 5
DRAFT_MASK_TOKEN = 163837
DRAFT_VERIFY_ROWS = 8
"""The verify rows of the drafter kernel for each sequence; a step of Q < 8 rows pads to 8."""
DRAFT_TAIL_ROWS = 7
"""The Markov greedy rows of the drafter tail (the drafted block); a k-draft step needs only k."""
DRAFT_MAX_SEQUENCES = 8
"""The most sequences of one drafter call; a larger batch runs in groups."""
DRAFT_MAX_LOOP_SEQUENCES = 4
"""The group size of a larger batch on the drafter loop path."""
MLA_PAGE_WIDTH = 128 * 576
"""Elements of one MLA kernel page: 128 rows of 576."""

FULL_VOCAB_MAX_ROWS = 128
"""The (row, vocab-chunk) layout of the sampler puts R rows times 128 // R chunks on the 128 partitions."""

KDA_PARAMS_COLUMNS_PER_HEAD = {"bf16": 2, "fp32": 14}
"""``kda_params`` columns of one head for each KDA conv dtype (``dt_bias``, ``A_log``, and the 3 x 4 fp32 conv taps);
the plain width adds the ``o_norm`` column: ``W = per_head * Hl + 1``."""

LNC = 1
"""The logical core count of each kernel call: the decode runs one core a rank."""

# --- the served forms of the 2-server decode

SERVED_FORMS = {"k3": (256, 4, 16), "k7b2": (256, 8, 16)}
"""form -> (world size, queries Q, verify rows of a replica Q x B)."""

FORM_BATCH = {"k3": 4, "k7b2": 2}
"""form -> sequences B of a replica."""

_MOE_GLUE = {
    "attn_res_sbuf_entries": 8,
    "last_expert_order": "up_first",
    "mx_load_order": "tile",
    "mx_scale_queue": "sp",
    "mx_scores_align": True,
    "norm_tail": "rows",
    "out_proj_early": True,
    "pe_keep_warm": {"matmuls": 6, "tile0": True},
    "routed_layout": "natural",
    "router_first": True,
    "router_norm_one_mul": True,
    "rows_queue": "act",
    "scale_rows_quadrants": True,
    "scale_rows_queue": "sync_hw",
    "selectors_once": True,
}

MOE_OUT_PROJ_FORM = "column"
"""The out projection of the MoE layers of every served form (the kernel's ``moe_out_proj_form``,
``tkg_moe_2srv.OUT_PROJ_FORMS``): ``column``. Each TP rank computes 224 hidden columns of its replica's rows from all
9,728 inputs in one FP32 accumulation, and a TP32 all-gather gives every rank the full rows. The ``reduced`` form
(the kernel default) adds 32 BF16 partials in a TP32 all-reduce instead: its verify takes longer in each k3 step.
The column form sums in another FP32 order, so its outputs do not have the bits of ``reduced``. The MoE weights follow
it (``weights_ffn.moe_layer``)."""

_VERIFY_COMMON = {
    "attn_res_ahead": True,
    "attn_res_ahead_min_entries": 9,
    "attn_res_ahead_mlp": True,
    "attn_res_bank_dma": "act",
    "attn_res_cached_bank_sums": True,
    "attn_res_early_sums": True,
    "attn_res_mlp_early_entries": 0,
    "attn_res_scalar_products": 6,
    "cc_pad256": True,
    "cc_source_pingpong": True,
    "ffn_weights_after_attn": True,
    "kda_out_ar_sbuf": False,
    "kda_slot_pages_first": True,
    "kda_slot_store_queues": 1,
    "kda_small_after_out_ar": False,
    "kda_state_loads_first": True,
    "kda_token_tiles": True,
    "kda_w_in_buffers": 8,
    "kda_w_in_prefetch": "after_ffn_done",
    "mla_cache_write_off_sp": True,
    "mla_critical_dma_priority": True,
    "mla_gemv_bands": True,
    "mla_k1_pack_options": {
        "fp8_pv_batched_exact": True,
        "fp8_pv_redo_rolled": True,
        "fp8_pv_rope_copy_vector": True,
        "no_if": True,
        "no_if_loops": True,
    },
    "mla_k1_packed": True,
    "mla_k1_rows": False,
    "mla_out_ar": True,
    "mla_p2_entry_act": True,
    "mla_pass_q_prefetch": 1,
    "mla_pass_setup_off_pool": True,
    "mla_q_gather_buffers": 2,
    "mla_state_bf16_o": True,
    "mla_step_items": True,
    "mla_tail_weights_after_rs": False,
    "mla_zeroed_pages": False,
    "moe_cross_expert_order": True,
    "moe_expert_prefetch": {"kda": 3, "kda_after_in_proj": 1, "mla": 3},
    "moe_expert_prefetch_start": {"mla": "front"},
    "moe_expert_queues": 1,
    "moe_gather": "dp_tp",
    "moe_latent_gather_mx": True,
    "moe_out_proj_form": MOE_OUT_PROJ_FORM,
    "moe_routed_own_rows": False,
    "moe_routed_reduce": "rs_ag",
    "moe_routed_stats_ag": False,
    "moe_router_rows_drop_selection": True,
    "moe_rows_latent_fp8": False,
    "moe_rows_load_split": 1,
    "moe_scale_buffers": None,
    "moe_transpose_block": True,
    "moe_transpose_lookahead": None,
    "moe_weight_buffers": 3,
    "mx_qk": "E",
    "norm_scalar_squares": 5,
}

VERIFY_LEVERS: dict[str, dict[str, Any]] = {
    "k3": {
        **_VERIFY_COMMON,
        "mla_fp8_pv": True,
        "mla_skip_stage_store": True,
        "mla_slot_rows_sbuf": True,
        "moe_glue": {**_MOE_GLUE, "dense_x_gather": "sharded"},
    },
    "k7b2": {**_VERIFY_COMMON, "moe_glue": dict(_MOE_GLUE)},
}
"""form -> the lever keywords of the verify call (``verify_kwargs`` of ``verify_block``): every lever with an explicit
value, so the program does not depend on the entry defaults. A None value gives the entry no argument."""

VERIFY_CALL: dict[str, dict[str, Any]] = {
    form: {
        "queries": SERVED_FORMS[form][1],
        "mla_mode": "k1",
        "mla_fp8_pages": True,
        "mla_loop_path": True,
        "lora_norm_eps": 1e-6,
        "kda_replay": "states",
        "kda_conv_dtype": "fp32",
        "kda_state_dtype": "bf16",
        "aux_layers": (2, 23, 47, 71, 89),
        "aux_tap": "prefix",
        "greedy": False,
        "return_logits": True,
        "return_aux": True,
        "moe_form": "moe_2srv",
        "routed_tp": 2,
        "live_rows": form == "k3",
    }
    for form in SERVED_FORMS
}
"""form -> the other keywords of the served verify call: the k1 MLA loop path over FP8 pages, the states replay, the
fp32 KDA conv, the drafter aux states, the sampled tail (logits) and the live-rows input (``live_rows``: k3 only). The
call takes no grammar mask: the decode serves no structured output."""

DRAFT_STAGE: dict[str, dict[str, Any]] = {
    "k3": {
        "tail_rows": 3,
        "context_done": False,
        "pe_moves": True,
        "merged_out_ag": True,
        "cc_pad256": True,
        "stack_rows": True,
        "pass_q_prefetch": 2,
        "q_gather_buffers": 2,
        "pass_tail_overlap": True,
        "seq_passes": True,
        "markov_w1_local": True,
        "prior_hoist": 3,
    },
    "k7b2": {
        "tail_rows": 7,
        "context_done": False,
        "pe_moves": True,
        "cc_pad256": True,
        "stack_rows": True,
        "q_call_one_pass": True,
        "markov_w1_local": True,
        "sample_unique_dst": True,
        "prior_hoist": 3,
    },
}
"""form -> the stage options of the drafter call (``draft_mla_step``) on the 1M table (256 local pages), window 49152,
CP32."""

DRAFT_CALL: dict[str, dict[str, Any]] = {
    form: {
        "window": 49152,
        "loop_path": False,
        "fp8_pages": True,
        "cp": 32,
        "window_read": False,
        "queries": SERVED_FORMS[form][1],
        "guard_indices": False,
        "embed_vocab": True,
        "sampled": True,
    }
    for form in SERVED_FORMS
}
"""form -> the other keywords of the served drafter call: the CP32 static path over FP8 rider pages, window 49152,
the vocab shard of the embedding in place of the block embedding (``embed_vocab``), the windowed rider (no index
guard) and the sampled drafts (``sampled``: ``sample_params`` and ``u_draft``)."""

SAMPLER_OPTIONS = {"skip_inactive_top_k": True, "cc_pad256": True}
"""The full-vocab sampler options of the served decode: ``skip_inactive_top_k`` on (the kernel skips its top-k search
on a step where no row sets top-k; same bits) and ``cc_pad256`` on (every all-reduce moves a multiple of 256 bytes;
same bits)."""


def served_form(world_size: int, queries: int, rows: int) -> str | None:
    """The served form of a verify of ``rows`` rows a replica at ``queries`` tokens a sequence, or None.

    Example:
        >>> served_form(256, 4, 16), served_form(256, 8, 16), served_form(256, 4, 8)
        ('k3', 'k7b2', None)
    """
    return next((name for name, shape in SERVED_FORMS.items() if shape == (world_size, queries, rows)), None)


# --- the strict bind check

_SIGNATURES: dict = {}
_BOUND: set = set()


def _signature(func: Any) -> inspect.Signature | None:
    """The signature of ``func``, read once, or None if it has none (then the call is not checked)."""
    try:
        return _SIGNATURES[func]
    except KeyError:
        pass
    except TypeError:
        return None
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        sig = None
    _SIGNATURES[func] = sig
    return sig


@torch._dynamo.assume_constant_result
def check_launch(func: Any, num_args: int, keywords: tuple) -> bool:
    """Raise ``TypeError`` when ``num_args`` positional arguments and the ``keywords`` do not bind to ``func``.

    Dynamo runs the function at trace time and keeps only its result (``assume_constant_result``), so the check adds no
    graph op. In eager mode each signature and call form is checked once.

    Example:
        >>> def kernel(a, b=1): ...
        >>> check_launch(kernel, 1, ("b",))
        True
    """
    key = (func, num_args, keywords)
    try:
        if key in _BOUND:
            return True
    except TypeError:
        key = None
    sig = _signature(func)
    if sig is None:
        return True
    try:
        sig.bind(*range(num_args), **dict.fromkeys(keywords))
    except TypeError as error:
        params = sig.parameters
        takes = {n for n, p in params.items() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
        var_kw = any(p.kind is p.VAR_KEYWORD for p in params.values())
        unknown = [] if var_kw else sorted(k for k in keywords if k not in takes)
        raise TypeError(f"The Kimi-K3 kernel launch of {getattr(func, '__qualname__', func)!r} does not bind ({error}). "
                        f"Keywords that the kernel does not take: {unknown}. wrap_nki would drop such an argument with "
                        "no error, so the option would do nothing.") from None
    if key is not None:
        _BOUND.add(key)
    return True


class StrictLaunch:
    """``[grid]`` and the call of ``wrap_nki(fn)``, behind the bind check."""

    def __init__(self, launch: Any, func: Any):
        self.launch = launch
        self.func = func

    def __getitem__(self, grid: Any) -> StrictLaunch:
        return StrictLaunch(self.launch[grid], self.func)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        check_launch(self.func, len(args), tuple(kwargs))
        return self.launch(*args, **kwargs)


def strict_wrap_nki(fn: Any) -> StrictLaunch:
    """``wrap_nki(fn)`` whose launches refuse a call that does not bind to the signature of ``fn`` (``fn.func`` of an
    ``nki.jit`` kernel, else ``fn``)."""
    func = fn.func if hasattr(fn, "func") else fn
    return StrictLaunch(stack.wrap_nki(fn), func)


# --- kernel source digests
# The NKI compile cache keys a kernel by its wrapper's source and arguments, not by the kernel files the wrapper calls
# into, and the compile-cache stamp (backend.compile_cache) guards torch-neuronx's NEFF caches only: each wrapper takes
# the digest of its kernels' sources (``kernel_digest``), so a kernel edit compiles the wrapper again.

SPEC_KERNEL_MODULES = (
    "verify_model",
    "draft_mla",
)
"""The modules (under the kernel package) whose import closure the drafter digest hashes."""

KDA_INIT_KERNEL_MODULES = ("tkg_kda",)
"""The modules whose import closure the KDA init digest hashes."""


def kernel_source_digest(paths: Iterable[Path], root: Path | None = None) -> str:
    """A 16-character hash of the given source files, in path order; each file adds its name (its path relative to
    ``root`` when given) and its bytes."""
    digest = hashlib.sha256()
    for path in sorted(Path(p) for p in paths):
        name = path.relative_to(root).as_posix() if root is not None else path.name
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _imported_names(source: str, package: str) -> list[str]:
    """The absolute module names that one file imports; ``from a import b`` gives ``a`` and ``a.b``."""
    names = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                parts = parts[: len(parts) - node.level + 1]
                base = ".".join(parts + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            names.append(base)
            names.extend(f"{base}.{alias.name}" for alias in node.names)
    return names


def _package_aliases(init: Path) -> dict[str, str]:
    """The module aliases (old -> new, relative to the package) that a package ``__init__.py`` installs with a call of
    ``__name__`` and a dict of names."""
    aliases = {}
    for node in ast.walk(ast.parse(init.read_text())):
        if not (isinstance(node, ast.Call) and len(node.args) == 2 and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "__name__" and isinstance(node.args[1], ast.Dict)):
            continue
        for key, value in zip(node.args[1].keys, node.args[1].values):
            if (isinstance(key, ast.Constant) and isinstance(value, ast.Constant) and isinstance(key.value, str)
                    and isinstance(value.value, str)):
                aliases[key.value] = value.value
    return aliases


def import_closure(modules: Iterable[str], root: Path, package: str = kernel_imports.KERNEL_ROOT) -> list[Path]:
    """The source files of the kernel package ``package`` (directory ``root``) that the given modules import,
    transitively: every import statement of each file (also inside functions), relative and absolute, the
    ``__init__.py`` of each package on the way, and an old module name that a package aliases gives the renamed file.

    Args:
        modules: Module names relative to ``package`` (for example ``draft_mla``).
        root: The directory of the package.
        package: The absolute name of the package.
    """
    aliases: dict[Path, dict[str, str]] = {}

    def file_of(name: str, depth: int = 0) -> Path | None:
        if name != package and not name.startswith(package + "."):
            return None
        parts = name[len(package) + 1:].split(".") if name != package else []
        base = root.joinpath(*parts)
        for candidate in (base / "__init__.py", base.with_suffix(".py")):
            if candidate.is_file():
                return candidate
        for split in range(len(parts) - 1, -1, -1):
            init = root.joinpath(*parts[:split], "__init__.py")
            if not init.is_file():
                continue
            if init not in aliases:
                aliases[init] = _package_aliases(init)
            new = aliases[init].get(".".join(parts[split:]))
            if new is not None and depth < 8:
                return file_of(".".join([package, *parts[:split], new]), depth + 1)
        return None

    def package_of(path: Path) -> str:
        parts = [package, *path.relative_to(root).with_suffix("").parts]
        return ".".join(parts[:-1])

    seen: set[Path] = set()
    todo = [path for path in (file_of(f"{package}.{m}") for m in modules) if path is not None]
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        parent = path.parent
        while parent != root.parent and (parent / "__init__.py").is_file():
            todo.append(parent / "__init__.py")
            parent = parent.parent
        source = path.read_text()
        names = _imported_names(source, package_of(path))
        if path.name == "__init__.py" and "pkgutil.iter_modules(__path__)" in source:
            names.extend(f"{package_of(path)}.{child.stem}" for child in path.parent.iterdir()
                         if child.suffix == ".py" or (child / "__init__.py").is_file())
        for name in names:
            found = file_of(name)
            if found is not None:
                todo.append(found)
    return sorted(seen)


def verify_kernel_sources(root: Path) -> list[Path]:
    """The sources the verify digest hashes: the modules at the root of the decode kernels, ``multinode`` and the
    routed-expert loop ``moe/routed``."""
    return [*root.glob("*.py"), *(root / "multinode").glob("*.py"), *(root / "moe" / "routed").rglob("*.py")]


def kernel_digests(root: Path | None = None, package: str = kernel_imports.KERNEL_ROOT) -> dict[str, str]:
    """The ``kernel_digest`` of each wrapper: ``verify`` (file names and bytes of ``verify_kernel_sources``), ``draft``
    and ``kda_a_exp_init`` (paths relative to ``root`` and bytes of the import closures of ``SPEC_KERNEL_MODULES`` and
    ``KDA_INIT_KERNEL_MODULES``)."""
    root = kernel_imports.kernel_root_dir() if root is None else Path(root)
    return {
        "verify": kernel_source_digest(verify_kernel_sources(root)),
        "draft": kernel_source_digest(import_closure(SPEC_KERNEL_MODULES, root, package), root),
        "kda_a_exp_init": kernel_source_digest(import_closure(KDA_INIT_KERNEL_MODULES, root, package), root),
    }


# --- the @nki.jit wrappers

_WRAPPERS: dict[str, Any] = {}
"""name -> ``@nki.jit`` wrapper, filled by ``prepare``."""
_LAUNCH: dict[str, Any] = {}
"""name -> the launch of a wrapper (``strict_wrap_nki`` / ``wrap_nki``), filled by ``prepare``."""
DIGESTS: dict[str, str] = {}
"""wrapper -> kernel source digest, filled by ``prepare``."""
SAMPLER_PARAMETERS: set[str] = set()
"""The parameter names of the full-vocab sampler kernel, filled by ``prepare`` (the token-major verify form needs
``token_major_queries``)."""


def _build_verify_wrapper(nki: Any, nl: Any, ReplicaGroup: Any, entry: Any) -> Any:
    @nki.jit
    def kimi_k3_verify_wrapper(
        stream: nl.ndarray,
        norm_weights: nl.ndarray,
        attn_res_weights: nl.ndarray,
        kda_w_in: nl.ndarray,
        kda_w_out: nl.ndarray,
        kda_w_small: nl.ndarray,
        kda_params: nl.ndarray,
        mla_in_proj: nl.ndarray,
        mla_q_a_norm: nl.ndarray,
        mla_q_b_proj: nl.ndarray,
        mla_kv_a_norm: nl.ndarray,
        mla_kv_b_proj: nl.ndarray,
        mla_o_proj: nl.ndarray,
        moe_in_proj: nl.ndarray,
        moe_out_proj: nl.ndarray,
        moe_vectors: nl.ndarray,
        expert_gate_up_weights: nl.ndarray,
        expert_gate_up_weights_scale: nl.ndarray,
        expert_down_weights: nl.ndarray,
        expert_down_weights_scale: nl.ndarray,
        dense_gate_up: nl.ndarray,
        dense_down: nl.ndarray,
        kda_state_page: nl.ndarray,
        kda_record_page: nl.ndarray,
        kda_replay_len: nl.ndarray,
        kda_state_position: nl.ndarray,
        mla_block_table: nl.ndarray,
        mla_slot: nl.ndarray,
        position_rows: nl.ndarray,
        rank_info: nl.ndarray,
        tp_ranks: tuple = (),
        dp_ranks: tuple = (),
        world_ranks: tuple = (),
        first_layer: int = 0,
        eps: float = 1e-5,
        queries: int = 4,
        dma_gates: bool = True,
        pool_of_layer: tuple = (),
        kv_pool_0: nl.ndarray = None,
        kv_pool_1: nl.ndarray = None,
        kv_pool_2: nl.ndarray = None,
        kv_pool_3: nl.ndarray = None,
        kv_pool_4: nl.ndarray = None,
        kv_pool_5: nl.ndarray = None,
        kv_pool_6: nl.ndarray = None,
        kv_pool_7: nl.ndarray = None,
        kv_pool_8: nl.ndarray = None,
        kv_pool_9: nl.ndarray = None,
        kv_pool_10: nl.ndarray = None,
        kv_pool_11: nl.ndarray = None,
        kv_pool_12: nl.ndarray = None,
        kv_pool_13: nl.ndarray = None,
        kv_pool_14: nl.ndarray = None,
        kv_pool_15: nl.ndarray = None,
        kv_pool_16: nl.ndarray = None,
        kv_pool_17: nl.ndarray = None,
        kv_pool_18: nl.ndarray = None,
        kv_pool_19: nl.ndarray = None,
        kv_pool_20: nl.ndarray = None,
        kv_pool_21: nl.ndarray = None,
        kv_pool_22: nl.ndarray = None,
        kv_pool_23: nl.ndarray = None,
        final_attn_res_weight: nl.ndarray = None,
        final_norm_weight: nl.ndarray = None,
        lm_head_weight: nl.ndarray = None,
        greedy: bool = False,
        return_logits: bool = False,
        return_aux: bool = False,
        aux_layers: tuple = (),
        aux_tap: str = "mixture",
        kernel_digest: str = "",
        world_size: int = 256,
        routed_tp: int = 2,
        moe_form: str = "moe_2srv",
        dp_size: int = 8,
        mla_mode: str = "k1",
        mla_fp8_pages: bool = None,
        mla_loop_path: bool = None,
        lora_norm_eps: float = 1e-5,
        kda_replay: str = "records",
        kda_state_slot_page: nl.ndarray = None,
        kda_write_state_slots: nl.ndarray = None,
        kda_conv_dtype: str = "bf16",
        kda_state_dtype: str = "bf16",
        attn_res_bank_in: nl.ndarray = None,
        return_attn_res_bank: bool = None,
        attn_res_fence_in: nl.ndarray = None,
        moe_routed_reduce: str = None,
        moe_out_proj_form: str = None,
        cc_source_pingpong: bool = None,
        routed_layers: tuple = None,
        moe_mx_tile: int = None,
        moe_routed_dynamic: bool = None,
        moe_rows_latent_fp8: bool = None,
        ffn_weights_after_attn: bool = None,
        moe_gather: str = "dp_tp",
        mla_out_ar: bool = None,
        mla_k1_rows: bool = None,
        mla_state_bf16_o: bool = None,
        attn_res_cached_bank_sums: bool = None,
        attn_res_early_sums: bool = None,
        moe_routed_own_rows: bool = False,
        norm_scalar_squares: int = None,
        attn_res_scalar_products: int = None,
        attn_res_bank_dma: str = None,
        moe_rows_load_split: int = None,
        moe_transpose_lookahead: int = None,
        kda_w_in_buffers: int = None,
        kda_w_in_prefetch: str = "off",
        kda_state_loads_first: bool = None,
        kda_slot_pages_first: bool = None,
        mla_k1_packed: bool = None,
        mla_k1_pack_options: tuple = None,
        attn_res_mlp_early_entries: int = None,
        moe_weight_buffers: int = None,
        moe_scale_buffers: int = None,
        moe_cross_expert_order: bool = None,
        moe_transpose_block: bool = None,
        moe_glue: tuple = None,
        moe_routed_stats_ag: bool = None,
        moe_router_rows_drop_selection: bool = None,
        moe_expert_queues: int = None,
        moe_expert_prefetch: tuple = None,
        moe_expert_prefetch_start: tuple = None,
        mla_zeroed_pages: bool = None,
        kda_w_in_scale: nl.ndarray = None,
        mla_in_proj_scale: nl.ndarray = None,
        mla_q_b_proj_scale: nl.ndarray = None,
        mla_kv_b_proj_scale: nl.ndarray = None,
        mla_o_proj_scale: nl.ndarray = None,
        router_dtype: str = "bf16",
        moe_router_weight: nl.ndarray = None,
        moe_in_proj_scale: nl.ndarray = None,
        kda_w_out_scale: nl.ndarray = None,
        moe_out_proj_scale: nl.ndarray = None,
        live_rows: nl.ndarray = None,
        kda_null_page: int = 0,
        kda_paired_pages: bool = None,
        kda_page_groups: tuple = (),
        kda_w_in_head_major: bool = False,
        pair_shared_cc_dsts: tuple = (),
        mla_critical_dma_priority: bool = None,
        mla_cache_write_off_sp: bool = None,
        mla_gemv_bands: bool = None,
        kda_token_tiles: bool = None,
        kda_slot_store_queues: int = None,
        moe_latent_gather_mx: bool = None,
        kda_out_ar_sbuf: bool = None,
        kda_small_after_out_ar: bool = None,
        mla_pass_q_prefetch: int = None,
        mla_q_gather_buffers: int = None,
        mla_tail_weights_after_rs: bool = None,
        mla_pass_setup_off_pool: bool = None,
        mla_slot_rows_sbuf: bool = None,
        mla_skip_stage_store: bool = None,
        mla_p2_entry_act: bool = None,
        mla_step_items: bool = None,
        mla_batched_cache_write: bool = None,
        mx_qk: str = None,
        mla_fp8_pv: bool = None,
        cc_pad256: bool = None,
        kda_chain: str = None,
        mla_qkv_parity: bool = None,
        attn_res_ahead: bool = None,
        attn_res_ahead_mlp: bool = None,
        kda_shared_state_ring: bool = None,
        attn_res_ahead_min_entries: int = None,
        mla_split: tuple = None,
        mla_split_stats: nl.ndarray = None,
        mla_split_own: nl.ndarray = None,
        mla_split_foreign: nl.ndarray = None,
        mla_split_table: nl.ndarray = None,
    ):
        """Build the replica groups from the rank tuples and run the 2-server verify entry."""
        tp_group = ReplicaGroup(tp_ranks)
        dp_group = ReplicaGroup(dp_ranks)
        world_group = ReplicaGroup(world_ranks)
        # A bool option is tri-state: None gives no argument (the entry default runs); True and False both go.
        flags = {}
        if aux_layers:
            flags["aux_layers"] = aux_layers
        if mla_fp8_pages is not None:
            flags["mla_fp8_pages"] = mla_fp8_pages
        if mla_loop_path is not None:
            flags["mla_loop_path"] = mla_loop_path
        if lora_norm_eps != 1e-5:
            flags["lora_norm_eps"] = lora_norm_eps
        if kda_replay != "records":
            flags["kda_replay"] = kda_replay
            flags["kda_state_slot_page"] = kda_state_slot_page
            flags["kda_write_state_slots"] = kda_write_state_slots
        if kda_conv_dtype != "bf16":
            flags["kda_conv_dtype"] = kda_conv_dtype
        if kda_state_dtype != "bf16":
            flags["kda_state_dtype"] = kda_state_dtype
        if attn_res_bank_in is not None:
            flags["attn_res_bank_in"] = attn_res_bank_in
            flags["attn_res_fence_in"] = attn_res_fence_in
        if return_attn_res_bank is not None:
            flags["return_attn_res_bank"] = return_attn_res_bank
        if moe_out_proj_form is not None:
            flags["moe_out_proj_form"] = moe_out_proj_form
        if routed_layers is not None:
            flags["routed_layers"] = routed_layers
        if moe_mx_tile is not None:
            flags["moe_mx_tile"] = moe_mx_tile
        if moe_routed_dynamic is not None:
            flags["moe_routed_dynamic"] = moe_routed_dynamic
        if cc_source_pingpong is not None:
            flags["cc_source_pingpong"] = cc_source_pingpong
        if moe_rows_latent_fp8 is not None:
            flags["moe_rows_latent_fp8"] = moe_rows_latent_fp8
        if moe_gather != "dp_tp":
            flags["moe_gather"] = moe_gather
        if mla_state_bf16_o is not None:
            flags["mla_state_bf16_o"] = mla_state_bf16_o
        if moe_latent_gather_mx is not None:
            flags["moe_latent_gather_mx"] = moe_latent_gather_mx
        if mla_qkv_parity is not None:
            flags["mla_qkv_parity"] = mla_qkv_parity
        for name, value in (
            ("attn_res_ahead", attn_res_ahead),
            ("attn_res_ahead_mlp", attn_res_ahead_mlp),
            ("kda_shared_state_ring", kda_shared_state_ring),
        ):
            if value is not None:
                flags[name] = value
        if attn_res_ahead_min_entries is not None:
            flags["attn_res_ahead_min_entries"] = attn_res_ahead_min_entries
        for name, value, default in (
            ("moe_routed_own_rows", moe_routed_own_rows, False),
            ("attn_res_bank_dma", attn_res_bank_dma, None),
            ("moe_transpose_lookahead", moe_transpose_lookahead, None),
            ("kda_w_in_prefetch", kda_w_in_prefetch, "off"),
        ):
            if value != default:
                flags[name] = value
        # The served-form levers: None keeps the entry default, and a given value goes, an off value too.
        for name, value in (
            ("moe_routed_reduce", moe_routed_reduce),
            ("ffn_weights_after_attn", ffn_weights_after_attn),
            ("mla_out_ar", mla_out_ar),
            ("mla_k1_rows", mla_k1_rows),
            ("attn_res_cached_bank_sums", attn_res_cached_bank_sums),
            ("attn_res_early_sums", attn_res_early_sums),
            ("norm_scalar_squares", norm_scalar_squares),
            ("attn_res_scalar_products", attn_res_scalar_products),
            ("moe_rows_load_split", moe_rows_load_split),
            ("kda_w_in_buffers", kda_w_in_buffers),
            ("kda_state_loads_first", kda_state_loads_first),
            ("kda_slot_pages_first", kda_slot_pages_first),
            ("mla_k1_packed", mla_k1_packed),
            ("attn_res_mlp_early_entries", attn_res_mlp_early_entries),
            ("moe_weight_buffers", moe_weight_buffers),
            ("moe_scale_buffers", moe_scale_buffers),
            ("moe_cross_expert_order", moe_cross_expert_order),
            ("moe_transpose_block", moe_transpose_block),
            ("moe_routed_stats_ag", moe_routed_stats_ag),
            ("moe_router_rows_drop_selection", moe_router_rows_drop_selection),
            ("moe_expert_queues", moe_expert_queues),
            ("mla_zeroed_pages", mla_zeroed_pages),
            ("kda_out_ar_sbuf", kda_out_ar_sbuf),
            ("kda_small_after_out_ar", kda_small_after_out_ar),
            ("mla_pass_q_prefetch", mla_pass_q_prefetch),
            ("mla_q_gather_buffers", mla_q_gather_buffers),
            ("mla_tail_weights_after_rs", mla_tail_weights_after_rs),
            ("mla_pass_setup_off_pool", mla_pass_setup_off_pool),
            ("mla_slot_rows_sbuf", mla_slot_rows_sbuf),
            ("mla_skip_stage_store", mla_skip_stage_store),
            ("mla_p2_entry_act", mla_p2_entry_act),
            ("mla_step_items", mla_step_items),
            ("mla_critical_dma_priority", mla_critical_dma_priority),
            ("mla_cache_write_off_sp", mla_cache_write_off_sp),
            ("mla_gemv_bands", mla_gemv_bands),
            ("kda_token_tiles", kda_token_tiles),
            ("kda_slot_store_queues", kda_slot_store_queues),
            ("mla_batched_cache_write", mla_batched_cache_write),
        ):
            if value is not None:
                flags[name] = value
        # The dict options come as (key, value) pairs.
        for name, pairs in (
            ("mla_k1_pack_options", mla_k1_pack_options),
            ("moe_glue", moe_glue),
            ("moe_expert_prefetch", moe_expert_prefetch),
            ("moe_expert_prefetch_start", moe_expert_prefetch_start),
        ):
            if pairs is not None:
                flags[name] = dict(pairs)
        if kda_w_in_scale is not None:
            flags["kda_w_in_scale"] = kda_w_in_scale
        if mla_in_proj_scale is not None:
            flags["mla_in_proj_scale"] = mla_in_proj_scale
        if mla_q_b_proj_scale is not None:
            flags["mla_q_b_proj_scale"] = mla_q_b_proj_scale
        if mla_kv_b_proj_scale is not None:
            flags["mla_kv_b_proj_scale"] = mla_kv_b_proj_scale
        if mla_o_proj_scale is not None:
            flags["mla_o_proj_scale"] = mla_o_proj_scale
        if router_dtype != "bf16":
            flags["router_dtype"] = router_dtype
            flags["moe_router_weight"] = moe_router_weight
        if moe_in_proj_scale is not None:
            flags["moe_in_proj_scale"] = moe_in_proj_scale
        if kda_w_out_scale is not None:
            flags["kda_w_out_scale"] = kda_w_out_scale
        if moe_out_proj_scale is not None:
            flags["moe_out_proj_scale"] = moe_out_proj_scale
        if live_rows is not None:
            flags["live_rows"] = live_rows
        if kda_null_page:
            flags["kda_null_page"] = kda_null_page
        if kda_paired_pages is not None:
            flags["kda_paired_pages"] = kda_paired_pages
        if pair_shared_cc_dsts:
            flags["pair_shared_cc_dsts"] = tuple(pair_shared_cc_dsts)
        if kda_page_groups:
            flags["kda_page_groups"] = tuple(kda_page_groups)
        # The head-major KDA input projection runs the head stages.
        if kda_w_in_head_major:
            flags["kda_w_in_head_major"] = True
            flags["kda_head_pipeline"] = "heads"
        if mx_qk is not None:
            flags["mx_qk"] = mx_qk
        if mla_fp8_pv is not None:
            flags["mla_fp8_pv"] = mla_fp8_pv
        if cc_pad256 is not None:
            flags["cc_pad256"] = cc_pad256
        if kda_chain is not None and kda_chain != "sequential":
            flags["kda_chain"] = kda_chain
        if mla_split is not None:
            flags["mla_split"] = dict(mla_split)
            flags["mla_split_stats"] = mla_split_stats
            if mla_split_own is not None:
                flags["mla_split_own"] = mla_split_own
            if mla_split_foreign is not None:
                flags["mla_split_foreign"] = mla_split_foreign
            if mla_split_table is not None:
                flags["mla_split_table"] = mla_split_table
        return entry(
            stream,
            norm_weights,
            attn_res_weights,
            kda_w_in,
            kda_w_out,
            kda_w_small,
            kda_params,
            mla_in_proj,
            mla_q_a_norm,
            mla_q_b_proj,
            mla_kv_a_norm,
            mla_kv_b_proj,
            mla_o_proj,
            moe_in_proj,
            moe_out_proj,
            moe_vectors,
            expert_gate_up_weights,
            expert_gate_up_weights_scale,
            expert_down_weights,
            expert_down_weights_scale,
            dense_gate_up,
            dense_down,
            kda_state_page,
            kda_record_page,
            kda_replay_len,
            kda_state_position,
            mla_block_table,
            mla_slot,
            position_rows,
            rank_info,
            tp_group,
            dp_group,
            world_group,
            first_layer,
            eps,
            queries=queries,
            dma_gates=dma_gates,
            pool_of_layer=pool_of_layer,
            kv_pool_0=kv_pool_0,
            kv_pool_1=kv_pool_1,
            kv_pool_2=kv_pool_2,
            kv_pool_3=kv_pool_3,
            kv_pool_4=kv_pool_4,
            kv_pool_5=kv_pool_5,
            kv_pool_6=kv_pool_6,
            kv_pool_7=kv_pool_7,
            kv_pool_8=kv_pool_8,
            kv_pool_9=kv_pool_9,
            kv_pool_10=kv_pool_10,
            kv_pool_11=kv_pool_11,
            kv_pool_12=kv_pool_12,
            kv_pool_13=kv_pool_13,
            kv_pool_14=kv_pool_14,
            kv_pool_15=kv_pool_15,
            kv_pool_16=kv_pool_16,
            kv_pool_17=kv_pool_17,
            kv_pool_18=kv_pool_18,
            kv_pool_19=kv_pool_19,
            kv_pool_20=kv_pool_20,
            kv_pool_21=kv_pool_21,
            kv_pool_22=kv_pool_22,
            kv_pool_23=kv_pool_23,
            final_attn_res_weight=final_attn_res_weight,
            final_norm_weight=final_norm_weight,
            lm_head_weight=lm_head_weight,
            greedy=greedy,
            return_logits=return_logits,
            return_aux=return_aux,
            aux_tap=aux_tap,
            mla_mode=mla_mode,
            world_size=world_size,
            routed_tp=routed_tp,
            moe_form=moe_form,
            dp_size=dp_size,
            **flags,
        )

    return kimi_k3_verify_wrapper


def _build_draft_wrapper(nki: Any, nl: Any, ReplicaGroup: Any, step: Any) -> Any:
    @nki.jit
    def kimi_k3_draft_mla_wrapper(
        aux: nl.ndarray,
        bonus: nl.ndarray,
        accepted: nl.ndarray,
        row0: nl.ndarray,
        draft_lo: nl.ndarray,
        rider_pool: nl.ndarray,
        block_table: nl.ndarray,
        verify_slots: nl.ndarray,
        rope_window: nl.ndarray,
        rope_base: nl.ndarray,
        context_proj: nl.ndarray,
        context_norm: nl.ndarray,
        kv_a_proj: nl.ndarray,
        kv_a_norm: nl.ndarray,
        mla_in_proj: nl.ndarray,
        mla_q_a_norm: nl.ndarray,
        mla_q_b_proj: nl.ndarray,
        mla_kv_a_norm: nl.ndarray,
        mla_kv_b_proj: nl.ndarray,
        mla_o_proj: nl.ndarray,
        input_norm: nl.ndarray,
        post_norm: nl.ndarray,
        dense_gate_up: nl.ndarray,
        dense_down: nl.ndarray,
        final_norm: nl.ndarray,
        markov_w1: nl.ndarray,
        markov_w2: nl.ndarray,
        lm_head: nl.ndarray,
        block_embed: nl.ndarray,
        rank_info: nl.ndarray,
        tp_ranks: tuple = (),
        softmax_scale: float = 0.0,
        layers: int = DRAFT_LAYERS,
        window: int = 0,
        mask_token: int = DRAFT_MASK_TOKEN,
        loop_path: bool = False,
        fp8_pages: bool = False,
        kernel_digest: str = "",
        cp: int = 32,
        window_read: bool = False,
        embed_vocab: bool = False,
        queries: int = DRAFT_VERIFY_ROWS,
        guard_indices: bool = False,
        tail_rows: int = DRAFT_TAIL_ROWS,
        context_done: bool = False,
        sample_params: nl.ndarray = None,
        u_draft: nl.ndarray = None,
        pe_moves: bool = False,
        win_trim: bool = False,
        last_layer_trim: bool = False,
        merged_out_ag: bool = False,
        cc_pad256: bool = False,
        stack_rows: bool = False,
        pass_q_prefetch: int = 0,
        q_gather_buffers: int = 1,
        pass_tail_overlap: bool = False,
        seq_passes: bool = False,
        q_call_one_pass: bool = False,
        markov_w1_local: bool = False,
        state_store_split: bool = False,
        sample_unique_dst: bool = False,
        prior_hoist: int = 0,
        ag_parity: bool = False,
    ):
        """View the FP8 rider pages, run the drafter step, and return the drafts and the bf16 pool.

        ``block_embed`` holds the vocab shard of the embedding when ``embed_vocab`` is True. Every option at its
        default gives the kernel no argument. With ``sample_params`` and ``u_draft`` the kernel draws the sampled
        drafts, and the q lists ``q_probs`` and ``q_ids`` follow.
        """
        flags = {"cp": cp} if cp != 32 else {}
        if window_read:
            flags["window_read"] = True
        flags["embed_vocab" if embed_vocab else "block_embed"] = block_embed
        if queries != DRAFT_VERIFY_ROWS:
            flags["queries"] = queries
        if guard_indices:
            flags["guard_indices"] = True
        if tail_rows != DRAFT_TAIL_ROWS:
            flags["tail_rows"] = tail_rows
        if context_done:
            flags["context_done"] = True
        if sample_params is not None:
            flags["sample_params"] = sample_params
            flags["u_draft"] = u_draft
        if pe_moves:
            flags["pe_moves"] = True
        if win_trim:
            flags["win_trim"] = True
        if last_layer_trim:
            flags["last_layer_trim"] = True
        if merged_out_ag:
            flags["merged_out_ag"] = True
        if cc_pad256:
            flags["cc_pad256"] = True
        if stack_rows:
            flags["stack_rows"] = True
        if pass_q_prefetch:
            flags["pass_q_prefetch"] = pass_q_prefetch
        if q_gather_buffers != 1:
            flags["q_gather_buffers"] = q_gather_buffers
        if pass_tail_overlap:
            flags["pass_tail_overlap"] = True
        if seq_passes:
            flags["seq_passes"] = True
        if q_call_one_pass:
            flags["q_call_one_pass"] = True
        if markov_w1_local:
            flags["markov_w1_local"] = True
        if state_store_split:
            flags["state_store_split"] = True
        if sample_unique_dst:
            flags["sample_unique_dst"] = True
        if prior_hoist:
            flags["prior_hoist"] = int(prior_hoist)
        if ag_parity:
            flags["ag_parity"] = True
        pool = rider_pool
        if fp8_pages:
            # The rider holds FP8 rows as bf16 pairs; the kernel reads FP8 pages of 128 rows.
            pages = rider_pool.shape[0] * rider_pool.shape[1] * 2 // MLA_PAGE_WIDTH
            pool = rider_pool.view(nl.float8_e4m3fn).reshape((pages, MLA_PAGE_WIDTH))
        outputs = step(
            aux=aux,
            bonus=bonus,
            accepted=accepted,
            row0=row0,
            draft_lo=draft_lo,
            rider_pool=pool,
            block_table=block_table,
            verify_slots=verify_slots,
            rope_window=rope_window,
            rope_base=rope_base,
            context_proj=context_proj,
            context_norm=context_norm,
            kv_a_proj=kv_a_proj,
            kv_a_norm=kv_a_norm,
            mla_in_proj=mla_in_proj,
            mla_q_a_norm=mla_q_a_norm,
            mla_q_b_proj=mla_q_b_proj,
            mla_kv_a_norm=mla_kv_a_norm,
            mla_kv_b_proj=mla_kv_b_proj,
            mla_o_proj=mla_o_proj,
            input_norm=input_norm,
            post_norm=post_norm,
            dense_gate_up=dense_gate_up,
            dense_down=dense_down,
            final_norm=final_norm,
            lm_head=lm_head,
            markov_w1=markov_w1,
            markov_w2=markov_w2,
            rank_info=rank_info,
            tp_group=ReplicaGroup(tp_ranks),
            softmax_scale=softmax_scale,
            layers=layers,
            window=window,
            mask_token=mask_token,
            loop_path=loop_path,
            **flags,
        )
        return (outputs[0], rider_pool, *outputs[2:])

    return kimi_k3_draft_mla_wrapper


def _build_full_vocab_shims(nki: Any, ReplicaGroup: Any, kernel: Any) -> dict[str, Any]:
    @nki.jit
    def full_vocab_shim(
        logits,
        params,
        rank_info,
        tp_ranks: tuple = (),
        tp_size: int = 1,
        return_masses: bool = False,
        skip_inactive_top_k: bool = False,
        cc_pad256: bool = False,
    ):
        """Rebuild the TP replica group from the rank mesh, then run the sampler."""
        tp_group = ReplicaGroup(tp_ranks) if tp_size > 1 else None
        pad = {"cc_pad256": True} if cc_pad256 else {}
        if not skip_inactive_top_k:
            return kernel(logits, params, rank_info, tp_group, tp_size, return_masses, **pad)
        return kernel(logits, params, rank_info, tp_group, tp_size, return_masses, skip_inactive_top_k=True, **pad)

    @nki.jit
    def full_vocab_logprobs_shim(
        logits,
        params,
        rank_info,
        tp_ranks: tuple = (),
        tp_size: int = 1,
        num_logprobs: int = 20,
        skip_inactive_top_k: bool = False,
        fold_topn: bool = False,
        skip_unasked_logprobs: bool = False,
        cc_pad256: bool = False,
    ):
        """Rebuild the TP replica group, then run the sampler with the raw logprobs outputs."""
        tp_group = ReplicaGroup(tp_ranks) if tp_size > 1 else None
        pad = {"cc_pad256": True} if cc_pad256 else {}
        if not (skip_inactive_top_k or fold_topn):
            return kernel(logits, params, rank_info, tp_group, tp_size, False, num_logprobs, **pad)
        if not fold_topn:
            return kernel(logits, params, rank_info, tp_group, tp_size, False, num_logprobs,
                          skip_inactive_top_k=True, **pad)
        return kernel(logits, params, rank_info, tp_group, tp_size, False, num_logprobs,
                      skip_inactive_top_k=skip_inactive_top_k, fold_topn=True,
                      skip_unasked_logprobs=skip_unasked_logprobs, **pad)

    @nki.jit
    def full_vocab_verify_shim(
        logits,
        params,
        rank_info,
        candidate_ids,
        excl_u,
        tp_ranks: tuple = (),
        tp_size: int = 1,
        skip_inactive_top_k: bool = False,
        cc_pad256: bool = False,
    ):
        """Rebuild the TP replica group, then run the sampler with the draft candidates of each row."""
        tp_group = ReplicaGroup(tp_ranks) if tp_size > 1 else None
        pad = {"cc_pad256": True} if cc_pad256 else {}
        return kernel(logits, params, rank_info, tp_group, tp_size, skip_inactive_top_k=skip_inactive_top_k,
                      candidate_ids=candidate_ids, excl_u=excl_u, **pad)

    @nki.jit
    def full_vocab_verify_tm_shim(
        logits,
        params,
        rank_info,
        candidate_ids,
        excl_u,
        tp_ranks: tuple = (),
        tp_size: int = 1,
        skip_inactive_top_k: bool = False,
        token_major_queries: int = 0,
        cc_pad256: bool = False,
    ):
        """Rebuild the TP replica group, then run the verify sampler on the token-major rows."""
        tp_group = ReplicaGroup(tp_ranks) if tp_size > 1 else None
        pad = {"cc_pad256": True} if cc_pad256 else {}
        return kernel(logits, params, rank_info, tp_group, tp_size, skip_inactive_top_k=skip_inactive_top_k,
                      candidate_ids=candidate_ids, excl_u=excl_u, token_major_queries=token_major_queries, **pad)

    return {"full_vocab": full_vocab_shim, "full_vocab_logprobs": full_vocab_logprobs_shim,
            "full_vocab_verify": full_vocab_verify_shim, "full_vocab_verify_tm": full_vocab_verify_tm_shim}


def _build_kda_a_exp_wrapper(nki: Any, nl: Any, kernel: Any) -> Any:
    @nki.jit
    def kimi_k3_kda_a_exp_init_wrapper(kda_params: nl.ndarray, conv_dtype: str = "bf16", kernel_digest: str = ""):
        return kernel(kda_params, conv_dtype=conv_dtype)

    return kimi_k3_kda_a_exp_init_wrapper


STRICT_WRAPPERS = ("verify", "draft", "kda_a_exp_init")
"""The wrappers launched through ``strict_wrap_nki``; the full-vocab shims go through ``wrap_nki`` directly."""


def prepare() -> None:
    """Import the kernels, build the wrappers and their launches, and compute the kernel digests; call outside any
    traced forward.

    Raises:
        KernelUnavailable: A kernel does not import.
    """
    if _LAUNCH:
        return
    kernel_imports.prepare()
    import nki
    import nki.language as nl
    from nki.collectives import ReplicaGroup

    wrappers = {
        "verify": _build_verify_wrapper(nki, nl, ReplicaGroup, kernel_imports.verify_entry()),
        "draft": _build_draft_wrapper(nki, nl, ReplicaGroup, kernel_imports.draft_mla_step()),
        "kda_a_exp_init": _build_kda_a_exp_wrapper(nki, nl, kernel_imports.kda_a_exp_init()),
        **_build_full_vocab_shims(nki, ReplicaGroup, kernel_imports.full_vocab_sample()),
    }
    DIGESTS.update(kernel_digests())
    sampler = kernel_imports.full_vocab_sample()
    SAMPLER_PARAMETERS.update((sampler.sign if hasattr(sampler, "sign") else inspect.signature(sampler)).parameters)
    _WRAPPERS.update(wrappers)
    _LAUNCH.update({name: strict_wrap_nki(fn) if name in STRICT_WRAPPERS else stack.wrap_nki(fn)
                    for name, fn in wrappers.items()})


def _launch(name: str) -> Any:
    launch = _LAUNCH.get(name)
    if launch is None:
        if torch.compiler.is_compiling():
            raise kernel_imports.KernelUnavailable(f"The Kimi-K3 decode wrapper {name} was not built before the "
                                                   "trace: call kernel_calls.prepare() when the programs are built.")
        prepare()
        launch = _LAUNCH[name]
    return launch[LNC]


def wrapper(name: str) -> Any:
    """The ``@nki.jit`` wrapper ``name`` (``prepare`` builds them)."""
    if not _WRAPPERS:
        prepare()
    return _WRAPPERS[name]


# --- the verify call


class VerifyOutputs(NamedTuple):
    """The outputs of one verify call.

    Attributes:
        stream: bf16 ``[R, H]`` (token-major rows).
        tokens: int32 greedy tokens (greedy tail), else None.
        logits: bf16 ``[R, V / TP]`` local logits (sampled tail), else None.
        aux: bf16 ``[5, R, H]`` drafter aux states (``return_aux``), else None.
        attn_res_bank: the AttnRes bank of a first split call, else None.
        attn_res_fence: the boundary fence of a first split call, else None.
        split_gathered, split_aux: the outputs of the cross-replica MLA split (not served), else None.
    """

    stream: Tensor
    tokens: Tensor | None = None
    logits: Tensor | None = None
    aux: Tensor | None = None
    attn_res_bank: Tensor | None = None
    attn_res_fence: Tensor | None = None
    split_gathered: Tensor | None = None
    split_aux: Tensor | None = None


def verify_wrapper_kwargs(verify_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """The lever keywords in the form of the verify wrapper: each dict option as its sorted ``(key, value)`` pairs."""
    return {name: tuple(sorted(value.items())) if name in VERIFY_PAIR_ARGUMENTS and value is not None else value
            for name, value in verify_kwargs.items()}


def verify_block(
    stream: Tensor,
    weights: Mapping[str, Tensor],
    *,
    kda_state_page: Tensor,
    kda_record_page: Tensor,
    kda_replay_len: Tensor,
    kda_state_position: Tensor,
    mla_block_table: Tensor,
    mla_slot: Tensor,
    position_rows: Tensor,
    rank_info: Tensor,
    pools: Sequence[Tensor],
    pool_of_layer: tuple[int, ...],
    eps: float,
    queries: int,
    tail_weights: Mapping[str, Tensor] | None,
    verify_kwargs: Mapping[str, Any],
    layout: DecodeLayout = DP8_LAYOUT,
    moe_form: str = "moe_2srv",
    routed_tp: int = 2,
    greedy: bool = False,
    return_logits: bool = False,
    return_aux: bool = True,
    aux_layers: Sequence[int] | None = None,
    aux_tap: str = "mixture",
    mla_fp8_pages: bool = False,
    mla_loop_path: bool = False,
    mla_mode: str = "k1",
    lora_norm_eps: float = 1e-5,
    kda_replay: str = "records",
    kda_state_slot_page: Tensor | None = None,
    kda_write_state_slots: Tensor | None = None,
    kda_conv_dtype: str = "bf16",
    kda_state_dtype: str = "bf16",
    first_layer: int = 0,
    attn_res_bank_in: Tensor | None = None,
    return_attn_res_bank: bool = False,
    attn_res_fence_in: Tensor | None = None,
    dma_gates: bool = True,
    mla_zeroed_pages: bool | None = None,
    live_rows: Tensor | None = None,
    mla_critical_dma_priority: bool = False,
    mla_cache_write_off_sp: bool = False,
    mla_gemv_bands: bool = False,
    kda_null_page: int = 0,
    kda_paired_pages: bool = False,
    kda_page_groups: Sequence[int] = (),
    kda_w_in_head_major: bool = False,
    pair_shared_cc_dsts: Sequence[str] = (),
) -> VerifyOutputs:
    """Run the 93 layers (and the tail) on Q tokens of each sequence of one replica through the 2-server verify.

    ``stream`` holds the token-major rows ``[Q x B, H]``. ``verify_kwargs`` are the lever keywords of the served form
    (``VERIFY_LEVERS``). ``pools`` are the page pools of the layers (``pool_of_layer`` names the pool of each layer).
    The tail (``greedy`` or ``return_logits``) needs ``tail_weights``. ``live_rows`` is the int32 ``[B, 1]`` vector of
    the request rows (1) and pad rows (0); ``kda_paired_pages`` needs an fp32 KDA state.

    Raises:
        ValueError: An input does not fit the verify.
        KeyError: A weight is missing.
    """
    if moe_form not in MOE_FORMS:
        raise ValueError(f"The MoE form {moe_form!r} is not a form of the 2-server verify ({', '.join(MOE_FORMS)}).")
    tail = return_logits or greedy
    if not 1 <= len(pools) <= MAX_POOLS:
        raise ValueError(f"The verify has {len(pools)} page pools; the kernel takes 1 to {MAX_POOLS}.")
    if tail and tail_weights is None:
        raise ValueError("greedy and return_logits need the tail weights.")
    rows = stream.shape[0]
    if rows > VERIFY_MAX_ROWS or rows % queries or layout.dp == 1:
        raise ValueError(f"The 2-server verify takes Q x B <= {VERIFY_MAX_ROWS} rows of a replica of a DP layout; the "
                         f"call has {rows} rows for Q = {queries} on {layout.name}.")
    missing = [name for name in WEIGHT_ARGUMENTS if name not in weights]
    if missing:
        raise KeyError(f"The verify weights have no {missing}.")
    if live_rows is not None and (tuple(live_rows.shape) != (rows // queries, 1) or live_rows.dtype != torch.int32):
        raise ValueError(f"The verify takes the live rows as int32 [{rows // queries}, 1]; the call gives "
                         f"{live_rows.dtype} {list(live_rows.shape)}.")
    if isinstance(kda_null_page, bool) or not isinstance(kda_null_page, int) or kda_null_page < 0:
        raise ValueError(f"The verify takes the KDA null page as a block number of 0 or more; the call gives "
                         f"{kda_null_page!r}.")
    if kda_paired_pages and kda_state_dtype == "bf16":
        raise ValueError("The KDA page pairs hold an fp32 state only, and the KDA state of the pages is bf16.")
    levers = [name for name, on in (("mla_critical_dma_priority", mla_critical_dma_priority),
                                    ("mla_cache_write_off_sp", mla_cache_write_off_sp),
                                    ("mla_gemv_bands", mla_gemv_bands)) if on]
    if levers and mla_mode != "k1":
        raise ValueError(f"The verify MLA levers {levers} act on the k1 MLA only; the MLA mode is {mla_mode!r}.")
    scales = {name: weights[name] for name in MX_SCALE_ARGUMENTS if weights.get(name) is not None}
    router = ({"router_dtype": "fp16", ROUTER_STACK: weights[ROUTER_STACK]}
              if weights.get(ROUTER_STACK) is not None else {})
    pair_sites = tuple(pair_shared_cc_dsts or ())
    tails = {name: (tail_weights or {}).get(name) for name in TAIL_ARGUMENTS}
    tp_ranks, dp_ranks, world_ranks = layout.rank_groups()
    padded_pools = list(pools) + [None] * (MAX_POOLS - len(pools))
    outputs = _launch("verify")(
        stream,
        *(weights[name] for name in WEIGHT_ARGUMENTS),
        kda_state_page,
        kda_record_page,
        kda_replay_len,
        kda_state_position,
        mla_block_table,
        mla_slot,
        position_rows,
        rank_info,
        tp_ranks,
        dp_ranks,
        world_ranks,
        int(first_layer),
        eps,
        int(queries),
        dma_gates,
        tuple(int(i) for i in pool_of_layer),
        *padded_pools,
        *(tails[name] if tail else None for name in TAIL_ARGUMENTS),
        greedy=greedy,
        return_logits=return_logits,
        return_aux=return_aux,
        aux_layers=tuple(int(layer) for layer in aux_layers or ()),
        aux_tap=aux_tap,
        kernel_digest=DIGESTS.get("verify", ""),
        world_size=layout.world_size,
        routed_tp=int(routed_tp),
        moe_form=moe_form,
        dp_size=layout.dp,
        mla_mode=mla_mode,
        mla_fp8_pages=mla_fp8_pages,
        mla_loop_path=mla_loop_path,
        lora_norm_eps=lora_norm_eps,
        kda_replay=kda_replay,
        kda_state_slot_page=kda_state_slot_page,
        kda_write_state_slots=kda_write_state_slots,
        kda_conv_dtype=kda_conv_dtype,
        kda_state_dtype=kda_state_dtype,
        attn_res_bank_in=attn_res_bank_in,
        return_attn_res_bank=bool(return_attn_res_bank),
        attn_res_fence_in=attn_res_fence_in,
        **{
            **verify_wrapper_kwargs(verify_kwargs),
            **({"mla_zeroed_pages": True} if mla_zeroed_pages else {}),
            **({"live_rows": live_rows} if live_rows is not None else {}),
            **({"kda_null_page": kda_null_page} if kda_null_page else {}),
            **({"kda_paired_pages": True} if kda_paired_pages else {}),
            **({"kda_page_groups": tuple(kda_page_groups)} if kda_page_groups else {}),
            **({"kda_w_in_head_major": True} if kda_w_in_head_major else {}),
            **({"pair_shared_cc_dsts": pair_sites} if pair_sites else {}),
            **dict.fromkeys(levers, True),
        },
        **scales,
        **router,
    )
    # Kernel outputs: stream, [tokens], [logits], [aux], [bank], [fence], *pools.
    values = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
    result = {"stream": values.pop(0)}
    if greedy:
        result["tokens"] = values.pop(0)
    if return_logits:
        result["logits"] = values.pop(0)
    if return_aux:
        result["aux"] = values.pop(0)
    if return_attn_res_bank:
        result["attn_res_bank"] = values.pop(0)
    if return_attn_res_bank:
        result["attn_res_fence"] = values.pop(0)
    return VerifyOutputs(**result)


# --- the drafter call


def _token_major_rows(rows_per_sequence: int, batch: int, start: int, stop: int, device) -> Tensor:
    """The rows ``j * batch + b`` of the sequences ``start .. stop - 1`` in the token-major order of a group."""
    j = torch.arange(rows_per_sequence, device=device).reshape(-1, 1)
    b = torch.arange(start, stop, device=device).reshape(1, -1)
    return (j * batch + b).reshape(-1)


def draft_groups(batch: int, most: int = DRAFT_MAX_SEQUENCES) -> list[tuple[int, int]]:
    """The ``(start, stop)`` sequences of each drafter call: equal groups of at most ``most`` sequences.

    Example:
        >>> draft_groups(4), draft_groups(16), draft_groups(12), draft_groups(16, 4)
        ([(0, 4)], [(0, 8), (8, 16)], [(0, 6), (6, 12)], [(0, 4), (4, 8), (8, 12), (12, 16)])
    """
    count = -(-batch // most)
    size = -(-batch // count)
    return [(start, min(start + size, batch)) for start in range(0, batch, size)]


def draft_mla_step(
    *,
    aux: Tensor,
    bonus: Tensor,
    accepted: Tensor,
    row0: Tensor,
    draft_lo: Tensor,
    rider_pool: Tensor,
    block_table: Tensor,
    verify_slots: Tensor,
    rope_window: Tensor,
    rope_base: Tensor,
    weights: Mapping[str, Tensor],
    lm_head_weight: Tensor,
    block_embed: Tensor | None,
    rank_info: Tensor,
    softmax_scale: float,
    layout: DecodeLayout = DP8_LAYOUT,
    window: int = 0,
    mask_token: int = DRAFT_MASK_TOKEN,
    loop_path: bool = False,
    fp8_pages: bool = False,
    cp: int = 32,
    window_read: bool = False,
    embed_vocab: Tensor | None = None,
    queries: int = DRAFT_VERIFY_ROWS,
    guard_indices: bool = False,
    tail_rows: int = DRAFT_TAIL_ROWS,
    context_done: bool = False,
    sample_params: Tensor | None = None,
    u_draft: Tensor | None = None,
    pe_moves: bool = False,
    win_trim: bool = False,
    last_layer_trim: bool = False,
    merged_out_ag: bool = False,
    cc_pad256: bool = False,
    stack_rows: bool = False,
    pass_q_prefetch: int = 0,
    q_gather_buffers: int = 1,
    pass_tail_overlap: bool = False,
    seq_passes: bool = False,
    q_call_one_pass: bool = False,
    markov_w1_local: bool = False,
    state_store_split: bool = False,
    sample_unique_dst: bool = False,
    prior_hoist: int = 0,
    ag_parity: bool = False,
) -> Tensor | tuple[Tensor, Tensor, Tensor]:
    """Run the DSpark MLA drafter for the B sequences of one replica; return the drafts ``[B, 7]`` int32, or with
    ``sample_params`` / ``u_draft`` ``(drafts, q_probs [B, 7, K] fp32, q_ids [B, 7, K] int32)``.

    The kernel writes the context rows of the verify rows of each sequence into ``rider_pool`` in place. A batch of
    more than ``DRAFT_MAX_SEQUENCES`` sequences runs as one call for each group of ``draft_groups``
    (``DRAFT_MAX_LOOP_SEQUENCES`` on the loop path); each group takes its sequences' rows in the token-major order of
    the group.

    Args:
        aux: bf16 ``[5, Q B, H]`` verify aux states, token-major (row ``j * B + b``).
        bonus, accepted, row0, draft_lo: int32 ``[B, 1]``: the anchor token, the accepted rows (1 + accepted drafts),
            the position of verify row 0, and the first valid context position.
        rider_pool: bf16 ``[5 * pages, W]`` rider pages of the 5 drafter layers (``W`` 73728, or 36864 for FP8 rows as
            bf16 pairs).
        block_table: int32 ``[B, NB]`` MLA kernel block table.
        verify_slots: int32 ``[Q B, 1]`` MLA slot of each verify row on this rank, else 0.
        rope_window: fp32 ``[B, W, 128]`` YaRN rows of the positions ``rope_base + i``; rope_base int32 ``[B, 1]``.
        weights: the per-rank drafter weights (``DRAFT_WEIGHT_ARGUMENTS``).
        lm_head_weight: bf16 ``[H, V / TP]`` lm_head shard of the target tail.
        block_embed: bf16 ``[7B, H]`` target embedding of the block (token-major), or None with ``embed_vocab``
            (bf16 ``[V / TP, H]``, the vocab shard of the embedding: the kernel looks up the block).
        rank_info: int32 ``[1, 4]``.
        softmax_scale: the drafter softmax scale (YaRN).
        The options are those of the kernel (``DRAFT_STAGE``, ``DRAFT_CALL``); each at its default gives the kernel no
        argument.

    Raises:
        ValueError: An option or an input does not fit the kernel.
        KeyError: A weight is missing.
    """
    if not 1 <= int(tail_rows) <= DRAFT_TAIL_ROWS:
        raise ValueError(f"tail_rows is {tail_rows}; the drafter tail runs 1 to {DRAFT_TAIL_ROWS} rows.")
    if context_done and row0.shape[0] > DRAFT_MAX_SEQUENCES:
        raise ValueError(f"The drafter context stage runs in one call of at most {DRAFT_MAX_SEQUENCES} sequences; the "
                         f"step has {row0.shape[0]}.")
    folded = embed_vocab is not None or queries != DRAFT_VERIFY_ROWS or guard_indices
    if (embed_vocab is None) == (block_embed is None):
        raise ValueError("The drafter takes the block embedding (block_embed) or the vocab shard (embed_vocab), "
                         "exactly one of them.")
    if not 0 <= int(pass_q_prefetch) <= 2 or int(q_gather_buffers) not in (1, 2):
        raise ValueError(f"pass_q_prefetch is {pass_q_prefetch} and q_gather_buffers is {q_gather_buffers}; the "
                         "drafter takes 0 to 2 prefetch tiles and 1 or 2 gather tiles.")
    if seq_passes and (not pe_moves or last_layer_trim):
        raise ValueError(f"seq_passes needs pe_moves and no last_layer_trim; the step has pe_moves {pe_moves} and "
                         f"last_layer_trim {last_layer_trim}.")
    if q_call_one_pass and (int(cp) != 32 or loop_path):
        raise ValueError(f"q_call_one_pass gathers the queries on the CP32 static path; the step has cp {cp} and "
                         f"loop_path {loop_path}.")
    if last_layer_trim and int(tail_rows) >= DRAFT_TAIL_ROWS:
        raise ValueError(f"last_layer_trim needs tail_rows below {DRAFT_TAIL_ROWS}; the tail runs {tail_rows} rows.")
    sampled = sample_params is not None
    if sampled != (u_draft is not None):
        raise ValueError("The sampled drafts take sample_params and u_draft together.")
    missing = [name for name in DRAFT_WEIGHT_ARGUMENTS if name not in weights]
    if missing:
        raise KeyError(f"The drafter weights have no {missing}.")
    tp_ranks, _, _ = layout.rank_groups()

    def call(seq: tuple[int, int] | None, pool: Tensor) -> tuple[Tensor, Tensor, tuple[Tensor, ...]]:
        per_seq = (aux, verify_slots) + ((block_embed,) if embed_vocab is None else ())
        args = (bonus, accepted, row0, draft_lo, block_table, rope_window, rope_base)
        sampling = (sample_params, u_draft) if sampled else ()
        if seq is not None:
            batch = row0.shape[0]
            per_seq = tuple(t.index_select(t.dim() - 2, _token_major_rows(t.shape[-2] // batch, batch, *seq,
                                                                          t.device)) for t in per_seq)
            args = tuple(t[seq[0]: seq[1]] for t in args)
            sampling = tuple(t[seq[0]: seq[1]] for t in sampling)
        if embed_vocab is not None:
            per_seq = per_seq + (embed_vocab,)
        group_aux, group_slots, group_embed = per_seq
        group_bonus, group_accepted, group_row0, group_lo, table, rope, base = args
        outputs = _launch("draft")(
            group_aux,
            group_bonus,
            group_accepted,
            group_row0,
            group_lo,
            pool,
            table,
            group_slots,
            rope,
            base,
            *(weights[name] for name in DRAFT_WEIGHT_ARGUMENTS),
            lm_head_weight,
            group_embed,
            rank_info,
            tp_ranks,
            float(softmax_scale),
            DRAFT_LAYERS,
            int(window),
            int(mask_token),
            bool(loop_path),
            bool(fp8_pages),
            DIGESTS.get("draft", ""),
            int(cp),
            window_read=bool(window_read),
            **({"embed_vocab": embed_vocab is not None, "queries": int(queries),
                "guard_indices": bool(guard_indices)} if folded else {}),
            **({"tail_rows": int(tail_rows)} if int(tail_rows) != DRAFT_TAIL_ROWS else {}),
            **({"context_done": True} if context_done else {}),
            **({"sample_params": sampling[0], "u_draft": sampling[1]} if sampled else {}),
            **({"pe_moves": True} if pe_moves else {}),
            **({"win_trim": True} if win_trim else {}),
            **({"last_layer_trim": True} if last_layer_trim else {}),
            **({"merged_out_ag": True} if merged_out_ag else {}),
            **({"cc_pad256": True} if cc_pad256 else {}),
            **({"stack_rows": True} if stack_rows else {}),
            **({"pass_q_prefetch": int(pass_q_prefetch)} if int(pass_q_prefetch) else {}),
            **({"q_gather_buffers": int(q_gather_buffers)} if int(q_gather_buffers) != 1 else {}),
            **({"pass_tail_overlap": True} if pass_tail_overlap else {}),
            **({"seq_passes": True} if seq_passes else {}),
            **({"q_call_one_pass": True} if q_call_one_pass else {}),
            **({"markov_w1_local": True} if markov_w1_local else {}),
            **({"state_store_split": True} if state_store_split else {}),
            **({"sample_unique_dst": True} if sample_unique_dst else {}),
            **({"prior_hoist": int(prior_hoist)} if prior_hoist else {}),
            **({"ag_parity": True} if ag_parity else {}),
        )
        values = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
        return values[0], values[1] if len(values) > 1 else pool, tuple(values[2:4]) if sampled else ()

    batch = row0.shape[0]
    if batch <= DRAFT_MAX_SEQUENCES:
        drafts, _, q_lists = call(None, rider_pool)
        return (drafts, *q_lists) if sampled else drafts
    # Each group writes its context rows into the pool that the group before returned, so the writes keep their order.
    most = DRAFT_MAX_LOOP_SEQUENCES if loop_path else DRAFT_MAX_SEQUENCES
    drafts, q_parts, pool = [], [], rider_pool
    for seq in draft_groups(batch, most):
        group, pool, q_lists = call(seq, pool)
        drafts.append(group)
        q_parts.append(q_lists)
    if not sampled:
        return torch.cat(drafts, dim=0)
    return torch.cat(drafts, dim=0), *(torch.cat([part[i] for part in q_parts], dim=0) for i in range(2))


# --- the full-vocab sampler calls


def _check_sampler_rows(rows: int, tp_size: int) -> None:
    if rows < 1 or rows > FULL_VOCAB_MAX_ROWS:
        raise ValueError(f"The full-vocab sampler takes 1 to {FULL_VOCAB_MAX_ROWS} rows; got {rows}.")
    if tp_size < 1:
        raise ValueError(f"The full-vocab sampler needs a TP size of at least 1; got {tp_size}.")


def full_vocab_sample(
    logits: Tensor,
    top_k: Tensor,
    top_p: Tensor,
    temperature: Tensor,
    draw: Tensor,
    tp_rank: Tensor,
    layout: DecodeLayout = DP8_LAYOUT,
    num_logprobs: int = -1,
    skip_inactive_top_k: bool = False,
    fold_topn: bool = False,
    logprobs_ask: Tensor | None = None,
    cc_pad256: bool = False,
) -> tuple[Tensor, ...]:
    """``(tokens [B] int32, flags [B] int32)`` from the sampler on this rank's vocab shard (TP group of ``layout``).

    The kernel takes the per-row columns as ``params [B, 4]`` = (top_k, top_p, temperature, draw) (column 4 the
    ``logprobs_ask`` when given) and the TP rank in ``rank_info [1, 4]`` column 0. With ``num_logprobs`` N (1 to 20)
    the result also has ``topn_ids [B, N]`` int32, ``topn_logits [B, N]``, ``logz [B]`` and ``sampled_logit [B]``.
    """
    tp_size = layout.attention_tp
    _check_sampler_rows(logits.shape[0], tp_size)
    pad = {"cc_pad256": True} if cc_pad256 else {}
    columns = [top_k, top_p, temperature, draw]
    if logprobs_ask is not None:
        columns.append(logprobs_ask)
    params = torch.stack([c.reshape(-1).to(torch.float32) for c in columns], dim=1)
    rank_info = torch.nn.functional.pad(tp_rank.reshape(1, 1).to(torch.int32), (0, 3))
    mesh = layout.rank_groups()[0]
    if num_logprobs < 1:
        args = (logits, params, rank_info, mesh, tp_size, False)
        if skip_inactive_top_k:
            args += (True,)
        tokens, flags = _launch("full_vocab")(*args, **pad)
        return tokens.reshape(-1).to(torch.int32), flags.reshape(-1).to(torch.int32)
    args = (logits, params, rank_info, mesh, tp_size, num_logprobs)
    keys = (skip_inactive_top_k, fold_topn, logprobs_ask is not None)
    count = 0
    for index, on in enumerate(keys):
        if on:
            count = index + 1
    args += keys[:count]
    tokens, flags, topn_ids, topn_logits, logz, sampled_logit = _launch("full_vocab_logprobs")(*args, **pad)
    return (tokens.reshape(-1).to(torch.int32), flags.reshape(-1).to(torch.int32), topn_ids.to(torch.int32),
            topn_logits, logz.reshape(-1), sampled_logit.reshape(-1))


def full_vocab_verify(
    logits: Tensor,
    top_k: Tensor,
    top_p: Tensor,
    temperature: Tensor,
    draw: Tensor,
    tp_rank: Tensor,
    candidate_ids: Tensor,
    excl_u: Tensor,
    layout: DecodeLayout = DP8_LAYOUT,
    skip_inactive_top_k: bool = False,
    direct: int | None = None,
    cc_pad256: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """``(tokens [B], flags [B], probs_at [B, K], excl_token [B])`` from one sampler call with the draft candidates.

    ``candidate_ids [B, K]`` int32 holds the candidates of each row in the kernel layout (column ``j * tp_size + s``
    = candidate ``j`` of vocab shard ``s``, or -1), and ``excl_u [B, 1]`` the host draw of the draw without them.
    ``direct = queries``: ``logits`` is the token-major verify output, which the kernel reads in request-major order
    (``token_major_queries``).
    """
    rows = logits.shape[0]
    tp_size = layout.attention_tp
    _check_sampler_rows(rows, tp_size)
    if candidate_ids.shape[-1] % tp_size:
        raise ValueError(f"The verify candidates take K a multiple of the TP size; got K = {candidate_ids.shape[-1]} "
                         f"and TP {tp_size}.")
    params = torch.stack([c.reshape(-1).to(torch.float32) for c in (top_k, top_p, temperature, draw)], dim=1)
    rank_info = torch.nn.functional.pad(tp_rank.reshape(1, 1).to(torch.int32), (0, 3))
    candidates = candidate_ids.to(torch.int32).contiguous()
    draws = excl_u.reshape(rows, 1).to(torch.float32).contiguous()
    pad = {"cc_pad256": True} if cc_pad256 else {}
    mesh = layout.rank_groups()[0]
    if direct is None:
        tokens, flags, probs_at, excl_token = _launch("full_vocab_verify")(
            logits, params, rank_info, candidates, draws, mesh, tp_size, bool(skip_inactive_top_k), **pad)
    else:
        queries = int(direct)
        if not _LAUNCH:
            prepare()
        if "token_major_queries" not in SAMPLER_PARAMETERS:
            raise ValueError("The vendored full-vocab sampler kernel has no token_major_queries input, so it cannot "
                             "read the token-major verify output directly; give the request-major rows (direct None).")
        if queries < 1 or rows % queries:
            raise ValueError(f"The token-major verify rows take queries that divide the rows; got {queries} queries "
                             f"and {rows} rows.")
        tokens, flags, probs_at, excl_token = _launch("full_vocab_verify_tm")(
            logits, params, rank_info, candidates, draws, mesh, tp_size, bool(skip_inactive_top_k), int(queries),
            **pad)
    return (tokens.reshape(-1).to(torch.int32), flags.reshape(-1).to(torch.int32), probs_at,
            excl_token.reshape(-1).to(torch.int32))


# --- the KDA exp(A_log) init


def kda_params_local_heads(width: int, conv_dtype: str) -> int:
    """The KDA heads ``Hl`` of a plain ``kda_params`` row of ``width`` columns.

    Example:
        >>> kda_params_local_heads(7, "bf16"), kda_params_local_heads(43, "fp32")
        (3, 3)
    """
    per_head = KDA_PARAMS_COLUMNS_PER_HEAD.get(conv_dtype)
    if per_head is None:
        raise ValueError(f"The KDA conv dtype {conv_dtype!r} is not one of {sorted(KDA_PARAMS_COLUMNS_PER_HEAD)}.")
    heads = (int(width) - 1) // per_head
    if heads < 1 or (int(width) - 1) % per_head:
        raise ValueError(f"The kda_params width {width} is not a plain width of the {conv_dtype} conv "
                         f"({per_head} * heads + 1).")
    return heads


def kda_params_wide_shape(shape: Sequence[int], conv_dtype: str) -> tuple[int, int, int]:
    """The shape ``[L, 128, W + Hl]`` of the wide stack of a plain ``kda_params`` stack ``[L, 128, W]``.

    Example:
        >>> kda_params_wide_shape((9, 128, 7), "bf16")
        (9, 128, 10)
    """
    if len(shape) != 3:
        raise ValueError(f"A kda_params stack is [L, 128, W]; this stack has the shape {tuple(shape)}.")
    layers, rows, width = int(shape[0]), int(shape[1]), int(shape[2])
    return layers, rows, width + kda_params_local_heads(width, conv_dtype)


def kda_a_exp_init(kda_params: Tensor, conv_dtype: str = "bf16") -> Tensor:
    """The wide ``kda_params`` stack ``[L, 128, W + Hl]`` (fp32) of the plain stack ``[L, 128, W]``: its last ``Hl``
    columns hold ``exp(A_log)``. One small program of its own, once for each stack after the weight load."""
    kda_params_wide_shape(kda_params.shape, conv_dtype)
    if kda_params.dtype != torch.float32:
        raise ValueError(f"A kda_params stack is fp32; this stack is {kda_params.dtype}.")
    return _launch("kda_a_exp_init")(kda_params, conv_dtype=conv_dtype, kernel_digest=DIGESTS.get("kda_a_exp_init", ""))


class KdaAExpInit(torch.nn.Module):
    """The init program: the wide ``kda_params`` stack of the plain stack it gets (``kda_a_exp_init``)."""

    def __init__(self, conv_dtype: str = "bf16") -> None:
        super().__init__()
        self.conv_dtype = conv_dtype

    def forward(self, kda_params: Tensor) -> Tensor:
        return kda_a_exp_init(kda_params, self.conv_dtype)
