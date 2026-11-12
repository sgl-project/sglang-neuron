# SPDX-License-Identifier: Apache-2.0
"""The launch rules of the GLM-5.2-family PD roles (catalog.ModelSpec.launch_hooks), each rule its role's.

The platform runs ``kv_page_size`` and ``launch_problems`` in the launching process, before any scheduler starts, and
applies ``launch_settings`` there; each rank takes ``rank_runtime_env``'s runtime settings before its Neuron runtime
starts (model_executor/runtime_init.py). The model host runs ``check_hf_config`` on a prefill when it is built, before
its parallel groups exist, and ``fleet_digest`` is what a prefill's ranks compare before they compile.

The prefill serves two points (stage_layout.PREFILL_POINTS), each a PD prefill server of pipeline stages at LNC1 whose
stages each run one SPMD program of the prefill giga kernel (models/glm52/kernels/prefill): PP4 x TP32
(EP32, split 18,20,20,20, 4096-token chunks with 1024- and 2048-token graphs for a prompt's last piece), the default,
and PP16 x TP8 (EP8, split 3,5 x 15, 2048-token chunks with a 1024-token graph). Every other prefill launch is refused
here, naming the flag to change, because the kernel's launch groups, the MoE placement (recipe.PLACEMENT_FILE) and the
graphs' collectives are built for those layouts alone. A decode (the decode giga kernel at TP128) is refused here only
for what its arguments and environment already state; the rest of the GK's layout is refused where its values
resolve: the serving policy's GK layout (serving_policy.giga_decode_layout), its launch settings and its host
(sglang_model.decode_problems, at SGLang's resolved context and request rows). The decode pairs with either prefill
point, which its --page-size names (paired_prefill_point). Both roles run on the pinned Neuron SDK release only
(release_problems). An aggregated server is refused: neither role's graphs serve it.

Imports at module level are stdlib only: the catalog names this module and the launching process imports it before
anything else is decided. The rest is imported where it is used."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# the default prefill point (stage_layout.PREFILL_POINTS[4]): a launch with --pp-size 16 runs the PP16 point
PP_SIZE = 4
STAGE_RANKS = 32  # --tp-size at the default point: the ranks of one stage, the kernel's context-parallel degree
# the samplers of the last prefill stage's first token (prefill.inputs.sample_first_token): the exact greedy selection,
# or topk_topp, the GK decode tail's draw from each request's (temperature, seed, top_k, top_p); the decode's are the
# catalog entry's giga_decode_samplers
FIRST_TOKEN_SAMPLERS = ("greedy", "topk_topp")


def served_prefill_point(server_args):
    """The served prefill point of the launch's --pp-size (stage_layout.PREFILL_POINTS), or None for another size, which
    _prefill_problems refuses naming the served ones."""
    from sglang_neuron.models.glm52.prefill import stage_layout

    return stage_layout.PREFILL_POINTS.get(_int(server_args, "pp_size"))


def boundary_carries_picks(pp_size: int) -> bool:
    """Whether the stage boundary of the ``pp_size``-stage prefill point carries each rank's top-k picks beside its
    (h, r) shard (GlmMoeDsaForCausalLM.pp_prefill_boundary): at PP16 only, whose five-layer stages mostly start on a
    shared indexer layer and take the picks of the stage before (StageLayout.picks_in). The PP4 stages all start on full
    layers, and the PP4 kernel knobs and recipe were never run with picks crossing a stage, so the PP4 boundary carries
    (h, r) alone and a PP4 split that starts a stage on a shared layer is refused (config_problems)."""
    return int(pp_size) == 16


def launch_settings(server_args) -> tuple:
    """The prefill's launch settings at its point (recipe.point_settings: the PP4 recipe, or the PP16 point's kernel
    knobs), as the catalog's LaunchSetting rows. The platform applies them on a prefill once launch_problems has passed,
    so an unserved --pp-size has been refused before and raises here; the decode keeps its giga-kernel settings
    (catalog.ModelSpec.giga_decode_launch_settings)."""
    from sglang_neuron.models import catalog
    from sglang_neuron.models.glm52.prefill import recipe
    from sglang_neuron.models.glm52.prefill import stage_layout

    point = stage_layout.prefill_point(server_args.pp_size)
    return tuple(catalog.LaunchSetting(*setting) for setting in recipe.point_settings(point.pp_size))


# the largest context the stage graphs serve: the 1M bucket, the one the kernel's stage step is measured at
MAX_CONTEXT = 1 << 20
# The kernel on the released toolchain gathers its KV statically and runs a whole-bucket indexer, so its work follows
# the bucket: the stage graphs come in context buckets, and each request runs the smallest that holds it. Each is a
# power of two, so every stage's per-rank KV context is one (the sparse attention's interleave decode shifts by it). The
# smallest is 128k: the first stage program of a 64k bucket that runs after a 2,048-token program stalled every rank at
# warm-up. A server compiles the buckets up to its --context-length.
CONTEXT_BUCKETS = (1 << 17, 1 << 18, 1 << 19, 1 << 20)
PARTITION_ENV = "SGLANG_PP_LAYER_PARTITION"
# the prefill point (stage_layout.PREFILL_POINTS) a decode launched without --page-size pairs with: the PP4 x TP32 one,
# which every decode launch before the PP16 point paired with, so those launches keep their 1024-token page
DEFAULT_PAIRED_PP = 4


def _int(server_args, name: str, default: int = 1) -> int:
    value = getattr(server_args, name, None)
    return default if value is None else int(value)


def _role(server_args) -> str:
    return str(getattr(server_args, "disaggregation_mode", None) or "null")


def kv_page_size(server_args) -> int:
    """The KV page, in tokens, SGLang allocates in: one 32-row block of the prefill kernel's context-parallel
    interleave on each rank of a stage, so a page is the same block id on every one of them (--tp-size x
    stage_layout.CP_INTERLEAVE: 1024 at PP4 x TP32, 256 at PP16 x TP8, 4096 for a 128-rank stage). The decode serves
    the page of the prefill point it pairs with (paired_prefill_point), whatever its own world: SGLang's PD refuses a
    pair whose pages differ (its KV manager's page size check, at the decode's first contact), and the PD hand-off
    writes one prefill rank's 32 rows of a page as one GK block (pd.placement). The GK itself reads no SGLang page (its
    caches are its own, decode.pools), so the decode's page only sizes SGLang's accounting of its slots and numbers
    the destination pages it publishes: 4096 pages of 1024 or 16,384 of 256 for 32 slots of the kernel's 131,072
    positions. The decode runs this page on SGLang's fake transfer too, so a decode measured alone serves the contract a
    PD pair serves. A decode page no prefill point serves is returned as stated, for _decode_problems to refuse naming
    the served ones (the platform checks the page, check_kv_cache, before it runs launch_problems)."""
    from sglang_neuron.models.glm52.prefill import stage_layout

    if _role(server_args) == "decode":
        point = paired_prefill_point(server_args)
        return int(server_args.page_size) if point is None else point.virtual_page
    return int(server_args.tp_size) * stage_layout.CP_INTERLEAVE


def paired_prefill_point(server_args):
    """The served prefill point (stage_layout.PREFILL_POINTS) a decode launch pairs with, which the launch names by its
    --page-size, a setting SGLang's PD already requires both servers to share (its first contact refuses a prefill of
    another page): the point whose page it is, the DEFAULT_PAIRED_PP point without one, and None for a page no point
    serves. The points' pages differ (1024, 256), so a page names one point. The decode needs no more of it than its
    stage ranks, the page over CP_INTERLEAVE, by which pd.placement places a prefill rank's rows; the rank mapping reads
    the prefill's own stages and ranks at first contact."""
    from sglang_neuron.models.glm52.prefill import stage_layout

    page = getattr(server_args, "page_size", None)
    if page is None:
        return stage_layout.PREFILL_POINTS[DEFAULT_PAIRED_PP]
    return next((point for point in stage_layout.PREFILL_POINTS.values() if point.virtual_page == int(page)), None)


def _served_decode_pages() -> str:
    from sglang_neuron.models.glm52.prefill import stage_layout

    return ", ".join(f"{point.virtual_page} ({point.name.upper()} x {point.stage_ranks} ranks"
                     f"{', the default' if pp == DEFAULT_PAIRED_PP else ''})"
                     for pp, point in sorted(stage_layout.PREFILL_POINTS.items()))


def _unpinned(environ, pinned) -> list:
    """The variables of ``environ``, sorted, that change a compiled kernel (kernels/compile_knobs.is_compile_knob) and
    that ``pinned`` does not name."""
    from sglang_neuron.models.glm52.kernels import compile_knobs

    return sorted(name for name in environ if compile_knobs.is_compile_knob(name) and name not in pinned)


def refuse_unpinned(environ, pp_size: int = PP_SIZE) -> list:
    """A problem naming every variable of ``environ`` that changes a compiled kernel (_unpinned) and that the recipe
    does not set: the kernel is validated at the recipe's values only, and the recipe
    refuses another value of its own names when the platform applies it (serving_policy.set_launch_setting)."""
    from sglang_neuron.models.glm52.prefill import recipe

    unpinned = _unpinned(environ, recipe.names(pp_size))
    if not unpinned:
        return []
    return [f"{', '.join(unpinned)}: the prefill kernel and its compiler run the recipe's values only "
            "(models/glm52/prefill/recipe.py), and these change the compiled program; remove them from the launch environment"]


def refuse_unpinned_gk(environ) -> list:
    """The decode's refuse_unpinned: a problem naming every variable of ``environ`` that changes a compiled kernel
    (_unpinned) and that the GK's static recipe (the catalog entry's
    giga_decode_launch_settings) does not set. The GK is validated at that recipe's values only, and the platform
    refuses another value of its names when it applies them (serving_policy.set_launch_setting). The GK's split is
    its decode point's (points.Point.gk_split), not a launch setting."""
    from sglang_neuron.models import catalog

    pinned = {setting.name for setting in catalog.MODELS["GlmMoeDsaForCausalLM"].giga_decode_launch_settings}
    unpinned = _unpinned(environ, pinned)
    if not unpinned:
        return []
    return [f"{', '.join(unpinned)}: the GLM decode giga kernel and its compiler run its static recipe's values only "
            "(giga_decode_launch_settings, models/catalog.py), and these change the compiled program; remove them "
            "from the launch environment"]


def _full_config(config):
    """The prefill model config (layout.prefill.config.GlmMoeDsaConfig) of a checkpoint's config (a dict or a transformers
    config), as the stage graphs read it: its field filter, its defaults, its derived indexer and MLP schedules."""
    from sglang_neuron.models.glm52.layout.prefill.config import GlmMoeDsaConfig
    from sglang_neuron.models.glm52.options import Glm52PrefillOptions

    return GlmMoeDsaConfig.from_configs(config, Glm52PrefillOptions())


def _stage_problems(full, partition, pp_size: int) -> list:
    """Each stage of ``partition`` (layer counts) of the ``pp_size``-stage point that the stage config refuses (a stage
    out of the model's layers, or one that starts on a shared indexer layer with no full layer before it), and, where
    the point's boundary carries no top-k picks (boundary_carries_picks), each stage that starts on a shared indexer
    layer: its first launch would take the picks of a stage boundary that holds none."""
    from sglang_neuron.models.glm52.layout.prefill.pipeline import stage_config

    picks = boundary_carries_picks(pp_size)
    problems, lo = [], 0
    for size in partition:
        hi = lo + size
        try:
            stage_config(full, lo, hi, shared_start=True)
        except ValueError as e:
            problems.append(f"layers [{lo}, {hi}): {e}")
        else:
            if not picks and full.indexer_types[lo] != "full":
                problems.append(f"layers [{lo}, {hi}): the stage starts on layer {lo}, a {full.indexer_types[lo]!r} "
                                "indexer layer, which takes the top-k picks of the stage before, and the "
                                f"{pp_size}-stage boundary carries none (only the 16-stage point's does); every stage "
                                "of this point must start on a full indexer layer")
        lo = hi
    return problems


def config_problems(config, pp_size: int = PP_SIZE) -> list:
    """What of a checkpoint's config the stage graphs of the ``pp_size``-stage point do not serve, one problem each ([]
    when they serve it): a tied lm_head, a model other than 78 layers with its MTP layer 78, an indexer or MLP schedule
    the model config refuses (another length, or an entry it does not know), and a stage of the point's split that its
    boundary does not serve (_stage_problems)."""
    from sglang_neuron.models.glm52.prefill import stage_layout

    try:
        full = _full_config(config)
    except ValueError as e:  # GlmMoeDsaConfig refuses a per-layer schedule that is not one known entry per layer
        return [str(e)]
    partition = stage_layout.prefill_point(pp_size).partition
    problems = []
    if full.tie_word_embeddings:
        problems.append("tie_word_embeddings: the last stage holds an lm_head weight of its own, and the embedding it "
                        "would tie to is the first stage's; serve a checkpoint whose lm_head is not tied")
    layers = sum(partition)
    if full.num_hidden_layers != layers or full.mtp_layer_index != layers:
        problems.append(f"num_hidden_layers {full.num_hidden_layers}, MTP layer {full.mtp_layer_index}: the stages "
                        f"split {layers} layers {','.join(map(str, partition))} and the last runs the MTP layer "
                        f"{layers}")
    else:
        problems += _stage_problems(full, partition, pp_size)
    return problems


def check_hf_config(hf_config, pp_size: int = PP_SIZE) -> None:
    """Refuse a checkpoint config the stage graphs of the ``pp_size``-stage point do not serve (config_problems): the
    model host's check_launch, which reads config facts only, since it runs before the parallel groups and the stage
    exist."""
    problems = config_problems(hf_config, pp_size)
    if problems:
        raise ValueError("the GLM-5.2 prefill stages do not serve this checkpoint:\n  - " + "\n  - ".join(problems))


def _sglang_split(partition: str | None, layers: int, pp_size: int = PP_SIZE) -> list | None:
    """The layer counts of SGLang's split of ``layers`` over the stages (distributed.utils.get_pp_indices): the
    launch's SGLANG_PP_LAYER_PARTITION, else SGLang's default, which gives an uneven split's extra layers to the last
    stages; None for a partition get_pp_indices would refuse itself."""
    if partition is not None:
        try:
            sizes = [int(size) for size in partition.split(",")]
        except ValueError:
            return None
        return sizes if len(sizes) == pp_size and sum(sizes) == layers else None
    from sglang.srt.distributed.utils import get_pp_indices

    # get_pp_indices reads the variable from this process's environment, the launch's, where it is unset
    return [hi - lo for lo, hi in (get_pp_indices(layers, stage, pp_size) for stage in range(pp_size))]


def _partition_problems(environ, checkpoint_config, point) -> list:
    """A launch whose SGLANG_PP_LAYER_PARTITION is not the ``point``'s split, with each stage of the split SGLang would
    then run that the point's boundary does not serve (_stage_problems)."""
    want = ",".join(map(str, point.partition))
    partition = environ.get(PARTITION_ENV)
    if partition == want:
        return []
    problem = (f"{PARTITION_ENV}={partition if partition is not None else '(unset)'}: the stages serve the split "
               f"{want}, which the kernel's launch groups and the MoE placement are built for; launch with "
               f"{PARTITION_ENV}={want}")
    split = _sglang_split(partition, sum(point.partition), point.pp_size)
    starts = [] if split is None else _stage_problems(_full_config(checkpoint_config), split, point.pp_size)
    return [problem] + [f"{PARTITION_ENV}: {start}" for start in starts]


def context_buckets(max_context: int = MAX_CONTEXT) -> tuple:
    """The contexts the stage graphs' block tables span for a server of ``max_context`` tokens, ascending; a table's
    width in rows is the kernel's static per-rank KV context (kv_ctx_local): the CONTEXT_BUCKETS up to the smallest that
    holds ``max_context`` (128k alone at the 128k point, 128k .. 1M at the 1M point, so a short request does not pay the
    1M bucket's whole-bucket indexer)."""
    for i, bucket in enumerate(CONTEXT_BUCKETS):
        if bucket >= max_context:
            return CONTEXT_BUCKETS[:i + 1]
    raise ValueError(f"a context of {max_context} tokens is past the largest bucket {CONTEXT_BUCKETS[-1]}")


def context_bucket() -> int:
    """The largest context any stage graph's block table spans (context_buckets' last, at the measured MAX_CONTEXT)."""
    return context_buckets()[-1]


def _toolchain_problems(environ) -> list:
    """[] on the released toolchain (toolchains.is_released), which it logs once here, in the launching process whose
    PATH and environment every rank inherits; else the refusals: the graphs and the recipes are validated on it alone,
    and the release ships no recipe for any other. torch-neuronx keys every cached NEFF on NEURONXCC_VERSION and keeps a
    value it finds already set, so a launch that presets another version than the installed package's is refused too:
    its NEFFs would be cached under a compiler they were not built by."""
    from sglang_neuron.release import toolchain
    from sglang_neuron.models.glm52 import toolchains

    tc = toolchain.compiler_toolchain(environ)
    problems = []
    preset = environ.get("NEURONXCC_VERSION")
    if preset is not None and preset != tc.neuronxcc_version:
        problems.append(f"NEURONXCC_VERSION={preset}: torch-neuronx keys every cached NEFF on it and keeps a preset "
                        f"value, while the installed neuronxcc is {tc.neuronxcc_version}; launch without it")
    if toolchains.is_released(tc):
        if not problems:
            logger.info("[glm52] released toolchain %s: the runtime's own microcode and GpSIMD libraries",
                        toolchains.describe(tc))
        return problems
    if not tc.neuronx_cc:
        problems.append("no neuronx-cc on PATH: the release compiles with the neuronx-cc console script of its "
                        "neuronxcc package; put that package's bin directory on PATH")
    elif not tc.same_install:
        runs = (f"runs the neuronxcc package at {tc.compiler_package}" if tc.compiler_package
                else "is no console script of an importable neuronxcc package")
        problems.append(f"neuronx-cc at {tc.neuronx_cc} {runs}, not the one this process imports "
                        f"({tc.imported_package}), whose version every cached NEFF is keyed on; put that package's "
                        "own neuronx-cc first on PATH")
    else:
        problems.append(f"neuronx-cc {tc.neuronxcc_version} with nki {tc.nki} is not the released toolchain this "
                        f"launch is validated on ({toolchains.released_names()}); install it")
    return problems


def release_problems(environ) -> list:
    """[] when the launch's compiler, nki, torch-neuronx line and Neuron runtime and collectives libraries are the
    pinned release's (release/neuron_sdk.toml), else one problem per part that is not. Both roles compile and run
    on them, and every number the release states was measured on them."""
    from sglang_neuron.release import neuron_sdk

    return (_toolchain_problems(environ) + neuron_sdk.torch_neuronx_problems()
            + neuron_sdk.runtime_problems(environ))


# The Neuron runtime's variables that make it load a microcode or engine library other than its own. The release runs
# only the runtime's own microcode and libraries, so a launch that inherits any of them (a shell set up for another
# build) is refused instead of loading them unseen.
RUNTIME_LIBRARY_VARIABLES = frozenset({"NEURON_RT_UCODE_LIB_PATH"})
RUNTIME_LIBRARY_PREFIX = "NEURON_RT_GPSIMD_"


def runtime_library_problems(environ) -> list:
    """[] when ``environ`` sets none of the runtime's microcode or engine library variables (RUNTIME_LIBRARY_VARIABLES,
    RUNTIME_LIBRARY_PREFIX), else one refusal naming each with its value."""
    names = sorted(name for name in environ if name in RUNTIME_LIBRARY_VARIABLES or
                   name.startswith(RUNTIME_LIBRARY_PREFIX))
    if not names:
        return []
    return [f"{', '.join(f'{name}={environ[name]}' for name in names)}: the release runs the Neuron runtime's own "
            "microcode and engine libraries, and these variables load others; launch "
            "without them"]


#: Runtime modes the release does not run: the experimental per-block collectives, which a collective under device
#: control flow needs (no served kernel puts one there).
UNRELEASED_RUNTIME_MODES = frozenset({"NEURON_RT_EXPERIMENTAL_CC_CONTROL_FLOW_ENABLE"})


def unreleased_runtime_problems(environ) -> list:
    """[] when ``environ`` turns on none of UNRELEASED_RUNTIME_MODES, else one refusal naming each with its value."""
    names = sorted(name for name in UNRELEASED_RUNTIME_MODES if environ.get(name, "0") not in ("", "0"))
    if not names:
        return []
    stated = ", ".join(f"{name}={environ[name]}" for name in names)
    return [f"{stated}: an experimental Neuron runtime mode the release does not run, and no served kernel needs; "
            "launch without it"]


def launch_problems(server_args, checkpoint_config: dict, environ=os.environ) -> list:
    """Every reason the launch is not a served layout of its role, each naming the flag or variable to change ([] when
    it is): a prefill's (_prefill_problems), a decode's that its arguments already state (_decode_problems), and
    either's runtime library variables (runtime_library_problems), unreleased runtime modes
    (unreleased_runtime_problems) and release (release_problems); or the role itself for an aggregated server. Runs on
    the launch's own environment, before the platform applies the role's settings."""
    role = _role(server_args)
    if role not in ("prefill", "decode"):
        return [f"--disaggregation-mode {role}: the GLM-5.2-family graphs serve the PD roles only, the prefill on the "
                "prefill stages and the decode on the GK, and no aggregated server; launch with --disaggregation-mode "
                "prefill or decode"]
    own = (_prefill_problems(server_args, checkpoint_config, environ) if role == "prefill"
           else _decode_problems(server_args, environ))
    # both roles: the decode's 128-rank NEFFs are as toolchain-specific as the prefill's (the compile cache keys on the
    # toolchain digest), and the GK decode is measured on the released toolchain only
    return own + runtime_library_problems(environ) + unreleased_runtime_problems(environ) + release_problems(environ)


def _decode_problems(server_args, environ) -> list:
    """What of a decode launch its arguments and environment already refuse: a world other than the GK's 128 ranks
    (any other is no giga-kernel decode, serving_policy.giga_decode_layout, and would take the prefill's settings),
    pipeline stages, which the catalog entry builds for the prefill alone, SGLang's prefill context parallelism, under
    which upstream's DSA provider re-plans the attention groups (sglang_patches/dsa_platform_adjustments): the GK runs every
    layer and its own context parallelism in one program; a compile knob its static recipe does not set
    (refuse_unpinned_gk);
    a KV transfer backend other than NIXL, whose WRITEs land in the GK's own caches (the decode pool registers them,
    decode.pools.gk_kv_regions), or SGLang's fake one for a decode measured alone; the decode's radix cache, since
    the GK's caches are slot-private and a prefix hit would address another slot's rows; a --page-size that is no
    served prefill point's page (paired_prefill_point), whose prefill this decode cannot pair with; a sampler the
    GK's tail does not serve (the catalog entry's giga_decode_samplers) or top logprobs; and the Quark AttnFP8 export,
    which the prefill refuses too and whose FP8 and MXFP4 attention and dense weights the decode's weight stores do
    not read (layout/quark_compat)."""
    from sglang_neuron import sampling
    from sglang_neuron.models import catalog
    from sglang_neuron.models.glm52.layout import quark_compat
    from sglang_neuron.models.glm52.layout.decode.gk_geometry import Glm52GkGeometry

    problems = []
    if paired_prefill_point(server_args) is None:
        problems.append(f"--page-size {server_args.page_size}: the GLM decode serves the page of the prefill it pairs "
                        "with, one 32-row block on each rank of a prefill stage, and the served prefills' pages are "
                        f"{_served_decode_pages()}; launch with the --page-size of the paired prefill, or without it "
                        "for the default")
    ranks, tp = Glm52GkGeometry().ranks, _int(server_args, "tp_size")
    if tp != ranks:
        problems.append(f"--tp-size {tp}: the decode giga kernel is one program over the node's {ranks} ranks; launch "
                        f"the decode with --tp-size {ranks} --ep-size {ranks}")
    pp = _int(server_args, "pp_size")
    if pp > 1:
        problems.append(f"--pp-size {pp}: the decode giga kernel runs every layer in one program over the node's 128 "
                        "ranks; launch the decode without --pp-size")
    if getattr(server_args, "enable_prefill_cp", False):
        problems.append("--enable-prefill-cp: the decode giga kernel runs its own context parallelism; launch without "
                        "it")
    problems += refuse_unpinned_gk(environ)
    if getattr(server_args, "disaggregation_decode_enable_radix_cache", False):
        problems.append("--disaggregation-decode-enable-radix-cache: the decode giga kernel's caches are slot-private, "
                        "so a prefix hit would point a request at another slot's rows; launch without it")
    mode = sampling.sampler_mode()
    samplers = catalog.MODELS["GlmMoeDsaForCausalLM"].giga_decode_samplers
    if mode not in samplers or sampling.MAX_TOP_LOGPROBS > 0:
        problems.append(f"SGLANG_NEURON_SAMPLER={mode}, SGLANG_NEURON_MAX_TOP_LOGPROBS={sampling.MAX_TOP_LOGPROBS}: "
                        "the decode giga kernel's LM-head tail samples on device, greedily (greedy) or with each "
                        "request's top_k, top_p and temperature over its 256 candidates (topk_topp), and returns no "
                        f"logprobs; launch with SGLANG_NEURON_SAMPLER={' or '.join(samplers)} and "
                        "SGLANG_NEURON_MAX_TOP_LOGPROBS=0")
    if quark_compat.checkpoint_format(server_args.model_path) == quark_compat.QUARK_ATTN_FP8:
        problems.append(f"{server_args.model_path} is a Quark AttnFP8 export: the decode's weight stores read the "
                        "attention and dense MLP as the MXFP4 export's bf16 tensors, as the prefill does; serve the "
                        "MXFP4 export")
    return problems


def _prefill_problems(server_args, checkpoint_config: dict, environ) -> list:
    """Every reason a prefill launch is not the served layout, each naming the flag or variable to change ([] when it
    is). Runs on the launch's own environment, before the platform applies the recipe."""
    from sglang_neuron import sampling, sglang_tree
    from sglang_neuron.models.glm52.prefill import stage_layout
    from sglang_neuron.models.glm52.layout import quark_compat

    problems = []
    point = served_prefill_point(server_args)
    if point is None:
        point = stage_layout.PREFILL_POINTS[PP_SIZE]
        problems.append(f"--pp-size {_int(server_args, 'pp_size')}: the stages serve "
                        f"{' or '.join(str(p) for p in sorted(stage_layout.PREFILL_POINTS))} pipeline stages; launch "
                        f"with --pp-size {PP_SIZE} (or 16)")
    ranks, chunk, smaller_graphs = point.stage_ranks, point.chunk, point.smaller
    for flag, name, want in (("--tp-size", "tp_size", ranks), ("--ep-size", "ep_size", ranks)):
        have = _int(server_args, name)
        if have != want:
            problems.append(f"{flag} {have}: the stages are {point.pp_size} pipeline stages of {ranks} ranks with "
                            f"{ranks}-way expert parallelism; launch with {flag} {want}")
    replicas, attention_dp = sglang_tree.dp_widths(server_args)
    if replicas > 1 or attention_dp > 1:
        problems.append("--dp-size / --enable-dp-attention: a stage's ranks run one sequence's context in parallel; "
                        "launch without them")
    if getattr(server_args, "enable_prefill_cp", False):
        problems.append("--enable-prefill-cp: a stage is the kernel's own context-parallel group (--tp-size); launch "
                        "without it")
    if _int(server_args, "dcp_size") > 1:
        problems.append(f"--dcp-size {_int(server_args, 'dcp_size')}: decode context parallelism has no prefill "
                        "role; launch without it")
    problems += _partition_problems(environ, checkpoint_config, point)
    have_chunk = getattr(server_args, "chunked_prefill_size", None)
    if have_chunk != chunk:
        problems.append(f"--chunked-prefill-size {have_chunk}: a stage's largest graph runs {chunk} tokens, "
                        f"{chunk // ranks} rows per rank; launch with --chunked-prefill-size {chunk}")
    counts = getattr(server_args, "cuda_graph_bs_prefill", None)
    if counts is None or sorted(int(count) for count in counts) != list(smaller_graphs):
        smaller = " ".join(map(str, smaller_graphs))
        problems.append(f"--cuda-graph-bs-prefill {counts}: the stages compile one graph each of {smaller} and "
                        f"{chunk} tokens; launch with --cuda-graph-bs-prefill {smaller}")
    page = point.virtual_page
    context = getattr(server_args, "context_length", None)
    bucket = context_bucket()
    if context is None or not 0 < int(context) <= bucket:
        # each step's block table spans a context bucket of the stage graphs (prefill.runtime), and the kernel's static
        # per-rank KV context is the table's width in rows (kv_ctx_local)
        problems.append(f"--context-length {context}: the stage graphs' block table holds the "
                        f"{bucket}-token bucket, {bucket // page} pages x {stage_layout.CP_INTERLEAVE} rows "
                        f"= {bucket // ranks} per rank, the kernel's static KV context, so a request fits "
                        f"it only up to {bucket} tokens; launch with --context-length {bucket} or less")
    total = getattr(server_args, "max_total_tokens", None)
    if total is None or int(total) % page:
        problems.append(f"--max-total-tokens {total}: the stage pools hold the stated tokens in whole {page}-token "
                        f"pages and size no pool from free memory; launch with --max-total-tokens, a multiple of "
                        f"{page}")
    if getattr(server_args, "radix_cache_backend", None) == "neuron-chunk":
        problems.append("--radix-cache-backend neuron-chunk: the stages keep SGLang's radix cache over whole pages; "
                        "launch without it")
    if sampling.sampler_mode() not in FIRST_TOKEN_SAMPLERS or sampling.MAX_TOP_LOGPROBS > 0:
        problems.append(f"SGLANG_NEURON_SAMPLER={sampling.sampler_mode()}, SGLANG_NEURON_MAX_TOP_LOGPROBS="
                        f"{sampling.MAX_TOP_LOGPROBS}: the last stage selects greedily or draws as the GK decode tail "
                        "draws over its vocabulary shards, and returns no top logprobs; launch with "
                        f"SGLANG_NEURON_SAMPLER={' or '.join(FIRST_TOKEN_SAMPLERS)} and "
                        "SGLANG_NEURON_MAX_TOP_LOGPROBS=0")
    if quark_compat.checkpoint_format(server_args.model_path) == quark_compat.QUARK_ATTN_FP8:
        problems.append(f"{server_args.model_path} is a Quark AttnFP8 export: the recipe's MXFP8 o_proj "
                        "(GLM_GK_PRJ_OMX=1) packs the o_proj rows past the loaders' FP8 scale upcast; serve the "
                        "MXFP4 export")
    problems += config_problems(checkpoint_config, point.pp_size)
    problems += refuse_unpinned(environ, point.pp_size)
    return problems


# The GK decode's collective DMA queues at the runtime's highest service level, 0 (lower is higher), ahead of the
# weight stream that keeps the decode's DMA queues busy for the whole layer: a shorter step than at the plugin's level
# 3 on the 128k decode (same node, 8k context x 32 requests, measured twice). Measured on this decode only, so it is
# this model's.
DECODE_RUNTIME_ENV = {"NEURON_RT_DBG_CC_DMA_QOS_SERVICE_LEVEL": "0"}


def validate_request(obj, preferred, server_args, vocab_size: int) -> None:
    """The model's own admission rules: a request that names golden tokens is admitted only on a logit test server
    (logit_probe.validate_request)."""
    from sglang_neuron.models.glm52 import logit_probe

    logit_probe.validate_request(obj, preferred, server_args, vocab_size)


def rank_runtime_env(role: str) -> dict:
    """The Neuron runtime settings a rank of ``role`` (serving_policy.ServingPolicy.role) starts with over the plugin's
    (runtime_init.runtime_defaults), unless the launch sets them: the decode's DECODE_RUNTIME_ENV, none on a prefill.
    Nothing else: on the released toolchain the runtime runs its own microcode and GpSIMD libraries, the GK's kernel
    settings are its launch settings (catalog.ModelSpec.giga_decode_launch_settings), and the toolchain is checked and
    logged once, in the launching process (_toolchain_problems), whose PATH and environment every rank inherits."""
    return dict(DECODE_RUNTIME_ENV) if role == "decode" else {}


def fleet_digest(pp_size: int = PP_SIZE) -> tuple:
    """What every rank of the prefill must hold the same (models.contract.assert_fleet_consistent): the recipe, the
    graphs' compiler arguments, the toolchain (toolchain_digest: the compiler, the package it runs and the versions), the
    kernel's digest (its files and compile knobs, as its wrapper keys the kernel), the placement file, the split and
    the logical-core degree."""
    from sglang_neuron.models.glm52 import kernels as _kernels
    from sglang_neuron.release import toolchain
    from sglang_neuron.models.glm52.prefill import compile_args
    from sglang_neuron.models.glm52.prefill import expert_placements as placements
    from sglang_neuron.models.glm52.prefill import recipe
    from sglang_neuron.models.glm52.prefill import stage_layout

    kernel = _kernels.kernel_digest(os.path.join(os.path.dirname(_kernels.__file__), "prefill"))
    placement = placements.file_md5(placements.PLACEMENT_FILE)
    point = stage_layout.prefill_point(pp_size)
    return (recipe.digest(pp_size), compile_args.stage_compiler_args(), toolchain.toolchain_digest(), kernel,
            placement if pp_size == PP_SIZE else None, list(point.partition),
            int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "0") or 0))
