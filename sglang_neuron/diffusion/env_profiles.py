# SPDX-License-Identifier: Apache-2.0
"""Per-model compiler and runtime environment defaults, written to ``os.environ`` before workers
spawn.

Each field name is the environment variable. Every value is a default: it is written only when
the variable is unset, so an exported value wins. The compiler variables are the exception
(``COMPILER_ENV``): the plugin sets them per graph, so the launch refuses an exported value
that is not the profile's (:func:`refuse_launch_compiler_env`). A profile may also force a few
variables (``EnvWrites.assign``): Wan2.2's two CC DMA packet-size variables.
"""

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnvWrites:
    """Environment writes: ``setdefault`` only when the variable is unset; ``assign`` always
    (a different exported value is replaced, with a warning)."""

    setdefault: Mapping[str, str] = field(default_factory=dict)
    assign: Mapping[str, str] = field(default_factory=dict)


@runtime_checkable
class EnvProfile(Protocol):
    def to_env(self) -> EnvWrites: ...


def merge(*sources: EnvWrites) -> EnvWrites:
    """Combine sources; a later source's value for a variable replaces an earlier one's."""
    setdefault: dict[str, str] = {}
    assign: dict[str, str] = {}
    for source in sources:
        setdefault.update(source.setdefault)
        assign.update(source.assign)
    return EnvWrites(setdefault=setdefault, assign=assign)


def thread_limits(tp_size: int) -> EnvWrites:
    """OMP/MKL thread caps of ``cpus // tp_size``; a value the launcher already set wins."""
    if hasattr(os, "sched_getaffinity"):
        cpus = len(os.sched_getaffinity(0))
    else:
        cpus = os.cpu_count() or 1
    threads = str(max(1, cpus // tp_size))
    return EnvWrites(setdefault={"OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads})


# The compiler variables the plugin sets per graph: compile_backend replaces both for the
# compile of a graph with its own compiler arguments (the DiT, the VAE).
COMPILER_ENV = ("NEURON_CC_FLAGS", "NEURON_COMPILER_OPT_LEVEL")


def refuse_launch_compiler_env(profile: EnvProfile | None, environ=None) -> None:
    """Refuse a launch environment that sets a compiler variable to another value than ``profile``.

    The DiT and the VAE compile with their own compiler arguments, which replace both
    variables for their compile; the other graphs take the profile's values. An exported value
    would reach some graphs and not others, so it is refused, naming it, as the LLM path refuses
    it (``backend.stack.refuse_launch_compiler_env``). The profile's own value is accepted.
    """
    environ = os.environ if environ is None else environ
    wanted = profile.to_env().setdefault if profile is not None else {}
    preset = sorted(n for n in COMPILER_ENV if n in environ and environ[n] != wanted.get(n))
    if preset:
        raise ValueError(
            f"{', '.join(preset)} in the launch environment: the Neuron diffusion platform sets "
            "the compiler arguments per graph (the DiT and the VAE have their own); unset "
            f"{'them' if len(preset) > 1 else 'it'}"
        )


def apply(*sources: EnvProfile | EnvWrites) -> None:
    """Merge every source into os.environ: a ``setdefault`` variable only when it is not already
    set, an ``assign`` one always (a different exported value is replaced, with a warning)."""
    writes = merge(*(s.to_env() if isinstance(s, EnvProfile) else s for s in sources))
    for name, value in writes.setdefault.items():
        os.environ.setdefault(name, value)
    for name, value in writes.assign.items():
        exported = os.environ.get(name)
        if exported is not None and exported != value:
            logger.warning("%s=%s replaced by the model profile's %s", name, exported, value)
        os.environ[name] = value


def effective(*sources: EnvProfile | EnvWrites) -> dict[str, str | None]:
    """The current value of every variable the sources write (for a launch log line)."""
    writes = merge(*(s.to_env() if isinstance(s, EnvProfile) else s for s in sources))
    names = sorted({*writes.setdefault, *writes.assign})
    return {name: os.environ.get(name) for name in names}


@dataclass(frozen=True)
class DiffusionEnvProfile:
    TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION: str | None = "1"
    NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE: str | None = "167772160"
    # Must match --hbm-scratchpad-page-size in NEURON_CC_FLAGS below; change both together.
    NEURON_SCRATCHPAD_PAGE_SIZE: str | None = "2048"
    NEURON_RT_DBG_CC_DMA_PACKET_SIZE: str | None = "2048"
    NEURON_LOGICAL_NC_CONFIG: str | None = "2"
    NEURON_PLATFORM_TARGET_OVERRIDE: str | None = None
    NEURON_CC_FLAGS: str | None = "-O1 --hbm-scratchpad-page-size=2048"
    TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS: str | None = "True"
    # CWD-relative, so callers whose working dir is not writable override it.
    TORCH_NEURONX_DEBUG_DIR: str | None = "./compile_dir"
    # The other ranks wait on this lock while one rank compiles a shared NEFF; the
    # torch-neuronx default (1200 s) is shorter than the largest compiles.
    TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT: str | None = "3600"

    extra: EnvWrites = field(default_factory=EnvWrites)

    def to_env(self) -> EnvWrites:
        """The fields as environment defaults, then ``extra``; a None field is skipped."""
        setdefault = {
            spec.name: getattr(self, spec.name)
            for spec in fields(self)
            if spec.name != "extra" and getattr(self, spec.name) is not None
        }
        return merge(EnvWrites(setdefault=setdefault), self.extra)


FLUX1_DEV = DiffusionEnvProfile(
    # A 4096 px DiT compile takes longer than the default lock budget.
    extra=EnvWrites(setdefault={"TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT": "14400"}),
)
MINIMAX_H3 = DiffusionEnvProfile()

# Wan2.2 A14B T2V and I2V: the class's defaults (the fallback switch, the RDH channel buffer, the
# 2 KiB scratchpad page and CC DMA packet, LNC 2, "-O1 --hbm-scratchpad-page-size=2048", preserved
# artifacts in ./compile_dir, and a 3600 s NEFF cache lock, which is longer than a 1080p DiT
# compile), plus:
# - The CC DMA packet size is forced to 2048 under both names the runtime reads (assigned, so an
#   exported value is replaced; the ring attention's collectives are tuned for it).
# - NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS is not set: NRT >= 2.35 ignores it (and warns on
#   every worker).
WAN22_CC_DMA_PACKET_SIZE = {
    "NEURON_RT_DBG_CC_DMA_PACKET_SIZE": "2048",
    "NEURON_RT_DBG_CC_DMA_COPY_DMA_PRIO_PKT_SIZE": "2048",
}
WAN22 = DiffusionEnvProfile(
    extra=EnvWrites(assign=WAN22_CC_DMA_PACKET_SIZE),
)

# LingBot-World-V2 real-time serving: every component at LNC 1 (the DiT TP8 x CP8 on 64 physical
# cores, the role workers' 16-rank world on cores 64-79). The Wan runtime debug buffers (intra-RDH
# channel buffer, CC DMA packet size) stay at the runtime defaults, as the pipeline was validated.
# A cold load compiles the 2 x 8 VAE / encoder world for a long time, hence the 14400 s NEFF
# lock. No compile artifacts under the working directory (a served process).
# LNC 1 is assigned, not defaulted: the Neuron container exports NEURON_LOGICAL_NC_CONFIG=2, and
# the core layout (TP8 x CP8 on 64 physical cores, the role workers on 64-79) is LNC 1's.
LINGBOT_WORLD = DiffusionEnvProfile(
    NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE=None,
    NEURON_RT_DBG_CC_DMA_PACKET_SIZE=None,
    NEURON_LOGICAL_NC_CONFIG=None,
    TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS=None,
    TORCH_NEURONX_DEBUG_DIR=None,
    TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT="14400",
    extra=EnvWrites(assign={"NEURON_LOGICAL_NC_CONFIG": "1"}),
)

# Process-wide defaults, written by the platform for every model before workers spawn.
PLATFORM_DEFAULTS = EnvWrites(
    setdefault={
        # Each worker creates hundreds of process groups at start-up, and Gloo connects every
        # group's full mesh eagerly. On 64-rank worlds (trn2.48xlarge) a rank can stall
        # mid-connect, and its peers then fail with "Gloo connectFullMesh failed ...
        # Connection closed by peer". Lazy init connects a group on its first collective.
        "TORCH_GLOO_LAZY_INIT": "1",
    }
)
