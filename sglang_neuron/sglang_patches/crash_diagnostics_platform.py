# SPDX-License-Identifier: Apache-2.0
"""A crashing Neuron server does not ask its schedulers for CUDA coredumps, nor wait 60 s for them.

Upstream's ``TokenizerManager.dump_requests_before_crash`` (SIGTERM, SIGQUIT from the subprocess watchdog, an unhandled
exception) reads ``SGLANG_CUDA_COREDUMP_BEFORE_CRASH`` (default on) without asking the platform: on Neuron it signals
every scheduler for a CUDA user coredump that never comes and then sleeps
``SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS`` (60 s) before the server exits, so every crash holds the node a minute
longer, and a launcher's restart waits on it. Here the method is upstream's, rebuilt with the coredump decision false:
the platform is not CUDA. The request dump and the py-spy dump (``SGLANG_PYSPY_DUMP_BEFORE_CRASH``) stay upstream's.

The tokenizer manager runs in the launching process, so the replacement is one of the launch hooks.

Upstream: a fix SGLang can take: gate the CUDA coredump on the platform's is_cuda.

Target, md5-pinned against upstream SGLang 43881a7: ``TokenizerManager.dump_requests_before_crash`` (rebuilt)."""

from __future__ import annotations

_TOKENIZER = "sglang.srt.managers.tokenizer_manager"
TARGET = f"{_TOKENIZER}.TokenizerManager.dump_requests_before_crash"
_TARGETS_MD5 = {
    (_TOKENIZER, "TokenizerManager.dump_requests_before_crash"): "d7d41abe1b6a8f89e00c73612583a4b0",
}
# upstream's coredump decision and its replacement: the Neuron platform is not CUDA
SUBSTITUTIONS = (
    ("    should_dump_cuda_coredump = envs.SGLANG_CUDA_COREDUMP_BEFORE_CRASH.get()\n",
     "    # sglang_neuron: no CUDA coredump on the Neuron platform (crash_diagnostics_platform)\n"
     "    should_dump_cuda_coredump = False\n"),
)


def dump_requests_before_crash():
    """TokenizerManager.dump_requests_before_crash, upstream's without the CUDA coredump and its wait."""
    from sglang_neuron.sglang_patches import source_guard

    cls = source_guard.resolve(_TOKENIZER, "TokenizerManager")
    return source_guard.with_substitutions(cls.__dict__["dump_requests_before_crash"], SUBSTITUTIONS)


def register(registry, replace) -> None:
    from sglang_neuron.sglang_patches import source_guard

    source_guard.check_targets(_TARGETS_MD5, __name__)
    registry.register(TARGET, dump_requests_before_crash(), replace)
