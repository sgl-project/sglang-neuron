# SPDX-License-Identifier: Apache-2.0
"""Gemma 4's launch rules (catalog.ModelSpec.launch_hooks). Module-level imports are stdlib only.

The KV pool is SGLang's uniform one (hybrid SWA memory off); the host declares how many pool heads a token needs
(sglang_model._declare_kv_layout)."""

PAGE_SIZE = 128


def kv_page_size(server_args) -> int:
    return PAGE_SIZE


def launch_problems(server_args, checkpoint_config: dict, environ=None) -> list:
    problems = []
    if not getattr(server_args, "disable_hybrid_swa_memory", False):
        problems.append("Gemma 4 on Neuron serves one uniform KV pool: launch with --disable-hybrid-swa-memory")
    return problems


def launch_settings(server_args) -> tuple:
    return ()


def rank_runtime_env(role: str) -> dict:
    return {}
