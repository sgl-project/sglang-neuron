# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Mini's launch rules (catalog.ModelSpec.launch_hooks). Module-level imports are stdlib only.

The graphs read and write 256-token pages (the conv-state record's unit, graph.StateLayout), a prompt is prefilled in
chunks of whole pages, and the ranks run one per physical NeuronCore (LNC1, the catalog entry), so a trn2.3xlarge
serves the model at ``--tp-size 8``. The checkpoint's config type (``inkling2_loop``) is newer than the image's
transformers: ``prepare_process`` registers it in every process before SGLang reads the config.
"""

import os

PAGE_SIZE = 256


def kv_page_size(server_args) -> int:
    return int(getattr(server_args, "page_size", None) or PAGE_SIZE)


def launch_problems(server_args, checkpoint_config: dict, environ=os.environ) -> list:
    problems = []
    page = int(getattr(server_args, "page_size", None) or PAGE_SIZE)
    if page != PAGE_SIZE:
        problems.append(f"--page-size {page}: the graphs read and write {PAGE_SIZE}-token pages")
    chunk = getattr(server_args, "chunked_prefill_size", None)
    if chunk and int(chunk) > 0 and int(chunk) % PAGE_SIZE:
        problems.append(f"--chunked-prefill-size {chunk}: a prefill chunk is whole {PAGE_SIZE}-token pages")
    mode = str(getattr(server_args, "disaggregation_mode", "null") or "null")
    if mode != "null":
        problems.append(f"--disaggregation-mode {mode}: the model serves aggregated only (its PD layout is not "
                        "validated)")
    if getattr(server_args, "enable_dp_attention", False):
        problems.append("--enable-dp-attention: the model serves attention TP over all ranks")
    if int(getattr(server_args, "dcp_size", 1) or 1) > 1:
        problems.append("--dcp-size above 1: the graphs read each request's pages whole on every rank")
    if int(getattr(server_args, "pp_size", 1) or 1) > 1:
        problems.append("--pp-size above 1: the looped decoder is not split into pipeline stages")
    if getattr(server_args, "speculative_algorithm", None):
        problems.append(f"--speculative-algorithm {server_args.speculative_algorithm}: the checkpoint's DSpark "
                        "draft is not served")
    tp = int(getattr(server_args, "tp_size", 1) or 1)
    if environ.get("SGLANG_NEURON_CPU_MODE", "0") != "1" and tp != 8:
        problems.append(f"--tp-size {tp}: one rank per physical NeuronCore (LNC1), so --tp-size 8 on a trn2.3xlarge")
    return problems


def launch_settings(server_args) -> tuple:
    return ()


def rank_runtime_env(role: str) -> dict:
    return {}


def prepare_process() -> None:
    """Register the checkpoint's config classes (``inkling2_loop``) with transformers in this process."""
    from sglang_neuron.models.inkling2.config import register_hf_configs

    register_hf_configs()
