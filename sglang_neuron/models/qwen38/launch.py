# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8's launch rules (catalog.ModelSpec.launch_hooks). Module-level imports are stdlib only.

The KV pool is paged (models/qwen38/kv_pool.py): the attention kernels read a request's pages through SGLang's block
table, so the page is a fixed size (``PAGE_SIZE``, a power of two of at least 128 tokens, which the kernels require)
and a prompt is prefilled in chunks of whole pages. The kernels run one rank per physical NeuronCore (LNC1), which
the catalog entry states, so a trn2.3xlarge serves this model at ``--tp-size 8``.
"""

import os


PAGE_SIZE = 512


def kv_page_size(server_args) -> int:
    """The KV page the attention kernels read through the block table: the launched --page-size (launch_problems
    checks it is a power of two of at least 128), else ``PAGE_SIZE``."""
    return int(getattr(server_args, "page_size", None) or PAGE_SIZE)


def launch_problems(server_args, checkpoint_config: dict, environ=os.environ) -> list:
    problems = []
    page = int(getattr(server_args, "page_size", None) or PAGE_SIZE)
    if page < 128 or page & (page - 1):
        problems.append(f"--page-size {page}: the attention kernels read pages of a power of two of at least 128")
    chunk = getattr(server_args, "chunked_prefill_size", None)
    if chunk and int(chunk) % page:
        problems.append(f"--chunked-prefill-size {chunk}: a prefill chunk is whole {page}-token pages")
    if not getattr(server_args, "disable_radix_cache", False):
        # the prefix cache keeps the GDN state only at a prefill forward's end, so its checkpoints must be the
        # prefill chunks (the platform checks the chunk grid and the tracking buffer, state_pool.radix_refusal)
        backend = getattr(server_args, "radix_cache_backend", None)
        if backend != "neuron-chunk":
            problems.append(f"--radix-cache-backend {backend}: Qwen3.8's prefix cache keeps the GDN state at prefill "
                            "chunk ends, the chunk-keyed tree: launch with --radix-cache-backend neuron-chunk "
                            "--mamba-radix-cache-strategy extra_buffer --mamba-track-interval <chunk>, or with "
                            "--disable-radix-cache")
        interval = getattr(server_args, "mamba_track_interval", None)
        if chunk and interval and int(interval) != int(chunk):
            problems.append(f"--mamba-track-interval {interval}: the GDN state is checkpointed at each prefill "
                            f"chunk's end, so the interval is the chunk ({chunk})")
    tp = int(getattr(server_args, "tp_size", 1) or 1)
    # CPU mode runs the kernels' torch references, any TP; on the device the kernels need one rank per core
    if environ.get("SGLANG_NEURON_CPU_MODE", "0") != "1" and tp != 8:
        problems.append(f"--tp-size {tp}: the kernels run one rank per physical NeuronCore (LNC1), so this model "
                        "serves at --tp-size 8 on a trn2.3xlarge")
    return problems


def launch_settings(server_args) -> tuple:
    """The settings the Qwen3.5 kernels read when they are imported. Their GDN prefill's widest matmul defaults to
    2048 moving columns, which the Trainium3 NeuronCore (gen4) takes; the Trainium2 NeuronCore (gen3) takes 512
    (neuronx-cc: "Matmul moving free dimension 2048 exceeds max 512 for nc_version gen3"). At 512 the kernel matches
    its reference on a trn2.3xlarge (notes: Q6)."""
    from sglang_neuron.models.catalog import LaunchSetting

    return (LaunchSetting("QWEN35_GDNP_WIDE", "512",
                          "the GDN prefill kernel's widest matmul; a Trainium2 NeuronCore takes 512 moving columns"),)


def rank_runtime_env(role: str) -> dict:
    return {}
