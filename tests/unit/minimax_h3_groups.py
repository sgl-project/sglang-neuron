# SPDX-License-Identifier: Apache-2.0
"""One-rank gloo groups for the MiniMax-H3 DiT's CPU tests (``SGLANG_NEURON_CPU_MODE=1``)."""

import contextlib
import socket

import pytest


@contextlib.contextmanager
def single_rank_groups():
    """SGLang's world, TP, SP and CFG groups of one rank, bound as the models' groups, torn down
    on exit. Skips when a default process group already exists, so it only destroys what it
    created."""
    import torch.distributed as dist
    from sglang.multimodal_gen.runtime.distributed import parallel_state as sglang_groups

    from sglang_neuron.diffusion.distributed import groups
    from sglang_neuron.diffusion.distributed.parallel_state import initialize_parallel_groups

    if dist.is_initialized() or groups.model_parallel_is_initialized():
        pytest.skip("a default process group is already initialized")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    try:
        sglang_groups.init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="gloo",
        )
        sglang_groups.initialize_model_parallel(
            sequence_parallel_degree=1, ring_degree=1, tensor_parallel_degree=1, backend="gloo"
        )
        initialize_parallel_groups("cpu", world_size=1, tensor_parallel_size=1)
        yield
    finally:
        # Best effort: a teardown error must not hide a setup or test failure.
        groups._WORLD = groups._TP = groups._CP = groups._CFG = None
        for teardown in (
            sglang_groups.destroy_model_parallel,
            sglang_groups.destroy_distributed_environment,
        ):
            with contextlib.suppress(Exception):
                teardown()
        if dist.is_initialized():
            dist.destroy_process_group()
