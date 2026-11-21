# SPDX-License-Identifier: Apache-2.0
"""CPU contract for the MiniMax-H3 conditioner's streamed blocks (``stream_layers``) and the
file-backed host copies they stream from (``host_weights.to_mapped``).

Streaming swaps each block's parameters between a host copy and a device copy around its
launch; whatever the chunk size, the blocks must compute what the resident stack does, and
between requests every block must point back at the host copy. The NKI blocks themselves need a
device, so a stand-in block with the same signature and parameter layout runs here.
"""

import multiprocessing as mp

import pytest
import torch
from torch import nn

pytest.importorskip("sglang.multimodal_gen")

from sglang_neuron.diffusion.models.minimax_h3 import encoders  # noqa: E402
from sglang_neuron.diffusion.models.minimax_h3.host_weights import (  # noqa: E402
    HostSwap,
    to_mapped,
)

HIDDEN, LAYERS = 16, 7


class _Block(nn.Module):
    """``NeuronQwen3VLDecoderLayer``'s signature, with nested parameters like its own."""

    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.o_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.mlp = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, hidden_states, residual, cos, sin):
        if residual is None:
            residual = hidden_states
        out = torch.tanh(self.self_attn.o_proj(hidden_states) * cos.mean() + sin.mean())
        return self.mlp(out), residual + out


def _model(stream_layers):
    model = object.__new__(encoders.NeuronQwen3VLTextModel)
    nn.Module.__init__(model)
    model._tp_size = 1
    model.stream_layers = stream_layers
    model._host_weights = None
    model._compiled_layers = None
    model._compiled_embed = None
    model.embed_tokens = nn.Embedding(11, HIDDEN)
    torch.manual_seed(0)
    model.layers = nn.ModuleList(_Block() for _ in range(LAYERS))
    return model


def _inputs():
    torch.manual_seed(1)
    return torch.randint(0, 11, (5,)), torch.randn(5, 4), torch.randn(5, 4), torch.zeros(1)


@pytest.mark.parametrize("stream_layers", [1, 2, 3, LAYERS])
@pytest.mark.parametrize("mapped", [False, True])
def test_streamed_blocks_match_the_resident_stack(stream_layers, mapped, tmp_path):
    reference = _model(0)
    streamed = _model(stream_layers)
    streamed.load_state_dict(reference.state_dict())
    streamed.keep_layers_on_host(str(tmp_path) if mapped else None, "te_r0of1")
    host = {
        (i, n): p.data_ptr()
        for i, layer in enumerate(streamed.layers)
        for n, p in layer.named_parameters()
    }

    input_ids, cos, sin, rank = _inputs()
    with torch.no_grad():
        expected = reference(input_ids, cos, sin, rank)
        for _ in range(2):  # a second request reuses the host copy
            actual = streamed.forward_streamed(torch.device("cpu"), input_ids, cos, sin, rank)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            after = {
                (i, n): p.data_ptr()
                for i, layer in enumerate(streamed.layers)
                for n, p in layer.named_parameters()
            }
            assert after == host  # every block points back at the host copy
    if mapped:
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "te_r0of1.bin",
            "te_r0of1.bin.done",
            "te_r0of1.bin.lock",
        ]


def _map_in_child(directory, queue):
    torch.manual_seed(2)
    tensors = {"a": torch.randn(3, 5, dtype=torch.bfloat16), ("b", 1): torch.arange(4)}
    out = to_mapped({k: v.clone() for k, v in tensors.items()}, "vae", directory)
    queue.put(all(torch.equal(out[k], v) and out[k].dtype == v.dtype for k, v in tensors.items()))


def test_ranks_with_the_same_key_share_one_file(tmp_path):
    context = mp.get_context("fork")
    queue = context.Queue()
    workers = [context.Process(target=_map_in_child, args=(str(tmp_path), queue)) for _ in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert [queue.get() for _ in workers] == [True] * 4
    assert sorted(p.name for p in tmp_path.iterdir()) == ["vae.bin", "vae.bin.done", "vae.bin.lock"]


@pytest.mark.parametrize("mapped", [False, True])
def test_host_swap_exchanges_and_restores_the_weights(mapped, tmp_path):
    torch.manual_seed(3)
    out_module, in_module = nn.Linear(4, 4), nn.Linear(4, 2)
    in_module.register_buffer("scale", torch.ones(2))
    expected_out = {k: v.clone() for k, v in out_module.state_dict().items()}
    expected_in = {k: v.clone() for k, v in in_module.state_dict().items()}
    directory = str(tmp_path) if mapped else None
    # On CPU every tensor is "on the host", so the device-resident filter is off here.
    out = HostSwap([out_module], torch.device("cpu"), directory, "dit", on_device=False)
    out.on_device = True
    into = HostSwap([in_module], torch.device("cpu"), directory, "vae", on_device=False)
    x = torch.randn(3, 4)
    reference = in_module(out_module(x))
    for _ in range(2):
        with HostSwap.exchange(out, into):
            torch.testing.assert_close(in_module(out_module(x)), reference, rtol=0, atol=0)
        for module, expected in ((out_module, expected_out), (in_module, expected_in)):
            for key, value in module.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


def test_host_swap_leaves_host_resident_tensors_alone():
    """``on_device``: only the tensors that are on the device take part (``adaln_on_host`` keeps
    the AdaLN projections on the host, and a swap must not move them)."""
    module = nn.Linear(4, 4)
    host_only = module.weight
    swap = HostSwap([module], torch.device("cpu"), None, "dit", on_device=True)
    assert swap._slots == []  # on CPU nothing is "on the device"
    swap.evict()
    swap.load()
    assert module.weight is host_only
