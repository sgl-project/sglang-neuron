# SPDX-License-Identifier: Apache-2.0
"""A size-changing reinterpret of a tensor on the device, at LNC1: a one-DMA NKI bitcast.

The GLM-5.2 prefill stage writes the fp8 indexer keys and the MTP prompt rows in the decode node's byte formats, which
takes reinterprets that change the element size (fp8 keys as uint8 bytes, the bytes as bf16 words of the key cache, a
latent row's bf16 and fp8 halves as one bf16 row). The FX -> HLO lowering has no size-changing bitcast, so it runs as a
one-DMA NKI kernel (_bitcast_nki, at grid [1]: the stage graph is an LNC1 program). It is the same device work the measured stage
step pays per full layer and per MTP row store, so it stays as it is for a like-for-like stage step (a torch view
lowering is not tried). On the host (the CPU tests and a CPU-mode process) it is Tensor.view.
"""

from __future__ import annotations

import torch

from sglang_neuron.backend.stack import wrap_nki

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
except Exception:  # noqa: BLE001 -- a host without the NKI DSL (the CPU tests): the bitcast is a view there
    nki = nl = nisa = None

#: The nki.language dtype of each torch dtype the stores reinterpret into, by name: nki.language is not importable on
#: the host.
TORCH_NKI_DTYPES = {
    torch.float8_e4m3fn: "float8_e4m3fn",
    torch.bfloat16: "bfloat16",
    torch.int32: "int32",
    # an fp32 router row as fp8 columns, unused; the map is kept whole
    torch.float32: "float32",
    # the bytes of a packed KV row or fp8 key
    torch.uint8: "uint8",
}

if nki is not None:

    @nki.jit
    def _bitcast_nki(data, nki_dtype):
        data = data.view(nki_dtype)
        data_new_dtype = nl.ndarray(data.shape, nki_dtype, buffer=nl.shared_hbm)
        nisa.dma_copy(data_new_dtype, data)

        return data_new_dtype

    # wrapped once at import: a wrap inside a traced call costs dynamo a trace of the kernel's signature
    _BITCAST_KERNEL = wrap_nki(_bitcast_nki)
else:
    _BITCAST_KERNEL = None


def bitcast(data: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``data`` reinterpreted as ``dtype``: the tensor itself for its own dtype, a view on the host, else the NKI
    kernel at grid [1] (LNC1, the stage graph's logical-core count) into a new shared-HBM tensor.

    Raises:
        ValueError: A dtype the kernel's map does not hold.
        RuntimeError: A device tensor in a process without the NKI DSL.
    """
    if data.dtype == dtype:
        return data
    if data.device.type == "cpu":
        return data.view(dtype)
    if dtype not in TORCH_NKI_DTYPES:
        raise ValueError(f"the device bitcast maps {sorted(map(str, TORCH_NKI_DTYPES))}, not {dtype}")
    if _BITCAST_KERNEL is None:
        raise RuntimeError(f"a bitcast of a {data.device.type} tensor needs the NKI DSL, which this process lacks")
    return _BITCAST_KERNEL[1](data, getattr(nl, TORCH_NKI_DTYPES[dtype]))
