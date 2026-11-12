import os


def logical_cores() -> int:
    """Logical NeuronCore count neuronx-cc compiles for. An NKI kernel's launch grid
    must match it: neuronx-cc rejects a grid-1 kernel under LNC=2 (NCC_EVRF066).

    Reads NEURON_LOGICAL_NC_CONFIG, which torch_neuronx sets at import when unset,
    so the call stays traceable inside torch.compile regions."""
    lnc = os.environ.get("NEURON_LOGICAL_NC_CONFIG")
    if lnc:
        return int(lnc)
    from torch_neuronx.utils import get_logical_neuron_cores

    return int(get_logical_neuron_cores())
