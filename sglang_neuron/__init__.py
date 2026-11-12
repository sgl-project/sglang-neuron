from importlib.util import find_spec


def activate() -> str | None:
    """Return the platform class when ``torch_neuronx`` is importable.

    sglang calls this from its plugin entry point; returning ``None``
    leaves the platform unset so other backends can claim it.
    """
    if find_spec("torch_neuronx") is None:
        return None
    return "sglang_neuron.platform.NeuronSRTPlatform"
