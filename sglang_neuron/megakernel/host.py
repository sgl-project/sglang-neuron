# SPDX-License-Identifier: Apache-2.0
"""The megakernel host's entry for the platform seams: ``NeuronSRTPlatform.install_megakernel_dispatch`` calls
``create_dispatch`` with the object SGLang built from the registered spec (docs/design/megakernel.md, Seams)."""
from __future__ import annotations

from .dispatch import NeuronMegakernelDispatch, attach


def create_dispatch(*, model, built, binding) -> NeuronMegakernelDispatch:
    """The whole-step dispatch of ``built``: a callable with ``model.forward``'s signature and ``prepare(model_runner)``,
    which SGLang keeps on the model and its EagerRunner runs for every step."""
    return attach(model, built, binding=binding)
