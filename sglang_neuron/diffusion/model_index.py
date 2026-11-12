# SPDX-License-Identifier: Apache-2.0
"""A diffusers checkpoint's ``model_index.json`` and component configs, local or on the Hub.

JSON reads only, no torch-neuronx: the launcher reads the checkpoint before it applies the
environment profiles, and the pipeline routing reads it before it builds a pipeline. For a Hub ID
only the file asked for is fetched, or its cached copy read.
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)


def read_model_index(model_path: str | None) -> dict:
    """``model_index.json`` of a local directory or a Hub ID; ``{}`` when there is none.

    A Hub ID goes through SGLang's ``maybe_download_model_index``, the reader its pipeline
    registry uses.
    """
    if not model_path:
        return {}
    if os.path.isdir(model_path):
        path = os.path.join(model_path, "model_index.json")
        if not os.path.isfile(path):
            return {}
        with open(path) as f:
            return json.load(f)
    from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
        maybe_download_model_index,
    )

    try:
        return dict(maybe_download_model_index(model_path))
    except Exception as error:  # Hub unreachable and not cached, not a diffusers pipeline
        logger.warning("Cannot read model_index.json of %s (%s)", model_path, error)
        return {}


def read_component_config(model_path: str, subfolder: str) -> dict | None:
    """``<subfolder>/config.json`` of a local directory or a Hub ID; ``None`` when there is none."""
    if os.path.isdir(model_path):
        path = os.path.join(model_path, subfolder, "config.json")
        if not os.path.isfile(path):
            return None
    else:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        try:
            path = hf_hub_download(model_path, "config.json", subfolder=subfolder)
        except EntryNotFoundError:  # no such file, or offline and not cached
            return None
        except Exception as error:
            logger.warning("Cannot read %s/config.json of %s (%s)", subfolder, model_path, error)
            return None
    with open(path) as f:
        return json.load(f)
