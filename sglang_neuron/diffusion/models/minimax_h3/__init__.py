# SPDX-License-Identifier: Apache-2.0
"""Neuron implementation of MiniMax-H3."""

# The checkpoint's ``model_index.json`` class and the Hub IDs SGLang maps to its H3 pipeline.
# Import-light: the platform reads them before any worker or pipeline module loads.
MINIMAX_H3_CLASS_NAME = "MiniMaxH3ModularPipeline"
MINIMAX_H3_HF_PATHS = ("MiniMaxAI/MiniMax-H3", "MiniMax/MiniMax-H3")
