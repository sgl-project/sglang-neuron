# SPDX-License-Identifier: Apache-2.0
"""The names of the plugin's HiCache storage backends (``--hicache-storage-backend dynamic``), apart from their code.

The launch reads them (platform._plugin_storage_backend) without importing the backends, so a change to a backend's
code does not change the compile caches' kernel-source digest (backend/compile_cache.py: the digest covers every plugin
module the kernels reach through their imports, which includes the platform)."""

MOONCAKE = {"backend_name": "mooncake_neuron", "module_path": "sglang_neuron.mem_cache.mooncake_store",
            "class_name": "NeuronMooncakeStore"}
