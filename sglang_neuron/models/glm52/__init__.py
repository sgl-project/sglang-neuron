# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 (``GlmMoeDsaForCausalLM``) on the GLM giga kernels vendored under sglang_neuron/models/glm52/kernels, one
host for both PD roles (sglang_model.py), each on its own kernel:

- the PD prefill on the prefill giga kernel, PP4 x TP32 (the default) or PP16 x TP8 (prefill/stage.py; launch.py,
  prefill/recipe.py and prefill/compile_args.py hold its launch);
- the PD decode on the decode giga kernel, TP128 (decode/graph.py), with its in-kernel MTP step in mtp/;
- the PD KV hand-off between them in pd/.

catalog.MODELS["GlmMoeDsaForCausalLM"] holds both roles' settings and launch.py their launch rules. layout/ holds the
host side of the kernels' layout contract."""
