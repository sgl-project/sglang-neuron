# Guides

How to use the runtime: what it serves, how to turn each feature on and how to profile a
workload. How the runtime works is in the [design docs](../design/index.md). Launch commands for each model are in the
[model recipes](../model-recipes/index.md).

| Guide | What it covers |
| --- | --- |
| [Features](features-guide.md) | Each feature: what it does, and whether it is always on or which flag turns it on. |
| [Feature and model compatibility](feature-model-compatibility.md) | Which features each model serves, which a launch refuses, and which are run on hardware. |
| [Configuration reference](reference-configuration.md) | Every environment variable the runtime reads or sets, the settings the platform fixes, the Neuron runtime's defaults and the server arguments the platform sets. |
| [HiCache remote DRAM tier](hicache-remote-dram.md) | Keeping prefix KV in another node's host DRAM below L2: the lease server, the launch flags, the security model and the limits. |
| [Container image](container.md) | The image's contents, checking a host, `docker run` for one node and for prefill/decode, serving the examples, checking accuracy, and the image's known limitations. |
| [Container runbook](container-runbook.md) | From a checkout to the SGLang-Neuron images, a serving model on one node, GPT-OSS-120B and GLM-5.2 in prefill/decode with the balancer, and an accuracy check. |
| [Profiling](profiling.md) | A system profile of a running server, profiles set at launch, opening them in Perfetto or Neuron Explorer, and the host trace. |
