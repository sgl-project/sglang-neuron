# sglang-neuron documentation

- [Getting started](getting-started/setup-guide.md): the container, SGLang's checkout and the venv, then the
  [quickstart](getting-started/quickstart.md) with an aggregated and a disaggregated launch.
- [NIXL patches](../patches/nixl/README.md): the NIXL overlay that `tools/build_nixl.sh` builds, which both
  disaggregated roles load ahead of SGLang on `PYTHONPATH`.
- [Tutorials](tutorials/disaggregated-1p1d.md): a prefill and a decode server on two nodes by hand, with the balancer, a
  request, a benchmark and a GSM8K accuracy check.
- [Design](design/index.md): [the runtime](design/runtime.md), [parallelism](design/parallelism.md), [async scheduling
  and execution](design/async-scheduling-and-execution.md) and [disaggregated
  inference](design/disaggregated-inference.md).
- [Guides](guides/index.md): each feature and how to turn it on, which models serve which feature, the [configuration
  reference](guides/reference-configuration.md) and profiling.
- [Model recipes](model-recipes/index.md): each served model's checkpoint, graphs, layouts, launch flags and refusals,
  LingBot-World-V2's real-time serving among them.
- [Model development](model-dev/index.md): onboarding a model, from its catalog entry to its docs.
