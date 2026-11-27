<!-- meta: description: How a model serves through upstream SGLang's MegakernelSpec on the Neuron platform: the megakernel host (built object, phase programs, pool layouts, dispatch) and SGLang's native megakernel integration -->
# Megakernels on the Neuron platform

## Overview

A megakernel is a whole-model (or whole-stage) NKI kernel: every decoder layer of a step in one kernel call, its
weights stacked on a leading layer axis and its KV pools written in place. Upstream SGLang carries
a shared megakernel contract. A plugin registers a `MegakernelSpec` by architecture and device target. SGLang selects
the provider, validates it against the resolved configuration, and builds it after model load. The Neuron platform
supplies the whole-step dispatch through its megakernel host (`sglang_neuron/megakernel/`). SGLang keeps scheduling,
the KV cache, PD disaggregation and result processing. The built object plans and runs whole steps.

Execution underneath stays the platform's: each phase is a `torch.compile` graph on the `neuron` backend whose body
calls the kernel once through `wrap_nki`, with SGLang's pool tensors passed in and the kernel's written pools aliased
back out, run by the platform's model runtime, whose decode family submits straight to the bound executor (direct
submit).

The plugin ships one spec, Qwen3.5's giga-kernel decode and prefill on `trn3`
([Supported configurations](#supported-configurations)). A whole-model kernel that needs a spec-owned loader or the
shared kernel-graph helpers lands with them at the extension point named under [Later seams](#later-seams). Every
other model in the catalog keeps the platform runtime as its host, the giga kernels of GPT-OSS and GLM-5.2 among them. They
use their existing Neuron graph paths.

## Key concepts

- **Spec** is SGLang's `MegakernelSpec`, including the provider's request validation and scheduler cohort requirements.
  The Qwen3.5 spec declares `trn3` in its requirements and does not use CUDA graph backends.
- **Device target**: the string specs are matched on, `NeuronSRTPlatform.get_megakernel_target()`: the platform
  target's family. The plugin registers each shipped spec on its target only (`megakernel/registry.py`).
- **Built megakernel**: `NeuronBuiltMegakernel` (`megakernel/built.py`), SGLang's `BuiltMegakernel` protocol on
  Neuron: `widths`, `summary`, `plan`, `forward`, the draft entries (None) and `graph_variants` (None), plus its phase
  programs, its pool layout and `prepare`.
- **Phase program**: `PhaseProgram`, one per covered step kind (`decode`, `prefill`, `idle`), naming the graph family
  that runs it (a key of the model's `model_graphs`), the `NeuronModelRuntime` subclass that builds its inputs, whether
  its steps go through the direct submit, and whether its graph samples in its forward (a fused sampling tail). An idle
  program runs a covered family's dummy step: an idle DP rank still enters the collectives the busy ranks run.
- **Pool layout**: `PoolLayout` (`megakernel/pools.py`). Its `pool_class(kind)` gives the KV pool classes SGLang's KV
  cache configurator builds for the model (the interface the seams read). Its `PoolEntry`s name the SGLang pool tensors
  the programs take, one kind each: `kv`, `kv_pair` (the GPT-OSS giga-kernel decode's paired buffers), `mla_latent`
  and `index_k` (an MLA pool's per-layer latent and a DSA indexer's per-layer keys, read through `pools.per_layer`;
  the shipped spec uses neither), and `recurrent_state` and `conv_state` (the state pool's tensors, read through
  `pools.state_pool`). An entry also names the global layer ids it holds, how many tensors it resolves to, whether the
  kernel writes them in place and returns them as aliases, and whether the kernel needs freed pages zeroed
  (`zero_on_free`: a gather of whole pages past a sequence's end must read zeros, since a masked score does not protect
  `0 * inf` in the PV product).
- **Neuron step**: `NeuronStep`, a subclass of SGLang's `Step` (`mode`, `rows`, `view`) that carries the graph family,
  the runtime's step plan (bucket, context bucket) and the step's attention metadata. Its `view` is a
  `NeuronForwardView`: the metadata, the bound pools and the forward batch.
- **Whole-step dispatch**: `NeuronMegakernelDispatch` (`megakernel/dispatch.py`), a callable with `model.forward`'s
  signature plus `prepare(model_runner)`; the platform returns it from `install_megakernel_dispatch` and SGLang keeps it
  on the model.

## How it works

### Install

At argument resolution, `handle_nvidia_megakernel` finds the spec registered for the model's architecture on the
device target and refuses the launch listing every reason the spec cannot serve the resolved configuration. After the
weights load, `install_megakernel` (before KV sizing) builds the spec, hands `built.pool_layout` to the platform, checks
rank agreement on widths and entries, and asks the platform for the dispatch: `megakernel/host.create_dispatch`, which
refuses a built object that is not a `NeuronBuiltMegakernel` or that was built for another model, makes the ranks of
the TP group agree on the build's digest (programs, widths, pool layout) and returns the dispatch. The KV configurator
then builds the model's pools with the layout's classes.

### Prepare

SGLang's `EagerRunner` (built after the pools) calls the dispatch's `prepare(model_runner)`, and the platform's graph
runner prepares it too (`NeuronModel._ensure_runtime`); the second call with the same runner is a no-op. Before it
prepares the dispatch, `_ensure_runtime` runs `dispatch.check_declared`: a model that declares its spec
(`NeuronModel.megakernel_spec`, as Qwen3.5 does) refuses to serve without that spec's dispatch, or with the dispatch of
another spec, since it built graphs that only its spec serves.

1. The pool layout binds against the live pools (`PoolLayout.bind`): each entry resolves to its declared number of
   tensors on one device, no tensor is bound twice, and their identities are recorded. A `zero_on_free` layout is
   refused over an allocator that does not zero the pages it frees.
2. `NeuronModel._build_runtimes` builds one runtime per graph family with the class its program names, marks the
   families whose program declares the direct submit, and compiles and warms them. Everything the platform runtime
   does at warm-up applies unchanged: the per-rank NEFF agreement (SPMD check), the model's post-warm-up check
   (`NeuronModel.neuron_after_warmup`) and the direct submit's first bind call.
3. The decode family's seqs buckets must be among the built widths.

### Steps

Every forward goes through the dispatch: SGLang's `EagerRunner.execute` calls it for every mode, the platform's
`NeuronGraphRunner` plans in `load_batch` and runs in `execute`, and `NeuronModel.forward` calls it.

- `plan` reads host state only: the attention backend's metadata and the family runtime's `plan_step`, whose phase and
  bucket agree on every DP rank. An empty batch plans as idle. A step kind no program covers plans None, and the
  dispatch refuses it with the reason; there is no eager path.
- `run` attaches the step's `NeuronForwardView` and calls `forward`, which checks that SGLang's pool objects are the
  ones bound and runs the step on its family's runtime (`run_decode`, `run_prefill` or `run_idle`). The model's own
  forward (`original`) is never called.

The device path of a step is the platform runtime's: the runtime builds the step's inputs, the compiled graph (or its
bound executor) is submitted on the default stream, and the async execution machinery owns completion.

### The kernel graph

A family's graph is the model's: a `torch.compile` body that calls the kernel once through `backend.stack.wrap_nki`.
A kernel whose collectives span a process group takes the group's whole partition as a static argument
(`backend.stack.resolve_full_mesh`), so every rank compiles one SPMD NEFF. The GPT-OSS giga-kernel decode and Qwen3.5's
programs build their graphs themselves; the shared kernel-graph helpers (stacked weights, static meshes, pool keyword
arguments, the alias split) are a [later seam](#later-seams).

### Compiling the kernel graphs

Each family compiles with the compiler arguments its host gives it (`neuron_compiler_args`,
[Compiler arguments](runtime.md#compiler-arguments)). A whole-model kernel that unrolls its layers at trace time, as
Qwen3.5's does, puts the instructions of every layer it runs in one program. Two settings shape that compile:

- **The modular flow over repeated segments.** A graph built as a chain of submodules of one module definition compiles
  in neuronx-cc's modular flow with the hierarchy splitter on. The splitter partitions the graph by its module
  hierarchy, one partition for each submodule, and a repeated module compiles once. Qwen3.5's prefill graph is one
  `Qwen35PrefillSegment` of 4 layers (3 GDN, 1 attention) for each group of layers, with the embedding and the
  first-token head as submodules of their own, so its prefill program compiles one segment
  (`sglang_model.prefill_compiler_args`). Its decode family is one kernel call and compiles as the GPT-OSS giga-kernel
  decode does: `-O2`, without the modular-flow threshold. Both families compile without the FSDP annotation of
  collectives (`--enable-annotate-fsdp-cc=false`).
- **The NKI compile mode.** NKI reads `NKI_COMPILE_MODE` when a graph's kernels compile, at warm-up. Qwen3.5's
  runtimes set `enable-alloc=legacy` around each family's warm-up (`models/qwen35/runtime.py`), which leaves the
  allocation and the instruction scheduling to neuronx-cc. Under NKI's own allocation the GDN prefill runs out of PSUM,
  and under NKI's own instruction scheduling the decode at 16 rows per replica uses up the Pool engine's registers
  (NCC_IGCA094).

A device loop (`nl.fori_loop`) whose body holds a collective fails at run time on trn3. So a kernel unrolls each loop
around a collective at trace time, as Qwen3.5's multistep decode unrolls its steps, and keeps a device loop only over a
region without collectives, as Qwen3.5's MoE expert loops do.

## Serving a whole-model megakernel

The launch reads how to serve a model's megakernel from its catalog entry (`catalog.ModelSpec.megakernel`, a
`MegakernelServing`) when SGLang selects its provider (`platform.megakernel_serving`). The serving policy
(`serving_policy.resolve`) then records `megakernel` and:

- **Runtime defaults.** Every role gets the giga-kernel decode's runtime settings (`runtime_init.runtime_defaults`:
  the DMA settings, 63 queued executions, no low-latency completion thread, the highest DMA service level), since every
  kernel call carries every layer's collectives over all the server's ranks. A setting the launch environment gives
  keeps its value.
- **DP attention from the exchange** (`dp_steps_from_exchange`). The model's ranks plan each step's kernel calls from
  SGLang's per-step DP-attention exchange (every replica's token and request counts), so a prefill replica and a
  decode replica run the same calls. An aggregated server with `--enable-dp-attention` is then served
  (`refuse_unserved_layouts`), and a decode with DP attention keeps the exchange instead of the DP identity, whose
  constant result would drop the counts; it runs without the identity's control fence, its shared-memory request
  broadcast and the rank-1 detokenizer sender. SGLang's exchange snapshot
  gives, for each participant, its forward mode, token count and request counts, and the longest KV history its step
  reads (`max_prefix_tokens`: an extend's longest cached prefix, or a decode's longest sequence plus its new token).
  Every rank holds the same snapshot, so every rank picks the same programs: Qwen3.5 takes the prefill bucket of the
  largest chunk and the prior tier of the longest prefix
  ([A chunk from a cached prefix](recurrent-state.md#a-chunk-from-a-cached-prefix)), and the decode context rung of
  the longest sequence.
- **Async steps** (`async_steps`). The programs run under the overlap schedule in the host-ids mode
  ([Async scheduling](async-scheduling-and-execution.md#a-megakernels-host-ids-mode)): a step's calls are launched
  without reads in between and return one pending result. A launch of a model without it, or with a speculative
  worker, must pass `--disable-overlap-schedule`; otherwise it is refused, naming why.
- **LNC degree.** `ModelSpec.logical_nc` (1 for a megakernel whose 128 ranks are a node's physical cores) sets
  `NEURON_LOGICAL_NC_CONFIG` for the launch.

### Beyond the host-ids mode

The host-ids mode takes SGLang's host work off the device's critical path, but the next decode still waits for the
previous step's ids: per step the device idles for the readback, the host embedding and the input upload. Three steps
remove the rest, each a change to a kernel graph (a recompile, validated on hardware):

1. **Device ids.** The decode graph embeds the ids itself from a device token vector, as the GPT-OSS giga-kernel
   decode does (`token_io.FixedDecodeTokenIO`: the previous step's tokens plus a `[2, W]` control that reorders them in
   the graph). The runtime then declares a token width instead of `host_input_ids`, and the platform's relay plans each
   decode from the device. Qwen3.5's decode graph does this under async execution (`models/qwen35/runtime.py`,
   `Qwen35TokenIO`). The embedding table must fit beside the weights: replicated, a vocabulary of 154k x 6144 bf16 takes
   about 1.9 GB per rank, so it shards over attention TP and all-gathers the rows.
2. **The result lag.** With device ids, the giga-kernel decode's result lag (`sglang_patches/pd_decode_result_lag.py`)
   applies to a megakernel decode too: process each step's result only after launching the in-flight bound of later
   steps, so a rank whose host runs late does not hold every rank at the kernel's collectives.
3. **Fewer program switches.** A rung change of the decode ladder, and an aggregated step that mixes prefill and
   decode calls, switch programs, and the program-switch fence drains the device there. One rung (and the direct
   submit) on a decode server, or a decode program that serves the prefill's positions too, keeps a steady decode free
   of fences.

A pipelined megakernel is not served: the host refuses `pp_proxy_tensors` at a step ([Limitations](#limitations)). Its
async path is the platform's PD pipeline one: a greedy (sharded argmax) tail in the last stage's decode graph writes its
tokens into the PP seam's "tk" slot, the first stage's graph assembles its ids from them (`pp_token_relay`), and the
prefill stages' overflow check moves from before the boundary send to the result's processing.

## Configuration

SGLang selects the registered Neuron provider automatically. Existing Neuron launches can still pass
`--enable-nvidia-megakernel` as a deprecated compatibility alias. The plugin registers its specs at load through
`sglang.kernels.plugin_kernels.megakernel.spec.register(architecture=..., path="module:attr", target="trn3")`.
An architecture-default registration omits `target`. The shipped specs are listed in `megakernel/registry.py`.

## Supported configurations

| Model | Architecture | Target | Role | Spec |
|---|---|---|---|---|
| Qwen3.5 | `Qwen3_5MoeForCausalLM` | trn3 | aggregated or either PD role (the giga-kernel decode and prefill) | `models/qwen35/megakernel.py` |

A model's spec registers in `megakernel/registry.py` (`SPECS`) with its programs and pool layout, and its row goes
here. Spec modules export their spec lazily via a module `__getattr__`, so importing one without its device
dependencies raises only when the spec itself is accessed.

## Limitations

- Every hooked function is upstream's without the flag, so the seams cost nothing on a launch without a megakernel;
  the megakernel launches themselves are qualified by the host, not by the seams.
- The decode CUDA-graph path (`_Dispatch`, `_CaptureCheck`, graph variants) does not run on Neuron: the platform always
  owns the dispatch. A spec's `decode_graph_backends` should be empty (accept any).
- A spec that states `resident_blocks` (a CUDA SM count) is refused out of tree; residency is the spec's
  `model_disable_reasons`.
- The GPT-OSS branch of `handle_model_specific_adjustments` still runs for a GPT-OSS spec; it accepts the Neuron
  attention backend (`gpt_oss_backend_check`).
- The paired-KV resolver (`pools.paired_kv`) reads the paired KV pool's per-pair tensors (`layer_pairs`), which the
  pool allocates when the model declares `kv_layout.KVLayout(paired_layers=True)`, as the GPT-OSS giga-kernel decode
  does.
- No pipeline stage: `pp_proxy_tensors` is refused at a step.
- One token per request: the host offers no draft entries, so a spec that accepts speculative decoding has no Neuron
  draft path.
- The direct submit serves a decode family only (its warmed shapes under tensor guards, one binding a shape).
- A pool layout's binding is checked per step against SGLang's pool objects; a tensor reallocated inside the same pool
  object is caught only by `BoundPools.recheck`.
- `zero_on_free` is a requirement on the allocator (`zeroes_freed_pages`). `NeuronPagedTokenToKVPoolAllocator`
  declares it when its pool, or a hybrid pool's full-attention leaf, is under the zero-page contract
  ([MLA and DSA pools](mla-dsa-pools.md#the-zero-page-contract)); a layout that needs it is refused over any other
  allocator.
- The host does not read SGLang's `MegakernelBinding` (`forward_view`, `assert_live`): its view is the platform's
  attention metadata, and the pool layout's binding checks liveness.

## Upstream SGLang interactions

The host imports version 3 of the shared contract from `sglang.kernels.plugin_kernels.megakernel.spec` through
`megakernel/upstream.py`.

### Seams

| Responsibility | Owner |
|---|---|
| Registry and provider selection | SGLang's shared registry and `sglang.srt.kernel_plugins.megakernel` select the provider. Neuron supplies the target and registers the shipped specs. |
| Build, pool layout and dispatch | SGLang builds the provider after model load and uses its layout when sizing and creating KV pools. `NeuronSRTPlatform` supplies the KV pool classes and whole-step dispatch. SGLang's eager runner calls that dispatch. |
| Request validation | SGLang calls the provider validator before admitting the request. Qwen3.5 rejects unsupported sampling and logprobs. |
| DP-attention cohort | SGLang gathers request counts and prefix metadata when requested by the provider, then carries the snapshot onto `ForwardBatch.mlp_sync_batch_metadata`. The Qwen3.5 planner reads that snapshot. |
| Recurrent state | Neuron's `state_pool_class` hook reads the built provider's layout through SGLang's `oot_megakernel_pool_layout` and constructs the declared state pool. The shared checkpoint and drain hooks still manage Neuron state-slot operations for megakernel and platform-runtime models. |

`sglang_neuron/megakernel/host.py` provides `create_dispatch`. The dispatch binds model-local pools, plans each step,
and runs the selected Neuron program. Its `prepare(model_runner)` runs after the pools exist.

### Later seams

These extensions are deferred until a provider needs them.

| Extension | Intended use |
|---|---|
| Startup loader and expert layout | A provider supplies its own weight loader or expert layout. |
| Token relay pool kind | A provider relays tokens through a pool tensor. |
| Kernel-graph helpers | Providers share code for stacked weights, static meshes, pool arguments and output aliasing. |

## Implementation

| Path | Contents |
|---|---|
| `sglang_neuron/megakernel/upstream.py` | The contract import and its API version check |
| `sglang_neuron/megakernel/built.py` | `PhaseProgram`, `NeuronStep`, `NeuronForwardView`, `NeuronBuiltMegakernel` |
| `sglang_neuron/megakernel/pools.py` | `PoolEntry`, `PoolLayout`, `BoundPools`, the pool resolvers |
| `sglang_neuron/megakernel/dispatch.py` | `attach`, `NeuronMegakernelDispatch`, `check_declared` |
| `sglang_neuron/megakernel/host.py` | `create_dispatch`, the platform's entry |
| `sglang_neuron/megakernel/registry.py` | The shipped specs by architecture and target |
| `sglang_neuron/models/base.py` | `runtime_class`, `_build_runtimes`, `neuron_after_warmup`, the dispatch in `_ensure_runtime` and `forward` |
| `sglang_neuron/model_executor/graph_runner.py` | The dispatch in `NeuronGraphRunner`, a program's runtime class (`runtime_class_of`), its direct submit (`build_compiled`) |
| `sglang_neuron/backend/direct_submit.py` | The direct submit of a one-shape decode graph to its bound executor |
