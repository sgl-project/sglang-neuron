# Integrating megakernels with SGLang's out-of-tree platform plugins

This plan uses Qwen3.5 on Trn3 as the first end-to-end integration. The Neuron
platform should register its spec through the existing SGLang plugin entry point,
build it through the shared megakernel binding, and run it through the Neuron
runners. SGLang should own spec selection and the lifecycle extension points.
Neuron should own graph compilation, step planning, weight packing, and the
physical KV and recurrent-state layouts.

## First uprev status

The uprev draft consumes the target registry, platform selection, provider
lifecycle, request validator, and attention-DP cohort changes. Its SGLang pin
`66f6844077` restores platform graph-runner selection for platforms with CPU
scheduler tensors. The draft also preserves Neuron's token plan during eager
batch copies and uses the verified 64K program in place of the affected 32K
program for KV CP16 with 2,048-token pages. [SGLang cohort PR](https://github.com/sgl-project/sglang/pull/53209) ·
[Neuron uprev PR](https://github.com/sgl-project/sglang-neuron/pull/23)

The two-node Trn3 run used 65,536-token context and NIXL LIBFABRIC over EFA.
Decode logs confirmed graph execution. All eight attention-DP replicas returned
the same chat-format answer and stop token. Long-context comparisons against
prefill passed through 33,041 input tokens, including cold and cached 16,400-token
requests. The live run used source changes applied to the serving containers.
The qualification steps below cover the rebuilt image and remaining serving
cases.

## Baselines

- SGLang-Neuron snapshot `4e24326` pins SGLang at `43881a7`. Its megakernel
  adapter requires API v2. The SGLang main snapshot inspected for this plan,
  `a3f0bff5ba`, uses API v3. Port the contract and lifecycle together when the
  candidate pin first crosses that boundary.
- Start the implementation on a fresh branch from current SGLang `main`.
  [RFC #46355](https://github.com/sgl-project/sglang/pull/46355) is a design
  reference for platform targets, dispatch ownership, and pool-layout
  activation. Implement the hooks directly against the current API v3 binding,
  using the Neuron patches to identify the eager dispatch and pool-class
  resolution requirements.

### What the megakernel API version covers

`MEGAKERNEL_API_VERSION` versions the Python interface between SGLang and a
megakernel provider. It covers how a provider declares requirements, what it
builds from a loaded model, and how the runtime plans and runs a step.

| Version | Provider and runtime contract |
| --- | --- |
| v2 at `43881a7` | `MegakernelSpec.build(model, hf_config, context)` returns a `BuiltMegakernel` with `plan` and `forward` for covered steps. The binding stores the built object directly on the model. Neuron supplies its post-pool preparation and eager dispatch through patches. |
| v3 at `a3f0bff5ba` | The binding stores an installed record and exposes the built object through `built_megakernel(model)`. A spec can supply an idempotent `prepare(built, view)` after pool allocation and opt into additional entry points, including prefill and tensor-entry verify. Built objects must expose `speculative_runtime_hooks`, which can be `None`. |

The cutover updates the adapter's version check, built objects, installed-object
lookups, and preparation wiring as one unit. Neuron can leave optional v3
features unset where it owns the corresponding execution path. The shared
preparation point still needs to run on Neuron's eager and graph paths.

[Neuron API adapter](../../sglang_neuron/megakernel/upstream.py) ·
[SGLang v2 spec](https://github.com/sgl-project/sglang/blob/43881a719e42165ff987fb0273a6bf12eaeb7662/python/sglang/kernels/plugin_kernels/nvidia/spec.py) ·
[SGLang v3 spec](https://github.com/sgl-project/sglang/blob/a3f0bff5babfe18d3862e134449cd8fd11fcb84a/python/sglang/kernels/plugin_kernels/nvidia/spec.py) ·
[SGLang v3 binding](https://github.com/sgl-project/sglang/blob/a3f0bff5babfe18d3862e134449cd8fd11fcb84a/python/sglang/srt/kernel_plugins/nvidia/binding.py#L1000)

## Patch inventory

| Neuron patch | Behavior to preserve | Proposed integration |
| --- | --- | --- |
| [`megakernel_platform.py`](https://github.com/sgl-project/sglang-neuron/blob/4e2432622bfab2c6663d12286a6813ef1a40bfe6/sglang_neuron/sglang_patches/megakernel_platform.py) | Platform target, lazy target-specific registrations, CUDA guards, pool activation, dispatch attachment, and rank agreement for opting into cohort metadata | Add the platform hooks and target keys to the current binding and registry. Move the eager callable and cohort opt-in to the lifecycle hooks below. |
| [`megakernel_eager_dispatch.py`](https://github.com/sgl-project/sglang-neuron/blob/4e2432622bfab2c6663d12286a6813ef1a40bfe6/sglang_neuron/sglang_patches/megakernel_eager_dispatch.py) | Whole-step execution through `EagerRunner`, including modes and shapes outside the platform graph runner, with forward context and a post-pool preparation call | Add an eager runner delegate for a model with a platform-owned dispatch. |
| [`megakernel_pool_layout.py`](https://github.com/sgl-project/sglang-neuron/blob/4e2432622bfab2c6663d12286a6813ef1a40bfe6/sglang_neuron/sglang_patches/megakernel_pool_layout.py) | Choose explicit MHA, MLA, or DSA classes from the installed model's layout for the pool kinds it actually instantiates | Resolve the model-scoped layout through the KV configurator at the point of construction. |
| [`state_pool_class.py`](../../sglang_neuron/sglang_patches/state_pool_class.py) | Select the recurrent/conv pool, bind checkpoint storage, and drain pending writes before slot reuse. Also used by models that run without a megakernel. | Add model-scoped recurrent-pool selection. Retain the shared Neuron state-lifecycle patch until its checkpoint and drain behavior has equivalent extension points. |
| [`megakernel_request_contract.py`](https://github.com/sgl-project/sglang-neuron/blob/4e2432622bfab2c6663d12286a6813ef1a40bfe6/sglang_neuron/sglang_patches/megakernel_request_contract.py) | Resolve once in the tokenizer and reject unsupported normalized requests before scheduling, with HTTP 400 on the serving path | Add an optional host-only validator on the selected spec and run it through the tokenizer's existing request-error path. |
| [`mlp_sync_cohort_snapshot.py`](https://github.com/sgl-project/sglang-neuron/blob/4e2432622bfab2c6663d12286a6813ef1a40bfe6/sglang_neuron/sglang_patches/mlp_sync_cohort_snapshot.py) | Opt into a rank-consistent DP-attention snapshot carrying per-replica request counts, decode counts, and maximum cached prefix lengths | Extend the existing scheduler exchange with opt-in typed metadata and carry it through batch construction and copies. |
| [`megakernel_oot_overrides.py`](https://github.com/sgl-project/sglang-neuron/blob/4e2432622bfab2c6663d12286a6813ef1a40bfe6/sglang_neuron/sglang_patches/megakernel_oot_overrides.py) | For a GPT-OSS megakernel launch, suppress stock attention/MoE backend choices while preserving the MXFP4 BF16 dtype | Add narrowly scoped override ownership when a GPT-OSS provider needs it. The shipped registry currently contains Qwen3.5 only. |

## Implementation sequence

### 1. Establish the platform and registry contract

Add `PlatformCapabilities.supports_megakernel: bool = False` to the current
SGLang platform interface. The capability says the platform can host the shared
binding. Model selection still depends on the registered spec and startup
validation. Keep the existing CUDA flag behavior and use the same flag as a
temporary compatibility alias on Neuron. The alias adds no new CLI switch.

| Platform | Selection rule |
| --- | --- |
| In-tree NVIDIA CUDA | Require both `supports_megakernel` and `--enable-nvidia-megakernel`. Retain the NVIDIA entry-point check and the existing flag's MSA routing. |
| OOT with the capability | Prefer a megakernel automatically when the model architecture and active device target have an exact registration. Use the regular path when there is no match and the launch does not require a megakernel. |
| Neuron with `--enable-nvidia-megakernel` | Select the same exact match and emit one deprecation warning from the launcher. If the requested match is missing, fail startup and name the architecture and target. |
| Platform without the capability | Use the regular platform path. An explicit NVIDIA request on an unsupported platform produces a startup error. |

Qwen3.5 on Trn3 is the first automatic registration. Separately, Neuron's
launch policy marks Qwen3.5 packed-weight serving as requiring a megakernel.
A missing provider for that configuration fails startup even if the legacy
flag was omitted. A selected spec that fails to import, uses an incompatible
API, or rejects the resolved configuration fails startup with its reason.

- Use a shared selection helper for argument resolution, frontend validation,
  and worker installation. For platform defaults that run before full argument
  resolution, expose the architecture/target registration check as a
  metadata-only query. Neuron can use that query with its own platform
  instance to choose serving defaults. Validate the resolved configuration
  later. Record the selected architecture, target, provider path, and API
  version in the resolved launch configuration so the frontend and workers
  can load and verify the same spec. Declare that a model requires a
  megakernel independently of its provider registration so a missing entry
  produces a startup error. Audit existing flag readers that check for an
  active megakernel, including speculative configuration and MoE compatibility.
- Use the existing `sglang.srt.platforms` and `sglang.srt.plugins` entry points.
  Neuron's platform returns a frontend-safe target string such as `trn3`.
  Its general plugin registers spec paths in launcher, tokenizer, and worker
  processes before their respective lookups. Import a provider spec only when
  the active target and architecture select its path.
- Start with an SGLang PR that moves the v3 contract and registry into
  `sglang.kernels.plugin_kernels.megakernel.spec`. Keep `nvidia.spec` as a
  compatibility import while `sgl-nvidia-mega` uses the old path. Coordinate
  the provider's source-contract checks with the move so they inspect the
  shared definitions. Remove the compatibility import after supported external
  providers use the new path. AMD continues to use its `MonoSpec` contract.
  Extend
  `register(architecture=..., path=...)` with an optional target and update
  Neuron's patched `isa=` registration call at cutover. For each HF
  architecture in order, use an exact target registration first and the
  architecture default when no exact entry exists. Preserve idempotent
  registration and reject conflicting paths for the same key. Keep the
  architecture-default lookup
  for legacy NVIDIA providers. Require explicit target registrations for
  automatic OOT selection so an unrelated architecture default cannot opt a
  model in. Report invalid exact entries directly and keep diagnostics aware of
  the target, including for optional draft specs.
- Honor the v3 built-object contract and rank checks in the Neuron adapter.
  Use public installed-object accessors in place of reading `model._megakernel`
  directly. Exercise both ordinary and tensor-entry build paths in the shared
  binding. Require no vendor SDK import to register a path or probe the Neuron
  target in the tokenizer.

### 2. Wire the model lifecycle, eager execution, and pools

1. **Install after model load, before cache sizing.** Build the selected spec,
   validate any declared layout for the model and role, and attach the dispatch
   to the installed model record. Resolve platform ownership before stock
   CUDA memory and residency probes, automatic `EagleBinding` selection, or
   stock forward wrapping. Keep those steps on the stock provider path when
   applicable. Give the platform runner an accessor for the installed
   dispatch. Use a typed optional eager delegate alongside platform graph
   ownership so a bare `True` cannot be mistaken for an eager implementation.
2. **Resolve pools from the installed model.** Apply the selected layout to
   the KV and recurrent-state configurators, including any sizing inputs that
   depend on the layout. Reuse `KVPoolKind` and the existing subclass checks.
   Require an explicit class for each pool kind actually built when the layout
   claims ownership. Carry the layout per model through target and draft
   construction. Reject a draft layout that conflicts with shared target pools.
   Preserve recurrent checkpoint binding, deferred sizing, and drain before
   free or reset while those responsibilities are still implemented in Neuron.
3. **Prepare once pools and attention metadata are ready.** Add or reuse a
   backend-neutral post-pool preparation point. Today's v3
   `prepare_megakernel()` is reached through CUDA warmup, so check the actual
   eager and platform graph entry points before using it for OOT. Both paths
   should share preparation for the same model runner. Preserve the Neuron
   host's idempotence and its rank agreement on programs, widths, and layout.
   Agree on rank-local build and pool validation outcomes before entering
   the next collective phase. Include rank-local exceptions in that agreement.
4. **Route every eligible step.** Let `NeuronGraphRunner` plan in `load_batch`
   and execute the plan in `execute`. Have `EagerRunner` call the same delegate
   under the existing model and forward contexts, with normal PP argument
   propagation. Skip the eager copy bank and warmup only for a delegate that
   owns those responsibilities. Check which modes the configured role can
   submit, including extend and idle where applicable. If a custom layout
   cannot be read by the stock path, refuse an uncovered configuration at
   startup where possible and raise an explicit error for a runtime coverage
   miss. Preserve standard fallback for providers that share compatible pools.

[Neuron dispatch](../../sglang_neuron/megakernel/dispatch.py) ·
[Neuron graph runner](../../sglang_neuron/model_executor/graph_runner.py) ·
[SGLang KV configurator](https://github.com/sgl-project/sglang/blob/a3f0bff5babfe18d3862e134449cd8fd11fcb84a/python/sglang/srt/mem_cache/kv_cache_configurator.py#L2268)

### 3. Complete the Qwen3.5 serving contract

- Resolve the same spec in the frontend and the workers. Run its opt-in
  validator after tokenization and default normalization, before enqueueing.
  Cover greedy requests, unsupported sampling or logprobs, the exact
  prompt-length rule for a launch with one prefill bucket, and a batched
  request containing an invalid item. Startup should explain a missing
  frontend spec without needing a device runtime.
- For Qwen3.5's aggregated DP-attention path, attach an immutable cohort
  snapshot to the batch. Agree on the metadata contract across the
  participants before changing the exchange shape. Preserve request and decode
  counts and maximum cached prefixes through scheduler copies, conversion to
  `ForwardBatch`, and eager staging. Refuse configurations that opt into the
  snapshot but skip the exchange. Keep these integrations or their compatibility
  patches in place before claiming full Qwen3.5 serving coverage.
- Handle GPT-OSS backend overrides in a separate change. Resolve target
  registration early enough for override collection, retain dtype declarations,
  and suppress only the backend settings the selected spec owns. Use the
  selected target entry and explicit ownership instead of the patch's
  architecture-only presence check. Treat named providers on stock CUDA and
  pre-load weight ownership as separate work.
  [Stock CUDA review discussion](https://github.com/sgl-project/sglang/pull/46355#issuecomment-6801771904)

### 4. Batch Neuron uprevs around working configurations

Submit focused, independently mergeable SGLang PRs for the registry and
selection contract, lifecycle extension points, frontend validator, and
cohort metadata. After a testable group lands, open one Neuron PR against the
earliest merged SGLang commit containing that group. Each merged Neuron pin
must boot and serve the configurations supported by the previous pin.

| Neuron batch | Merge condition and patch retirement |
| --- | --- |
| Upstream compatibility | Bring up ordinary Qwen3-0.6B on four NeuronCores as the smoke test for every candidate pin. Use the development checkout to identify dependency, source-guard, and moved-symbol changes. A pin that crosses to v3 also needs a working Qwen3.5 adapter and lifecycle before merge. |
| Registry and lifecycle cutover | After shared selection, install, pool resolution, prepare, and dispatch hooks land, update the API adapter and switch Qwen3.5 on Trn3 to native selection. Declare the capability on `NeuronSRTPlatform` and use the same selection in serving policy and pipeline checks. Retire the corresponding platform, eager dispatch, and KV layout replacements once the native paths pass. |
| Qwen3.5 validator and cohort exchange | Consume the public hooks in the tokenizer and workers and remove the matching replacements once the frontend, aggregated DP-attention, and disaggregated prefill/decode cases pass. Fold these changes into the lifecycle cutover if partial adoption would break a supported Qwen3.5 configuration. |
| Shared state lifecycle and GPT-OSS | Keep the shared state patch until checkpoint and drain hooks cover its megakernel and regular-model users. Handle GPT-OSS override ownership with its provider in a separate change. |

For each Neuron uprev PR:

1. Record the old and new full SGLang SHAs, the merged upstream PRs included,
   and the patches each change supersedes. Update `SGLANG_COMMIT`, the image
   version metadata, affected dependency locks, and build and setup
   instructions. Keep the Neuron SDK and PyTorch pins while resolving any
   new SGLang runtime dependencies. Build and test the candidate image.
2. Review the upstream source changes for every retained patch. Port changed
   hooks before refreshing their guards. Use `tools/pin_upstream_sources.py
   --write` and `--check` for the JSON-backed guards and audit inline guards
   separately.
3. Remove the matching wrappers and replacements from `general_plugin.py`
   once the launcher, tokenizer, or workers exercise the native hook. Track
   retirement by hook, upstream PR, owning process, and validation case so
   a module covering several seams can be retired incrementally. Verify
   that each hook has one owner in the candidate image. If required seams
   land in different batches, defer the pin bump or retain a tested bridge
   with a named removal dependency.
4. Run the ordinary-model smoke test and the affected serving cases on Trn3.
   The first Neuron uprev PR requires a successful two-node Qwen3.5
   prefill/decode run through the balancer before merge. Run the rest of the
   Qwen3.5 qualification below for the API cutover. Keep the previous image
   and exact pin for parity checks and rollback.
5. In the automatic-selection uprev, remove the flag from the checked-in
   prefill and decode scripts. Exercise both flagless launches and the legacy
   commands before merge.

  [Image pin](../../docker/Dockerfile#L25) ·
  [Setup guide](../getting-started/setup-guide.md) ·
  [Source pin tool](../../tools/pin_upstream_sources.py) ·
  [Plugin registration](../../sglang_neuron/general_plugin.py) ·
  [Shipped specs](../../sglang_neuron/megakernel/registry.py)

## Validation and exit criteria

| Stage | Checks |
| --- | --- |
| CPU contract tests | Use an SDK-free fake OOT platform and a fake CUDA provider for the same architecture. Verify automatic selection, the Neuron compatibility alias and its one-time warning, regular execution for unregistered models, required-spec and invalid-spec errors, and the NVIDIA capability/flag combinations. Cover lazy imports, target precedence, duplicates, and legacy draft behavior. Instrument CUDA probes to fail if invoked by the fake OOT path. |
| CPU lifecycle tests | Check install → pool sizing/construction → prepare → dispatch ordering. Exercise eager-only and graph-owned steps, unsupported modes, idempotent preparation, layout isolation across target/draft models, unavailable pool classes, and unchanged stock dispatch. Test frontend rejection before enqueue and cohort metadata survival through eager copies. |
| Distributed tests | Inject disagreements in spec, layout, and cohort opt-in, plus an exception on a single rank during local build or pool validation before the first agreement. Verify that every rank reports the failure without hanging. Exercise a mixed extend/decode/idle cohort and confirm the ranks choose compatible programs. Check cache slot reuse after pending state writes complete. |
| Trn3 qualification | Record the candidate and baseline image digests, checkpoint pack, launch commands, and fixed input cases. Require exact greedy token parity on those cases, an eager-path run, aggregated DP-attention, and a successful two-node prefill/decode handoff through the balancer using the checked-in Qwen3.5 recipe. Verify cache hits and state reuse. Report startup memory and steady-state latency or throughput under the same workload. |
| Regression and cutover | Run the SGLang NVIDIA megakernel unit suite and a stock CUDA serving smoke test, including legacy-flag graph capture and fallback. Confirm that the native Neuron launch no longer installs the replaced megakernel hooks. Document any remaining shared patches and why they still run. |

[Existing Neuron design](megakernel.md) ·
[Qwen3.5 PD qualification scripts](../../examples/qwen35/README.md) ·
[SGLang unit suite](https://github.com/sgl-project/sglang/tree/a3f0bff5babfe18d3862e134449cd8fd11fcb84a/test/registered/unit/kernel_plugins/nvidia)
