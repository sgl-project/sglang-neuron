# Kimi-K3 PD on Neuron

Kimi-K3 runs disaggregated on the plugin's [runtime](../design/runtime.md). The prefill server is one node of
Trainium3, 128 ranks at LNC1 (one rank per physical NeuronCore), eight pipeline stages of a 16-rank TP group. The
decode server is two nodes, 256 ranks at LNC1, with DSPARK speculative decoding ([PD decode](#pd-decode)). The code is
in `sglang_neuron/models/kimi_k3/` (the decode in `decode/`), and the NKI kernels it calls are in `kernels/`
(`kernels/PROVENANCE.md`).

## The model

Kimi-K3 is a hybrid MoE transformer of 93 layers: 69 KDA layers (Kimi delta attention, a recurrent linear attention)
and 24 MLA layers, and each layer's attention residuals (AttnRes) read a bank of earlier block outputs. The checkpoint's
experts are MXFP4 (`compressed-tensors`). The prefill also writes the drafter's context rows, which a decode's
speculative drafter reads.

| State | Where it lives |
|---|---|
| MLA latent rows | SGLang's hybrid linear-attention pool, with the Neuron MLA pool as its full-attention leaf: FP8 rows (`--kv-cache-dtype fp8_e4m3`), 128-token pages, stored in DCP placement over the stage's 16 ranks |
| KDA state | SGLang's request-table state pool, built as `mem_cache/state_pool.NeuronStatePool` with conv channel-major `[3 heads/tp dim, K - 1]` (the KDA kernels' layout) and upstream's temporal `[heads, dim, dim]`, both bf16 (`--mamba-ssm-dtype bfloat16`) |
| Drafter rows | one row pool per drafter layer at the MLA pool's pages, on the drafter writer stage |

## The parallel layout

| Axis | Size | What it splits |
|---|---|---|
| PP | 8 | layers, `SGLANG_PP_LAYER_PARTITION=12,12,12,12,12,12,12,9` (three MLA layers a stage) |
| TP | 16 | heads, experts and the vocabulary; between layers the tokens are sequence-parallel over the 16 ranks |
| DCP | 16 | each stage's MLA rows: 128-token stripe `j` of a request lives on rank `j % 16` |

Each stage is one compiled graph a bucket (2048 and 4096 tokens). The stage boundaries move over the PP seam as three
declared kinds (`NeuronModel.pp_prefill_boundary_kinds`): the hidden rows, the attention-residual bank (one more block
a stage) and the drafter accumulator. The catalog entry sets `NEURON_SCRATCHPAD_PAGE_SIZE=512` for the stage graphs.

## Environment

The Kimi-K3 prefill has run with these versions only, and the decode with the same versions on two such nodes:

| Piece | Version |
|---|---|
| Node | one Trainium3 node (16 devices, 128 physical NeuronCores) |
| Neuron SDK | 2.33.0: neuronx-cc 2.28.7631.0, nki 0.7.0, torch-neuronx 2.14.3.0.282+dc70ffa, runtime 2.35.30.0, collectives 2.35.43.0 |
| torch, Python | 2.14.0+cu130, 3.14.6 |
| SGLang | `43881a7`, from source as in the [setup guide](../getting-started/setup-guide.md) |
| NIXL | with the patches in [NIXL patches](../../patches/nixl/README.md): the overlay `tools/build_nixl.sh` builds |

- The container image has the same SGLang, the same Neuron SDK 2.33.0 libraries and compiler, and NIXL with the same
  patches, with the release's torch-neuronx build (2.14.3.0.1+b1b6c66). The
  commands below are in the image's form ([container guide](../guides/container.md)); in a development environment,
  run them with `SGLANG_PLATFORM=neuron` and `<nixl overlay>:<sglang checkout>/python` on `PYTHONPATH`.
- The [example](../../examples/kimi_k3/) has the three nodes' scripts, the balancer and a chat front for
  OpenAI chat clients (the prefill takes token ids only).
- The setup guide's validated container (neuronx-cc 2.28.4568.0, torch-neuronx 2.14.3.0.2115) is a different build of
  the compiler and of torch-neuronx; Kimi-K3 has not run on it.
- Place the decode's two nodes in one UltraServer (`neuron-ls` in the container prints each node's UltraServer); the
  decode has not run over two UltraServers. The prefill node has run in the decode's UltraServer and in another one,
  where its KV goes to the decode over EFA between the UltraServers.
- The PP seam moves every stage boundary over NIXL, so the prefill needs the patched NIXL with the fake transfer
  backend too: the image's, or in a development environment `<nixl overlay>` ahead of SGLang on `PYTHONPATH`, with
  `NIXL_PLUGIN_DIR` unset. Without NIXL the launch stops at
  `No module named 'nixl'`. A container's own NIXL wheel is not served: one that bundles protobuf crashes at import
  next to torch-neuronx, and the seam relies on the fixes in `patches/nixl`.
- Leave `NEURON_LOGICAL_NC_CONFIG` unset: the catalog entry runs the schedulers at LNC1, one rank per physical
  NeuronCore, and a launch that sets another value than torch-neuronx's default for the host is refused.
- On an empty compile cache (`TORCH_NEURONX_NEFF_CACHE_DIR`, `TORCH_NEURONX_HLO_CACHE_DIR`) the first launch compiles
  every stage graph before it answers `/health`. The cache entries depend on where the plugin is installed, so a
  development environment serves from one checkout path to reuse them (in the image the path is fixed).

## Launch

The served prefill: a PD prefill with the kimi-prompt prefix cache and its host tier, paired with the
[PD decode](#pd-decode) over NIXL.

```bash
SGLANG_PP_LAYER_PARTITION=12,12,12,12,12,12,12,9 \
SGLANG_NEURON_KIMI_PROMPT_HOST_SLOTS=18000 SGLANG_NEURON_KIMI_PROMPT_HOST_STATE_SLOTS=2000 \
python3 -m sglang.launch_server --model-path /models/Kimi-K3 --trust-remote-code --skip-tokenizer-init \
  --disaggregation-mode prefill --disaggregation-transfer-backend nixl --disaggregation-bootstrap-port 8998 \
  --host <prefill address> --port 30000 \
  --pp-size 8 --tp-size 16 --dcp-size 16 --page-size 128 --kv-cache-dtype fp8_e4m3 \
  --chunked-prefill-size 4096 --mamba-ssm-dtype bfloat16 \
  --radix-cache-backend kimi-prompt --max-mamba-cache-size 512 \
  --max-total-tokens 524288 --max-running-requests 32 --mem-fraction-static 0.85 \
  --speculative-draft-model-path /models/<drafter checkpoint>
```

- `SGLANG_PP_LAYER_PARTITION=12,12,12,12,12,12,12,9` is the one split the stages serve: every stage ends after an MLA
  layer and has its AttnRes table. A launch without it, or with another split, is refused.
- `--skip-tokenizer-init` makes the prefill take token ids only (`/generate` with `input_ids`). For
  `/v1/chat/completions`, or `/v1/completions` with text, launch the prefill without it: the balancer sends the same
  request to both servers, and a prefill without a tokenizer fails a chat request.
- `--max-total-tokens` counts one DCP rank's rows: 524,288 tokens a rank is 8,388,608 tokens a stage.
- `--max-running-requests` bounds the requests the prefill holds, those in the stages and those whose KV is still
  going to the decode. Below `2 x --pp-size + 3` a stage idles between forwards; without the flag a pipelined PD
  prefill takes `4 x --pp-size` (32 at PP8).
- `--radix-cache-backend kimi-prompt --max-mamba-cache-size 512` is the prefix cache. The two
  `SGLANG_NEURON_KIMI_PROMPT_HOST_*` values are its host tier: 18,000 page slots and 2,000 state slots, about
  1,080 GB of host memory a node ([The kimi-prompt cache](#the-kimi-prompt-cache)).
- `--speculative-draft-model-path <drafter checkpoint>` writes the drafter's context rows, which the DSPARK decode
  reads (the decode takes the same flag); without it the prefill writes none. The prefill runs no speculative step,
  so it takes no `--speculative-algorithm`. The drafter row pools sit beside SGLang's pools on the last stage (about
  1.41 GiB a rank at 524,288 tokens). Each rank charges them against its own free memory, so a launch with the
  drafter states `--max-total-tokens`.
- The launch rules (`models/kimi_k3/launch.py`) set both roles' sampler, `SGLANG_NEURON_SAMPLER=topk_topp` (each
  request's temperature, top-k, top-p and seed on the device), and the 512 MB scratchpad page the graphs compile for;
  a launch that sets either to another value is refused.
- `--disaggregation-transfer-backend fake` in place of `nixl --disaggregation-bootstrap-port 8998` runs the prefill
  without a decode peer, for a prefill-only benchmark: no KV moves, and the PP seam still moves the stage boundaries
  over NIXL.
- The launch rules also set `SGLANG_NEURON_PP_PREFILL_EARLY_LAST_SEND=1` on the prefill when the launch does not set
  it: every stage sends a request's last chunk at its own forward ([The hand-off](#the-hand-off-to-the-decode)).

## The hand-off to the decode

The prefill's last stage samples each request's first output token with the request's own temperature, top-p, top-k and
seed (the `topk_topp` device sampler). The draw is keyed by the seed and the position of the last prompt token, so a
seeded request repeats. With a top-k, top-p or min-p filter, token 0 comes from the top 256 candidates, and the decode
filters the full vocabulary: the two differ only when the filters keep more than 256 tokens. The metadata row carries
the token to the decode, which keeps its own sampler. The settings below let the prefill send a request's last chunk and
answer the request before its release.

- `SGLANG_NEURON_PP_PREFILL_EARLY_LAST_SEND=1`: a stage other than the last sends a request's last chunk right after it
  launches that chunk's forward, with no metadata row. The last stage sends its chunk and the metadata row as soon as
  its outputs are on the host. Without it, each stage sends at its processing, after the outputs have come round the
  PP ring, and under load stage s sends about s + 1 stage times after the forward. The decode fails a room whose
  metadata row does not land in 30 s after every source reported (`sglang_patches/pd_metadata_wait`).
- `SGLANG_NEURON_PP_PREFILL_EARLY_ANSWER` (on by default): the prefill answers a request right after a stage processes
  its last chunk, if every stage sent that chunk at its forward. Without it, the prefill answers at the release, about
  pp_size + 1 chunk times later under load. SGLang's router reads the prefill's response before the decode's, and a
  chat or completions stream sends its 200 only with its first chunk, so both hold the client's first token until the
  prefill answers.
- `SGLANG_NEURON_PP_PREFILL_INFLIGHT_WAIT_PASSES` (on by default): a request whose prompt starts with the whole prompt
  of a request that waits for its release waits for that release when the release inserts more of that prompt into
  the prefix cache than the cache holds now. The next turn of a conversation then takes the hit. On the chunk-keyed tree
  this applies only to a prompt that ends on a whole chunk: the tree inserts each other whole chunk when the chunk is
  computed.
- The PD load balancer (`sglang_neuron/disaggregation/balancer.py`) streams the decode's answer as soon as the decode
  answers 200, and it watches the prefill until the decode's first event. After the stream, it keeps the prefill's
  connection open until the prefill answers, because a closed connection aborts the prefill's request.

## Serving a decode over NIXL

With `--disaggregation-transfer-backend nixl` the prefill hands each request to a decode server whose DCP size is a
multiple of 16, by whole page stripes (`disaggregation/dcp_stripes.py`, through `sglang_patches/dcp_page_transfer.py`):
MLA stripe `j` goes from prefill rank `j % 16` to decode rank `j % Cd`, the drafter rows of the decode's drafter window
go into the request's ring of rider pages, and each KDA layer's state goes as four parts (the q, k and v conv parts and
the recurrent state) to the request's decode state slot. The decode announces where these land (a page rule in its
registration), and the prefill refuses at registration a decode whose rule does not match its own pools. On hardware
the bytes the two-node DCP32 decode ([PD decode](#pd-decode)) received were bitwise equal to the prefill's.

The flags of the launch above that pair the prefill with a decode:

- `--disaggregation-transfer-backend nixl --disaggregation-bootstrap-port 8998`: the decode registers with the
  prefill's bootstrap server on that port.
- `--host <prefill address>`: SGLang binds the bootstrap server on `--host`, and the decode ranks on other nodes reach it and
  the prefill's NIXL listeners there; on loopback no decode can pair. The HTTP port is then reachable on that network
  too.
- Both servers load the patched NIXL of the [Environment](#environment) section.
- A decode can restart under a running prefill. The prefill's first libfabric WRITEs to some restarted decode ranks
  complete as canceled (EFA status 1, `Operation canceled`, prov_errno 1; ai-dynamo/nixl#2505), and later writes to
  them pass. The Kimi-K3 prefill posts such a batch's failed writes again (`sglang_patches/nixl_repost_failed_writes`,
  up to 4 rounds; each logs `NIXL re-post round`; the catalog entry's `pd_prefill_reposts_failed_writes`), so those
  requests are served. A prefill restarted under a running decode is served too.
- A decode the prefill refuses at registration (a page rule that does not match, another page size or KV dtype) is
  refused alone: the prefill logs why, fails that decode's requests with the reason on both servers, and keeps serving
  every other decode (`sglang_patches/nixl_peer_refusal`).

## PD decode

The decode server spans two Trainium3 nodes, the one decode layout the catalog entry declares
(`models/kimi_k3/decode/capability.TWO_NODE_DP8`):

| Axis | Size | What it splits |
|---|---|---|
| TP | 256 | every rank of both nodes at LNC1, as 8 DP-attention groups of attention TP 32 (four groups a node) |
| EP | 256 | the experts over all ranks |
| DCP | 32 | each attention-TP group's MLA rows |

Each step runs DSPARK with `k = 3` (4 requests a DP group, 4 verify rows each) or `k = 7` (2 requests, 8 rows) as two
programs on every rank: the verify (the 93 layers in one kernel call, the full-vocab sampler and the acceptance) and
the draft (the 5-layer MLA drafter over the request's ring of rider pages). The model's own SGLang speculative worker
(`decode/spec_worker.py`) runs them under SGLang's scheduler and overlap schedule.

Launch the decode on both nodes with the same flags and each node's `--node-rank`; node 0 serves HTTP:

```bash
python3 -m sglang.launch_server --model-path /models/Kimi-K3 --trust-remote-code \
  --nnodes 2 --node-rank <0|1> --dist-init-addr <node 0 address>:29500 \
  --tp-size 256 --dp-size 8 --enable-dp-attention --ep-size 256 --dcp-size 32 \
  --disaggregation-mode decode --disaggregation-transfer-backend nixl --disaggregation-bootstrap-port 8998 \
  --speculative-algorithm DSPARK --speculative-dspark-block-size 3 --speculative-num-draft-tokens 4 \
  --speculative-draft-model-path /models/<drafter checkpoint> \
  --enable-dp-lm-head --moe-a2a-backend none --kv-cache-dtype fp8_e4m3 --page-size 128 --disable-radix-cache \
  --context-length 1048576 --enable-dp-attention-local-control-broadcast --scheduler-recv-interval 1 \
  --mem-fraction-static 0.9 --host <this node address> --port 30000
```

- `--speculative-dspark-block-size 7 --speculative-num-draft-tokens 8` runs `k = 7`.
- The launch rules (`models/kimi_k3/launch.py`) set what the decode programs fix where the launch leaves it unset: the
  pools' tokens a rank (`--max-total-tokens`, the programs' page layout), the running requests
  (`--max-running-requests`, 32 at `k = 3` and 16 at `k = 7`) and the request slots beside them
  (`--disaggregation-decode-extra-slots`: a replica holds as many requests as its rider pool has rings). They refuse
  another value of each, a `k` the programs are not built for, a launch without the drafter or without the patched
  NIXL, and the runtime switches the programs are not built with (`NEURON_RT_A2AV_ONE_RANK_PER_CHIP`,
  `NEURON_RT_DBG_EARLY_POSTING`, `NKI_SIMULATOR`). Every rank starts its runtime with the layout's settings
  (`capability.RUNTIME_ENV`).
- Unless the launch environment sets them, the decode runs with the EFA libfabric settings (`FI_PROVIDER=efa`,
  `FI_EFA_USE_DEVICE_RDMA=1`, `FI_EFA_FORK_SAFE=1`, `FI_EFA_ENABLE_SHM_TRANSFER=0`) and `PYTHONHASHSEED=42` for its
  schedulers.
- SGLang fails a request whose KV has not arrived `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` seconds (300 by default)
  after the prefill took it, and that time includes the wait behind the prompts queued ahead of it. A cold 1M-token
  prompt alone takes most of the default. A workload that queues more (several long prompts at once) raises it on the
  decode. The AgentX traces at `--max-running-requests 32` on the prefill are such a workload: their first minutes
  queue the cold first turns of every session, up to 32 prompts of about 100K tokens at once, and at 300 s some of
  those requests failed with no tokens. Run them with `SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600` on the decode and
  `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600` on the prefill. The prefill fails a request whose decode has not
  sent its KV indices within the bootstrap timeout (300 s by default), and the decode then waits for KV that the
  prefill does not send; the client gets no error before the decode's timeout.
- Pair the decode (its node 0) with the prefill through the plugin's balancer:
  `python3 -m sglang_neuron.disaggregation.balancer --prefill http://<prefill address>:30000 --decode
  http://<decode node 0 address>:30000`. The prefill takes token ids only, so chat clients go through a front that
  applies the chat template; the [example](../../examples/kimi_k3/) runs the balancer and that front.
- The decode samples with each request's temperature, top-p, top-k and seed. A `/v1/chat/completions` request samples
  with the Kimi-K3 contract, as the Kimi-K3 API does: temperature 1.0 when it renders the think channel (0.6 when its
  `thinking` template kwarg is false) and top-p 0.95, whatever it sets (`SGLANG_NEURON_KIMI_K3_SAMPLING=client` keeps
  the client's values; models/kimi_k3/launch.chat_sampling_params). The contract changes the speculative acceptance:
  on real AgentX turns, which set no sampling value, top-p 1.0 in place of 0.95 gave a lower mean acceptance length
  (k = 3, temperature 1.0). A request that asks for structured
  output, min-p, `min_new_tokens`, a penalty, a logit bias, a custom logit processor, logprobs or hidden states is
  refused with HTTP 400 at the tokenizer of both servers, before the prefill computes its prompt.
- A chat request with `tools` and `tool_choice` `auto` or `none` is served. With `--tool-call-parser kimi_k3` on both
  servers, the parser reads the model's calls into `tool_calls`; without it, the calls stay in `content` in the
  model's markup. The servers apply no grammar of the tool-call parser (models/kimi_k3/launch.CHAT_TOOL_CALL_GRAMMAR),
  so a request with `tool_choice` `required` or a named tool, which needs the grammar to force the call, is refused
  with HTTP 400.
- `--disaggregation-transfer-backend fake` runs the decode alone, without a prefill: each request's KV receive
  completes at once over the pages the decode holds, for a decode-only benchmark. `SGLANG_NEURON_KIMI_K3_FORCED_AL`
  forces a mean acceptance length, for benchmarks only.
- A launch over several nodes in any other layout is refused, naming the flags it misses.

## Prefix cache

The served launch uses the kimi-prompt cache ([below](#the-kimi-prompt-cache)): a hit ends at the exact end of an
earlier prompt. The chunk-keyed tree stays available. The stage graphs keep a KDA state checkpoint at the end of every
chunk, so the chunk-keyed tree (`--radix-cache-backend neuron-chunk`) keys whole prefill chunks. A hit copies the
checkpoint at its last whole chunk into the request's state slot and restarts from there over the shared MLA pages; the
tokens after the prompt's last whole chunk (up to 4,095) are computed again. `--max-mamba-cache-size 512` gives each
stage rank 512 checkpoint slots in HBM (about 0.86 GiB a rank); a slot holds one 4,096-token chunk's state, so the slots
cover 2,097,152 tokens of chunks. The flag is required under PP: every stage's state pool must hold the same slots, or
stages evict different checkpoints.

A prompt that hits nothing takes the same time on this tree as with `--disable-radix-cache`, which runs every prompt
cold (for example a cold-prefill benchmark; drop the host-tier flags with it).

On this tree the prefill keeps at most 4 checkpoints on each cached prompt's path (`--mamba-max-states-per-path`
defaults to 4 here; SGLang's default -1 keeps one at every chunk). After each insert SGLang drops the path's shallowest
interior checkpoints and keeps the one at the prompt's end, so the state slots hold the ends of the cached prompts and a
next turn that extends a prompt hits its whole chunks. With a checkpoint at every chunk, the interior checkpoints of
long prompts fill the slots and the pool's LRU frees the leaves, the checkpoints at the prompts' ends. A value set at
launch is kept; `-1` cannot be passed (it reads as unset).

The HiCache host tier (`--enable-hierarchical-cache --hicache-ratio 4.7`) holds MLA pages, drafter rows and state
checkpoints on the host. At ratio 4.7 it takes 828 GiB of host memory a node: MLA pages and drafter rows of 19.7M tokens
a stage, and 2,406 checkpoint slots a rank (9.9M tokens of chunks). A hit from the host serves a 131,072-token prefix
faster than a cold prefill, and a prompt that hits nothing takes about as long as without the tier. The ranks
of a node share one budget for their host pools: the node's MemAvailable less SGLang's 10 GiB reserve, read before any
host pool is built and taken as the smallest across the nodes (the log prints it as `the 128 ranks on this host share
... GiB`). A tier that does not fit is refused, naming the bytes the node has left. On the nodes measured the budget
read 1,348 GiB, and ratio 6.25 at `--max-total-tokens 262144` starts on it without a new compile: MLA pages and drafter
rows of 26.2M tokens a stage and 3,201 checkpoint slots a rank (13.1M tokens of chunks), 1,101 GiB a node.

### The kimi-prompt cache

`--radix-cache-backend kimi-prompt` (`mem_cache/kimi_prompt_backend.py`) replaces the chunk-keyed tree and its HiCache
tier. It keeps END entries of finished prompts: an entry holds the KV pages of the tokens the prefill computed and
the KDA state of its last computed token (the entry takes the request's state slot). A next turn of a session hits the
exact end of the turn before and computes only its new tokens; on the chunk-keyed tree it computes again up to 4,095
tokens of the turn before.

- A hit shares the entry's whole pages. The request writes its new rows into the page that holds the hit's last
  tokens in place when no other request or entry uses that page past the hit (a next turn of a session). Else the page
  is copied into a page of the request first (its MLA rows, and its drafter rows on the drafter writer stage), through
  one host page with the runtime's DMA. The entry's state is copied into the request's state slot (SGLang's
  copy-on-write) before the request's first forward. The stage graphs read the prior of
  each rank from the positions its stripes hold, so a chunk can start at any token.
- Each forward that does not end its prompt also writes its end state into the request's tracking slot, and the cache
  keeps it as a chunk-end entry. A prompt that shares only those chunks with a running or finished prompt hits there.
- An entry that a next turn extends stays in the group of the new entry (keep_previous); the eviction takes the least
  recent entries first.
- The host tier (`SGLANG_NEURON_KIMI_PROMPT_HOST_SLOTS`, `SGLANG_NEURON_KIMI_PROMPT_HOST_STATE_SLOTS`) keeps the END
  entries that the device drops: their pages, their drafter rows and their KDA state. The copies to the host complete
  before the pages and the slot go back to the allocator. A host hit restores the entry and is exact too.
- The launch takes `--max-mamba-cache-size` (each entry holds one state slot) and no HiCache flags. The backend
  serves the PD prefill only.
- The host tier has separate page slots and state slots. Agentic traffic fills the state slots first: at 26,000 page
  slots and 800 state slots the state slots were full while the page slots were two-thirds empty, and the tier dropped
  the entries of sessions idle for about 15 minutes. 18,000 page slots and 2,000 state slots use the same host memory
  and keep those entries.

## What a prefill launch refuses

Another role than the PD prefill or decode; DP attention; a `--pp-size` other than 8, or a layer split other than the
served one; a TP size that does not split the KDA heads, MLA heads, experts, vocabulary or MLP widths; an EP size other
than 1 or the TP size; a DCP size other than the TP size; a chunk that is not whole stripe cycles (`128 x --dcp-size`
tokens) or is above the MX linear kernel's 8,192 rows; SGLang's default radix tree (keyed on KV pages, it would hit
inside a chunk where no state checkpoint exists); a drafter checkpoint the context rows do not implement, or a drafter
without `--max-total-tokens`; a `--speculative-algorithm`; a NIXL other than the patched overlay; a checkpoint config
without the experts' quantization, or without `rms_norm_eps`, `linear_attn_config.gate_lower_bound` or the SiTU betas;
a platform target other than Trainium3.
