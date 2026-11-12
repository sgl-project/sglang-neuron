# GLM-5.2 decode giga kernel

One NKI kernel runs a decode step's 78 layers on every rank: DSA sparse MLA attention with its indexer, the MXFP4
all-expert MoE, the LM-head tail and its sampler. The `decode/mtp/` fork is the MTP draft layer's
rows-layout chain, with attention, indexer, MLP and MoE sub-kernels of its own.

Every served decode point (`models/glm52/points.py`: `128k`, `1m`) runs this copy;
`models/glm52/decode/kernel_limits.py` (`DECODE_KERNEL_PACKAGE`) pins it. `models/glm52/ops/gk_decode.py` wraps the giga kernel
and `ops/mtp_draft.py` the draft chain.

## What the host reads from the copy

- Its limits, from the giga kernel's and the draft chain's source without importing them
  (`models/glm52/decode/kernel_limits.py`): the indexer keys a rank serves, the latent stacks, and the options it takes.
  `check_kernel_copy` refuses a geometry above them before any trace.
- The 640-byte static KV row (`decode/attention/mla/kv_row_layout.py`): bf16 `k_pe` divided by
  the static scale, then the e4m3fn latent in the kernel's channel order. It is the only row the copy reads, and the
  prefill writes it directly, so a page moves from the prefill to the decode without translation.
- The MXFP4 weight conversion (`decode/mxfp4_checkpoint.py`), which the weight store uses.

## Knobs

The copy reads `GLM52K_*` and `GLM52_*` knobs when it is imported or traced, each with its own default (most off). A
served point's values are its launch settings (`models/catalog.py` from `points.py`); `kernels/compile_knobs.py` puts
each set knob into the kernel digest that keys the NKI compile cache, so a knob change recompiles.

## The indexer's top-k

A rank's indexer keys are dealt over the CP group in `idx_block_size`-key blocks. The local stage selects the
rank's top `index_topk`; a shard above 32,768 keys is ranked in a power of two of equal chunks, then the chunk
winners. The packed CP merge selects the global top-k in NKI (`../topk_select.py`), each winner carrying its shard-local
key as the payload; a chunked shard takes the wide rounds, each rank keeping its share of the global top-k. At the
1M point the chunk-dealt pool spreads a sequence's chunks over the pool members (`models/glm52/layout/decode/idx_pool.py`).

## Provenance

`../PROVENANCE.md` lists where the copy comes from and every edit to it; `MANIFEST.json` records every file's sha256.
