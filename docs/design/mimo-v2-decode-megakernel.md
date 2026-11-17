# MiMo-V2 decode megakernel on trn2

Goal: a MiMo-V2.6-Flash decode step on one trn2.48xlarge (TP64 / EP64, LNC2) with **no XLA ops**: every layer's work,
its collectives included, inside NKI kernels, and in the end state one kernel call per step, at batches of at least 8.

**Status:** stage 2 below runs and is accurate (GSM8K 89.8% against 89.5% for the two half kernels; the
[recipe](../model-recipes/mimo-v2.md#the-decode) has the numbers) but is about 1.1 ms a step slower than the two half
kernels with the graph's all-reduces (TPOT 20.55 against 19.45 ms at c=1, 22.51 against 21.32 ms at c=8; the per-layer
graph 22.17 and 25.36 ms). Its 96 in-kernel all-reduces a step are the cost to cut next.

## What the per-layer graph runs today, per layer

| Piece | Today | Where it goes |
| --- | --- | --- |
| input RMSNorm, QKV, RoPE, attention with sinks | nkilib `attention_block_tkg` (256-channel padded heads) | layer kernel, own attention at Q/K 192 / V 128 |
| new token's K/V into the pages | NKI scatter kernel (`kv_write.py`) | layer kernel, DGE row scatter with OOB skip |
| decode mask | `gen_mask_tkg` kernel + XLA | in kernel, from positions and the table (position mode) |
| `o_proj`, residual add | XLA matmul | layer kernel, this rank's partial |
| attention all-reduce (64 ranks) | XLA collective | in-kernel collective (trnrt `cc_world` pattern) |
| post-attention RMSNorm, router (fp32 sigmoid, bias, top 8, renorm, affinity scatter) | XLA | layer kernel (max8 / find_index8 top 8, exp-only activations) |
| 4 local experts, FP8 ROW scales | nkilib `moe_tkg` | layer kernel body, the two programs splitting the experts' I columns |
| MoE all-reduce | XLA collective | in-kernel collective |
| embedding, final norm, LM head, sampler | XLA + device sampler | head / tail kernels, then folded in |

## Stages

| Stage | State |
| --- | --- |
| 1. layer kernel, the all-reduces outside | done: `kernels/attn_half.py`, `kernels/moe_half.py` (the default decode) |
| 2. the all-reduces inside, no XLA in a layer | done: `kernels/layer.py` (`SGLANG_NEURON_MIMO_DECODE_LAYER_KERNEL=1`), slower than stage 1 so far |
| 3. one call a step | not started |
| 4. own attention, FP8 KV | not started |

Two traps stage 2 met, both now in `kernels/layer.py`:

- every shared HBM tensor of a kernel needs a name of its own: two unnamed ones a helper made landed on one buffer (the
  layer's output on its attention residual; 12% off, the same on every rank);
- a kernel that returns a cache it wrote in place beside its output gave the caller another tensor in the output's place
  (130% off); the in-place write lands without the return.

### The plan of each stage

1. **Layer kernel** (attention half + MoE half in one call, all-reduce to all-reduce): `emulate()` torch reference of the
   model's served form, `nki.simulate` at grid 1 and 2 (bit-equal, planted mutations caught), one NeuronCore against the
   emulation, then a graph of 48 calls with XLA all-reduces between them. Accuracy gate: GSM8K and needles equal to the
   per-layer graph.
2. **Collectives inside**: the two all-reduces of a layer as in-kernel collectives over the 64 ranks (fenced send buffers
   and destinations, program 0 builds the sends; XLA's pairwise tree order where bit-equality is wanted). The step graph
   then holds kernel calls only.
3. **One call a step**: the 48 layers looped in one kernel over `[L, ...]`-stacked weights (one packed blob per dtype),
   embedding gather and LM-head + sampling tail inside; served through the platform's giga-decode seams
   (`samples_in_forward`, `runtime_class`), as GPT-OSS's and GLM-5.2's are on trn3.
4. **Own attention**: Q/K at 192 and V at 128 channels (no padding: 1.6x fewer KV bytes and reads), FP8 KV (unpacked),
   position-mode early exit at the batch's longest prior.

## Rules carried over from trnrt's trn2 kernels (trnrt/docs/MEGAKERNELS.md)

- Split every weight stream over the two programs (one program streams 340-400 GB/s, two together ~700 GB/s).
- Static DMA descriptors; row gathers by slot through the hardware DGE.
- exp and rsqrt only on the scalar engine (a table switch is ~1.3 us); rsqrt with one Newton step.
- NaN containment before any transpose over rows; sanitized top-k ids before a gather.
- trnrt gives every kernel a `rev` (a CRC of its source), since its compile cache keys the graph only; this plugin's
  cache refuses a directory built from other kernel sources (`backend/compile_cache.py`), so its kernels need none.

