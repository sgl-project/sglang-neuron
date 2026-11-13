# Model recipes

Every LLM runs on the same [runtime](../design/runtime.md), as an aggregated server or as prefill and decode servers;
FLUX.1-dev and MiniMax-H3 run on SGLang Diffusion.
A recipe covers what is particular to a model: its checkpoint and graphs, the layouts it serves, the SGLang changes it
needs and what its launch refuses.

- [GPT-OSS-120B](gpt-oss.md): MXFP4 experts, an FP8 KV cache, expert parallelism, and a disaggregated decode with DP
  attention.
- [DeepSeek-V2-Lite](deepseek-v2.md): MLA over SGLang's MLA pool, served on attention TP4.
- [Qwen3](qwen3.md): Qwen3-0.6B and Qwen3-8B on the shared dense transformer, TP4 (and TP8 for Qwen3-8B).
- [Qwen3-MoE](qwen3-moe.md): Qwen3-30B-A3B, the dense transformer with a sparse MoE block, TP4 with EP4.
- [GLM-4](glm-4.md): GLM-4-9B-0414 on the dense transformer, with QKV biases, a fused gate and up projection, two more
  norms and a partial RoPE.
- [Gemma 4](gemma4.md): Gemma 4 31B (text), sliding and global layers of different head dims over one uniform KV
  pool, torch attention with indirect-DMA cache access, TP4 on a trn2.3xlarge.
- [Qwen3.5-397B PD](qwen35-pd.md): the giga megakernel as disaggregated prefill and decode, with a kv_cp=16 (DCP)
  decode for 256K context.
- [Qwen3.5 radix](qwen35-radix.md): the giga megakernel with the radix cache.
- [Kimi-K3](kimi-k3.md): the PD prefill, PP8 x TP16 with the MLA rows in DCP16 placement and the KDA state in SGLang's
  state pool, a chunk-keyed prefix cache with state checkpoints.
- [FLUX.1-dev](flux1-dev.md): text-to-image on SGLang Diffusion's own runtime, not the LLM runtime above, TP8 x CP8 on
  the 64 logical NeuronCores of a trn2.48xlarge.
- [MiniMax-H3](minimax-h3.md): text and keyframes to video and audio on SGLang Diffusion, TP8 x CP8 on the 64 logical
  NeuronCores of a trn2.48xlarge.
- [Wan2.2 A14B](wan22.md): text-to-video and image-to-video on SGLang Diffusion's own runtime, TP4 x CP8 x CFG2 on
  the 64 logical NeuronCores of a trn2.48xlarge.
- [FLUX.3-Action](flux3-action.md): the DROID robot policy on SGLang Diffusion's `Flux3ActionPipeline`, `sd` and `gd` at
  TP4 and TP4 x CP2 and `base` at TP4 x CFG2, on 4 or 8 NeuronCores of a trn2.48xlarge.
- [LingBot-World-V2](lingbot-world-v2.md): real-time image-to-world on SGLang Diffusion over its WebSocket, a TP8 x CP8
  DiT at LNC 1 over a paged K/V history with UMT5 and the streaming VAE on 16 more cores of a trn2.48xlarge.
- [GLM-5.2](glm52.md): GLM-5.2-MXFP4, MLA with DeepSeek sparse attention, disaggregated only: a prefill of four
  TP32 or sixteen TP8 pipeline stages on the vendored prefill giga kernel in front of the vendored decode giga
  kernel at TP128, at 32 slots of 128k or 1M positions, with async MTP K7, on the pinned Neuron SDK 2.33.0.

[Onboarding a model](../model-dev/onboarding-models.md) covers adding one.
