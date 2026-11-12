# Wan2.2-T2V-A14B

Text to video on one trn2.48xlarge with SGLang Diffusion: one launch generates one 480p video, with 64 workers (one per
logical NeuronCore) as TP4 x CFG-parallel 2 x context-parallel 8. Start the image for one node as
[examples/README.md](../README.md) says, with a directory for the videos also mounted at `/out`
(`-v <output dir>:/out`).

| role | node | script |
|---|---|---|
| generation | the trn2.48xlarge | `run_generate.sh` |

## Launch

```bash
bash /opt/sglang-neuron/examples/wan22/run_generate.sh
```

## Check

```bash
ls -la /out/t2v_480p.mp4    # the video: 81 frames at 832 x 480
```

## Notes

- Defaults: `MODEL_PATH=/models/Wan2.2-T2V-A14B-Diffusers` (the `Wan-AI/Wan2.2-T2V-A14B-Diffusers` checkpoint),
  `OUTPUT_DIR=/out`, `PROMPT` and `NEGATIVE_PROMPT` the recipe's. Arguments after the script go to `sglang generate`.
- The script sets `SGLANG_NEURON_WAN_VAE_PATCH_PARALLEL_SIZE=16`, the VAE's patch parallelism for 480p; other
  resolutions take other values ([the Wan2.2 recipe](../../docs/model-recipes/wan22.md) lists them).
- [The Wan2.2 recipe](../../docs/model-recipes/wan22.md) explains the layout, image to video (`Wan2.2-I2V-A14B`), and
  serving many requests with `sglang serve`.
