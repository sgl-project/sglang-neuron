# GLM-5.2

Prefill/decode on two Trn3 nodes: one pipelined prefill server, 4 stages of 32 ranks (`PP=4`, the default) or 16 stages
of 8 ranks (`PP=16`), and one giga-kernel decode server of 128 ranks with MTP, whose 32 sequence slots each serve any
context up to 1,048,448 tokens. Start the image on each node and run the scripts in the order
[the examples page](../README.md) gives. The decode node also mounts the weight store ([below](#the-weight-store)).

| role | node | script |
|---|---|---|
| decode | decode node | `run_decode.sh` |
| prefill | prefill node | `run_prefill.sh` |
| balancer | prefill node | `run_balancer.sh` |
| client, accuracy check | prefill node | `run_client.sh`, `run_gsm8k.sh` |

## Launch

Give both servers the same `PP` (4 or 16, default 4): the decode's KV pages match the prefill's. The decode
also needs `WEIGHT_STORE`, the store's path in the container. `MODEL_PATH` defaults to `/models/GLM-5.2-MXFP4`.

```bash
PP=4 HOST=<decode node address> WEIGHT_STORE=/stores/glm52-1m-lat8 \
  bash /opt/sglang-neuron/examples/glm52/run_decode.sh                                     # decode node
PP=4 HOST=<prefill node address> bash /opt/sglang-neuron/examples/glm52/run_prefill.sh    # prefill node
# prefill node, second shell, once both servers are ready
PREFILL_URL=http://<prefill node address>:30000 DECODE_URL=http://<decode node address>:30000 \
  bash /opt/sglang-neuron/examples/glm52/run_balancer.sh
```

## Check

```bash
# prefill node, third shell (docker exec -it sglang bash)
bash /opt/sglang-neuron/examples/glm52/run_client.sh    # the answer: 391
bash /opt/sglang-neuron/examples/glm52/run_gsm8k.sh     # Accuracy: about 0.96 on the first 100 questions
```

[The GLM-5.2 recipe](../../docs/model-recipes/glm52.md) explains the flags. Both servers run `--reasoning-parser glm45`,
so a chat answer's thinking is in `reasoning_content` and the answer alone in `content`. The thinking counts against
`max_tokens`: a request that runs out of tokens while thinking returns an empty `content`.

## The weight store

The decode loads its weights from a weight store: the checkpoint converted to the giga kernel's per-rank layout, with
the MTP draft's files. It serves the 1M store with a latent split of 8, made from the 1M and the 128k stores. Build
them once, before the first launch, in a shell of the decode node's container started with `-v <stores dir>:/stores`
added; the build runs on the CPU. It can be resumed, and each worker holds one layer's experts in memory, so lower
`workers` on a host with less memory:

```bash
SGLANG_NEURON_CPU_MODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -c '
from sglang_neuron.models.glm52.points import POINTS
from sglang_neuron.models.glm52.layout.decode.weight_loaders_mxfp4 import build_rank_store
from sglang_neuron.models.glm52.layout.decode.weight_loaders_mtp import build_mtp_rank_store, build_mtp_fp8_tensor_store
ckpt = "/models/GLM-5.2-MXFP4"
for point in ("1m", "128k"):
    store, geo = f"/stores/glm52-{point}", POINTS[point].geometry()
    build_rank_store(ckpt, store, geo, workers=48)
    build_mtp_rank_store(ckpt, store, geo)
    build_mtp_fp8_tensor_store(ckpt, store, geo)'
python3 /opt/sglang-neuron/tools/glm52/build_latent_store.py \
  /stores/glm52-1m /stores/glm52-128k /stores/glm52-1m-lat8 8
```

## Notes

- The last step links the new store's files into the other two with absolute paths: keep both stores, and mount the
  stores directory at `/stores` on the decode node, the path the links were made under.
- The decode refuses a store built for another point, latent split or checkpoint, and names the field that differs.
- A server refuses its peer at the first request when the two servers run different `PP`.
- An empty `content`: the thinking used the whole `max_tokens`; raise it.
