#!/bin/bash
# Qwen3.5-397B-A17B, prefill/decode on two Trn3 nodes at 256K context: the DECODE server, the prefill's layout with
# each replica's KV split over its 16 ranks and a radix tree of its own. Run it inside the image on the decode node.
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/models/Qwen3.5-397B-A17B-FP8-text}
PACK=${PACK:?set PACK to the Qwen3.5 weight pack (the packed per-rank weights), such as /models/qwen35-pack}
HOST=${HOST:?set HOST to an address of this node that the other node and the balancer reach}
PORT=${PORT:-30000}
# a persistent compile cache: the same path on every start, one directory per image version and node role
CACHE_DIR=${CACHE_DIR:-/cache}
export TORCH_NEURONX_NEFF_CACHE_DIR=$CACHE_DIR/neff TORCH_NEURONX_HLO_CACHE_DIR=$CACHE_DIR/hlo \
    NKI_COMPILE_CACHE_URL=$CACHE_DIR/nki
# a role's compile can outlast the compile-cache lock's default timeout; this raises it
export TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT=10800
export SGLANG_NEURON_QWEN35_PACK=$PACK
export NEURON_RT_MAP_HBM=1 TORCH_NEURONX_ENABLE_STREAMS=1 QWEN35_ATTN_PAGE_Q=auto QWEN35_ATTN_SLOT_DGE=hw
# the KV moves over EFA with LIBFABRIC; the patched NIXL's unsolicited write receive keeps long transfers alive
export SGLANG_DISAGGREGATION_NIXL_BACKEND=LIBFABRIC NIXL_LIBFABRIC_EFA_UNSOLICITED_WRITE_RECV=1
# 128 ranks x 16 EFA rails would drain the huge pages; one host thread a rank keeps the launches fast
export FI_EFA_USE_HUGE_PAGE=0 FI_EFA_USE_DEVICE_RDMA=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

COMMON_FLAGS=(--device cpu --attention-backend neuron --sampling-backend pytorch
    --tp-size 128 --dp-size 8 --enable-dp-attention --ep-size 128 --enable-nvidia-megakernel
    --page-size 2048 --context-length 262144 --chunked-prefill-size 16384 --radix-cache-backend neuron-chunk
    --mamba-radix-cache-strategy extra_buffer --mamba-track-interval 2048 --mem-fraction-static 0.8
    --disable-custom-all-reduce --trust-remote-code --watchdog-timeout 7200 --stream-interval 1
    --disaggregation-transfer-backend nixl --disaggregation-bootstrap-port 8998 --port "$PORT")
# each replica context-parallels its KV over its 16 ranks
export SGLANG_NEURON_QWEN35_KV_CP=16

exec python3 -m sglang.launch_server --model-path "$MODEL_PATH" "${COMMON_FLAGS[@]}" \
    --disaggregation-mode decode --max-running-requests 512 --max-total-tokens 299008 --max-mamba-cache-size 2880 \
    --disaggregation-decode-enable-radix-cache --cuda-graph-bs-decode 16 32 64 \
    --host "$HOST" "$@"
