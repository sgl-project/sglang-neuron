#!/bin/bash
# GLM-5.2 prefill/decode pair: the balancer, which sends each request to both servers. Run it once both are up, on the
# prefill node or any host that reaches both; it needs no Neuron device.
set -euo pipefail

PREFILL_URL=${PREFILL_URL:?set PREFILL_URL to the prefill server, http://<prefill node>:30000}
DECODE_URL=${DECODE_URL:?set DECODE_URL to the decode server, http://<decode node>:30000}

exec python3 -m sglang_neuron.disaggregation.balancer --prefill "$PREFILL_URL" --decode "$DECODE_URL" "$@"
