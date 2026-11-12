#!/bin/bash
# The balancer in front of the prefill server and the decode server (its node 0): it sends each request to both and
# returns the decode's answer. It needs no Neuron device; run it on the prefill node.
set -euo pipefail

PREFILL_URL=${PREFILL_URL:?set PREFILL_URL to the prefill server, http://<prefill node>:30000}
DECODE_URL=${DECODE_URL:?set DECODE_URL to the decode server on its node 0, http://<decode node 0>:30000}

# listens on 127.0.0.1:8000; for clients on other hosts pass --host <address> (and --port) as arguments, reachable
# only by them.
exec python3 -m sglang_neuron.disaggregation.balancer --prefill "$PREFILL_URL" --decode "$DECODE_URL" "$@"
