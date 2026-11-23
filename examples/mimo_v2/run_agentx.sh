#!/bin/bash
# One InferenceX AgentX point against the server: SemiAnalysis's AgentX harness (agentx-harness 1.0.12, on AIPerf),
# its built-in agentx scenario over the Claude Code traces (semianalysis_cc_traces_weka_062126): 3,600 s at one
# concurrency, streamed chat completions with the server's usage. Run it on a separate CPU host: the harness is
# CPU-heavy, and the requests a minute it reaches depend on its client's cores, so state them next to any number.
#
#   pip install agentx-harness==1.0.12
#   URL=http://<server>:30000 CONCURRENCY=8 MAX_CONTEXT=262144 bash run_agentx.sh
#
# MAX_CONTEXT is the AgentX point (262144 or 1048576) and must not exceed the server's --context-length. Score the
# artifacts with InferenceX's own code (inferencex-e2e/infx/results/agentic in SemiAnalysisAI/InferenceX); per chip, one
# trn2.48xlarge is 16 Trainium2 chips.
set -euo pipefail

URL=${URL:?set URL to the server, http://<server>:30000}
SERVED_NAME=${SERVED_NAME:-MiMo-V2.6-Flash-RL}
TOKENIZER=${TOKENIZER:-XiaomiMiMo/MiMo-V2.6-Flash-RL}
CONCURRENCY=${CONCURRENCY:-8}
MAX_CONTEXT=${MAX_CONTEXT:-262144}
OUT=${OUT:-agentx_ctx${MAX_CONTEXT}_c${CONCURRENCY}}

exec agentx profile --url "$URL" --endpoint /v1/chat/completions --model "$SERVED_NAME" \
    --tokenizer "$TOKENIZER" --tokenizer-trust-remote-code \
    --max-context-length "$MAX_CONTEXT" --concurrency "$CONCURRENCY" \
    --public-dataset semianalysis_cc_traces_weka_062126 --output-artifact-dir "$OUT" "$@"
