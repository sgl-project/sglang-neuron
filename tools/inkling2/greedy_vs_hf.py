# Greedy tokens from the SGLang server for a reference file's prompts (token ids in, N tokens out), against HF's.
# SPDX-License-Identifier: Apache-2.0
# usage: inkl_sgen.py REF OUT [N] [PORT]
import json, sys, urllib.request
ref, out = json.load(open(sys.argv[1])), sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
port = sys.argv[4] if len(sys.argv) > 4 else "30000"
recs = []
for r in ref[:n]:
    body = {"input_ids": r["prompt"], "sampling_params": {"max_new_tokens": len(r["tokens"]), "temperature": 0,
                                                          "ignore_eos": True}}
    o = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/generate",
        json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600))
    toks = o.get("output_ids") or (o.get("meta_info") or {}).get("output_ids") or []
    same = sum(int(a == b) for a, b in zip(toks, r["tokens"]))
    first = next((i for i, (a, b) in enumerate(zip(toks, r["tokens"])) if a != b), None)
    print(f"prompt {len(r['prompt'])}: {len(toks)} tokens, {same}/{len(r['tokens'])} equal to HF's greedy "
          f"(first diff {first}); text {o.get('text', '')[:100]!r}", flush=True)
    recs.append({"prompt": r["prompt"], "tokens": toks})
json.dump(recs, open(out, "w"))
