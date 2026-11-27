# Radix-cache exactness on the SGLang server: a prompt run cold, then again (a full prefix hit), then a second prompt
# SPDX-License-Identifier: Apache-2.0
# that shares its first pages and diverges after (a partial hit), each compared with its cold greedy tokens; prints the
# cached tokens SGLang reports. usage: inkl_radix_check.py REF [PORT]
import json, sys, urllib.request
ref = json.load(open(sys.argv[1])); port = sys.argv[2] if len(sys.argv) > 2 else "30000"
def gen(ids, n=24):
    body = {"input_ids": ids, "sampling_params": {"max_new_tokens": n, "temperature": 0, "ignore_eos": True}}
    o = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/generate",
        json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600))
    return o.get("output_ids") or [], (o.get("meta_info") or {}).get("cached_tokens")
def flush():
    urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/flush_cache", b"", {}), timeout=600)
for r in ref:
    p = r["prompt"]
    if len(p) < 600:
        continue
    flush(); cold, c0 = gen(p)
    warm, c1 = gen(p)
    # B: A's first 2/3 (whole pages) then other tokens; cold run of B after a flush, then B after A's run
    cut = (2 * len(p) // 3) // 256 * 256
    b = p[:cut] + list(reversed(p[cut:cut + 300]))
    flush(); b_cold, _ = gen(b)
    flush(); gen(p); b_hit, c2 = gen(b)
    print(f"RADIX prompt {len(p)}: repeat == cold {warm == cold} (cached {c1}); shared-prefix B ({len(b)} tokens, "
          f"{cut} shared) hit == cold {b_hit == b_cold} (cached {c2}); cold vs HF {sum(int(x == y) for x, y in zip(cold, r['tokens']))}/24",
          flush=True)
