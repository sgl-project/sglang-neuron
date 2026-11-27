# TTFT of a conversation turn on the SGLang server with the radix cache: a history of H tokens plus a new turn of N
# SPDX-License-Identifier: Apache-2.0
# tokens, cold (cache flushed) and warm (the history ran before). usage: inkl_turn_ttft.py [PORT]
import json, sys, time, random, urllib.request
port = sys.argv[1] if len(sys.argv) > 1 else "30000"
U = f"http://127.0.0.1:{port}"
def ttft(ids):
    body = {"input_ids": ids, "stream": True, "sampling_params": {"max_new_tokens": 8, "temperature": 0, "ignore_eos": True}}
    t0 = time.perf_counter(); first = None; cached = None
    with urllib.request.urlopen(urllib.request.Request(U + "/generate", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800) as r:
        for raw in r:
            s = raw.decode().strip()
            if s.startswith("data: ") and s != "data: [DONE]":
                first = first or time.perf_counter() - t0
                cached = (json.loads(s[6:]).get("meta_info") or {}).get("cached_tokens", cached)
    return first, cached
rng = random.Random(0)
for H in (1024, 4096, 8192):
    hist = [rng.randrange(1000, 150000) for _ in range(H)]
    turn = [rng.randrange(1000, 150000) for _ in range(128)]
    urllib.request.urlopen(urllib.request.Request(U + "/flush_cache", b"", {}), timeout=600)
    cold, _ = ttft(hist + turn)
    urllib.request.urlopen(urllib.request.Request(U + "/flush_cache", b"", {}), timeout=600)
    ttft(hist)
    warm, c = ttft(hist + turn)
    print(f"TURN history {H} + turn 128: TTFT cold {cold:.2f} s, warm {warm:.2f} s (cached {c})", flush=True)
