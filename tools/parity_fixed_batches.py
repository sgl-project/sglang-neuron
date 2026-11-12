"""Greedy token ids under controlled batch composition, for exact-token A/B between two
servers (e.g. this plugin on SGLang main vs a baseline).

A: every prompt alone (sequential). B: waves of 8 sent together with the same output
length, so each wave is one batch from admission to finish. Batch composition is then
the same on both servers, so batch-shape-dependent numerics cancel out.

Usage: python tools/parity_fixed_batches.py <port> <out.json>
       python tools/parity_compare.py <a.json> <b.json>
"""
import concurrent.futures as cf
import json
import sys
import urllib.request

port, out_path = int(sys.argv[1]), sys.argv[2]
topics = ["volcanoes", "the French revolution", "photosynthesis", "binary search",
          "the Roman empire", "black holes", "jazz music", "coral reefs", "TCP/IP",
          "the immune system", "glaciers", "chess openings", "neural networks",
          "the printing press", "honeybees", "plate tectonics"]
shared = "You are a helpful assistant. Answer carefully and concisely.\n\n" * 3
prompts = [(shared if i % 2 == 0 else "") +
           f"Question {i}: explain {topics[i % len(topics)]} in a few sentences.\nAnswer:"
           for i in range(24)]


def gen(text, n):
    body = {"text": text, "sampling_params": {"max_new_tokens": n, "temperature": 0.0,
                                              "ignore_eos": True}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())["output_ids"]


solo = [gen(p, 64) for p in prompts]
waves = []
for w in range(3):
    batch = prompts[w * 8:(w + 1) * 8]
    with cf.ThreadPoolExecutor(8) as ex:
        waves.extend(ex.map(lambda p: gen(p, 64), batch))
json.dump({"solo": solo, "waves": waves}, open(out_path, "w"))
print("wrote", out_path)
