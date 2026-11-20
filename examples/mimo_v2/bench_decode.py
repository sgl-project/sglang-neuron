# SPDX-License-Identifier: Apache-2.0
"""Decode speed at a few concurrencies: each of ``c`` requests streams ``--out`` tokens (ignore_eos) after a prompt of
about ``--in`` tokens, all sent at once; per request its TTFT (first streamed token) and TPOT (the rest of its time
over the rest of its tokens), the medians reported, and the output tokens a second over the batch's span.

    python3 bench_decode.py --url http://127.0.0.1:30000 --concurrency 1 8 --in 1000 --out 256

Distinct prompts per request (a counter at the front), so the radix cache does not share them."""

import argparse
import json
import statistics
import threading
import time
import urllib.request


def one(url: str, prompt: str, out_tokens: int, result: list) -> None:
    body = json.dumps({"text": prompt, "stream": True,
                       "sampling_params": {"max_new_tokens": out_tokens, "temperature": 0, "ignore_eos": True}})
    request = urllib.request.Request(f"{url}/generate", data=body.encode(),
                                     headers={"Content-Type": "application/json"})
    start = time.time()
    first = None
    count = 0
    with urllib.request.urlopen(request, timeout=3600) as response:
        for line in response:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            if first is None:
                first = time.time()
            count = json.loads(line[5:])["meta_info"]["completion_tokens"]
    end = time.time()
    result.append((first - start, (end - first) / max(count - 1, 1), count, start, end))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--in", dest="in_tokens", type=int, default=1000)
    parser.add_argument("--out", type=int, default=256)
    args = parser.parse_args()
    for c in args.concurrency:
        results, threads = [], []
        for i in range(c):
            prompt = f"{time.time_ns()}-{i} " + "hello world " * (args.in_tokens // 2)
            threads.append(threading.Thread(target=one, args=(args.url, prompt, args.out, results)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        ttft = statistics.median(r[0] for r in results)
        tpot = statistics.median(r[1] for r in results)
        span = max(r[4] for r in results) - min(r[3] for r in results)
        tokens = sum(r[2] for r in results)
        print(f"c={c}: median TTFT {ttft * 1000:.0f} ms, median TPOT {tpot * 1000:.2f} ms, "
              f"{tokens / span:.1f} output tok/s")


if __name__ == "__main__":
    main()
