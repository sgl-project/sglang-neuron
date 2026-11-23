# SPDX-License-Identifier: Apache-2.0
"""A needle in a long haystack, through the server's chat API: the long-context check of the full-attention layers
(the sliding layers see 128 tokens only, so a needle far back is found through the 9 full layers or not at all).

    python3 run_needle.py --url http://127.0.0.1:30000 --tokens 32000 120000 250000 --depths 0.1 0.5 0.9

Each point puts "The secret number is <n>." at a depth of a filler text of about ``tokens`` tokens (4 characters a
token is the estimate; the server's usage reports the real count) and asks for the number, greedy."""

import argparse
import json
import random
import urllib.request

FILLER = ("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
          "A quiet morning by the river, the boats drift slowly and nobody is in a hurry. ")


def ask(url: str, model: str, prompt: str, max_tokens: int) -> tuple:
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    request = urllib.request.Request(f"{url}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=3600) as response:
        out = json.load(response)
    message = out["choices"][0]["message"]
    # the answer, or where the model ran out of tokens still reasoning, its reasoning
    return (message.get("content") or message.get("reasoning_content") or ""), out["usage"]["prompt_tokens"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="MiMo-V2.6-Flash-RL")
    parser.add_argument("--tokens", type=int, nargs="+", default=[8000, 32000])
    parser.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    # MiMo reasons before it answers; a small budget can end inside the reasoning
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    found = total = 0
    for tokens in args.tokens:
        for depth in args.depths:
            secret = random.Random(tokens * 10 + int(depth * 10)).randint(100000, 999999)
            haystack = (FILLER * (tokens * 4 // len(FILLER) + 1))[:tokens * 4]
            cut = int(len(haystack) * depth)
            prompt = (haystack[:cut] + f" The secret number is {secret}. " + haystack[cut:]
                      + "\n\nWhat is the secret number? Answer with the number only.")
            answer, prompt_tokens = ask(args.url, args.model, prompt, args.max_tokens)
            ok = str(secret) in answer
            found, total = found + ok, total + 1
            verdict = "found" if ok else "MISSED"
            print(f"{prompt_tokens:>8} tokens, depth {depth:.1f}: {verdict} ({answer.strip()[:60]!r})")
    print(f"{found}/{total} found")


if __name__ == "__main__":
    main()
