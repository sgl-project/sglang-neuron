# SPDX-License-Identifier: Apache-2.0
"""An OpenAI chat-completions front for the Kimi-K3 prefill/decode pair. The prefill runs with
``--skip-tokenizer-init`` and takes token ids only, so this front applies the checkpoint's chat template and sends the
ids to the balancer's ``/generate``.

    python3 chat_front.py --upstream <balancer URL> --model <checkpoint> --port 8021 [--log <jsonl>]

It binds 127.0.0.1 only. Per ``/v1/chat/completions`` request:

* the messages through the checkpoint's chat template (``apply_chat_template``, ``add_generation_prompt``, the
  template's defaults) as token ids;
* the upstream's ``/generate`` with those ids, the request's temperature, top_p, top_k, max_tokens (or
  ``max_completion_tokens``) and seed (``sampling_seed``), and the end-of-message and EOS ids as stop token ids;
* the reply: the output up to the first stop id, decoded and split as a reasoning parser does. When the prompt opens
  the think channel, the output up to ``<|close|>think<|sep|>`` is ``reasoning_content`` and the rest ``content``, its
  channel markers removed (an unclosed think channel gives empty content). The request's stop strings cut the
  reasoning first, then the content.

With ``--log``, every request appends one JSON line: its prompt and completion tokens, finish reason, the server's
``spec_accept_length``, seconds, and any error."""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional, Tuple

THINK_CLOSE = re.compile(r"<\|close\|>\s*think\s*<\|sep\|>")
THINK_OPEN_END = re.compile(r"<\|open\|>\s*think\s*<\|sep\|>\s*$")
CHANNEL = re.compile(r"<\|(?:open|close)\|>\s*(?:response|message|think)[^<]*?<\|sep\|>")
SPECIAL = re.compile(r"<\|[A-Za-z_]+\|>|\[EOS\]")
END_OF_MESSAGE = 163586
"""The checkpoint config's ``eos_token_id``: the end of an assistant message."""


class ChatFront:
    def __init__(self, upstream: str, tokenizer, model: str, log: Optional[str] = None, timeout_s: float = 7200.0):
        self.upstream, self.tok, self.model = upstream.rstrip("/"), tokenizer, model
        self.stop_ids = sorted({END_OF_MESSAGE, int(tokenizer.eos_token_id)})
        self.log_path, self.timeout_s = log, timeout_s
        self.lock = threading.Lock()

    def log(self, record: dict) -> None:
        if self.log_path:
            with self.lock, open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    def sampling_params(self, body: dict) -> dict:
        """The upstream's sampling params of a chat request."""
        params = {"max_new_tokens": int(body.get("max_completion_tokens") or body.get("max_tokens") or 2048),
                  "temperature": float(body.get("temperature", 1.0)), "stop_token_ids": self.stop_ids}
        for key in ("top_p", "top_k"):
            if body.get(key) is not None:
                params[key] = body[key]
        if body.get("seed") is not None:
            params["sampling_seed"] = int(body["seed"])
        return params

    def generate(self, ids: List[int], body: dict) -> dict:
        payload = {"input_ids": ids, "sampling_params": self.sampling_params(body)}
        request = urllib.request.Request(self.upstream + "/generate", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            reply = json.loads(response.read())
        return reply[0] if isinstance(reply, list) else reply

    def cut(self, out: List[int]) -> Tuple[List[int], bool]:
        for i, t in enumerate(out):
            if int(t) in self.stop_ids:
                return out[:i], True
        return out, False

    @staticmethod
    def stop_at(text: str, stops) -> Tuple[str, bool]:
        hits = [text.find(s) for s in stops or () if s and text.find(s) >= 0]
        return (text[:min(hits)], True) if hits else (text, False)

    def chat(self, body: dict) -> dict:
        ids = self.tok.apply_chat_template(body["messages"], tokenize=True, add_generation_prompt=True,
                                           **dict(body.get("chat_template_kwargs") or {}))
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        ids = [int(i) for i in ids]
        reply = self.generate(ids, body)
        output = [int(t) for t in reply.get("output_ids", [])]
        out, stopped = self.cut(output)
        full = self.tok.decode(out, skip_special_tokens=False)
        # the output starts inside the think channel only when the prompt opens it (thinking on)
        thinking = bool(THINK_OPEN_END.search(self.tok.decode(ids[-8:], skip_special_tokens=False)))
        m = THINK_CLOSE.search(full)
        if not thinking and not m:
            reasoning, content = "", full
        else:
            reasoning, content = (full[:m.start()], full[m.end():]) if m else (full, "")
        reasoning = SPECIAL.sub("", CHANNEL.sub("", reasoning))
        content = SPECIAL.sub("", CHANNEL.sub("", content))
        stops = body.get("stop")
        stops = [stops] if isinstance(stops, str) else stops
        reasoning, hit = self.stop_at(reasoning, stops)
        if hit:
            content = ""
        else:
            content, hit = self.stop_at(content, stops)
        meta = reply.get("meta_info", {})
        message = {"role": "assistant", "content": content.strip(), "reasoning_content": reasoning}
        return {"id": f"chat-{uuid.uuid4().hex[:16]}", "object": "chat.completion", "created": int(time.time()),
                "model": self.model,
                "choices": [{"index": 0, "message": message, "finish_reason": "stop" if (stopped or hit) else "length"}],
                "usage": {"prompt_tokens": len(ids), "completion_tokens": len(output),
                          "total_tokens": len(ids) + len(output)},
                "server_meta": {k: meta.get(k) for k in ("spec_accept_length", "spec_verify_ct", "e2e_latency")}}


def handler(front: ChatFront):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def send(self, code: int, obj: dict) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.rstrip("/")
            if path in ("/v1/models", "/models"):
                return self.send(200, {"object": "list", "data": [{"id": front.model, "object": "model"}]})
            if path == "/health":
                return self.send(200, {"ok": True})
            return self.send(404, {"error": self.path})

        def do_POST(self):
            start = time.monotonic()
            record = {"path": self.path, "ts": time.time()}
            try:
                if not self.path.endswith("/chat/completions"):
                    return self.send(404, {"error": self.path})
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                out = front.chat(body)
                record.update(prompt_tokens=out["usage"]["prompt_tokens"],
                              completion_tokens=out["usage"]["completion_tokens"],
                              finish=out["choices"][0]["finish_reason"], **out["server_meta"])
                self.send(200, out)
            except Exception as error:  # noqa: BLE001 - reported to the client and the log
                record["error"] = repr(error)
                self.send(500, {"error": {"message": repr(error)}})
            finally:
                record["seconds"] = round(time.monotonic() - start, 3)
                front.log(record)

    return Handler


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8021)
    ap.add_argument("--log")
    ap.add_argument("--timeout", type=float, default=7200.0)
    args = ap.parse_args(argv)
    from transformers import AutoTokenizer

    front = ChatFront(args.upstream, AutoTokenizer.from_pretrained(args.model, trust_remote_code=True), args.model,
                      args.log, args.timeout)
    # a listen backlog for the client's bursts (the default 5 resets connections of 32 at once)
    ThreadingHTTPServer.request_queue_size = 256
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(front))
    print(f"chat front on 127.0.0.1:{args.port} -> {args.upstream}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
