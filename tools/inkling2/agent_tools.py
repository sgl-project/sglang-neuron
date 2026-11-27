# SPDX-License-Identifier: Apache-2.0
# A tool-calling agent loop against the local SGLang server (127.0.0.1 only). SGLang's Inkling tool-call parser
# constrains calls with a grammar the device sampler does not apply, so the loop applies the chat template (with the
# tools) itself, generates through /generate and parses the template's format from the raw output (special tokens kept):
# <|message_model|>NAME<|content_invoke_tool_json|>{"name": ..., "args": {...}}<|end_message|>.
# Tools are simulated. Prints, per task and effort: turns, tool calls made, per-turn TTFT / time / prompt tokens,
# and the final answer. usage: agent_tools.py [effort ...]   (effort: default, none, low, medium, high)
import json, re, sys, time, urllib.request

URL = "http://127.0.0.1:30000/generate"
from transformers import AutoTokenizer
TOK = AutoTokenizer.from_pretrained("/models/Inkling-2-Mini-text")
TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather of a city.",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "calculator", "description": "Evaluates an arithmetic expression.",
     "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "search_orders", "description": "Looks up a customer's orders by email.",
     "parameters": {"type": "object", "properties": {"email": {"type": "string"}}, "required": ["email"]}}},
]
WEATHER = {"paris": {"temp_c": 14, "sky": "cloudy"}, "tokyo": {"temp_c": 21, "sky": "clear"},
           "sao paulo": {"temp_c": 27, "sky": "rain"}}
ORDERS = {"ana@example.com": [{"id": "A-101", "total": 129.9, "status": "shipped"},
                              {"id": "A-117", "total": 45.5, "status": "processing"}]}


def run_tool(name, args):
    if name == "get_weather":
        return WEATHER.get(str(args.get("city", "")).lower(), {"error": "unknown city"})
    if name == "calculator":
        expr = str(args.get("expression", ""))
        if not re.fullmatch(r"[0-9+\-*/(). ]+", expr):
            return {"error": "bad expression"}
        return {"result": eval(expr)}  # noqa: S307 (digits and operators only)
    if name == "search_orders":
        return {"orders": ORDERS.get(str(args.get("email", "")).lower(), [])}
    return {"error": f"unknown tool {name}"}


TASKS = [
    "What's the weather in Paris and in Tokyo right now? Then tell me the temperature difference in Celsius.",
    "Customer ana@example.com asks how much she has spent in total across her orders and which one is not shipped yet.",
    "Qual o clima em Sao Paulo agora? Responda em portugues, em uma frase.",
]
CALL = re.compile(r"<\|message_model\|>([A-Za-z_][\w]*)<\|content_invoke_tool_json\|>(\{.*?\})<\|end_message\|>", re.S)


def chat(messages, effort):
    kw = {} if effort == "default" else {"reasoning_effort": effort}
    text = TOK.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True, tokenize=False, **kw)
    ids = TOK.encode(text, add_special_tokens=False)
    body = {"input_ids": list(ids), "stream": True,
            "sampling_params": {"max_new_tokens": 1024, "temperature": 0, "skip_special_tokens": False}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; text = ""; usage = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            s = raw.decode().strip()
            if not s.startswith("data: ") or s == "data: [DONE]":
                continue
            d = json.loads(s[6:])
            ttft = ttft or time.perf_counter() - t0
            text = d.get("text", text)
            m = d.get("meta_info") or {}
            usage = {"prompt_tokens": m.get("prompt_tokens"), "completion_tokens": m.get("completion_tokens")}
    out = [text]
    return "".join(out), ttft or 0.0, time.perf_counter() - t0, usage


def agent(task, effort, max_turns=6):
    messages = [{"role": "system", "content": "You are a helpful assistant. Use the tools when they help; answer "
                 "concisely once you have what you need."}, {"role": "user", "content": task}]
    turns, ncalls = [], 0
    for _ in range(max_turns):
        text, ttft, dt, usage = chat(messages, effort)
        calls = CALL.findall(text)
        think = sum(len(m) for m in re.findall(r"<\|content_thinking\|>(.*?)(?:<\|end_message\|>|$)", text, re.S))
        turns.append((ttft, dt, usage.get("prompt_tokens"), usage.get("completion_tokens"), len(calls), think))
        if not calls:
            final = re.findall(r"<\|content_text\|>(.*?)(?:<\|end_message\|>|<\|content_model_end_sampling\|>|$)", text, re.S)
            return turns, ncalls, (final[-1] if final else text).strip()
        tcs = []
        for i, (name, js) in enumerate(calls):
            try:
                args = json.loads(js).get("args", {})
            except json.JSONDecodeError:
                args = {}
            tcs.append({"id": f"call_{len(turns)}_{i}", "type": "function", "function": {"name": name, "arguments": args}})
        messages.append({"role": "assistant", "content": "", "tool_calls": tcs})
        for tc in tcs:
            ncalls += 1
            res = run_tool(tc["function"]["name"], tc["function"]["arguments"])
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(res)})
    return turns, ncalls, "(no final answer within the turn limit)"


for effort in (sys.argv[1:] or ["default", "low"]):
    for task in TASKS:
        t0 = time.perf_counter()
        try:
            turns, ncalls, final = agent(task, effort)
        except Exception as e:  # noqa: BLE001
            print(f"AGENT effort={effort} task={task[:40]!r} ERROR {type(e).__name__}: {str(e)[:300]}", flush=True)
            continue
        per = " | ".join(f"TTFT {a:.2f}s gen {b:.1f}s prompt {c} out {d} calls {e} think {f}c" for a, b, c, d, e, f in turns)
        print(f"AGENT effort={effort} turns={len(turns)} tool_calls={ncalls} total {time.perf_counter() - t0:.1f}s\n"
              f"  task: {task}\n  turns: {per}\n  answer: {final[:400]!r}", flush=True)
