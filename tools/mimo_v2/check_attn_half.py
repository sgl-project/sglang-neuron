#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hardware check of MiMo-V2's decode attention half kernel (models/mimo_v2/kernels/attn_half.py) on one NeuronCore,
against a torch reference of the served form: the input RMSNorm, the QKV projection, RoPE over the padded head's halves
(models/mimo_v2/rope.py), attention over each row's paged prior and its own token with the layer's window and sinks,
and the output projection of each head's first v_head_dim channels. It also checks that the new token's K and V land
in its slot and that no other cache row changes.

Rows of 5, 130, 300 and 1000 prior tokens over disjoint pages; a full layer (no window, no sink) and a sliding layer
(window 128, sinks). A control with the sinks dropped from the reference must fail on the sliding layer.

    NEURON_RT_VISIBLE_CORES=0 python3 tools/mimo_v2/check_attn_half.py

Run on a free NeuronCore."""

import sys

D, VD, H, PAGE = 256, 128, 4096, 128
LENGTHS = [5, 130, 300, 1000]
BLOCKS = 16


def rms(x, w, eps):
    import torch

    x32 = x.float()
    return (w * (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)).to(torch.bfloat16)).to(torch.bfloat16)


def reference(x, ln_w, eps, w_qkv, w_o, k_cache, v_cache, table, positions, cos, sin, scale, window, sink):
    """[B, H] fp32 partial and the new K, V [B, D]."""
    import torch

    from sglang_neuron.models.mimo_v2.rope import rotate

    rows = x.shape[0]
    qkv = rms(x, ln_w, eps).float() @ w_qkv.float()
    q, k, v = (t.view(rows, 1, D) for t in qkv.split(D, dim=-1))
    q, k = rotate(q, cos, sin)[:, 0], rotate(k, cos, sin)[:, 0]
    q, k, v = q.bfloat16().float(), k.bfloat16().float(), v[:, 0].bfloat16().float()
    out = torch.zeros(rows, H)
    for b in range(rows):
        pos = int(positions[b])
        pages = table[b][: pos // PAGE + 1]
        keys = k_cache[pages][:, 0].reshape(-1, D)[:pos].float()
        vals = v_cache[pages][:, 0].reshape(-1, D)[:pos].float()
        keys, vals = torch.cat([keys, k[b:b + 1]]), torch.cat([vals, v[b:b + 1]])
        idx = torch.arange(pos + 1)
        scores = (keys @ q[b]) * scale
        if window:
            scores = scores.masked_fill(idx <= pos - window, float("-inf"))
        if sink is not None:
            scores = torch.cat([scores, sink.float().reshape(1)])
        probs = torch.softmax(scores, dim=0)[: pos + 1]
        head = (probs @ vals).bfloat16().float()
        out[b] = head[:VD] @ w_o.float()
    return out, k, v


def main() -> int:
    from sglang_neuron.process_init import init_stack

    init_stack()
    import torch

    from sglang_neuron.backend import stack
    from sglang_neuron.backend.stack import wrap_nki
    from sglang_neuron.model_executor.graph_runner import COMPILER_ARGS
    from sglang_neuron.models.mimo_v2.kernels.attention_block import decode_masked
    from sglang_neuron.models.mimo_v2.kernels.attn_half import attn_half
    from sglang_neuron.models.mimo_v2.rope import RotaryTables
    from sglang_neuron.ops.attention.attention_decode_mask import gen_attention_decode_mask, qk_swap_layout
    from sglang_neuron.utils.hardware_config import get_logical_nc_config

    g = torch.Generator().manual_seed(0)
    rows, eps, scale = len(LENGTHS), 1e-6, 192 ** -0.5
    pages_total = rows * BLOCKS + 1
    k_cache = (torch.randn(pages_total, 1, PAGE, D, generator=g)).to(torch.bfloat16)
    v_cache = (torch.randn(pages_total, 1, PAGE, D, generator=g)).to(torch.bfloat16)
    k_cache[..., 192:] = 0
    v_cache[..., VD:] = 0
    table = torch.stack([1 + b * BLOCKS + torch.arange(BLOCKS) for b in range(rows)]).to(torch.int32)
    positions = torch.tensor(LENGTHS)
    slots = (table.gather(1, (positions // PAGE).view(-1, 1)).view(-1).long() * PAGE + positions % PAGE).to(torch.int32)
    x = (torch.randn(rows, H, generator=g)).to(torch.bfloat16)
    ln_w = (1 + 0.1 * torch.randn(H, generator=g)).to(torch.bfloat16)
    w_qkv = (0.02 * torch.randn(H, 3 * D, generator=g))
    for part in range(3):  # the padded channels' weights are zero, as the loader writes them
        w_qkv[:, part * D + (192 if part < 2 else VD):(part + 1) * D] = 0
    w_qkv = w_qkv.to(torch.bfloat16)
    w_o = (0.02 * torch.randn(VD, H, generator=g)).to(torch.bfloat16)
    device = torch.device("neuron", 0)
    failures = []
    for window, sink in ((None, None), (128, torch.tensor([1.5]))):
        cos, sin = RotaryTables(1e4 if window else 1e7, 64)(positions)
        start = None if window is None else (positions - window + 1).clamp(min=0).reshape(1, rows).float()
        mask = gen_attention_decode_mask(positions.reshape(1, rows).float(), bs=rows, q_head=1, s_active=1,
                                         s_prior=BLOCKS * PAGE, block_len=PAGE, start_pos=start)
        mask = qk_swap_layout(mask, d_head=D, fp8_kv=False, fp8_packed=False)
        want, k_new, v_new = reference(x, ln_w, eps, w_qkv, w_o, k_cache, v_cache, table, positions, cos, sin, scale,
                                       window, sink)
        inputs = [x.view(rows, 1, H), ln_w.view(1, H).float(), w_qkv, cos.t().unsqueeze(-1).contiguous(),
                  sin.t().unsqueeze(-1).contiguous(), table, k_cache.clone(), v_cache.clone(), mask]
        inputs += ([] if sink is None else [sink.reshape(1, 1).float()]) + [slots.view(rows, 1), w_o]
        dev = [t.contiguous().to(device) for t in inputs]
        has_sink = sink is not None

        def fn(X, gamma, wq, c, s, tbl, kc, vc, m, *rest):
            sk, sl, wo = (rest[0], rest[1], rest[2]) if has_sink else (None, rest[0], rest[1])
            return wrap_nki(attn_half)[get_logical_nc_config()](X, gamma, wq, c, s, tbl, kc, vc, m, sk, sl, wo,
                                                                rmsnorm_X_eps=eps, softmax_scale=scale, v_head_dim=VD)

        with stack.compile_scope(COMPILER_ARGS):
            compiled = torch.compile(fn, backend=stack.compile_backend_name(), fullgraph=True, dynamic=False)
            out, kc, vc = compiled(*dev)
        torch._dynamo.reset()
        out, kc, vc = out.cpu().float(), kc.cpu(), vc.cpu()

        # the served per-layer path's attention kernel (kernels/attention_block.decode_masked), its heads projected
        # on the host as attention.MiMoAttention._project_out does: a second reference that ran GSM8K
        def served(X, gamma, wq, c, s, tbl, kc_, vc_, m, *rest):
            sk = rest[0] if has_sink else None
            return wrap_nki(decode_masked)[get_logical_nc_config()](
                X=X, X_hidden_dim_actual=H, rmsnorm_X_eps=eps, rmsnorm_X_gamma=gamma, W_qkv=wq, cos=c, sin=s,
                active_blocks_table=tbl, K_cache=kc_, V_cache=vc_, attention_mask=m, sink=sk, softmax_scale=scale)

        dev2 = [t.contiguous().to(device) for t in inputs]
        with stack.compile_scope(COMPILER_ARGS):
            heads, _, _ = torch.compile(served, backend=stack.compile_backend_name(), fullgraph=True,
                                        dynamic=False)(*dev2)
        torch._dynamo.reset()
        heads = heads.cpu().float().reshape(rows, 1, D, 1)[:, 0, :, 0]
        served_out = heads[:, :VD].bfloat16().float() @ w_o.float()
        rel_served = ((served_out - want).norm() / want.norm()).item()
        rel_kernel_served = ((out - served_out).norm() / served_out.norm()).item()
        print(f"  window {window}: served path vs reference {rel_served:.2e}, fused kernel vs served path "
              f"{rel_kernel_served:.2e}")
        rel = ((out - want).norm() / want.norm()).item()
        flat = lambda c: c.view(-1, D)  # noqa: E731 - one KV head: a slot is a row
        k_ok = torch.allclose(flat(kc)[slots.long()].float(), k_new, atol=2e-2, rtol=2e-2)
        v_ok = torch.allclose(flat(vc)[slots.long()].float(), v_new, atol=2e-2, rtol=2e-2)
        others = torch.ones(flat(kc).shape[0], dtype=torch.bool)
        others[slots.long()] = False
        untouched = torch.equal(flat(kc)[others], flat(k_cache)[others]) and torch.equal(flat(vc)[others],
                                                                                         flat(v_cache)[others])
        ok = rel < 2e-2 and k_ok and v_ok and untouched
        line = (f"window {window} sink {sink is not None}: relative L2 {rel:.2e}, K written {k_ok}, V written {v_ok}, "
                f"other rows untouched {untouched}")
        if sink is not None:
            control, _, _ = reference(x, ln_w, eps, w_qkv, w_o, k_cache, v_cache, table, positions, cos, sin, scale,
                                      window, None)
            rel_c = ((out - control).norm() / control.norm()).item()
            ok = ok and rel_c > 5 * rel
            line += f"; vs the no-sink control {rel_c:.2e}"
        print(("PASS " if ok else "FAIL ") + line)
        if not ok:
            failures.append(window)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
