"""PyTorch reference for one rank's Qwen3.5 gated full-attention PREFILL (one sequence).

HF ``Qwen3_5MoeAttention`` over L new tokens at positions ``start .. start+L-1`` of one sequence
whose cache lives in slot ``slot`` of slot-contiguous caches ``k_cache`` / ``v_cache`` [B, S, d]
(token-major here; the kernel stores K transposed). Token t attends to cached history
``0 .. start-1`` plus new tokens ``0 .. t``; the L new K/V rows are written at ``start + t``.
Same rounding points as ``attn_decode_torch`` (and equal to running it token by token).
"""

import torch
import torch.nn.functional as F

from .attn_decode_torch import HEAD, ROT, _rmsnorm_zc, _rotate_half, rope_cos_sin


def attn_prefill_torch(x, w_q, w_gate, w_k, w_v, q_norm_w, k_norm_w, w_o, k_cache, v_cache, slot, start=0):
    """Returns (out [L, N] bf16 partial, new k_cache, new v_cache)."""
    bf = torch.bfloat16
    L, H = x.shape
    nq = w_q.shape[1] // HEAD
    xb = x.to(bf).float()
    q = (xb @ w_q.to(bf).float()).to(bf).view(L, nq, HEAD)
    gate = (xb @ w_gate.to(bf).float()).to(bf).view(L, nq * HEAD)
    k = (xb @ w_k.to(bf).float()).to(bf).view(L, 1, HEAD)
    v = (xb @ w_v.to(bf).float()).to(bf).view(L, 1, HEAD)
    q = _rmsnorm_zc(q, q_norm_w)
    k = _rmsnorm_zc(k, k_norm_w)
    cos, sin = rope_cos_sin(torch.arange(start, start + L))
    cos, sin = cos[:, None, :], sin[:, None, :]

    def rope(t):
        tr, tp = t[..., :ROT].float(), t[..., ROT:]
        tr = tr * cos + _rotate_half(tr) * sin
        return torch.cat([tr.to(t.dtype), tp], dim=-1)

    q, k = rope(q), rope(k)
    k_cache, v_cache = k_cache.clone(), v_cache.clone()
    k_cache[slot, start : start + L] = k[:, 0].to(k_cache.dtype)
    v_cache[slot, start : start + L] = v[:, 0].to(v_cache.dtype)
    T = start + L
    ks = k_cache[slot, :T].float()  # [T, d]
    vs = v_cache[slot, :T].float()
    s = torch.einsum("lhd,td->hlt", q.float(), ks) * HEAD**-0.5  # [nq, L, T]
    allowed = torch.arange(T)[None, :] <= (start + torch.arange(L))[:, None]  # [L, T]
    s = s.masked_fill(~allowed[None], float("-inf"))
    o = torch.einsum("hlt,td->lhd", F.softmax(s, dim=-1), vs).to(bf).reshape(L, nq * HEAD)
    o = (o.float() * torch.sigmoid(gate.float())).to(bf)
    res = (o.float() @ w_o.to(bf).float()).to(bf)
    return res, k_cache, v_cache


def attn_prefill_paged_torch(x, w_q, w_gate, w_k, w_v, q_norm_w, k_norm_w, w_o, k_pages, v_pages, write_pages,
                             prior_pages=None, prior_len=0):
    """The paged prefill (attn_prefill_tkg ``write_pages`` / ``prior_pages`` / ``prior_len``) over token-major page
    pools ``k_pages`` / ``v_pages`` [NPAGE, PG, d]: the chunk at positions prior_len .. prior_len + L - 1 attends to
    the request's pages before it and writes its K/V rows into ``write_pages``. Returns (out, k_pages, v_pages)."""
    PG = k_pages.shape[1]
    L = x.shape[0]
    start = int(prior_len)
    if start % PG or L % PG:
        raise ValueError(f"the chunk ({start}, {L}) is not on the {PG}-token page grid")
    write = [int(p) for p in torch.as_tensor(write_pages).reshape(-1)]
    prior = [] if start == 0 else [int(p) for p in torch.as_tensor(prior_pages).reshape(-1)[: start // PG]]
    pages = prior + write
    k_c = k_pages[pages].reshape(1, -1, k_pages.shape[2])
    v_c = v_pages[pages].reshape(1, -1, v_pages.shape[2])
    out, k_c, v_c = attn_prefill_torch(x, w_q, w_gate, w_k, w_v, q_norm_w, k_norm_w, w_o, k_c, v_c, 0, start)
    k_pages, v_pages = k_pages.clone(), v_pages.clone()
    for j, p in enumerate(write):
        rows = slice(start + j * PG, start + (j + 1) * PG)
        k_pages[p] = k_c[0, rows]
        v_pages[p] = v_c[0, rows]
    return out, k_pages, v_pages
