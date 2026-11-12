"""PyTorch references for the Qwen3.5 GDN prefill kernel (``gdn_prefill``).

``gdn_prefill_torch`` IS the decode recurrence (``gdn_decode_torch``) applied token by token, so its
final states are by construction what decode would hold after the prompt: the P -> D contract.
``chunk_gated_delta_fp32`` is the chunked (WY) algorithm the kernel implements, in fp32, used to check
the algebra against the token loop.
"""

import torch

from .gdn_decode_torch import HEAD_DIM, gdn_decode_torch


def gdn_prefill_torch(
    x,  # [B*L, H] bf16 sequence-major
    w_qkvz, w_a, w_b, conv_w,
    conv_state,  # [B, Cc, K-1] or None (zeros)
    a_log, dt_bias, norm_w,
    rec_state,  # [B, vh, 128, 128] fp32 or None (zeros)
    w_out,
    seq_len,
    eps: float = 1e-6,
    valid_len=None,
):
    """Returns (out [B*L, N_out] bf16, new_conv [B, Cc, K-1], new_rec [B, vh, 128, 128] fp32).

    ``valid_len`` (int): rows t >= valid_len of every sequence are padding: the states skip them and their outputs
    are zero (the kernel's are unspecified)."""
    T, H = x.shape
    L = seq_len
    B = T // L
    vh = a_log.shape[0]
    Vd = vh * HEAD_DIM
    Kd = (w_qkvz.shape[1] - 2 * Vd) // 2
    Cc = 2 * Kd + Vd
    K = conv_w.shape[1]
    conv = torch.zeros(B, Cc, K - 1, dtype=torch.bfloat16) if conv_state is None else conv_state
    rec = torch.zeros(B, vh, HEAD_DIM, HEAD_DIM) if rec_state is None else rec_state.float()
    xs = x.view(B, L, H)
    outs = []
    n = L if valid_len is None else int(valid_len)
    for t in range(L):
        if t >= n:
            outs.append(torch.zeros_like(outs[-1]))
            continue
        o, conv, rec = gdn_decode_torch(xs[:, t], w_qkvz, w_a, w_b, conv_w, conv, a_log, dt_bias, norm_w, rec, w_out, eps)
        outs.append(o)
    out = torch.stack(outs, dim=1).reshape(T, -1)
    return out, conv, rec


def chunk_gated_delta_fp32(q, k, v, g, beta, S0, C=128):
    """Chunked gated delta rule (the kernel's algebra) in fp32 for one head.

    q, k [L, K] (l2-normed, q scaled), v [L, V], g, beta [L], S0 [K, V]. Returns (o [L, V], S [K, V]).
    """
    L = q.shape[0]
    S = S0.clone().float()
    outs = []
    tril = torch.tril(torch.ones(C, C), -1)
    triu = torch.triu(torch.ones(C, C))
    for c0 in range(0, L, C):
        sl = slice(c0, c0 + C)
        K_, Q_, V_, b_ = k[sl].float(), q[sl].float(), v[sl].float(), beta[sl].float()
        gc = torch.cumsum(g[sl].float(), 0)
        D = torch.exp(torch.clamp(gc[:, None] - gc[None, :], max=0.0))
        A = b_[:, None] * (K_ @ K_.T) * D * tril
        T = torch.linalg.inv(torch.eye(C) + A)
        U = T @ (b_[:, None] * V_)
        W = T @ (b_[:, None] * torch.exp(gc)[:, None] * K_)
        dlt = U - W @ S
        o = torch.exp(gc)[:, None] * (Q_ @ S) + ((Q_ @ K_.T) * D * triu) @ dlt
        S = torch.exp(gc[-1]) * S + (torch.exp(gc[-1] - gc)[:, None] * K_).T @ dlt
        outs.append(o)
    return torch.cat(outs), S
