# SPDX-License-Identifier: Apache-2.0
"""``graph.PagedIO`` (Inkling-2-Mini's caches over SGLang's paged pool) against contiguous caches, on a small random
configuration in fp32 on the CPU: chunked prefill then decode (with a padded decode row), and a radix-cache prefix
hit (a second request on the first one's whole pages continues exactly as a cold run of its whole sequence). Every KV
layer reads one block table (-1 past the request's pages) and one slot mapping (slot 0, SGLang's reserved page 0, for
padding). Run: python -m pytest tools/inkling2/test_paged_io.py (needs torch; sglang_neuron importable)."""

import types

import pytest
import torch

from sglang_neuron.models.inkling2 import ops
from sglang_neuron.models.inkling2.config import Inkling2Config
from sglang_neuron.models.inkling2 import graph as G

BS = 32            # page of the test (the server uses 256)
NBLK = 512         # pages in the pool (page 0 reserved)


def tiny():
    # one KV head a rank, as on the server (3 query heads)
    return Inkling2Config(hidden_size=256, num_layers=18, vocab_size=1024, unpadded_vocab_size=1000, num_heads=3,
                          num_kv_heads=1, head_dim=64, d_rel=8, rel_extent=64, window=64, log_floor=32,
                          num_experts=8, top_k=2, moe_inter=128, loop_start=6, loop_end=12, passes=3, gate_rank=16,
                          full_layers=(5, 11, 17), tp_size=1)


def params(cfg, g):
    def R(*s, scale=0.05):
        return torch.randn(*s, generator=g) * scale

    H, d, hq, hkv, E, I = cfg.hidden_size, cfg.head_dim, cfg.q_heads_per_rank, cfg.kv_heads_per_rank, \
        cfg.num_experts, cfg.inter_per_rank
    layers = []
    for L in range(cfg.num_layers):
        full = cfg.is_full(L)
        p = dict(wq=R(H, hq * d), q_scale=R(hq, d) + 1, wk=R(H, hkv * d), wv=R(H, hkv * d), wo=R(hq * d, H),
                 wgate=R(H, hq * d), wsink=R(H, hq), wr=R(H, hq * cfg.d_rel), r_scale=R(hq, cfg.d_rel) + 1,
                 r_proj=R(cfg.d_rel, cfg.rel_extent), post_norm=R(H) + 1, mlp_query=R(1, H),
                 post_attn_conv=R(H, 4), post_mlp_conv=R(H, 4), router=R(H, E), router_bias=R(E),
                 gate_up=R(E, H, 2 * I), down=R(E, I, H))
        if L > 0:
            p["attn_query"] = R(3 if full else 1, H)
        if full:
            p.update(k_static=R(4, hkv * d), v_static=R(4, hkv * d), k_wproj=R(H, hkv * 4), v_wproj=R(H, hkv * 4),
                     q_norm=R(H) + 1, k_norm=R(H) + 1, v_norm=R(H) + 1)
        else:
            p.update(k_conv=R(hkv * d, 4), v_conv=R(hkv * d, 4), in_norm=R(H) + 1)
        if cfg.loop_start <= L < cfg.loop_end:
            for n in ("q", "k", "v", "mlp"):
                p[f"gate_{n}"] = (R(2 * H, cfg.gate_rank), R(cfg.gate_rank, H), R(cfg.gate_rank, H))
        layers.append(p)
    return {"layers": layers, "message_query": R(cfg.loop_layers, 4, H), "final_query": R(1, H),
            "norm": R(H) + 1, "embed": R(cfg.vocab_size, H, scale=1.0), "embed_norm": R(H) + 1}


class ContiguousIO(ops.LayerIO):
    def __init__(self, cfg):
        self.cfg, self.k, self.v, self.win = cfg, {}, {}, {}

    def kv(self, c, k, v, window=None):
        if c in self.k:
            k, v = torch.cat([self.k[c], k], 2), torch.cat([self.v[c], v], 2)
        self.k[c], self.v[c] = k, v
        return k, v, torch.arange(k.shape[2]).view(1, -1).expand(k.shape[0], -1)

    def window(self, c, slot):
        if (c, slot) in self.win:
            return self.win[(c, slot)]
        w = self.cfg.kv_heads_per_rank * self.cfg.head_dim if slot < 2 else self.cfg.hidden_size
        return torch.zeros(self.positions.shape[0], ops.CONV_WINDOW, w)

    def put_window(self, c, slot, seq):
        self.win[(c, slot)] = ops.next_window(seq, seq.shape[1] - ops.CONV_WINDOW)



def stub(cfg):
    n = G.kv_layers(cfg, BS)
    caches = [tuple(torch.zeros(2, NBLK, 1, BS, cfg.head_dim).unbind(0)) for _ in range(n)]
    g = types.SimpleNamespace(config=cfg, state_layout=G.StateLayout(cfg, BS), moe_kernel=False,
                              tp_group=types.SimpleNamespace(world_size=1, device_group=None))
    g.kv_caches, g.state_caches = caches[:cfg.cache_layers], caches[cfg.cache_layers:]
    return g


class Request:
    def __init__(self, free, pages=None):
        self.free, self.pages = free, list(pages or [])

    def grow(self, n_tokens):
        while len(self.pages) * BS < n_tokens:
            self.pages.append(self.free.pop())


def metadata(rows, positions, S, n=None, width=64):
    """attn_metadata of one step: rows (a Request, or None for a padding row) at ``positions [B]``; tokens past ``n``
    of a prefill chunk are padding (slot 0)."""
    n = S if n is None else n
    table, slots = [], []
    for b, r in enumerate(rows):
        q = int(positions[b])
        if r is None:
            table.append([0] + [-1] * (width - 1))
            slots += [0] * S
            continue
        table.append(r.pages + [-1] * (width - len(r.pages)))
        slots += [r.pages[(q + s) // BS] * BS + (q + s) % BS if s < n else 0 for s in range(S)]
    return {"block_table_tensor": torch.tensor(table), "slot_mapping": torch.tensor(slots)}


def logits(prm, cfg, ids, io, dense):
    h = ops.rms_norm(prm["embed"][ids], prm["embed_norm"], cfg.eps, dtype=torch.float32)
    return ops.text_model(prm, cfg, h, io, dense)


def step_prefill(g, prm, cfg, ids, r, t0, S, n):
    r.grow(t0 + n)
    pos = torch.tensor([t0])
    io = G.PagedIO(g, metadata([r], pos, S, n), pos.view(1, 1) + torch.arange(S), torch.tensor([n]),
                   torch.tensor([[t0]]) > 0, 1, S)
    tok = ids[:, t0:t0 + S]
    if tok.shape[1] < S:
        tok = torch.cat([tok, tok.new_zeros(1, S - tok.shape[1])], 1)
    out = logits(prm, cfg, tok, io, True)
    io.publish()
    return out[:, :n]


def step_decode(g, prm, cfg, ids, r, t, pad_row=False):
    r.grow(t + 1)
    rows = [r, None] if pad_row else [r]
    pos = torch.tensor([t] + ([0] if pad_row else []))
    B = len(rows)
    io = G.PagedIO(g, metadata(rows, pos, 1), pos.view(B, 1), 1, None, B, 1)
    out = logits(prm, cfg, ids[:, t:t + 1].expand(B, 1), io, False)[:1]
    io.publish()
    return out


def contiguous(prm, cfg, ids):
    ref = ContiguousIO(cfg)
    ref.positions = torch.arange(ids.shape[1]).view(1, -1)
    return logits(prm, cfg, ids, ref, True)


@pytest.fixture(autouse=True)
def fp32(monkeypatch):
    monkeypatch.setattr(G, "DTYPE", torch.float32)


@pytest.mark.parametrize("pad_row", [False, True])
def test_chunked_prefill_and_decode_match_contiguous(pad_row):
    g_ = torch.Generator().manual_seed(1)
    cfg = tiny()
    prm = params(cfg, g_)
    g = stub(cfg)
    T, chunk, n_dec = 160, 64, 24
    ids = torch.randint(0, 1000, (1, T + n_dec), generator=g_)
    r = Request(list(torch.randperm(NBLK - 1, generator=g_).add(1).tolist()))
    with torch.no_grad():
        out = [step_prefill(g, prm, cfg, ids, r, t0, chunk, min(chunk, T - t0)) for t0 in range(0, T, chunk)]
        out += [step_decode(g, prm, cfg, ids, r, t, pad_row) for t in range(T, T + n_dec)]
        want = contiguous(prm, cfg, ids)
    got = torch.cat(out, 1)
    d = (got - want).abs().max().item()
    assert torch.isfinite(got).all() and d < 1e-3 * max(1.0, want.abs().max().item()), d


def test_radix_prefix_hit_matches_contiguous():
    """Request B's prompt shares request A's first 3 pages (96 tokens: SGLang hands B those pages, as the radix
    cache does); B prefills from 96 and decodes; its logits match a cold contiguous run of B's whole sequence."""
    g_ = torch.Generator().manual_seed(2)
    cfg = tiny()
    prm = params(cfg, g_)
    g = stub(cfg)
    P, T, n_dec = 96, 192, 8
    ids_a = torch.randint(0, 1000, (1, 160), generator=g_)
    ids_b = torch.cat([ids_a[:, :P], torch.randint(0, 1000, (1, T + n_dec - P), generator=g_)], 1)
    free = list(torch.randperm(NBLK - 1, generator=g_).add(1).tolist())
    a = Request(free)
    with torch.no_grad():
        step_prefill(g, prm, cfg, ids_a, a, 0, 64, 64)
        step_prefill(g, prm, cfg, ids_a, a, 64, 64, 64)       # page 2 (tokens 64..95) is filled inside this chunk
        step_prefill(g, prm, cfg, ids_a, a, 128, 64, 32)      # A goes on past the shared prefix
        b = Request(free, a.pages[:P // BS])
        out = [step_prefill(g, prm, cfg, ids_b, b, P, 64, 64)]
        out.append(step_prefill(g, prm, cfg, ids_b, b, P + 64, 32, 32))
        out += [step_decode(g, prm, cfg, ids_b, b, t) for t in range(T, T + n_dec)]
        want = contiguous(prm, cfg, ids_b)[:, P:]
    got = torch.cat(out, 1)
    d = (got - want).abs().max().item()
    assert torch.isfinite(got).all() and d < 1e-3 * max(1.0, want.abs().max().item()), d
