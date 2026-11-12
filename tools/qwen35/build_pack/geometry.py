# SPDX-License-Identifier: Apache-2.0
"""Per-rank shard maps of the giga kernels (one trn3 node, 128 ranks, LNC=1).

A ``RankLayout`` fixes, for every rank, which heads / experts / vocab rows it
holds. Ranks of a group are consecutive (rank r is TP position ``r % tp`` of
group ``r // tp``), the convention of every kernel test and of
``megakernel/layout.py``.

* decode (``DECODE``): GDN TP64 x DP2 (v head ``r % 64``, its k head
  ``(r % 64) // 4`` recomputed), attention TP8 x DP16 (q heads ``4j..4j+3``,
  KV head ``j // 4``), MoE EP128 (experts ``4r..4r+3``), shared expert sliced
  ``Is / 128`` columns per rank, lm_head vocab-sharded ``V / 128`` rows.
* prefill (``PREFILL``): GDN TP16 x 8 prompt groups (k head ``t``, v heads
  ``4t..4t+3``), attention TP8 subgroups, MoE EP64 x 2 replicas (experts
  ``8p..8p+7`` of EP position ``p = r % 64``), shared expert replicated
  (the moe_prefill block runs the whole shared expert on the home rank).
"""

from __future__ import annotations

import dataclasses

from tools.qwen35.build_pack.config import Qwen35Config


@dataclasses.dataclass(frozen=True)
class RankLayout:
    name: str
    world: int = 128
    attn_tp: int = 8
    gdn_tp: int = 64
    ep: int = 128
    shared: str = "slice"  # "slice": Is/ep columns per rank; "replicated": the whole shared expert
    fold: bool = True  # router fold operands (decode high-C kernel, QWEN35_ROUTER_FOLD=1)

    def __post_init__(self):
        for what, n in (("attention TP", self.attn_tp), ("GDN TP", self.gdn_tp), ("EP", self.ep)):
            if self.world % n:
                raise ValueError(f"{self.name}: {what} {n} does not divide world {self.world}")
        if self.shared not in ("slice", "replicated"):
            raise ValueError(f"shared={self.shared!r}")

    # -- attention -------------------------------------------------------------
    def attn_pos(self, rank: int) -> int:
        return rank % self.attn_tp

    def attn_group(self, rank: int) -> int:
        return rank // self.attn_tp

    def q_heads(self, cfg: Qwen35Config, pos: int) -> range:
        per = cfg.n_q_heads // self.attn_tp
        return range(pos * per, (pos + 1) * per)

    def kv_head(self, cfg: Qwen35Config, pos: int) -> int:
        if self.attn_tp < cfg.n_kv_heads:
            raise ValueError("more than one KV head per rank is not a kernel layout")
        return pos * cfg.n_kv_heads // self.attn_tp

    # -- GDN -------------------------------------------------------------------
    def gdn_pos(self, rank: int) -> int:
        return rank % self.gdn_tp

    def gdn_group(self, rank: int) -> int:
        return rank // self.gdn_tp

    def gdn_vh(self, cfg: Qwen35Config) -> int:
        if cfg.gdn_v_heads % self.gdn_tp:
            raise ValueError(f"GDN TP{self.gdn_tp} does not divide {cfg.gdn_v_heads} v heads")
        return cfg.gdn_v_heads // self.gdn_tp

    def gdn_kh(self, cfg: Qwen35Config) -> int:
        return max(1, self.gdn_vh(cfg) // cfg.gqa)

    def gdn_v_heads(self, cfg: Qwen35Config, pos: int) -> range:
        vh = self.gdn_vh(cfg)
        return range(pos * vh, (pos + 1) * vh)

    def gdn_k_heads(self, cfg: Qwen35Config, pos: int) -> range:
        """k heads whose q / k channels this shard holds (recomputed when tp > k heads)."""
        vs = self.gdn_v_heads(cfg, pos)
        k0 = vs.start // cfg.gqa
        return range(k0, k0 + self.gdn_kh(cfg))

    def gdn_channels(self, cfg: Qwen35Config, pos: int) -> list:
        """Rows of in_proj_qkv / conv1d of this shard, in the kernel's q | k | v order."""
        ks, vs = self.gdn_k_heads(cfg, pos), self.gdn_v_heads(cfg, pos)
        dk, dv, K = cfg.gdn_k_dim, cfg.gdn_v_dim, cfg.key_dim
        q = [h * dk + i for h in ks for i in range(dk)]
        k = [K + h * dk + i for h in ks for i in range(dk)]
        v = [2 * K + h * dv + i for h in vs for i in range(dv)]
        return q + k + v

    def gdn_conv_dim(self, cfg: Qwen35Config) -> int:
        return 2 * self.gdn_kh(cfg) * cfg.gdn_k_dim + self.gdn_vh(cfg) * cfg.gdn_v_dim

    # -- MoE -------------------------------------------------------------------
    def ep_pos(self, rank: int) -> int:
        return rank % self.ep

    def experts_per_rank(self, cfg: Qwen35Config) -> int:
        if cfg.n_experts % self.ep:
            raise ValueError(f"EP{self.ep} does not divide {cfg.n_experts} experts")
        return cfg.n_experts // self.ep

    def experts(self, cfg: Qwen35Config, rank: int) -> range:
        e = self.experts_per_rank(cfg)
        p = self.ep_pos(rank)
        return range(p * e, (p + 1) * e)

    def shared_cols(self, cfg: Qwen35Config, rank: int) -> range:
        if self.shared == "replicated":
            return range(cfg.shared_inter)
        per = cfg.shared_inter // self.ep
        if per * self.ep != cfg.shared_inter:
            raise ValueError(f"shared expert {cfg.shared_inter} does not split over EP{self.ep}")
        p = self.ep_pos(rank)
        return range(p * per, (p + 1) * per)

    # -- lm_head ---------------------------------------------------------------
    def vocab_rows(self, cfg: Qwen35Config, rank: int) -> range:
        per = cfg.vocab // self.world
        if per * self.world != cfg.vocab:
            raise ValueError(f"vocab {cfg.vocab} does not split over {self.world} ranks")
        return range(rank * per, (rank + 1) * per)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


DECODE = RankLayout("decode", attn_tp=8, gdn_tp=64, ep=128, shared="slice", fold=True)
PREFILL = RankLayout("prefill", attn_tp=8, gdn_tp=16, ep=64, shared="replicated", fold=False)


def layout_by_name(name: str, **overrides) -> RankLayout:
    base = {"decode": DECODE, "prefill": PREFILL}[name]
    return dataclasses.replace(base, **overrides) if overrides else base


def perm_rows(w, part: int = 128):
    """[K, C] -> [128, K/128, C] with out[p, k1, c] = w[k1*128 + p, c] (the kernels'
    pre-permuted weight rows)."""
    import numpy as np

    w = np.asarray(w)
    K = w.shape[0]
    if K % part:
        raise ValueError(f"{K} rows are not a multiple of {part}")
    return np.ascontiguousarray(w.reshape(K // part, part, w.shape[1]).transpose(1, 0, 2))


def unperm_rows(wp):
    import numpy as np

    wp = np.asarray(wp)
    return np.ascontiguousarray(wp.transpose(1, 0, 2).reshape(wp.shape[0] * wp.shape[1], wp.shape[2]))
