#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hardware check of DCP at the decode attention op (nkilib's attention_block_tkg with CP).

Runs seeded decode cases through ops.attention.attention_decode on NeuronCores, so the kernel merges the partial
softmaxes over real collectives: a reduce-scatter at D = 2, an all-to-all at D = 4. Each rank keeps a page-interleaved
share of every sequence's KV (decode_context_parallel.DcpLayout), masks by its own share with the decoded token on its
owner only, and passes the group's sinks (real on rank 0 only). Per case and rank it checks that:

* the DCP decode of this rank's heads matches the single-rank kernel decode and the host torch path, for full and
  sliding-window attention, over the plain and the packed FP8 K cache;
* the decoded token's K and V land in its owner's slot;
* sinks counted on every rank, or a missing owner gate, push the result well past the decode's own error.

The world splits into consecutive DCP groups of --dcp ranks, as parallel.state builds them, so a collective over the
wrong group shows as a mismatch or a hang.

usage: torchrun --nproc-per-node <world> tools/check_dcp_decode_attention.py --dcp <D> --output <result.json>
Run on free NeuronCores only (NEURON_RT_VISIBLE_CORES is set per rank from --first-core).
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace


def _rank_environment(rank: int, world: int, first_core: int, cache_root: str) -> None:
    """This rank's Neuron environment: one core, the world's collectives bootstrap, and its own compile caches."""
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(first_core + rank)
    os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", "127.0.0.1:62183")
    os.environ["SGLANG_NEURON_CPU_MODE"] = "0"
    for name, sub in (("TORCH_NEURONX_NEFF_CACHE_DIR", "neff"), ("TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR", "neff_local"),
                      ("NKI_COMPILE_CACHE_URL", "nki")):
        os.environ[name] = os.path.join(cache_root, sub, f"rank{rank}")


class _Cases:
    """The decode cases at the kernel's shapes: a hidden size the LNC shards in 128-row tiles, the serving page, and
    rows that span several virtual pages at that page, so every rank of a group holds part of each long one. Every
    rank's case is the same (seeded); ``_decode`` runs attention_decode as ops.attention exposes it, so ``main`` can
    put the device op in its place."""

    D_HEAD, Q_PER_RANK = 64, 2
    LENGTHS = [5, 300, 700, 1000]  # each row's prior tokens; the decoded token is at that position
    WINDOW = 50

    def __init__(self, hidden: int, block: int):
        self.HIDDEN, self.BLOCK = hidden, block

    @staticmethod
    def _quantized(t):
        import torch

        return t.clamp(-200, 200).to(torch.float8_e4m3fn)

    def _case(self, size):
        """One decode step's weights, inputs and every row's prior K/V."""
        import torch

        g = torch.Generator().manual_seed(17 + size)
        heads, rows, hidden, d = size * self.Q_PER_RANK, len(self.LENGTHS), self.HIDDEN, self.D_HEAD
        return SimpleNamespace(
            x=torch.randn(rows, 1, hidden, generator=g),
            wq=torch.randn(hidden, heads * d, generator=g) / hidden ** 0.5,
            wkv=torch.randn(hidden, 2 * d, generator=g) / hidden ** 0.5,
            w_out=torch.randn(heads * d, hidden, generator=g) / d,
            sinks=torch.randn(heads, generator=g),
            cos=torch.randn(d // 2, rows, 1, generator=g), sin=torch.randn(d // 2, rows, 1, generator=g),
            k=[self._quantized(torch.randn(n, d, generator=g) * 2) for n in self.LENGTHS],
            v=[self._quantized(torch.randn(n, d, generator=g) * 2) for n in self.LENGTHS],
            pages=torch.randperm(64, generator=g) + 1,  # page 0 is SGLang's reserved page
        )

    def _caches(self, case, layout, width):
        """K/V caches (as the pool holds them, [blocks, 1, BLOCK, D_HEAD]) and the block table of every row: a virtual
        page of size x BLOCK tokens per table column, this rank's block of it at the page's index (layout None: one
        rank, its blocks the pages)."""
        import torch

        size, rank = (1, 0) if layout is None else (layout.size, layout.rank)
        block_len = self.BLOCK
        page = size * block_len
        k_cache = torch.zeros(65, 1, block_len, self.D_HEAD, dtype=torch.float8_e4m3fn)
        v_cache = torch.zeros_like(k_cache)
        table = torch.full((len(self.LENGTHS), width), -1, dtype=torch.int32)
        used = 0
        for b, n in enumerate(self.LENGTHS):
            pages = case.pages[used:used + n // page + 1].tolist()  # the decoded token's page too
            used += len(pages)
            table[b, :len(pages)] = torch.tensor(pages, dtype=torch.int32)
            for p in range(n):
                if (p % page) // block_len == rank:
                    block, row = pages[p // page], p % block_len
                    k_cache[block, 0, row], v_cache[block, 0, row] = case.k[b][p], case.v[b][p]
        return k_cache, v_cache, table

    def _slots(self, layout, table):
        """Each row's slot of its decoded token: its block row on the owner, the reserved slot elsewhere."""
        import torch

        from sglang_neuron.distributed.decode_context_parallel import DcpLayout

        layout = layout or DcpLayout(1, 0, self.BLOCK)
        virtual = torch.tensor([int(table[b, n // layout.page]) * layout.page + n % layout.page
                                for b, n in enumerate(self.LENGTHS)])
        return layout.local_slots(virtual), virtual

    def _decode(self, case, heads, layout, group=None, window=None, fp8_packed=False, sinks=None, gate=True):
        """This rank's heads ``heads`` decoded by attention_decode: over the whole cache (layout None), or over this
        rank's share of it within ``group``."""
        import torch

        from sglang_neuron.distributed.decode_context_parallel import group_sinks
        from sglang_neuron.ops.attention import attention_decode as op_module
        from sglang_neuron.ops.attention.attention_decode import swizzle_packed_k
        from sglang_neuron.ops.attention.attention_decode_mask import gen_attention_decode_mask

        hidden, d, block_len = self.HIDDEN, self.D_HEAD, self.BLOCK
        size = 1 if layout is None else layout.size
        width = 8 if size == 1 else 16 // size  # the table's prior: 8 pages on one rank, 16 // D on a DCP rank
        k_cache, v_cache, table = self._caches(case, layout, width)
        if fp8_packed:
            k_cache = swizzle_packed_k(k_cache)
        slots, _ = self._slots(layout, table)
        lengths = torch.tensor(self.LENGTHS)
        first = (lengths - window + 1).clamp(min=0) if window else None
        if layout is None:
            end, start, active_gate, q_head = lengths, first, None, len(heads)
            sink = case.sinks[heads]
        else:
            end = layout.local_count(lengths)
            start = None if first is None else layout.local_count(first)
            owner = layout.owner(lengths) == layout.rank
            active_gate, q_head = (owner if gate else torch.ones_like(owner)), len(heads) * size
            sink = group_sinks(case.sinks, len(heads), layout.rank, size, layout.rank) if sinks is None else sinks
        mask = gen_attention_decode_mask(end.reshape(1, -1).float(), bs=len(self.LENGTHS), q_head=q_head, s_active=1,
                                         s_prior=width * block_len, block_len=block_len,
                                         start_pos=None if start is None else start.reshape(1, -1).float(),
                                         active_gate=active_gate)
        wq = case.wq.view(hidden, -1, d)[:, heads].reshape(hidden, -1)
        w_qkv = torch.cat([wq, case.wkv], dim=1)
        rows = case.w_out.view(-1, d, hidden)[heads].reshape(-1, hidden)
        out = op_module.attention_decode(
            case.x, X_hidden_dim_actual=hidden, rmsnorm_X_gamma=torch.ones(1, hidden), rmsnorm_X_eps=1e-6,
            W_qkv=w_qkv, bias_qkv=torch.zeros(1, w_qkv.shape[1]), cos=case.cos, sin=case.sin,
            active_blocks_table=table, K_cache=k_cache, V_cache=v_cache, attention_mask=mask, sink=sink.reshape(-1, 1),
            softmax_scale=d ** -0.5, kv_cache_update_idx=slots.view(-1, 1).to(torch.uint32), k_scale=torch.ones(1, 1),
            v_scale=torch.ones(1, 1), fp8_packed=fp8_packed, W_out=rows, bias_out=torch.zeros(1, hidden),
            dcp_group=group)
        return out, (k_cache, v_cache), slots


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dcp", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-core", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--cache-root", default=os.path.expanduser("~/.cache/sglang_neuron/dcp_check"))
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if world % args.dcp:
        parser.error(f"--dcp {args.dcp} must divide the world size {world}")
    _rank_environment(rank, world, args.first_core, args.cache_root)

    from sglang_neuron.process_init import init_stack

    init_stack()
    import torch
    import torch.distributed as dist

    from sglang_neuron.backend import stack
    from sglang_neuron.distributed.decode_context_parallel import NULL_SLOT, DcpLayout
    from sglang_neuron.model_executor.graph_runner import COMPILER_ARGS
    from sglang_neuron.ops.attention import attention_decode as op_module
    from sglang_neuron.ops.attention.attention_decode import unswizzle_packed_k
    from sglang_neuron.parallel.neuron_init import init_neuron_ranks
    from sglang_neuron.parallel.state import _make_group

    init_neuron_ranks(rank, world, rank, world)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    cells = [list(range(i, i + args.dcp)) for i in range(0, world, args.dcp)]
    group = _make_group(cells, rank, f"dcp_check_{args.dcp}")
    dcp_rank = rank % args.dcp
    assert list(group.ranks) == cells[rank // args.dcp] and group.rank_in_group == dcp_rank
    device = torch.device("neuron", rank)
    cases = _Cases(args.hidden, args.block)
    real_op = op_module.attention_decode
    torch_op = {"on": False}

    def device_op(X, **kw):
        """attention_decode on this rank's NeuronCore with the host op's contract (caches updated in place, float32
        output on the host); the op's torch path while ``torch_op`` is on."""
        if torch_op["on"]:
            return real_op(X, **kw)
        bf16 = {"X", "W_qkv", "bias_qkv", "W_out", "bias_out", "cos", "sin", "k_scale", "v_scale"}
        kw = {**kw, "X": X}
        names = sorted(k for k, v in kw.items() if isinstance(v, torch.Tensor))
        static = {k: v for k, v in kw.items() if not isinstance(v, torch.Tensor)}
        inputs = [(kw[n].to(torch.bfloat16) if n in bf16 else kw[n].to(torch.int64) if n == "kv_cache_update_idx"
                   else kw[n]).contiguous().to(device) for n in names]

        def fn(*tensors):
            named = dict(zip(names, tensors))
            named["kv_cache_update_idx"] = named["kv_cache_update_idx"].to(torch.uint32)
            return real_op(**named, **static)

        compiled = torch.compile(fn, backend=stack.compile_backend_name(), fullgraph=True, dynamic=False)
        failed = torch.zeros(1)
        try:
            from torch_neuronx.compiler import compile_only

            with compile_only(), stack.compile_scope(COMPILER_ARGS):
                torch.compile(fn, backend=lambda gm, ex: torch._dynamo.lookup_backend(
                    stack.compile_backend_name())(gm, ex), fullgraph=True, dynamic=False)(*inputs)
        except Exception:  # noqa: BLE001 - reported below, after every rank has compiled or failed
            traceback.print_exc()
            failed += 1
        dist.all_reduce(failed)  # no rank enters the kernel's collectives unless every rank has its NEFF
        if int(failed):
            raise RuntimeError("a rank failed to compile this graph")
        with stack.compile_scope(COMPILER_ARGS):
            out = compiled(*inputs)
        for name in ("K_cache", "V_cache"):
            kw[name].copy_(inputs[names.index(name)].cpu())
        torch._dynamo.reset()
        return out.cpu().float()

    op_module.attention_decode = device_op
    case = cases._case(args.dcp)
    # sinks that hold a real share of each row's attention, so a sink counted on every rank of the group shows
    case.sinks = case.sinks + 4.0
    layout = DcpLayout(args.dcp, dcp_rank, args.block)
    heads = list(range(dcp_rank * cases.Q_PER_RANK, (dcp_rank + 1) * cases.Q_PER_RANK))
    rows, failures = [], []
    tol = dict(atol=3e-2, rtol=3e-2)  # bf16 operands
    for window in (None, cases.WINDOW):
        for packed in (False, True):
            where = f"world {world} dcp {args.dcp} rank {rank} window {window} packed {packed}"
            row = {"rank": rank, "window": window, "packed": packed}
            try:
                ref, (ref_k, ref_v), ref_slots = cases._decode(case, heads, None, window=window, fp8_packed=packed)
                out, (k_cache, v_cache), slots = cases._decode(case, heads, layout, group, window=window,
                                                               fp8_packed=packed)
                torch_op["on"] = True
                try:
                    host, _, _ = cases._decode(case, heads, layout, group, window=window, fp8_packed=packed)
                finally:
                    torch_op["on"] = False  # a failed host decode leaves the later cases on the kernel
                row["dcp_vs_ref"] = float((out - ref).abs().max())
                row["dcp_vs_torch"] = float((out - host).abs().max())
                torch.testing.assert_close(out, ref, **tol, msg=lambda m: f"DCP vs one rank, {where}: {m}")
                torch.testing.assert_close(out, host, **tol, msg=lambda m: f"DCP kernel vs torch, {where}: {m}")
                if packed:
                    k_cache, ref_k = unswizzle_packed_k(k_cache), unswizzle_packed_k(ref_k)
                for b, (s, r) in enumerate(zip(slots.tolist(), ref_slots.tolist())):
                    if s != NULL_SLOT:
                        here, there = (s // args.block, 0, s % args.block), (r // args.block, 0, r % args.block)
                        torch.testing.assert_close(k_cache[here].float(), ref_k[there].float(), **tol)
                        torch.testing.assert_close(v_cache[here].float(), ref_v[there].float(), **tol)
                if window is None and not packed:
                    bad_sink, _, _ = cases._decode(case, heads, layout, group,
                                                   sinks=case.sinks[: args.dcp * cases.Q_PER_RANK].float())
                    ungated, _, _ = cases._decode(case, heads, layout, group, gate=False)
                    row["bad_sink_vs_ref"] = float((bad_sink - ref).abs().max())
                    row["ungated_vs_ref"] = float((ungated - ref).abs().max())
                    # each well above the decode's own bf16 error (the decoded token is one key of hundreds, so
                    # counting it twice moves a row less than a sink counted twice does)
                    noise = max(row["dcp_vs_ref"], 1e-3)
                    assert row["bad_sink_vs_ref"] > 10 * noise, (where, row)
                    assert row["ungated_vs_ref"] > 4 * noise, (where, row)
                row["pass"] = True
            except Exception:  # noqa: BLE001 - every case is reported
                row["pass"], row["error"] = False, traceback.format_exc(limit=4)
                failures.append(where)
            print(json.dumps(row), flush=True)
            rows.append(row)
    gathered = [None] * world
    dist.all_gather_object(gathered, rows)
    if rank == 0:
        flat = [r for g in gathered for r in g]
        summary = {"world": world, "dcp": args.dcp, "collective": op_module.dcp_collective_mode(args.dcp).name.lower(),
                   "cases": len(flat), "passed": sum(r["pass"] for r in flat),
                   "max_dcp_vs_ref": max(r.get("dcp_vs_ref", float("nan")) for r in flat)}
        summary["pass"] = summary["passed"] == summary["cases"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"summary": summary, "ranks": gathered}, indent=1) + "\n")
        print(json.dumps(summary), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
