"""Hy3's decode RMSNorm of a few rows in one NKI kernel: ``out = x * rsqrt(mean(x^2) + eps) * w``.

The torch RMSNorm lowers to a chain of upcasts, broadcasts and DMA reshapes, about 23 us a layer on a decode step's
32 rows, most of it on the critical path. Here the rows sit one per partition: one squared-sum reduction, one rsqrt,
one fused multiply by the row's scale and the weight. Each of the LNC programs normalizes every row (a row's sum needs
all of H) and stores its own half of the columns.

  x    [T, H] bf16, T <= 128
  w    [1, H] bf16
  eps  compile-time constant
"""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128


@nki.jit
def rmsnorm_rows(x, w, eps=1e-5):
    T, H = x.shape
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    HH = H // n_prg
    h0 = pid * HH

    out = nl.ndarray((T, H), dtype=x.dtype, buffer=nl.shared_hbm)
    xf = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=xf[0:T, 0:H], src=x.ap(pattern=[[H, T], [1, H]], offset=0))
    gam = nl.ndarray((P, HH), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=gam[0:T, 0:HH], src=w.ap(pattern=[[0, T], [1, HH]], offset=h0))
    sq = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)
    ss = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq[0:T, 0:H], op=nl.square, data=xf[0:T, 0:H], reduce_op=nl.add,
                    reduce_res=ss[0:T, 0:1], reduce_cmd=nisa.reduce_cmd.reset_reduce)
    rr = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rr[0:T, 0:1], op=nl.rsqrt, data=ss[0:T, 0:1], scale=1.0 / H, bias=eps)
    nrm = nl.ndarray((P, HH), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=nrm[0:T, 0:HH], data=xf[0:T, h0:h0 + HH], op0=nl.multiply,
                              operand0=rr[0:T, 0:1], op1=nl.multiply, operand1=gam[0:T, 0:HH])
    nisa.dma_copy(dst=out.ap(pattern=[[H, T], [1, HH]], offset=h0), src=nrm[0:T, 0:HH])
    return out
