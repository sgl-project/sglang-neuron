# SPDX-License-Identifier: Apache-2.0
"""Compile small-shape cases of the Qwen3.5 kernels (sglang_neuron/models/qwen35/kernels) to a NEFF with the
installed nki and neuronx-cc, and report the trace time, the compile time and the NEFF size of each (JSON lines on
stdout).

No device is used: the kernel is traced to BIR (nki.compiler.driver.compile_to_bir, the path nki-library's test
harness takes) and compiled to a NEFF for one rank. Run it with the serving interpreter from a checkout:

    PYTHONPATH=$PWD python tools/kernels/compile_smoke.py --out <dir> [--case <name> ...] [--target trn3]
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path


def _bf16():
    import ml_dtypes
    import numpy as np

    return np.dtype(ml_dtypes.bfloat16)


def _zeros(shape, dtype):
    import numpy as np

    return np.zeros(shape, dtype=dtype)


def case_qwen35_gdn_decode():
    """Qwen3.5's GDN decode layer body: 8 rows, H = 4096, one key head and four value heads."""
    import numpy as np

    bf16, f32 = _bf16(), np.float32
    b, h, kh, vh, head, k_win = 8, 4096, 1, 4, 128, 4
    kd, vd = kh * head, vh * head
    cc = 2 * kd + vd
    mod = importlib.import_module("sglang_neuron.models.qwen35.kernels.giga.gdn_decode")
    return mod.gdn_decode_tkg, {
        "x": _zeros((b, h), bf16),
        "w_proj_p": _zeros((128, h // 128, 2 * kd + 2 * vd + 2 * vh), bf16),
        "conv_w": _zeros((cc, k_win), bf16),
        "conv_state": _zeros((b, cc, k_win - 1), bf16),
        "a_log": _zeros((vh,), f32),
        "dt_bias": _zeros((vh,), f32),
        "norm_w": _zeros((head,), f32),
        "rec_state": _zeros((b, head, vh, head), f32),
        "w_out": _zeros((vd, h), bf16),
    }


CASES = {
    "qwen35_gdn_decode": case_qwen35_gdn_decode,
}


def compile_case(name: str, out_dir: Path, target: str) -> dict:
    from nki.compiler.driver import compile_to_bir
    from nki.compiler.frontend import TracerFrontend
    from nki.compiler.ncc_driver import CompileOptions, compile_bir_to_neff

    record = {"case": name, "target": target}
    work = out_dir / name
    work.mkdir(parents=True, exist_ok=True)
    try:
        kernel, inputs = CASES[name]()
        record["kernel"] = f"{getattr(kernel, '__module__', '?')}.{getattr(kernel, '__name__', kernel)}"
        opts = CompileOptions(target=target, lnc=1, output_path=str(work / "file.neff"),
                              artifacts_dir=str(work / "artifacts"))
        if hasattr(opts, "disable_backend_optimizations"):  # nki-library's harness default
            opts = opts.disable_backend_optimizations()
        t0 = time.monotonic()
        bir = compile_to_bir(kernel_func=kernel, frontend=TracerFrontend(), inputs=inputs, compile_opts=opts)
        t1 = time.monotonic()
        args = [s.name for s in bir.descriptor.input_specs]
        outs = [s.name for s in bir.descriptor.output_specs]
        compiled = compile_bir_to_neff(opts, bir, [], args, outs)
        t2 = time.monotonic()
        neff = Path(compiled.neff_path)
        record.update(status="ok", trace_s=round(t1 - t0, 2), neuronx_cc_s=round(t2 - t1, 2),
                      neff=str(neff), neff_bytes=neff.stat().st_size if neff.exists() else None,
                      inputs=len(args), outputs=len(outs))
    except Exception as err:  # noqa: BLE001 -- report every case
        record.update(status="failed", error=f"{type(err).__name__}: {err}"[:2000],
                      traceback=traceback.format_exc()[-4000:])
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--case", action="append", choices=sorted(CASES))
    ap.add_argument("--target", default=os.environ.get("NKI_COMPILE_TARGET", "trn3"))
    args = ap.parse_args(argv)
    failed = 0
    for name in args.case or list(CASES):
        rec = compile_case(name, args.out, args.target)
        failed += rec["status"] != "ok"
        print(json.dumps(rec), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
