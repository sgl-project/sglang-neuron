"""Compare two tools/parity_fixed_batches.py outputs token by token."""
import json
import sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
for mode in ("solo", "waves"):
    same = 0
    for i, (x, y) in enumerate(zip(a[mode], b[mode])):
        if x == y:
            same += 1
        else:
            d = next(k for k in range(min(len(x), len(y))) if x[k] != y[k])
            print(f"{mode} req{i:02d}: first difference at token {d}")
    print(f"{mode}: {same}/{len(a[mode])} identical")
