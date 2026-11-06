"""Reports total and non-zero parameter counts for the compiled net.

Reads weights/weights.npz (already compiled by scripts/compile_weights.py)
and prints a breakdown. Read-only: never rebuilds, never touches a
label, never runs the net on text.

Run: python scripts/report_params.py
"""

import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
WEIGHTS_PATH = os.path.join(ROOT, "weights", "weights.npz")


def main() -> None:
    if not os.path.exists(WEIGHTS_PATH):
        sys.exit(f"{WEIGHTS_PATH} does not exist yet -- run scripts/compile_weights.py first.")

    with np.load(WEIGHTS_PATH) as data:
        E = data["E"]
        comp1_W = data["comp1_W"]
        comp1_b = data["comp1_b"]
        fusion_w = data["fusion_w"]
        output_scale = float(data["output_scale"])
        output_bias = float(data["output_bias"])

    rows = [
        ("embedding table (E)", E.size, int(np.count_nonzero(E))),
        ("composition layer 1 weights", comp1_W.size, int(np.count_nonzero(comp1_W))),
        ("composition layer 1 biases", comp1_b.size, int(np.count_nonzero(comp1_b))),
        ("fusion weights", fusion_w.size, int(np.count_nonzero(fusion_w))),
        ("alpha, beta", 2, 2),
        ("output_scale, output_bias", 2, int(output_scale != 0.0) + int(output_bias != 0.0)),
    ]

    total = sum(r[1] for r in rows)
    total_nonzero = sum(r[2] for r in rows)

    name_width = max(len(r[0]) for r in rows)
    for name, size, nonzero in rows:
        print(f"{name:<{name_width}}  total={size:>7}  nonzero={nonzero:>6}")
    print(f"{'TOTAL':<{name_width}}  total={total:>7}  nonzero={total_nonzero:>6}")


if __name__ == "__main__":
    main()
