"""Paired bootstrap over problems, per the spec.

Mean +/- std across three seeds has three points of resolution and treats the
problem set as fixed. Resampling problems instead uses all ~30-500 of them and
is paired, so it cancels the large per-problem difficulty variance that
otherwise swamps a few-point accuracy gap on AIME-sized benchmarks.

    python -m math_opd.eval.compare --a runs/vanilla/eval.json --b runs/v0/eval.json
"""

import argparse
import json

import numpy as np


def paired_bootstrap(a: list[float], b: list[float], n_boot: int, rng: np.random.Generator) -> dict:
    """Bootstrap the paired difference `b - a` over problem indices."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = (b[idx] - a[idx]).mean(axis=1)
    observed = float(b.mean() - a.mean())
    # Two-sided p: how often a resampled difference lands on the other side of zero.
    p = float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return {
        "acc_a": float(a.mean()),
        "acc_b": float(b.mean()),
        "delta": observed,
        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "p_two_sided": min(p, 1.0),
        "n_problems": len(a),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Paired bootstrap between two eval runs")
    ap.add_argument("--a", type=str, required=True, help="baseline eval json")
    ap.add_argument("--b", type=str, required=True, help="treatment eval json")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    ra = json.load(open(args.a, encoding="utf-8"))
    rb = json.load(open(args.b, encoding="utf-8"))
    rng = np.random.default_rng(args.seed)

    out = {"a": ra["model"], "b": rb["model"], "benchmarks": {}}
    for name in sorted(set(ra["results"]) & set(rb["results"])):
        pa, pb = ra["results"][name]["per_problem"], rb["results"][name]["per_problem"]
        if len(pa) != len(pb):
            raise ValueError(f"{name}: problem counts differ ({len(pa)} vs {len(pb)}); not paired")
        out["benchmarks"][name] = paired_bootstrap(pa, pb, args.n_boot, rng)

    print(f"{'benchmark':<12}{'a':>8}{'b':>8}{'delta':>9}{'95% CI':>20}{'p':>8}")
    for name, r in out["benchmarks"].items():
        ci = f"[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}]"
        print(f"{name:<12}{r['acc_a']:>8.4f}{r['acc_b']:>8.4f}{r['delta']:>+9.4f}{ci:>20}{r['p_two_sided']:>8.3f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
