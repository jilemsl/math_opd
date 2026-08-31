"""Phase 0 probe -- the whole project's risk, measured before anything is trained.

Writes the five gate metrics to `results/phase0.json`:

  1. Retention rate per variant.                 GATE: v2 >= 1%
  2. Positional histogram, 20 bins.              FLAG: mass in the last 15%
  3. Span-length distribution (undelimited).
  4. Delimiter coverage of all M tokens.         FLAG: > 80%
  5. Post-trigger content split.

Plus the truncation rate at max_response_length, which Phase 1 needs to justify
dropping from 16,384/32,768 to 4,096.

Run against ~500 rollouts from the *untrained* student:

    python -m math_opd.probe.coverage --rollouts rollouts.jsonl --tokenizer Qwen/Qwen3-1.7B

`rollouts.jsonl` needs one object per line with a `completion` field. Use
`--source hendrycks` only to exercise the plumbing offline -- MATH reference
solutions are not student rollouts and their numbers do not satisfy the gate.
"""

import argparse
import json
from bisect import bisect_left
from collections import Counter
from pathlib import Path

import numpy as np

from ..masks.align import char_ranges_to_token_mask, completion_offsets
from ..masks.charclass import M, atomic_units
from ..masks.spans import find_delimited_spans, find_spans
from ..masks.variants import TRIGGER_RE, VARIANTS, v2_spans, variant_char_ranges


N_BINS = 20


def load_rollouts(args) -> list[str]:
    if args.source == "jsonl":
        texts = []
        with open(args.rollouts, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(json.loads(line)[args.field])
        return texts[: args.limit]

    # Plumbing check only. Reference solutions, not rollouts.
    from datasets import load_dataset

    ds = load_dataset("EleutherAI/hendrycks_math", "algebra", split="train")
    return [r["solution"] for r in ds.select(range(min(args.limit, len(ds))))]


def probe(texts: list[str], tokenizer, max_len: int = 4096) -> dict:
    retained = dict.fromkeys(VARIANTS, 0)
    total_tokens = 0
    pos_hist = {v: np.zeros(N_BINS) for v in VARIANTS}
    run_span_units: list[int] = []
    m_inside = m_total = 0
    openers = unbalanced = 0
    trig_terminal = trig_rearrange = trig_empty = 0
    trig_units: list[int] = []
    n_truncated = 0
    lengths: list[int] = []

    for text in texts:
        offsets = completion_offsets(tokenizer, text)
        n_tok = len(offsets)
        if n_tok == 0:
            continue
        lengths.append(n_tok)
        if n_tok >= max_len:
            n_truncated += 1
        total_tokens += n_tok

        for v in VARIANTS:
            mask = char_ranges_to_token_mask(offsets, variant_char_ranges(text, v), len(text))
            retained[v] += int(mask.sum())
            if mask.any():
                # 2. Normalized position of every masked token.
                bins = np.clip((np.flatnonzero(mask) / n_tok * N_BINS).astype(int), 0, N_BINS - 1)
                np.add.at(pos_hist[v], bins, 1)

        # 3. Span-length distribution, undelimited mode.
        spans, stats = find_spans(text)
        run_span_units.extend(len(s.units) for s in spans if s.kind == "run")
        openers += stats["openers"]
        unbalanced += stats["unbalanced"]

        # 4. Fraction of all M matches sitting inside delimiters.
        _, dstats = find_delimited_spans(text)
        covered = dstats["covered"]
        for s, _e in atomic_units(text, M):
            m_total += 1
            if any(cs <= s < ce for cs, ce in covered):
                m_inside += 1

        # 5. Is the post-trigger run a final value (`= 47`) or a rearrangement
        #    (`= 3x + 2 - 5`)? Different stories; we need to know which we tell.
        runs = v2_spans(text)  # one per trigger that had a run, in trigger order
        starts = [s.start for s in runs]
        for tm in TRIGGER_RE.finditer(text):
            i = bisect_left(starts, tm.end())
            run = runs[i] if i < len(runs) and runs[i].start < tm.end() + 8 else None
            if run is None:
                trig_empty += 1
                continue
            trig_units.append(len(run.units))
            rest = text[run.end :].lstrip()
            if rest[:1].isalpha() or rest[:1] == "\\":
                trig_rearrange += 1
            else:
                trig_terminal += 1

    n_trig = max(trig_terminal + trig_rearrange + trig_empty, 1)
    out = {
        "n_rollouts": len(texts),
        "total_tokens": total_tokens,
        "mean_response_tokens": float(np.mean(lengths)) if lengths else 0.0,
        "truncation_rate": n_truncated / max(len(lengths), 1),
        "retention": {v: retained[v] / max(total_tokens, 1) for v in VARIANTS},
        "positional_histogram": {v: (pos_hist[v] / max(pos_hist[v].sum(), 1)).round(5).tolist() for v in VARIANTS},
        "last_15pct_mass": {
            v: float((pos_hist[v][int(N_BINS * 0.85) :].sum()) / max(pos_hist[v].sum(), 1)) for v in VARIANTS
        },
        "run_span_units": {
            "mean": float(np.mean(run_span_units)) if run_span_units else 0.0,
            "median": float(np.median(run_span_units)) if run_span_units else 0.0,
            "frac_le_2": float(np.mean([u <= 2 for u in run_span_units])) if run_span_units else 0.0,
            "histogram": dict(Counter(min(u, 10) for u in run_span_units)),
        },
        "delimiter_coverage": m_inside / max(m_total, 1),
        "unbalanced_delimiter_rate": unbalanced / max(openers, 1),
        "post_trigger": {
            "terminal": trig_terminal / n_trig,
            "rearrangement": trig_rearrange / n_trig,
            "empty": trig_empty / n_trig,
            "mean_units": float(np.mean(trig_units)) if trig_units else 0.0,
        },
    }

    out["gates"] = {
        "v2_retention_ge_1pct": out["retention"]["v2"] >= 0.01,
        "no_positional_mask": {v: out["last_15pct_mass"][v] < 0.30 for v in VARIANTS},
        "delimiters_not_redundant": out["delimiter_coverage"] <= 0.80,
        "unbalanced_under_5pct": out["unbalanced_delimiter_rate"] <= 0.05,
        "truncation_under_2pct": out["truncation_rate"] <= 0.02,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 0 mask coverage probe")
    ap.add_argument("--rollouts", type=str, default=None, help="JSONL with a `completion` field per line")
    ap.add_argument("--field", type=str, default="completion")
    ap.add_argument("--source", choices=["jsonl", "hendrycks"], default="jsonl")
    ap.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.source == "jsonl" and not args.rollouts:
        ap.error("--rollouts is required with --source jsonl")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    texts = load_rollouts(args)
    res = probe(texts, tokenizer, args.max_len)
    res["source"] = args.source
    res["tokenizer"] = args.tokenizer
    if args.source == "hendrycks":
        res["WARNING"] = "Reference solutions, not student rollouts. Plumbing check only; does not satisfy the gate."

    out_path = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "results" / "phase0.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print(f"n={res['n_rollouts']} tokens={res['total_tokens']} mean_len={res['mean_response_tokens']:.0f}")
    print("retention:  " + "  ".join(f"{v}={res['retention'][v]:.4f}" for v in VARIANTS))
    print("last-15%:   " + "  ".join(f"{v}={res['last_15pct_mass'][v]:.3f}" for v in VARIANTS))
    print(f"delim_cov={res['delimiter_coverage']:.3f}  unbalanced={res['unbalanced_delimiter_rate']:.3f}")
    print(f"run_units mean={res['run_span_units']['mean']:.2f} frac<=2={res['run_span_units']['frac_le_2']:.3f}")
    print(f"post-trigger: {res['post_trigger']}")
    print(f"truncation={res['truncation_rate']:.4f}")
    print("\ngates:")
    for k, v in res["gates"].items():
        print(f"  {'PASS' if (v is True or (isinstance(v, dict) and all(v.values()))) else 'CHECK'}  {k}: {v}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
