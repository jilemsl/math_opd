"""Bottom-k and top-k selection by the trainer's own per-token divergence.

Sections 0b/4 show accuracy running *against* gradient-mass coverage over the
masks tried so far. This probe asks what the axis looks like at its endpoints:
at a fixed budget alpha, the mask that captures the least divergence possible
(bottom-k by `D_t`) and the most (top-k). Neither is a candidate method -- both
need the teacher's and the student's full vocabulary projection before they can
select, which is the cost the lexical masks exist to avoid -- but they bound the
axis and, more usefully, they separate two things `v0` confounds: being
*low-divergence* and being *mathematical notation*.

Three numbers decide whether the training runs are worth launching:

- `mass_per_token`, the value being minimized: coverage / retention.
- `loss_scale_ratio`, mean `D_t` over selected positions / mean over all of
  them. The loss is `(1/|M|) sum_{t in M} D_t`, so bottom-k shrinks the loss by
  construction and with it the gradient. A ratio far below 1 means a fixed-LR
  run is a disguised learning-rate change and cannot be read as a mask effect.
- `composition`, what the selected tokens *are*. Bottom-k by divergence may be
  mostly whitespace, punctuation and sub-word continuations, which would make it
  junk rather than "the determined part of the derivation".

`D_t` is the trainer's per-token loss, not a proxy for it: `per_token_divergence`
is imported from `gradmass` so the branch-for-branch `beta` handling stays
shared. At the default `beta=1` that is KL(student || teacher). Note the `kl`
arm in `masked_trainer.py` scores the *other* direction, so it is not the
quantity minimized here.

    python -m math_opd.probe.oracle_mass --rollouts rollouts.jsonl --out results/oracle_mass.json
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..masks.align import char_ranges_to_token_mask
from ..masks.variants import variant_char_ranges
from ..train.mask_utils import offsets_from_ids
from .gradmass import jaccard, per_token_divergence


#: Categories are ordered: the first pattern that matches a token wins, so
#: `\frac` counts as latex rather than as an alphabetic word.
_CATEGORIES = [
    ("whitespace", re.compile(r"^\s+$")),
    ("latex", re.compile(r"[\\${}^_]")),
    ("digit", re.compile(r"^\s*\d+$")),
    ("operator", re.compile(r"^\s*[+\-*/=<>()\[\]|]+$")),
    ("punctuation", re.compile(r"^\s*[.,;:!?'\"`]+$")),
    ("word", re.compile(r"^\s*[A-Za-z]{2,}$")),
]


def categorize(piece: str) -> str:
    """Coarse surface class of one decoded token; `subword` is the catch-all."""
    for name, pattern in _CATEGORIES:
        hit = pattern.search(piece) if name == "latex" else pattern.match(piece)
        if hit:
            return name
    return "subword"


def extreme_k_mask(scores: np.ndarray, k: int, *, largest: bool) -> np.ndarray:
    """Boolean mask over the `k` largest (or smallest) entries of `scores`."""
    out = np.zeros(len(scores), dtype=bool)
    if k <= 0:
        return out
    order = np.argsort(scores)
    out[order[-k:] if largest else order[:k]] = True
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Endpoints of the gradient-mass axis at a fixed budget")
    ap.add_argument("--rollouts", type=str, required=True)
    ap.add_argument("--student", type=str, default="Qwen/Qwen3-1.7B")
    ap.add_argument("--teacher", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.638, help="budget to match; v0's measured retention")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--examples", type=int, default=40, help="selected tokens to record verbatim, per arm")
    ap.add_argument(
        "--drop-fractions",
        type=str,
        default="0.01,0.02,0.05,0.10,0.15,0.20,0.30,0.362,0.50",
        help="delta grid for the drop-the-top-delta curve: keep the (1-delta) lowest-`D_t` tokens",
    )
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.student)
    student = AutoModelForCausalLM.from_pretrained(args.student, dtype=torch.bfloat16, device_map="cuda").eval()
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.bfloat16, device_map="cuda").eval()

    rows = [json.loads(line) for line in open(args.rollouts, encoding="utf-8") if line.strip()][: args.limit]
    arms = ["oracle_min", "oracle_max", "v0"]
    mass = dict.fromkeys(arms, 0.0)
    kept = dict.fromkeys(arms, 0)
    composition = {a: {} for a in arms}
    examples = {a: [] for a in arms}
    overlap_min_v0, overlap_max_v0 = [], []
    deltas = [float(x) for x in args.drop_fractions.split(",")]
    drop_mass = dict.fromkeys(deltas, 0.0)
    drop_kept = dict.fromkeys(deltas, 0)
    total_mass = 0.0
    total_tokens = 0

    for row in rows:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["problem"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(row["completion"], add_special_tokens=False)["input_ids"]
        if not completion_ids:
            continue
        text, offsets = offsets_from_ids(tokenizer, completion_ids)
        input_ids = torch.tensor([prompt_ids + completion_ids], device="cuda")
        div, _ = per_token_divergence(student, teacher, input_ids, len(prompt_ids), args.beta)

        # Per rollout, so the budget matches the trainer's per-sequence draw.
        k = int(round(args.alpha * len(completion_ids)))
        masks = {
            "oracle_min": extreme_k_mask(div, k, largest=False),
            "oracle_max": extreme_k_mask(div, k, largest=True),
            "v0": char_ranges_to_token_mask(offsets, variant_char_ranges(text, "v0"), len(text)),
        }

        total_mass += float(div.sum())
        total_tokens += len(completion_ids)
        for a in arms:
            m = masks[a]
            mass[a] += float(div[m].sum())
            kept[a] += int(m.sum())
            for idx in np.flatnonzero(m):
                piece = tokenizer.decode([completion_ids[idx]])
                cat = categorize(piece)
                composition[a][cat] = composition[a].get(cat, 0) + 1
                if len(examples[a]) < args.examples:
                    examples[a].append(piece)
        overlap_min_v0.append(jaccard(masks["oracle_min"], masks["v0"]))
        overlap_max_v0.append(jaccard(masks["oracle_max"], masks["v0"]))

        # "Drop the top delta by divergence" is the same family as bottom-k,
        # reparameterized: keeping the (1-delta) lowest is dropping the top
        # delta. What the curve settles is which delta leaves a trainable loss.
        order = np.argsort(div)
        for delta in deltas:
            keep = order[: len(div) - int(round(delta * len(div)))]
            drop_mass[delta] += float(div[keep].sum())
            drop_kept[delta] += len(keep)

    mean_div_all = total_mass / max(total_tokens, 1)
    out = {
        "n_rollouts": len(rows),
        "beta": args.beta,
        "alpha": args.alpha,
        "total_tokens": total_tokens,
        "mean_divergence_all_tokens": mean_div_all,
        "arms": {},
        "jaccard_with_v0": {
            "oracle_min": float(np.mean(overlap_min_v0)),
            "oracle_max": float(np.mean(overlap_max_v0)),
        },
    }
    for a in arms:
        retention = kept[a] / max(total_tokens, 1)
        coverage = mass[a] / max(total_mass, 1e-12)
        mean_div_selected = mass[a] / max(kept[a], 1)
        out["arms"][a] = {
            "retention": retention,
            "gradient_mass_coverage": coverage,
            "mass_per_token": coverage / retention if retention else 0.0,
            # The go/no-go number: how far the loss (and so the gradient) moves
            # relative to training on every token.
            "loss_scale_ratio": mean_div_selected / mean_div_all if mean_div_all else 0.0,
            "composition": {k: v / max(kept[a], 1) for k, v in sorted(composition[a].items(), key=lambda kv: -kv[1])},
            "example_tokens": examples[a],
        }

    # For each delta: what fraction of tokens survives, what fraction of the
    # divergence survives with them, and how far the mean per-token loss -- and
    # so the gradient, and so the effective learning rate -- moves.
    out["drop_top_delta_curve"] = {
        f"{delta:g}": {
            "retention": drop_kept[delta] / max(total_tokens, 1),
            "gradient_mass_coverage": drop_mass[delta] / max(total_mass, 1e-12),
            "mass_per_token": (drop_mass[delta] / max(total_mass, 1e-12))
            / max(drop_kept[delta] / max(total_tokens, 1), 1e-12),
            "loss_scale_ratio": (drop_mass[delta] / max(drop_kept[delta], 1)) / mean_div_all,
        }
        for delta in deltas
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    for a in arms:
        d = out["arms"][a]
        print(
            f"{a:11s} r={d['retention']:.4f} cov={d['gradient_mass_coverage']:.4f} "
            f"mass/tok={d['mass_per_token']:.4f} loss_scale={d['loss_scale_ratio']:.4f}"
        )
        print(f"            {dict(list(d['composition'].items())[:5])}")
    print("\ndrop-top-delta curve:")
    print(f"  {'delta':>7} {'retention':>10} {'coverage':>10} {'mass/tok':>10} {'loss_scale':>11}")
    for delta in deltas:
        c = out["drop_top_delta_curve"][f"{delta:g}"]
        print(
            f"  {delta:7g} {c['retention']:10.4f} {c['gradient_mass_coverage']:10.4f} "
            f"{c['mass_per_token']:10.4f} {c['loss_scale_ratio']:11.4f}"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
