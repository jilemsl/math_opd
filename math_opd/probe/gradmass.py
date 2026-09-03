"""Gradient-mass coverage and mask overlap -- the non-accuracy metrics.

A mask that captures disproportionate gradient mass for free is the strongest
claim available here; DEAR's figures to beat are entropy 39.1% and random 35.9%.

"Mass" is the per-token divergence the trainer actually reduces, summed over the
positions a mask keeps and divided by the total. That is a proxy for gradient
norm, not the norm itself: the true per-token parameter gradient is intractable
to enumerate, but a position's contribution to the loss is what scales its
contribution to the gradient, and it is the same quantity DEAR reports.

Also computes the Jaccard overlap between each lexical mask and the entropy
top-k mask at matched budget. High overlap would mean the lexical masks are a
zero-cost *approximation* of entropy selection -- still a result, but it must be
claimed as that rather than as something new.

    python -m math_opd.probe.gradmass --rollouts rollouts.jsonl --out results/gradmass.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..masks.align import char_ranges_to_token_mask
from ..masks.baselines import match_retention, random_stratified_mask
from ..masks.variants import VARIANTS, variant_char_ranges
from ..train.mask_utils import offsets_from_ids


CHUNK = 256


@torch.no_grad()
def per_token_divergence(student, teacher, input_ids, n_prompt, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-position divergence and student entropy.

    The divergence mirrors `_chunked_divergence_loss` in `distillation_trainer.py`
    branch for branch, including the endpoint special-cases: at `beta=1` the
    mixture equals the teacher and the generalized form collapses to numerical
    noise, so the trainer uses the reverse KL directly and so does this.
    `F.kl_div(input, target, log_target=True)` is `target.exp() * (target - input)`,
    hence the swapped argument order.
    """
    s_hidden = student.model(input_ids=input_ids).last_hidden_state
    t_hidden = teacher.model(input_ids=input_ids).last_hidden_state
    s_head, t_head = student.get_output_embeddings(), teacher.get_output_embeddings()

    n = input_ids.size(1)
    idx = torch.arange(n_prompt - 1, n - 1, device=input_ids.device)
    div, ent = [], []
    for start in range(0, idx.numel(), CHUNK):
        sl = idx[start : start + CHUNK]
        s_logp = torch.log_softmax(s_head(s_hidden[0, sl]).float(), dim=-1)
        t_logp = torch.log_softmax(t_head(t_hidden[0, sl]).float(), dim=-1)
        if beta == 0.0:
            jsd = F.kl_div(s_logp, t_logp, reduction="none", log_target=True)
        elif beta == 1.0:
            jsd = F.kl_div(t_logp, s_logp, reduction="none", log_target=True)
        else:
            b = torch.tensor(beta, dtype=s_logp.dtype, device=s_logp.device)
            mixture = torch.logsumexp(torch.stack([s_logp + torch.log1p(-b), t_logp + torch.log(b)]), dim=0)
            jsd = b * F.kl_div(mixture, t_logp, reduction="none", log_target=True) + (1 - b) * F.kl_div(
                mixture, s_logp, reduction="none", log_target=True
            )
        # KL is non-negative; float error can put a position a hair below zero.
        div.append(jsd.sum(-1).clamp_min(0))
        ent.append(-(s_logp.exp() * s_logp).sum(-1))
    return torch.cat(div).cpu().numpy(), torch.cat(ent).cpu().numpy()


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Gradient-mass coverage and mask overlap")
    ap.add_argument("--rollouts", type=str, required=True)
    ap.add_argument("--student", type=str, default="Qwen/Qwen3-1.7B")
    ap.add_argument("--teacher", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.student)
    student = AutoModelForCausalLM.from_pretrained(args.student, dtype=torch.bfloat16, device_map="cuda").eval()
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.bfloat16, device_map="cuda").eval()
    rng = np.random.default_rng(args.seed)

    rows = [json.loads(line) for line in open(args.rollouts, encoding="utf-8") if line.strip()][: args.limit]
    arms = list(VARIANTS) + ["entropy", "random"]
    mass = dict.fromkeys(arms, 0.0)
    kept = dict.fromkeys(arms, 0)
    overlap = {v: [] for v in VARIANTS}
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
        div, ent = per_token_divergence(student, teacher, input_ids, len(prompt_ids), args.beta)

        masks = {v: char_ranges_to_token_mask(offsets, variant_char_ranges(text, v), len(text)) for v in VARIANTS}
        valid = np.ones(len(completion_ids), dtype=bool)
        # Baselines are budget-matched to v0, per the design.
        masks["entropy"] = match_retention(ent, masks["v0"], valid)
        masks["random"] = random_stratified_mask(masks["v0"], valid, rng=rng)

        total_mass += float(div.sum())
        total_tokens += len(completion_ids)
        for a in arms:
            m = masks[a]
            mass[a] += float(div[m].sum())
            kept[a] += int(m.sum())
        for v in VARIANTS:
            overlap[v].append(jaccard(masks[v], masks["entropy"]))

    out = {
        "n_rollouts": len(rows),
        "beta": args.beta,
        "student": args.student,
        "teacher": args.teacher,
        "total_tokens": total_tokens,
        "arms": {
            a: {
                "retention": kept[a] / max(total_tokens, 1),
                "gradient_mass_coverage": mass[a] / max(total_mass, 1e-12),
                "mass_per_token_ratio": (mass[a] / max(total_mass, 1e-12))
                / max(kept[a] / max(total_tokens, 1), 1e-12),
            }
            for a in arms
        },
        "jaccard_with_entropy_topk": {v: float(np.mean(overlap[v])) for v in VARIANTS},
        "dear_reference": {"entropy": 0.391, "random": 0.359},
    }

    print(f"{'arm':<9}{'retention':>11}{'grad mass':>11}{'mass/token':>12}{'jaccard(ent)':>14}")
    for a in arms:
        r = out["arms"][a]
        j = out["jaccard_with_entropy_topk"].get(a)
        print(
            f"{a:<9}{r['retention']:>11.4f}{r['gradient_mass_coverage']:>11.4f}"
            f"{r['mass_per_token_ratio']:>12.3f}{(f'{j:.4f}' if j is not None else '-'):>14}"
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
