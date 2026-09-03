"""Teacher scoring over student rollouts — Phase 1.

The spec's secondary claim is that a mask known *before* scoring lets the
teacher pass skip unmasked positions, which logit-based selectors cannot do.
That claim is about the `lm_head` projection, not the whole forward: a causal
backbone must run over every position to produce any position's hidden state.
Only the projection to the 151k-token vocabulary can be restricted. This module
measures both parts separately so the saving is reported rather than asserted.

Scores are cached over the union of every variant's spans, so `v0`/`v1`/`v0ms`/
`v2` all derive from one teacher pass and boundary policy stays a post-hoc flag.

    python -m math_opd.train.score --rollouts rollouts.jsonl --out scores.jsonl
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..masks.align import char_ranges_to_token_mask
from ..masks.variants import VARIANTS, variant_char_ranges
from .mask_utils import offsets_from_ids


#: Project this many positions per `lm_head` call, mirroring the trainer's
#: chunked path so peak memory does not scale with the sequence length.
CHUNK = 1024


def union_mask(text: str, offsets: list[tuple[int, int]]) -> np.ndarray:
    """Positions any variant might select. Caching this keeps boundary policy post-hoc."""
    ranges = [r for v in VARIANTS for r in variant_char_ranges(text, v)]
    return char_ranges_to_token_mask(offsets, ranges, len(text))


@torch.no_grad()
def score_batch(teacher, lm_head, input_ids, attention_mask, keep, n_prompt):
    """Teacher log-prob and entropy at the `keep` completion positions.

    `keep` is a boolean mask over completion positions. The backbone runs over
    everything; only the selected positions are projected to vocab.
    """
    t0 = time.perf_counter()
    hidden = teacher.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    torch.cuda.synchronize()
    t_backbone = time.perf_counter() - t0

    # Position i predicts token i+1, so the completion token at index j is
    # predicted by the hidden state at n_prompt + j - 1.
    idx = keep.nonzero(as_tuple=True)[0] + n_prompt - 1
    targets = input_ids[0, keep.nonzero(as_tuple=True)[0] + n_prompt]

    t0 = time.perf_counter()
    logps, entropies = [], []
    for start in range(0, idx.numel(), CHUNK):
        sl = idx[start : start + CHUNK]
        logits = lm_head(hidden[0, sl]).float()
        logprobs = torch.log_softmax(logits, dim=-1)
        logps.append(logprobs.gather(-1, targets[start : start + CHUNK, None]).squeeze(-1))
        entropies.append(-(logprobs.exp() * logprobs).sum(-1))
    torch.cuda.synchronize()
    t_project = time.perf_counter() - t0

    if not logps:
        return np.zeros(0), np.zeros(0), t_backbone, t_project
    return (
        torch.cat(logps).cpu().numpy(),
        torch.cat(entropies).cpu().numpy(),
        t_backbone,
        t_project,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Teacher scoring over student rollouts")
    ap.add_argument("--rollouts", type=str, required=True)
    ap.add_argument("--teacher", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--student", type=str, default="Qwen/Qwen3-1.7B", help="tokenizer source for the rollout ids")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--full", action="store_true", help="also time an unmasked pass, for the saving comparison")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.student)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.bfloat16, device_map="cuda").eval()
    lm_head = teacher.get_output_embeddings()

    rows = [json.loads(line) for line in open(args.rollouts, encoding="utf-8") if line.strip()][: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_completion = n_kept = 0
    t_backbone = t_project = t_project_full = 0.0
    t_wall = time.perf_counter()

    with out_path.open("w", encoding="utf-8") as f:
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
            keep = torch.from_numpy(union_mask(text, offsets)).cuda()

            input_ids = torch.tensor([prompt_ids + completion_ids], device="cuda")
            attention_mask = torch.ones_like(input_ids)
            logps, ents, tb, tp = score_batch(teacher, lm_head, input_ids, attention_mask, keep, len(prompt_ids))
            t_backbone += tb
            t_project += tp
            n_completion += len(completion_ids)
            n_kept += int(keep.sum())

            if args.full:
                _, _, _, tpf = score_batch(
                    teacher, lm_head, input_ids, attention_mask, torch.ones_like(keep), len(prompt_ids)
                )
                t_project_full += tpf

            f.write(
                json.dumps(
                    {
                        "problem": row["problem"],
                        "completion": row["completion"],
                        "positions": keep.nonzero(as_tuple=True)[0].tolist(),
                        "teacher_logp": [round(float(x), 5) for x in logps],
                        "teacher_entropy": [round(float(x), 5) for x in ents],
                    }
                )
                + "\n"
            )
            f.flush()

    wall = time.perf_counter() - t_wall
    summary = {
        "n_rollouts": len(rows),
        "completion_tokens": n_completion,
        "scored_positions": n_kept,
        "union_retention": n_kept / max(n_completion, 1),
        "wall_s": round(wall, 2),
        "backbone_s": round(t_backbone, 2),
        "projection_s": round(t_project, 2),
        "projection_full_s": round(t_project_full, 2) if args.full else None,
    }
    if args.full and t_project_full > 0:
        saved = (t_project_full - t_project) / (t_backbone + t_project_full)
        summary["teacher_pass_saving"] = round(saved, 4)
    Path(str(out_path) + ".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
