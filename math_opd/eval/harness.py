"""Phase 4 evaluation.

Per-problem correctness is written out, not just the mean: the spec calls for a
paired bootstrap over problems, which needs the per-problem vector. Prompts use
the same DAPO instruction the arms trained on -- evaluating under a different
instruction would measure a distribution the student was never trained on.

    python -m math_opd.eval.harness --model runs/v0/checkpoint-625 --suite dev
"""

import argparse
import ast
import json
from pathlib import Path

from datasets import load_dataset
from math_verify import parse, verify
from transformers import AutoTokenizer

from ..train.rollout import INSTRUCTIONS


#: name -> (hub id, config, split, problem field, answer field, samples per problem)
BENCHMARKS = {
    "math500": ("HuggingFaceH4/MATH-500", None, "test", "problem", "answer", 4),
    "amc23": ("knoveleng/AMC-23", None, "train", "problem", "answer", 32),
    "aime24": ("Maxwell-Jia/AIME_2024", None, "train", "Problem", "Answer", 32),
    "aime25": ("yentinglin/aime_2025", None, "train", "problem", "answer", 32),
    "aime26": ("MathArena/aime_2026", None, "train", "problem", "answer", 32),
    "hmmt_feb26": ("MathArena/hmmt_feb_2026", None, "train", "problem", "answer", 32),
    "hmmt_nov25": ("MathArena/hmmt_nov_2025", None, "train", "problem", "answer", 32),
    # Open-ended, text-only, English maths. The multimodal and theorem-proving
    # configs are out of scope: the student takes no images and is not scored
    # on proofs.
    "olympiad": ("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", "train", "question", "final_answer", 4),
}

#: Checkpoint selection and training curves. Cheap and adequately powered at 1.7B.
SUITES = {"dev": ("math500", "amc23"), "full": tuple(BENCHMARKS)}


def gold_answer(raw) -> str:
    """OlympiadBench stores `final_answer` as a stringified list; others are plain.

    Multi-answer problems are scored on their first answer, which understates
    accuracy slightly on the few that have several.
    """
    s = str(raw).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)) and parsed:
                return str(parsed[0])
        except (ValueError, SyntaxError):
            pass
    return s


def score(completion: str, gold: str) -> bool:
    """`math_verify` handles both `\\boxed{}` and DAPO's `Answer: X`."""
    try:
        return bool(verify(parse(f"${gold}$"), parse(completion)))
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on the math suites")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-1.7B")
    ap.add_argument("--suite", choices=tuple(SUITES) + tuple(BENCHMARKS), default="dev")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--samples", type=int, default=0, help="override the per-benchmark sample count")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    names = SUITES.get(args.suite, (args.suite,))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_tokens + 2048,
        seed=args.seed,
    )

    results = {}
    for name in names:
        path, config, split, pf, af, n_default = BENCHMARKS[name]
        ds = load_dataset(path, config, split=split) if config else load_dataset(path, split=split)
        n = args.samples or n_default
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": INSTRUCTIONS["dapo"].format(problem=r[pf])}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for r in ds
        ]
        outputs = llm.generate(
            prompts,
            SamplingParams(
                n=n, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, seed=args.seed
            ),
        )

        # Per problem: the fraction of its `n` samples that are correct. The mean
        # of these is accuracy; the vector itself is what the bootstrap resamples.
        per_problem = []
        for row, out in zip(ds, outputs, strict=True):
            gold = gold_answer(row[af])
            correct = [score(s.text, gold) for s in out.outputs]
            per_problem.append(sum(correct) / len(correct))
        results[name] = {
            "n_problems": len(per_problem),
            "samples_per_problem": n,
            "accuracy": sum(per_problem) / len(per_problem),
            "per_problem": per_problem,
        }
        print(f"{name:<11} n={len(per_problem):<4} k={n:<3} accuracy={results[name]['accuracy']:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"model": args.model, "results": results}, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
