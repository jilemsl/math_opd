"""Student rollouts, in the format `probe/coverage.py` and Phase 1 both read.

Sampling matches the spec (T=1.0, top_p=1.0) and the `DistillationConfig`
defaults, so the text the probe measures is the text the trainer will produce.
Anything else measures a distribution training never sees.

    python -m math_opd.train.rollout --n-prompts 500 --out rollouts.jsonl
    python -m math_opd.probe.coverage --rollouts rollouts.jsonl --tokenizer Qwen/Qwen3-1.7B
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


#: The training distribution, per the spec: DAPO-Math-17K, English subset.
DATASET = "open-r1/DAPO-Math-17k-Processed"

#: DAPO ships its own instruction (`Answer: $Answer`). `boxed` is the convention
#: the Phase 4 benchmarks score against, and `\boxed{` is a v2 trigger, so the
#: choice moves v2's retention. Phase 0 measured both: `dapo` halves the
#: truncation rate (3.0% vs 6.4% at 4,096) and costs v2 nothing -- its retention
#: is *higher* (0.176 vs 0.171), because the `=` triggers carry the arm and
#: `\boxed{` is not load-bearing. `dapo` is also not a prompt we added, which is
#: what the spec's gotcha list asks for.
INSTRUCTIONS = {
    "dapo": (
        "Solve the following math problem step by step. The last line of your response should be of the form "
        "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n{problem}"
    ),
    "boxed": "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate student rollouts")
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    ap.add_argument("--n-prompts", type=int, default=500)
    ap.add_argument("--n-samples", type=int, default=1, help="G, rollouts per prompt")
    ap.add_argument("--instruction", choices=tuple(INSTRUCTIONS), default="dapo")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--thinking", action="store_true", help="Qwen3 hybrid thinking mode; the spec says off")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    ds = load_dataset(DATASET, "en", split="train")
    rows = [ds[i] for i in random.Random(args.seed).sample(range(len(ds)), min(args.n_prompts, len(ds)))]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    template = INSTRUCTIONS[args.instruction]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": template.format(problem=r["prompt"])}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.thinking,
        )
        for r in rows
    ]

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_tokens + 2048,
        seed=args.seed,
    )
    outputs = llm.generate(
        prompts,
        SamplingParams(
            n=args.n_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=args.seed,
        ),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lengths, n_clipped = [], 0
    with out_path.open("w", encoding="utf-8") as f:
        for row, output in zip(rows, outputs, strict=True):
            for sample in output.outputs:
                lengths.append(len(sample.token_ids))
                n_clipped += sample.finish_reason == "length"
                record = {
                    "problem": row["prompt"],
                    "answer": row["solution"],
                    "completion": sample.text,
                    "n_tokens": len(sample.token_ids),
                    "finish_reason": sample.finish_reason,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"wrote {len(lengths)} rollouts to {out_path}")
    print(f"mean_len={sum(lengths) / len(lengths):.0f} max_len={max(lengths)} clipped={n_clipped}/{len(lengths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
