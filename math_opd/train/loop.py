"""Training entry point — one arm, one seed.

`MaskedDistillationTrainer` is the whole method; this only builds the dataset
and the config. `DistillationConfig` has no `num_generations`, so G rollouts per
prompt come from repeating each prompt G times in the dataset.

    python -m math_opd.train.loop --arm v0 --seed 0 --output-dir runs/v0-s0
"""

import argparse
import json
import random
from pathlib import Path

from datasets import Dataset, load_dataset

from .masked_trainer import MaskedDistillationConfig, MaskedDistillationTrainer
from .rollout import DATASET, INSTRUCTIONS


def build_dataset(n_prompts: int, g: int, seed: int) -> Dataset:
    """`n_prompts` DAPO problems, each repeated `g` times, under DAPO's own instruction."""
    ds = load_dataset(DATASET, "en", split="train")
    rows = [ds[i] for i in random.Random(seed).sample(range(len(ds)), min(n_prompts, len(ds)))]
    prompts = [
        {"prompt": [{"role": "user", "content": INSTRUCTIONS["dapo"].format(problem=r["prompt"])}]}
        for r in rows
        for _ in range(g)
    ]
    random.Random(seed).shuffle(prompts)
    return Dataset.from_list(prompts)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train one arm of the lexical-mask OPD experiment")
    ap.add_argument("--arm", type=str, default="v0")
    ap.add_argument(
        "--budget-variant", type=str, default="v0", help="mask whose per-batch retention the baselines must match"
    )
    ap.add_argument(
        "--random-retention",
        type=float,
        default=None,
        help="for `--arm random`: keep this flat fraction of completion tokens instead of matching a mask's budget",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--student", type=str, default="Qwen/Qwen3-1.7B")
    ap.add_argument("--teacher", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--n-prompts", type=int, default=5000)
    ap.add_argument("--g", type=int, default=4, help="rollouts per prompt")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--max-completion-length", type=int, default=4096)
    ap.add_argument("--per-device-train-batch-size", type=int, default=4)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=1e-6)
    ap.add_argument("--lr-scheduler-type", type=str, default="linear")
    ap.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.25)
    ap.add_argument("--logging-steps", type=int, default=1)
    ap.add_argument("--save-steps", type=int, default=0, help="0 disables checkpointing")
    ap.add_argument("--output-dir", type=str, required=True)
    args = ap.parse_args()

    config = MaskedDistillationConfig(
        output_dir=args.output_dir,
        arm=args.arm,
        budget_variant=args.budget_variant,
        random_retention=args.random_retention,
        seed=args.seed,
        mask_seed=args.seed,
        # Qwen3's chat template defaults to *thinking* mode: without this the
        # template omits the empty `<think></think>` prefill and the student
        # generates a reasoning trace. Those traces are prose-heavy and
        # LaTeX-sparse, which drops v0's retention from 0.63 to 0.24 -- a
        # different distribution from the one Phase 0 gated, and not the
        # non-thinking student the spec specifies.
        chat_template_kwargs={"enable_thinking": False},
        # Sampling per the spec; these are also the `DistillationConfig` defaults,
        # so the rollouts match what Phase 0 measured.
        temperature=1.0,
        top_p=1.0,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        # A short probe under the default linear decay would compare schedule
        # shapes rather than learning rates; `constant` isolates the rate.
        lr_scheduler_type=args.lr_scheduler_type,
        max_steps=args.max_steps,
        num_train_epochs=1,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit",
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        teacher_model_init_kwargs={"dtype": "bfloat16"},
        model_init_kwargs={"dtype": "bfloat16"},
        logging_steps=args.logging_steps,
        # Checkpoints carry `trainer_state.json`, so the loss and
        # `mask_retention` history survives a crash mid-run.
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=2,
        report_to="none",
    )

    trainer = MaskedDistillationTrainer(
        model=args.student,
        teacher_model=args.teacher,
        args=config,
        train_dataset=build_dataset(args.n_prompts, args.g, args.seed),
    )
    trainer.train()

    log = [h for h in trainer.state.log_history if "loss" in h]
    out = Path(args.output_dir) / "train_log.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"arm": args.arm, "seed": args.seed, "history": log}, indent=2), encoding="utf-8")

    losses = [h["loss"] for h in log]
    retentions = [h["mask_retention"] for h in log if "mask_retention" in h]
    print(f"ARM={args.arm} SEED={args.seed} steps={len(losses)}")
    print(f"  loss first={losses[0]:.5f} last={losses[-1]:.5f} min={min(losses):.5f} max={max(losses):.5f}")
    print(f"  any NaN: {any(x != x for x in losses)}")
    if retentions:
        print(
            f"  mask_retention first={retentions[0]:.4f} last={retentions[-1]:.4f} mean={sum(retentions) / len(retentions):.4f}"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
