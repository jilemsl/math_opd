# Lexical-Mask On-Policy Distillation

Implementation of `MATH_MASK_EXPERIMENT.md`. Selection is computed from surface
text alone — a regex over the decoded completion, no forward pass, no logits,
no verifier.

## Status

| Phase | Deliverable | State |
|---|---|---|
| 0 | `masks/` + coverage probe | **built, gate not yet cleared** |
| 1 | Rollout + teacher scoring | not started |
| 2 | Training loop | trainer built; `entropy`/`kl` scoring pass stubbed |
| 3 | 8 arms × 3 seeds | not started |
| 4 | Eval suite | not started |

The Phase 0 gate needs ~500 rollouts from the untrained 1.7B student, which
needs a GPU. `results/phase0.json` currently holds a **plumbing check** run
against MATH reference solutions (`--source hendrycks`) — it exercises every
code path but its numbers do not satisfy the gate, and the JSON says so in a
`WARNING` field. Do not treat those retention figures as the real ones:
reference solutions are far denser in LaTeX than student rollouts.

## Layout

```
masks/
  charclass.py   M, M_EXT, atomic units
  spans.py       delimited interiors + gap-tolerant runs
  variants.py    v0, v1, v0ms, v2 -> char ranges
  align.py       char spans -> token indices via offset_mapping
  baselines.py   random-stratified, top-k, budget matching
probe/
  coverage.py    Phase 0, writes results/phase0.json
train/
  mask_utils.py     batch mask construction over generated ids
  masked_trainer.py DistillationTrainer subclass
tests/            45 tests, offline
```

## Running

```bash
# Phase 0 gate (needs real rollouts: one JSON object per line, `completion` field)
python -m math_opd.probe.coverage --rollouts rollouts.jsonl --tokenizer Qwen/Qwen3-1.7B

# plumbing check, offline, NOT the gate
python -m math_opd.probe.coverage --source hendrycks --limit 500

python -m pytest math_opd/tests/ -q
```

## Design notes

**Masks are character spans, never token ids.** Qwen3 splits `12.5` into
`1`,`2`,`.`,`5` and emits ` $` as one token covering the space *and* the
delimiter. Building on characters and mapping through `offset_mapping` makes
both problems disappear and keeps the code tokenizer-portable.

**The trainer changes one thing.** `DistillationTrainer` already reduces over
`completion_mask * tool_mask`, so an arm is installed by composing its mask
into `tool_mask` and calling `super()._compute_loss`. No part of the JSD,
the chunked `lm_head` projection, or the grad-accum normalization is
reimplemented.

**Loss normalization is rescaled to the selected count.** The base trainer
divides by `num_items_in_batch`, gathered from the *unmasked* completion
tokens. Left alone, a 19%-retention arm would train at ~19% of the effective
learning rate of a 59%-retention arm, and `v0` vs `v2` would compare learning
rates rather than masks. `normalize_by_selected=True` (default) corrects this.

**`entropy` and `kl` pay for their scoring pass explicitly.**
`_selection_scores` is a separate no-grad forward rather than a free read of
the training forward's logits. Folding it in would hide exactly the compute
asymmetry the paper claims. It is stubbed until Phase 0 is reviewed.

## Two things Phase 0 should decide

1. **Gap tolerance `k=2` bridges more than intended.** On the plumbing run,
   run-spans have `frac_le_2 = 0.55` but a mean of ~14 atomic units — bimodal.
   The rule counts whitespace-separated chunks, so `\boxed{47} and 3 cases`
   bridges across `and` into the next clause and `3` joins the boxed span. That
   is faithful to the spec ("short connectives"), but it weakens `v0ms`: a lone
   numeral in prose gets absorbed into a neighbouring span instead of being
   dropped, which is the exact contrast `v0ms` exists to isolate. Check the
   span-length histogram on real rollouts before Phase 3.

2. **v2's stop rule is strict.** `= 3x + 2` yields just `3` — it stops at the
   first non-M_EXT token, per spec. The plumbing run splits post-trigger content
   roughly 59% terminal / 25% rearrangement, so a quarter of v2's triggers are
   truncated to their first unit. Worth knowing which story the paper tells
   before committing to the arm.
