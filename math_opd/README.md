# Lexical-Mask On-Policy Distillation

Implementation of `MATH_MASK_EXPERIMENT.md`. Selection is computed from surface
text alone — a regex over the decoded completion, no forward pass, no logits,
no verifier.

## Status

| Phase | Deliverable | State |
|---|---|---|
| 0 | `masks/` + coverage probe | **cleared, 4 gates of 5** |
| 1 | Rollout + teacher scoring | **gate cleared** (100 prompts end-to-end) |
| 2 | Training loop | trainer built; `entropy`/`kl` scoring pass stubbed |
| 3 | 8 arms × 3 seeds | **on hold** — pilot says the setup is underpowered |
| 4 | Eval suite | built (`eval/harness.py`, `eval/compare.py`, `probe/gradmass.py`) |

`results/phase0.json` is the gate run: 500 rollouts from the untrained
Qwen3-1.7B student, non-thinking, on **DAPO-Math-17K (English)** — the Phase 1
training distribution — under DAPO's own instruction, sampled at T=1.0/top_p=1.0,
751,834 completion tokens.

| gate | value | threshold | |
|---|---|---|---|
| v2 retention | 0.176 | ≥ 0.01 | pass |
| last-15% mass | v0 .158, v1 .157, v0ms .160, v2 .206 | < 0.30 | pass |
| delimiter coverage | 0.763 | ≤ 0.80 | pass |
| unbalanced delimiters | 0.001 | ≤ 0.05 | pass |
| truncation @ 4,096 | **0.030** | ≤ 0.02 | **fail** |

Retention is v0 0.632, v1 0.548, v0ms 0.604, v2 0.176; mean response 1,504
tokens, median 1,197, p90 2,881.

**The truncation gate does not clear and 4,096 is kept anyway.** DAPO is far
harder than MATH — the same student averages 809 tokens on MATH reference
prompts and 1,504 here. 3.0% of rollouts hit the cap and they hold 8.2% of all
completion tokens. The spec's conditional was "if under 2%, the reduction from
16,384/32,768 is uncontroversial"; at 3.0% it is a stated limitation instead.
Truncation removes the final answer from the hardest traces, and it bites v2
(back-loaded mass) harder than v0, so the paper should say so rather than let a
reviewer find it.

**The prompt is DAPO's own, not `\boxed{}`.** Both were measured on 500 rollouts
each. DAPO's native instruction halves truncation (3.0% vs 6.4%), and costs v2
nothing — retention is *higher* (0.176 vs 0.171) with less back-loaded mass
(0.206 vs 0.224), because the relational triggers carry the arm and `\boxed{`
is not load-bearing. It is also not a prompt we added, per the spec's gotcha.

**v0 is 87% "everything inside `$…$`".** The delimiter gate counts M *matches*
inside delimiters (0.763, passes). Measured on the tokens that actually receive
gradient it is 88.4% delimited / 11.6% run-span, because delimited interiors are
kept wholesale and pull in non-M material. With `frac_le_2 = 0.594`, the spec's
metric-3 warning — "if most spans are 1–2 atomic units the undelimited arm
contributes little" — is met on both counts. Worth stating plainly: v0 is
mostly a LaTeX-region mask.

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
  rollout.py        vLLM generation over DAPO-Math-17K
  score.py          teacher log-probs, masked positions only
  mask_utils.py     batch mask construction over generated ids
  masked_trainer.py DistillationTrainer subclass
tests/            46 tests, offline
```

## Running

```bash
# Phase 0 gate: generate the rollouts, then probe them.
python -m math_opd.train.rollout --n-prompts 500 --out rollouts.jsonl
python -m math_opd.probe.coverage --rollouts rollouts.jsonl --tokenizer Qwen/Qwen3-1.7B

# plumbing check, offline, NOT the gate
python -m math_opd.probe.coverage --source hendrycks --limit 500

# Phase 1 gate: teacher scoring over those rollouts.
python -m math_opd.train.score --rollouts rollouts.jsonl --limit 100 --full --out scores.jsonl

python -m pytest math_opd/tests/ -q
```

## Phase 1 — what the teacher pass actually costs

`results/phase1.json`. 100 prompts end-to-end: rollout (563 s, mean 1,725
tokens) then teacher scoring with Qwen3-4B-Instruct-2507 on one A6000 over
172,450 completion tokens.

**The secondary claim has a ceiling of 11.6%, and it is not the one the spec
states.** The teacher *forward* cannot be restricted to masked positions — a
causal backbone must run over every position to produce any position's hidden
state. Only the `lm_head` projection can be skipped, and for Qwen3-4B
(hidden 2560, vocab 151,936) that projection is **11.6% of the teacher pass**:

| | time | share of an unmasked pass |
|---|---|---|
| backbone | 21.59 s | 88.4% |
| `lm_head` projection | 2.84 s | 11.6% |

Projection cost is linear in positions (16.5 µs/1k full, 16.8 µs/1k masked), so
the saving per arm follows directly:

| arm | retention | teacher-pass saving |
|---|---|---|
| v0 | 0.639 | 4.2% |
| v0ms | 0.615 | 4.5% |
| v1 | 0.560 | 5.1% |
| v2 | 0.190 | 9.4% |
| *a mask selecting nothing* | 0.000 | *11.6%* |

The claim survives — `entropy` and `kl` must project everything before they can
select, so they save 0%, and the lexical arms save 4–9%. But it is a
single-digit percentage of the teacher pass, not of training, and the paper
should give the ceiling rather than let a reviewer compute it.

Note also that TRL already does this: `_chunked_divergence_loss` sorts masked
positions to the front and projects only those, so the saving is in the baseline
trainer, not something this work adds.

**One shared cache cannot serve every arm cheaply.** The spec asks both to score
masked positions only and to cache over full spans so boundary policy stays
post-hoc. The union of all four variants has retention 0.6395 — indistinguishable
from v0's 0.6394, because v2's relational triggers sit almost entirely inside the
delimited interiors v0 keeps wholesale. So a shared cache gives v0/v1/v0ms one
teacher pass for free but charges v2 v0's price, cutting v2's saving from 9.4% to
4.2%.

## The pilot — why Phase 3 is on hold

`results/pilot.json`. Two arms, one seed, full spec size (5,000 prompts × G=4,
625 steps, 4,096 tokens), 51 GPU-hours. Dev suite, paired bootstrap over
problems.

| model | MATH500 ×4 | AMC23 ×32 |
|---|---|---|
| base (untrained) | 0.6990 | 0.4297 |
| vanilla | 0.7065 | 0.4680 |
| v0 | 0.6940 | 0.4672 |

| comparison | benchmark | Δ | 95% CI | p |
|---|---|---|---|---|
| v0 vs vanilla | AMC23 | −0.001 | [−0.038, +0.037] | 0.964 |
| v0 vs vanilla | MATH500 | −0.013 | [−0.035, +0.009] | 0.274 |
| vanilla vs base | AMC23 | +0.038 | [−0.018, +0.095] | 0.188 |
| vanilla vs base | MATH500 | +0.008 | [−0.015, +0.031] | 0.526 |

**v0 does not beat vanilla**, which is what the gradient-mass probe predicted.
But **vanilla does not beat the untrained student either**, and that is the
result that matters: there is no effect to divide between arms, so a Phase 3
null would be indistinguishable from "OPD did not train".

More steps will not fix it. Both arms' loss plateaued by the second quintile
(vanilla .373→.325, v0 .207→.178) with grad norm falling throughout, so the
models had stopped moving long before step 625. The likely cause is the one the
spec's own scope statement names: 4B→1.7B is a narrow capability gap — the pilot
shows it is below the resolution of these benchmarks at this training scale.

Two side effects worth carrying into any redesign. Responses inflate badly under
OPD — mean length 1,521→2,401 for vanilla and 1,525→2,155 for v0, with clipping
at 4,096 rising from ~2-3% to 28% and 19% respectively — so the 4,096 budget that
Phase 0 justified at 3.0% truncation ends the run at 5-10× that. And v0's mask
retention decays monotonically, 0.617→0.597; small enough to keep this run, but
it is the direction the spec warns invalidates a run, and it is not a truncation
artifact (correlation with clipping is +0.03).

## Gradient-mass coverage — the headline is negative

`results/gradmass.json`, 200 untrained-student rollouts on the Phase 0 gate set.
The paper's central claim is that a free lexical mask captures *disproportionate*
gradient mass. Measured, it captures disproportionately little.

| arm | retention | mass coverage | mass/token | Jaccard w/ entropy |
|---|---|---|---|---|
| v0 | 0.638 | 0.348 | **0.546** | 0.323 |
| v1 | 0.553 | 0.279 | 0.505 | 0.282 |
| v0ms | 0.609 | 0.326 | 0.535 | 0.308 |
| v2 | 0.187 | 0.078 | 0.414 | 0.106 |
| entropy (matched) | 0.638 | 0.942 | 1.477 | — |
| random (matched) | 0.638 | 0.547 | 0.858 | — |

A ratio of 1.0 is neutral. Every lexical variant is well below it, and a
budget-matched **random** mask captures 0.547 against v0's 0.348. Random
reproduces v0's positional profile by construction, so the gap is not
positional: within the same positions, the lexical criterion selects
lower-divergence tokens.

The mechanism is that digits and operators inside LaTeX are among the *most
predictable* tokens in a derivation. After `\frac{1}{`, the `2}` is nearly
determined; teacher and student agree and the divergence is small. The mass sits
in the reasoning prose where the models disagree.

Jaccard with entropy top-k is only 0.28–0.32, so these masks are not a zero-cost
approximation of entropy selection either — they select something genuinely
different, which happens to be the low-information half.

**Comparing to DEAR needs care.** Its quoted 39.1% / 35.9% are coverages at a
budget this note does not state, so they are not comparable to our coverages.
The budget-independent quantity is mass/token: ours are entropy 1.477, random
0.858.

This is a prediction for Phase 3, not an excuse after it: if v0 trains on tokens
the models already agree on, it should not beat vanilla or its random control.

## The retention cross-check

Phase 0's measured retention is a live invariant, not just a reported number.
The trainer logs `mask_retention` every step; if it does not sit near the Phase 0
figure for that arm (v0 0.632, v1 0.548, v0ms 0.604, v2 0.176), the student is
not generating the distribution the gate was computed on and the run is invalid.

This already caught one bug. Qwen3's chat template defaults to *thinking* mode:
with no `enable_thinking` kwarg it omits the empty `<think></think>` prefill and
the model emits its own reasoning trace. Those traces are prose-heavy and
LaTeX-sparse, so v0's retention logged at **0.24 instead of 0.63** — a different
student, a different text distribution, and none of Phase 0's gates applicable.
`loop.py` now sets `chat_template_kwargs={"enable_thinking": False}` on the
config, which the trainer threads into the generation path. Watch this number.

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

## What Phase 0 decided

1. **Gap tolerance `k=2` is not over-bridging.** The worry came from a plumbing
   run against MATH reference solutions, where run-spans averaged ~14 atomic
   units and looked bimodal. On real rollouts they are short: mean 3.05, median
   2.0, `frac_le_2 = 0.594`, only ~3% at 10 or more units. `k=2` stays.

2. **`v1` was keeping the entry tokens it exists to drop; fixed.**
   `_subtract` removed the first unit from the span's *char range*, but a
   delimited span keeps its interior wholesale, so the whitespace in front of
   the unit survived — and Qwen3 emits that space glued to the following token
   (` x`, ` a`, ` n`), which then overlapped the mask and was selected anyway.
   `v1` dropped only 49% of its entry tokens. It now subtracts
   `(span.start, units[0][1])`, which removes the padding too and leaves run
   spans untouched:

   | | before | after |
   |---|---|---|
   | entry token correctly dropped | 48.98% | **82.07%** |
   | unavoidable (token straddles a kept unit) | 22.95% | 17.92% |
   | defect (whitespace leak) | 28.07% | **0.01%** |

   The remainder is irreducible at token granularity: one token covering both
   the dropped unit and a kept one cannot be dropped. `v1` retention falls from
   0.580 to 0.548 as a result. Regression test:
   `test_v1_drops_padding_before_the_entry_unit`.

   One leak is deliberately left. `v1` still re-admits the *closing* delimiter
   on interiors padded like `$ 47 $`, because the trailing whitespace also
   survives the subtraction — so `v1` is still not a subset of `v0ms` (0.93% of
   `v1`). Closing it means `v1` keeping `units[1:]` rather than the interior
   minus a hole, which changes what the arm means; that is a decision, not a
   bug fix.

3. **v2's stop rule is strict about letters, not operators.** `= 3x + 2` does
   yield just `3`. But `=` is itself in `M_EXT`, so `= 8 + 3 = 11` runs through
   the second `=` and consumes the whole chain. Post-trigger content splits
   58.5% terminal / 30.5% rearrangement / 10.9% empty at a mean of 4.80 units,
   so v2 is not a "final answer only" mask; it is "everything from the first
   relation onward, until a word". The paper should say that.

4. **The probe path and the training path agree.** `coverage.py` builds offsets
   from `offset_mapping`, `mask_utils.py` by decoding ids one at a time. Over
   the 500 gate rollouts they disagree on 106–318 tokens per variant, 0.02–0.24%
   of selected, so the retention above is the retention Phase 3 trains at. The
   `align.py` claim that delimiters fall out for free holds at ~96%: Qwen3 emits
   ` $\` as one token on `is $\boxed{4}$` and the `\` is interior, so 3.98% of
   v0-selected tokens carry a delimiter character.

5. **Budget matching is exact.** The random-stratified baseline reproduces v0's
   per-sequence count with zero mismatch and its 20-bin positional profile to
   L1 = 0.0000, while overlapping v0 only 74.4% — a different selection, not a
   copy. Caveat: with `n_bins = 20`, a completion shorter than ~40 tokens puts
   one position per bin and the "random" mask degenerates into an exact copy of
   its target. Harmless at the real 1,504-token length; a trap if completions
   are ever shortened.
