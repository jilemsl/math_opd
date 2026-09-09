# Results

**Headline: gradient-mass coverage — the metric used to select and justify token
masks — anti-predicts accuracy.** Three masks spanning 0.55–1.77 mass/token, trained
identically (lr 1e-5, 120 steps, 3 seeds) on MATH500:

| mask | mass/token | MATH500 | over untrained |
|---|---|---|---|
| v0 (maths) | 0.55 *(worst)* | **0.7580 ± 0.0120** | +0.059 |
| plain OPD (all tokens) | 1.00 | 0.7363 ± 0.0090 | +0.037 |
| v0c (prose) | 1.77 *(best)* | 0.7072 ± 0.0210 | +0.008 |

v0c vs v0: −0.051, p=0.022. v0c vs untrained: p=0.570. Ranked by mass the order is
v0c > plain > v0; by accuracy it is exactly reversed. See `results/mass_vs_accuracy.json`.

**What survives a change of learning rate, and what does not** (§0b). The *budget* null is
robust: a uniform random mask matches full-token OPD at every retention from 0.20 to 1.00, and
at both 1e-5 and 3e-6. The *selection* effect is not: math_only beats a budget-matched random
mask by +0.024 at 1e-5 and by −0.006 at 3e-6. Every arm at either rate tops out at 0.758, so
the mask recovers what an over-large rate loses rather than beating a tuned baseline. The
anti-prediction above stands as a statement about the metric, not as a better mask.

## 0b. Matched-budget control: budget is free at every rate, selection only at 1e-5

`results/retention_sweep.json`. A **flat-rate random mask at rate r** gives gradient to
`round(r·n)` of each rollout's `n` completion tokens, drawn uniformly without replacement and
redrawn every optimizer step — so it holds the *budget* fixed while making the *selection*
uninformative. Measured retention over 120 steps: 0.6399 against a 0.64 target.

lr 1e-5 constant, 120 steps, 32 rollouts/step, 3 seeds, MATH500 ×4.

### The budget axis: swept 0.20 → 1.00, nothing happens

Uniform random selection at each rate, so only the budget varies.

| retention | MATH500 | seeds | vs vanilla@1.00 | p |
|---|---|---|---|---|
| 0.20 | 0.7348 ± 0.0053 | 2 | −0.0016 | 0.821 |
| 0.41 | 0.7220 ± 0.0031 | 3 | −0.0143 | 0.098 |
| 0.64 | 0.7337 ± 0.0081 | 3 | −0.0027 | 0.722 |
| 0.85 | 0.7348 ± 0.0088 | 2 | −0.0016 | 0.861 |
| 1.00 | 0.7363 ± 0.0090 | 3 | — | — |

Total span **0.0143** across a 5× range of budget, against a seed SD of ~0.008, and
non-monotone — 0.20 sits above 0.41. No cell is significantly below full-token OPD. **How many
tokens carry gradient does not measurably matter.** Four fifths of the completion can be
dropped at random for free.

### The selection axis: same budget, different tokens

| budget | lexical mask | mass/token, lexical vs random | Δ vs random | p |
|---|---|---|---|---|
| 0.64 | math_only | 0.55 vs 0.86 | **+0.0243** | **0.052** |
| 0.41 | text_only | 1.77 vs 1.24 | −0.0148 | 0.345 |

| arm | retention | mass/token | MATH500 |
|---|---|---|---|
| v0 / math_only | 0.638 | 0.55 | **0.7580 ± 0.0120** |
| random @0.64 | 0.640 | 0.86 | 0.7337 ± 0.0081 |
| vanilla | 1.000 | 1.00 | 0.7363 ± 0.0090 |
| untrained | — | — | 0.6990 |

Against vanilla the decomposition is exact:
math_only − vanilla = (math_only − random@0.64) + (random@0.64 − vanilla)
= +0.0243 + (−0.0027) = +0.0216.

**Selection is the only axis here that moves accuracy, and it moves opposite to gradient-mass
coverage**: the lower-divergence selection beats random, the higher-divergence one loses to it.
The text_only contrast is directionally consistent but not significant on its own (its SD is
0.021 — one seed came in at 0.683).

In context: OPD buys +0.059 over the untrained student. A budget-matched random mask captures
+0.035 of that; math_only captures all of it. The regex recovers the ~41% an uninformative
selection leaves on the table — while capturing *less* divergence than the mask it beats.

This also removes the obvious objection to §0, where the three arms differed in budget as well
as selection. Here budget is held fixed and the lower-mass selection still wins: the metric
anti-predicts **within** a matched budget.

### The same control at 3e-6: budget replicates, selection does not

3 seeds per cell, otherwise identical.

| lr | selection (math_only − random@0.64) | p | budget (random@0.64 − vanilla) | p |
|---|---|---|---|---|
| 3e-6 | **−0.0058** | 0.173 | −0.0030 | 0.599 |
| 1e-5 | **+0.0243** | 0.052 | −0.0027 | 0.722 |

**The budget result is rate-robust** — zero at both rates, on top of the flat 0.20→1.00 sweep.
**The selection result is not.** It exists at 1e-5 and is gone, slightly negative, at 3e-6.

Cells at 3e-6: math_only 0.7492 ± 0.0045, random@0.64 0.7550 ± 0.0041, vanilla 0.7580 ± 0.0079.

The pattern across both rates: vanilla scores 0.7580 at 3e-6 and degrades to 0.7363 at 1e-5;
math_only scores 0.7580 at 1e-5. **Every arm at either rate tops out at the same 0.758.** The
mask recovers ground an over-large rate loses — it does not add anything a well-tuned baseline
lacks. This is consistent with the length-inflation reading (§4): math_only clips at 0.569 vs
vanilla's 0.584 at 1e-5, and the gap closes at lower rates.

**Caveat.** p=0.052 at 1e-5 is marginal, though the effect clears the 3-seed minimum detectable
difference of 0.018. random@0.20 and random@0.85 have 2 seeds, not 3.

## 1. Headline: v0 vs vanilla — 18 runs, 3 seeds per cell

Statistics at the **seed** level (Welch t on seed means). MATH500, mean ± SD over 3 seeds.

| lr | vanilla | v0 | Δ | p |
|---|---|---|---|---|
| 3e-6 | 0.7580 ± 0.0079 | 0.7492 ± 0.0049 | −0.0088 | 0.188 |
| 1e-5 | 0.7363 ± 0.0090 | 0.7580 ± 0.0120 | +0.0217 | 0.072 |
| 3e-5 | 0.7190 ± 0.0078 | 0.7305 ± 0.0067 | +0.0115 | 0.127 |

**Null.** Pooled across rates: +0.0081, p=0.10. Both arms peak at exactly **0.7580**
(base 0.6990). LR-sensitivity difference +0.0115, 95% CI [−0.0043, +0.0273] — contains
zero.

**What the single-seed data got wrong.** Differences were +0.0330 / +0.0280 at one
seed, now +0.0217 / +0.0115. Vanilla's seed 0 was unlucky at both higher rates, v0's
lucky. The earlier p-values (0.001, 0.002) resampled *problems* with the model fixed,
so they said nothing about run-to-run variance. The 1e-5 p moved 0.040 → 0.072 on one
added seed: n=3 is fragile for effects this size.

**What stands.** OPD works (+0.059 over base, far outside the 0.0079 seed SD); the
learning rate moves accuracy 0.039, five times any arm difference; and a zero-cost
regex matches full-token OPD everywhere. That last is the defensible claim — *no loss*
from regex selection, not a gain.

## 2. Learning-rate sweep (vanilla, 120 steps, constant LR)

| lr | loss@120 | clipped@4096 | MATH500 | AMC23 | p vs base (MATH500) |
|---|---|---|---|---|---|
| 1e-6 | 0.3403 | 0.209 | 0.7185 | 0.4453 | 0.086 |
| 3e-6 | 0.2804 | 0.544 | 0.7565 | 0.4664 | 0.000 |
| 1e-5 | 0.2531 | 0.584 | 0.7260 | 0.4875 | 0.051 |
| 3e-5 | 0.2271 | 0.613 | 0.7100 | 0.4961 | 0.431 |

Loss is **anti-correlated** with MATH500 accuracy past 3e-6: the lowest-loss run
(3e-5, 0.227) is second-worst on MATH500. Lower JSD is bought with length
inflation, and length inflation costs accuracy. Do not select on loss.

## 3. The first pilot was undertrained

625 steps at lr 1e-6 with linear decay, 5,000 prompts × G=4, 51 GPU-hours.

| model | MATH500 | AMC23 |
|---|---|---|
| base (untrained) | 0.6990 | 0.4297 |
| pilot vanilla, 625 steps @1e-6 | 0.7065 | 0.4680 |
| pilot v0, 625 steps @1e-6 | 0.6940 | 0.4672 |

Nothing there was significant (vanilla vs base: MATH500 p=0.526, AMC23 p=0.188).
**120 steps at 3e-6 beats 625 steps at 1e-6 on both benchmarks** and clears
significance (MATH500 +0.0575, p=0.000). The null was a learning rate, not an
exhausted teacher-student gap.

## 4. Gradient-mass coverage — the prediction that failed

200 untrained-student rollouts. `results/gradmass.json`.

| arm | retention | mass coverage | mass/token |
|---|---|---|---|
| v0 | 0.6380 | 0.3483 | **0.546** |
| v1 | 0.5532 | 0.2792 | **0.505** |
| v0ms | 0.6092 | 0.3259 | **0.535** |
| v2 | 0.1870 | 0.0775 | **0.414** |
| entropy | 0.6380 | 0.9424 | **1.477** |
| random | 0.6380 | 0.5471 | **0.858** |

A ratio of 1.0 is neutral. Every lexical mask is well below it and budget-matched
random beats them all. This predicted v0 would *underperform* vanilla. **It did
not** — v0 wins at two of three rates. So gradient-mass coverage is not what
determines the outcome here, and the paper should not lead with it as the mechanism.
§0b sharpens this: at a *single* fixed budget of 0.64, v0 (mass/token 0.55) beats the
budget-matched random mask (0.86) by +0.024, so the metric anti-predicts within a matched
budget, not only across arms spending different amounts.

A partial replacement: v0 inflates less than vanilla (clip 0.438 vs 0.544 at 3e-6,
0.569 vs 0.584 at 1e-5), and accuracy falls as inflation rises. Withholding
gradient from prose appears to damp verbosity. **Incomplete**: at 3e-5 both clip at
0.613 and v0 still wins by +0.028.

Jaccard with budget-matched entropy top-k is 0.28–0.32 (v2: 0.11), so the lexical
masks are not a cheap approximation of entropy selection.

## 5. Teacher-forward saving is capped at 11.6%

`results/phase1.json`. The teacher *forward* cannot be restricted to masked
positions — a causal backbone needs every position. Only the `lm_head` projection
can be skipped, and for Qwen3-4B that is 11.6% of the pass (backbone 88.4%).

| arm | retention | teacher-pass saving |
|---|---|---|
| v0 | — | 4.2% |
| v0ms | — | 4.5% |
| v1 | — | 5.1% |
| v2 | — | 9.4% |
| ceiling_zero_retention | — | 11.6% |

`entropy`/`kl` save 0% (they must project everything to select), so the advantage
is real but single-digit. TRL's `_chunked_divergence_loss` already does the masked
projection, so this saving is in the baseline trainer, not added by this work.

## 6. AIME 24/25/26 — 3 seeds per arm

`results/aime_seeds.json`. math_only vs vanilla, both lr 1e-5 constant, 120 steps, 3 training
seeds each. 30 problems × 32 samples per year, eval seed fixed at 0 so cells differ only by
training seed.

| year | vanilla | math_only | Δ | p |
|---|---|---|---|---|
| AIME24 | 0.1688 ± 0.0053 | 0.1754 ± 0.0075 | +0.0066 | 0.289 |
| AIME25 | 0.1774 ± 0.0069 | 0.1732 ± 0.0084 | −0.0042 | 0.543 |
| AIME26 | 0.1455 ± 0.0044 | 0.1472 ± 0.0102 | +0.0017 | 0.810 |

Untrained base: 0.1073 / 0.0854 / 0.0740.

**Null on all three years**, signs inconsistent, every |Δ| ≤ 0.008. Training lifts AIME well
clear of base (0.107 → ~0.17 on AIME24), but **the MATH500 math_only advantage at 1e-5
(+0.0217) does not transfer.** Consistent with §0b's 3e-6 control: the mask recovers ground an
over-large rate loses rather than adding a transferable gain.

**Correction to an earlier claim.** This section previously called AIME underpowered from a
±0.10 binomial CI. That is the CI on *one model's absolute accuracy* over 30 problems — the
wrong statistic for a contrast. Both arms are scored on the same 30 problems, so problem
difficulty is common to both and cancels. The limiting quantity is the training-seed SD,
measured here at **0.004–0.011**, giving a 3-seed MDE near **0.03** — wider than MATH500's
0.018, but far tighter than 0.10. AIME is precise enough to have seen a MATH500-sized effect.
It isn't there.

## 6b. Seed noise floor — is any of this real?

`results/seed_noise.json`. Plain OPD, 3e-6, 120 steps, run three times with different
seeds (seed varies both the prompt sample and training randomness).

| seed | MATH500 |
|---|---|
| 0 | 0.7565 |
| 1 | 0.7665 |
| 2 | 0.7510 |
| mean | 0.7580 |
| **SD** | **0.0079** |

Minimum detectable difference at 80% power: **3 seeds → 0.018**, 5 → 0.014, 8 → 0.011,
12 → 0.009.

The measured effects (0.028, 0.033) are ~4x the seed SD and clear the 3-seed
threshold. **They are not run-to-run noise.** Two corollaries:

- The tie at 3e-6 (−0.0045) is inside one SD — a genuine null, not an undertuned
  baseline.
- Best-vs-best is a tie against a proper estimate: plain OPD at 3e-6 averages 0.7580
  over three seeds, v0's best is 0.7590.

Accuracy is far more stable across seeds than training metrics: final-loss SD 0.0067,
clipped-ratio SD 0.0290. The clipping SD exceeds the whole accuracy effect, so
single-seed differences in answer length are not evidence of a systematic difference —
this weakens the length-regularization explanation.

## 6c. Difficulty control

MATH500 labels problems 1–5. Correlation between level and (v0 − plain OPD):
**+0.015 / −0.027 / +0.044** at 3e-6 / 1e-5 / 3e-5, all 95% CIs containing zero, sign
inconsistent. At 1e-5 the advantage is largest on the *easiest* problems. The effect
does not grow with difficulty.

## 7. Cost model

`results/gpu_scaling.json`, `results/batch_sweep.json`.

- Tensor parallelism is **3× slower than one GPU** for a 1.7B model (TP=4: 1,201 tok/s vs 3,630). Use data parallelism: 1.85× at 2 GPUs, 2.86× at 4.
- Batching helps 1.84× at best and flattens past 64 rollouts/step.
- At 120 steps and 32 rollouts/step an arm costs ~4.1 GPU-h, so 8 arms × 3 seeds ≈ **98 GPU-h**. At the original 625 steps it was 441–680.
- Cluster gotchas: set `GLOO_SOCKET_IFNAME=lo`/`NCCL_SOCKET_IFNAME=lo` (four interfaces per node, gloo picks a routed one and blocks 1800s), derive `MASTER_PORT` from the job id (co-located jobs collide on 29500), and never import `trl`+`vllm` on the login node (QEMU CPU, segfaults).

## What is not done

- **Three arms untested for accuracy.** `v1`, `v0ms`, `v2` and the budget-matched
  `entropy` / `kl` selectors all run and are budget-matched, but were never trained
  to completion. Only `vanilla`, `v0`, `v0c` and flat-rate `random` have accuracy numbers.
- **OlympiadBench and HMMT** are wired into the harness but not run.
- **`v1`'s trailing-delimiter leak** (0.93% of v1) is documented, not fixed — closing it changes what the arm means.
- **One model pair, one training set.** Qwen3-4B-Instruct-2507 (4.02 B) → Qwen3-1.7B
  (2.03 B) on DAPO-Math-17K (en); MATH500 is the only benchmark that resolves the
  effect sizes involved.
