# Results

Every number here is reproducible from the JSON in this directory. Student
Qwen3-1.7B (non-thinking), teacher Qwen3-4B-Instruct-2507, DAPO-Math-17K (en),
T=1.0/top_p=1.0, 4,096 max completion. **One seed per point** — the spec asks
for three, and no claim below survives that caveat unchanged.

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

## 6. AIME 24/25/26 (×32)

| model | AIME24 | AIME25 | AIME26 |
|---|---|---|---|
| base | 0.1073 | 0.0854 | 0.0740 |
| van3e6 | 0.1844 | 0.1625 | 0.1406 |
| v03e6 | 0.1958 | — | — |
| van1e5 | 0.1729 | 0.1854 | 0.1490 |
| v01e5 | 0.1635 | 0.1635 | — |

**Underpowered at this model scale, as the spec anticipated.** 30 problems per
year gives 95% CIs of roughly ±0.10 — wide enough to contain both zero and the
+0.03 effect MATH500 resolves. Training clearly lifts AIME over base
(0.107 → 0.16–0.20 on AIME24), but the v0-vs-vanilla ordering is inconsistent
across years and should not be read as signal.

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

- **Three seeds.** Everything above is one seed. The two significant results (p=0.001, 0.002) need replication.
- **Six of eight arms untested for accuracy.** `v1`, `v0ms`, `v2`, `random`, `entropy`, `kl` all run and are budget-matched, but only `vanilla` and `v0` have been trained to completion and evaluated.
- **OlympiadBench and HMMT** are wired into the harness but not run.
- **`v1`'s trailing-delimiter leak** (0.93% of v1) is documented, not fixed — closing it changes what the arm means.
