# Lexical-Mask On-Policy Distillation
## Slide content, one section per slide

---

## 1. Title

**Can a regular expression choose which tokens to learn from?**

Lexical-mask on-policy distillation: experiment and results

---

## 2. The question, in one slide

When a small model learns by imitating a bigger one, it normally learns from *every*
token it produces.

Recent work says: don't. Pick a **subset** of tokens and learn only from those.
Every published method picks that subset by **running a model and reading its output
probabilities**. That costs a forward pass before you can choose.

**Our question:** can you pick the subset with a **regular expression over the text**
instead — no model, no probabilities, essentially free — and do just as well?

A clean "no" is a publishable answer. We are not trying to make it work.

---

## 3. On-policy distillation (OPD), concretely

Two models:
- **Student** — Qwen3-1.7B. The model being trained.
- **Teacher** — Qwen3-4B-Instruct-2507. Frozen, never updated.

One training step:
1. Take a batch of maths problems.
2. The **student** writes out full solutions ("rollouts"). These are the student's
   *own* words — that is what "on-policy" means.
3. At every token position of those solutions, ask both models: *what would you
   predict next here?* Each answers with a probability distribution over the whole
   151,936-token vocabulary.
4. Nudge the student's distribution toward the teacher's.

The quantity minimised at position t is a divergence D_t between the two
distributions. Summed over positions, that is the loss.

---

## 4. What "selective" OPD changes

Standard OPD sums D_t over **all** positions.

Selective OPD sums over a **chosen subset** M:

    loss = (1 / |M|) * sum over t in M of D_t

The published methods choose M by scoring: highest student uncertainty, highest
teacher-student disagreement, and so on. **All of these require model outputs before
they can choose.**

We choose M with a regex over the decoded text.

---

## 5. Definition 1 — RETENTION

Ambiguous on its own. Precisely:

    retention r = (number of completion tokens the mask selects)
                  / (number of completion tokens)

- **Completion tokens only.** The problem statement is excluded — the student did not
  write it and is not trained on it.
- **Padding excluded.** Batches are ragged; padding never counts.
- Measured **per batch and logged every step**, not once.

Why logged every step: the mask keys off LaTeX punctuation, and nothing in the loss
rewards the student for continuing to *emit* LaTeX. If its formatting drifts, the mask
quietly shrinks and the run stops being the experiment we designed.

---

## 6. Definition 2 — the TEACHER FORWARD PASS has two halves

Running the teacher over a sequence is **two different computations**, and the whole
compute argument turns on telling them apart.

1. **Backbone.** The transformer layers. Produces one 2,560-number hidden vector per
   position. *Position t depends on all positions before it* — you cannot compute a
   subset.
2. **lm_head projection.** One matrix multiply per position, 2,560 x 151,936, turning
   that hidden vector into a probability for every vocabulary token. *Independent per
   position* — you **can** compute a subset.

So "restrict the teacher pass to the masked positions" can only ever mean *skip the
projection at unmasked positions*.

Measured split: **backbone 88.4%, projection 11.6%** of the pass.

---

## 7. Definition 3 — GRADIENT-MASS COVERAGE

Each position t contributes D_t (its divergence) to the loss.

    coverage(M)  = (sum over t in M of D_t) / (sum over all t of D_t)
    mass/token   = coverage(M) / retention

**mass/token is the number to read:**
- = 1.0 — the selected tokens are exactly average.
- > 1.0 — the mask concentrates on informative tokens. Good.
- < 1.0 — the mask concentrates on tokens that matter *less* than average.

**Honest caveat.** D_t is the per-token *loss*, used as a stand-in for gradient
magnitude. It is not a per-parameter gradient norm — that is intractable to enumerate
per token. A position's loss contribution is what scales its gradient contribution,
and this is the same proxy the prior work reports.

---

## 8. Definition 4 — GATE

A **gate** is a pass/fail threshold **written down in advance**, checked before
spending the next phase's GPU hours.

Example: *"the v2 mask must select at least 1% of tokens."* If it selects 0.2%, that
arm is cut before it costs 15 GPU-hours, not after.

Gates are a budgeting device, not a result. Five were defined for the mask design;
four passed.

---

## 9. What the regex actually selects

Two ingredients:

- **Delimited regions** — anything inside `$...$`, `$$...$$`, `\[...\]`. Kept whole,
  including variable names and `\frac`. The `$` signs themselves are excluded.
- **Runs** — outside those regions, stretches of digits and operators
  `+ - * / = < > ( ) ^` etc., allowing up to two ordinary words inside a run without
  breaking it (so `f(x) = 3x + 2` stays one span).

No model is consulted. This is a regex over the decoded string, then a map from
character positions to token positions.

---

## 10. A worked example

Student text, brackets = selected by v0:

    To solve $[2][x][ +][ ][3][ =][ ][1][1]$, subtract [3] from both
    sides to get $[2][x][ =][ ][8]$. Dividing by [2] gives $[x][ =][ ][4]$.

Two things this shows:
- `12` is split by the tokenizer into `1`,`2`. The mask is built on **characters**,
  then mapped to tokens, so numbers are never half-selected.
- The prose ("subtract", "from both sides") is **not** selected. This turns out to
  matter a great deal.

---

## 11. The four mask variants

| variant | what it is |
|---|---|
| **v0** | Everything above. The headline mask. |
| **v1** | v0 minus the *first* unit of every span — tests whether the token that *enters* a formula matters. |
| **v0ms** | v0 minus spans shorter than two units — drops lone numerals in prose ("we have 3 cases"). |
| **v2** | Only what follows a relational operator (`=`, `\boxed{`, ...), up to the first word. Much smaller. |

**v1 vs v0ms** isolates the entry-token effect, because v1 differs from v0 along two
axes at once and v0ms differs along only one of them.

---

## 12. The eight arms

| arm | needs a model to choose? | role |
|---|---|---|
| `vanilla` | — | All tokens. The reference. |
| `random` | no | Random subset, matched to v0's count *and* position profile. |
| `entropy` | **yes** | Tokens where the student is most uncertain. |
| `kl` | **yes** | Tokens where student and teacher disagree most. |
| `v0` `v1` `v0ms` `v2` | **no** | The lexical masks. |

**Budget matching is the crux.** Every baseline is forced to select the *same number*
of tokens as v0, retuned on every batch. Otherwise a difference in accuracy is just a
difference in how much data each arm saw. Verified exact: baselines hit v0's count
with zero mismatch.

---

## 13. Result 1 — what the masks select

500 solutions from the untrained student, 752k tokens.

| mask | retention | from `$...$` | from runs |
|---|---|---|---|
| v0 | 0.632 | 88.4% | 11.6% |
| v1 | 0.548 | | |
| v0ms | 0.604 | | |
| v2 | 0.176 | | |

- v0 keeps roughly **two-thirds of all tokens** — it is not a sparse mask.
- **v0 is essentially "everything inside dollar signs"**: the outside-runs machinery
  contributes only 12% of what is selected.
- Four of five gates passed. The truncation gate failed (3.0% of solutions hit the
  length cap, threshold was 2%); kept by decision and reported.

---

## 14. Result 2 — the central hypothesis, tested before training

*Does the free mask concentrate on the informative tokens?* Measured on 200 solutions.

| arm | retention | coverage | **mass/token** |
|---|---|---|---|
| v0 | 0.638 | 0.348 | **0.546** |
| v1 | 0.553 | 0.279 | **0.505** |
| v0ms | 0.609 | 0.326 | **0.535** |
| v2 | 0.187 | 0.078 | **0.414** |
| random (matched) | 0.638 | 0.547 | 0.858 |
| entropy (matched) | 0.638 | 0.942 | **1.477** |

**The hypothesis fails, and not narrowly.** Every lexical mask is *below* 1.0 — they
select tokens that matter *less* than average. A budget-matched **random** mask
captures more.

---

## 14b. The inversion — a free rule that DOES concentrate the signal

If the mathematical tokens carry *less* signal than average, the tokens that are not
mathematical must carry more. Testing that, and a family of other free rules, against
the same divergence vector. Entropy and random are matched to **each rule's own
budget** (a greedy top-k concentrates harder when it picks fewer tokens, so unmatched
comparison is meaningless).

| free rule | % of tokens | mass/token | vs random | of paid entropy | overlap w/ entropy |
|---|---|---|---|---|---|
| sentence-initial word | 2.6% | **3.62** | 3.12x | 75% | 0.10 |
| connectives (so, thus, since...) | 2.9% | 3.07 | 2.42x | 65% | 0.11 |
| all words >= 2 letters | 28.7% | 1.83 | 1.48x | 63% | 0.38 |
| **v0c — the prose (NOT maths)** | 41.0% | **1.77** | 1.43x | 77% | 0.58 |
| **v0 — the maths** | 63.8% | **0.55** | **0.63x** | 37% | 0.32 |

**The sign flips.** The mask this project was built on captures *less* signal than a
random mask at the same budget (0.63x). Its complement captures 1.43x.

**Sentence-initial tokens are 2.6% of the text and carry 9.4% of the signal.** One
regex, no model, no logits.

---

## 14c. Does capturing the signal actually help? No.

Three masks spanning the whole range of mass/token, trained identically (lr 1e-5,
120 steps, **3 seeds each**) and evaluated on MATH500.

| mask | mass/token | MATH500 | over untrained |
|---|---|---|---|
| **v0 — the maths** | **0.55** (worst) | **0.7580 +/- 0.0120** | **+0.059** |
| plain OPD — all tokens | 1.00 | 0.7363 +/- 0.0090 | +0.037 |
| **v0c — the prose** | **1.77** (best) | **0.7072 +/- 0.0210** | **+0.008** |

untrained student = 0.6990

- v0c vs v0: **-0.051, p = 0.022**
- v0c vs untrained: +0.008, **p = 0.570** — prose masking barely learns at all

**Ranked by gradient mass the order is v0c > plain > v0. Ranked by accuracy it is
exactly reversed.**

---

## 14d. The actual result

**Gradient-mass coverage does not predict accuracy. Here it anti-predicts it.**

This is the metric the subfield selects masks by — DEAR's published targets are
entropy 39.1%, random 35.9%, and our own project was justified by it. On these three
arms it would have chosen **the worst one**, and that one is statistically
indistinguishable from not training.

**Why.** High divergence marks *arbitrariness* as readily as it marks information.
Prose diverges because phrasing is underdetermined: many wordings are correct, so
teacher and student disagree without either being wrong, and matching the teacher's
word choice teaches no mathematics. Notation agrees because it is *determined* — and
training there reinforces correct computation without fighting over style.

**What to take from it.** A selection metric computed from divergence needs an
accuracy check before it justifies anything. Ours did not have one until now.

*Caveat: one learning rate, one benchmark, three seeds, one teacher-student pair. The
ordering is clean; the mechanism is inferred, not demonstrated.*

---

## 15. Why — and what it predicted

**Mechanism.** Digits and operators inside LaTeX are among the *most predictable*
tokens in a derivation. After `\frac{1}{`, the `2}` is nearly determined: teacher and
student already agree, so D_t is small. The disagreement lives in the reasoning prose.

**This gave a falsifiable prediction:** v0 trains on the tokens the two models already
agree about, so it should *underperform* plain OPD.

We wrote that down before running the comparison. **It turned out to be wrong** — see
Result 4.

One useful side result: overlap between the lexical masks and the entropy-selected set
is only 0.28-0.32. They are **not** a cheap approximation of entropy selection; they
select something genuinely different.

---

## 16. Result 3 — the learning rate mattered more than the mask

Our first full comparison found **nothing**: no arm beat any other, and neither beat
the untrained model. We nearly concluded the setup was hopeless.

It was the learning rate. Same arm, 120 steps, varying only the rate:

| learning rate | final loss | % solutions cut off | MATH500 | AMC23 |
|---|---|---|---|---|
| 1e-6 (original) | 0.340 | 21% | 0.7185 | 0.4453 |
| **3e-6** | 0.280 | 54% | **0.7565** | 0.4664 |
| 1e-5 | 0.253 | 58% | 0.7260 | 0.4875 |
| 3e-5 | **0.227** | 61% | 0.7100 | 0.4961 |

**Lowest loss, second-worst accuracy.** Loss and accuracy point in *opposite*
directions past 3e-6: the model buys lower loss by writing longer and longer answers
that get cut off. **Never select a checkpoint on this loss.**

---

## 17. Result 4 — the headline comparison (18 runs, 3 seeds per cell)

v0 vs plain OPD, matched on learning rate, batch size, schedule and data order.
120 steps. **Statistics across training runs (seeds), not across problems** — the
earlier problem-level p-values held the trained model fixed and could not speak to
reproducibility.

MATH500, mean +/- SD over 3 seeds:

| learning rate | plain OPD | v0 | difference | p |
|---|---|---|---|---|
| 3e-6 | 0.7580 +/- 0.0079 | 0.7492 +/- 0.0049 | -0.0088 | 0.188 |
| 1e-5 | 0.7363 +/- 0.0090 | 0.7580 +/- 0.0120 | +0.0217 | 0.072 |
| 3e-5 | 0.7190 +/- 0.0078 | 0.7305 +/- 0.0067 | +0.0115 | 0.127 |

**Nothing is significant.** Pooled across rates the advantage is +0.0081 (p = 0.10).
Both arms peak at **exactly 0.7580**.

---

## 17b. What one seed got wrong

| | single seed | three seeds |
|---|---|---|
| difference at 1e-5 | +0.0330 | +0.0217 |
| difference at 3e-5 | +0.0280 | +0.0115 |
| p-values reported | 0.001, 0.002 | 0.072, 0.127 |

Plain OPD's first seed happened to be **unlucky** at both higher rates; v0's happened
to be **lucky**. The earlier p-values resampled *problems* with the model held fixed,
so they measured problem-sampling noise and were silent about run-to-run variance.

The 1e-5 p-value moved **0.040 -> 0.072 on one added seed**. Three seeds is fragile
for differences this size; five or more would be needed.

## 18. Robustness to learning rate — also not established

The remaining hypothesis was that the mask makes training less sensitive to the
learning rate. Loss across each arm's own best-to-worst rate:

| | best | worst | drop |
|---|---|---|---|
| plain OPD | 0.7580 @3e-6 | 0.7190 @3e-5 | 0.0390 |
| v0 | 0.7580 @1e-5 | 0.7305 @3e-5 | 0.0275 |

Difference in sensitivity: **+0.0115, 95% CI [-0.0043, +0.0273]** (bootstrap over
seeds). **The interval contains zero.**

Directionally v0 is flatter, but the study cannot establish it. Note also that the
learning rate itself moves accuracy by 0.039 — **five times larger than any
difference between the arms.**

## 19. Result 5 — the compute saving is capped

The secondary claim: because the mask is known *before* any model runs, the teacher
pass can skip unmasked positions.

True, but bounded by the backbone/projection split:

| arm | share of teacher pass saved |
|---|---|
| v0 | 4.2% |
| v2 | 9.4% |
| *a mask selecting nothing at all* | *11.6%* |
| `entropy` / `kl` | 0% |

`entropy` and `kl` save nothing because they must project *every* position before they
can choose. So the advantage is real and it is single-digit. It should be reported
with its ceiling.

Also: the training framework already skips the projection at unselected positions.
This saving is in the **baseline**, not added by us.

---

## 20. Result 6 — harder benchmarks cannot resolve this

AIME 2024/2025/2026, 30 problems each, 32 samples per problem.

| model | AIME24 | AIME25 | AIME26 |
|---|---|---|---|
| untrained | 0.107 | 0.085 | — |
| plain @3e-6 | 0.184 | — | — |
| v0 @3e-6 | 0.196 | — | — |
| plain @1e-5 | 0.173 | 0.185 | 0.149 |
| v0 @1e-5 | 0.164 | 0.164 | — |

Training clearly helps: 0.107 -> 0.16-0.20 on AIME24.

**But the benchmark cannot compare the arms, and one row proves it.** `plain @1e-5`
is a *single model* scored on three years: **0.149 to 0.185, a range of 0.036** —
larger than the 0.033 difference we are trying to detect. Year-to-year variation of
one model exceeds the effect size.

Runs were stopped at 12 of 15 cells; the missing cells would not change this.

---

## 20b. A control: does the effect depend on problem difficulty?

MATH500 labels every problem 1-5 for difficulty. If the mask helped by "focusing on
the maths", its advantage should grow with difficulty.

Correlation between difficulty level and (v0 - plain OPD), per problem:

| learning rate | correlation | 95% CI |
|---|---|---|
| 3e-6 | +0.015 | [-0.066, +0.103] |
| 1e-5 | -0.027 | [-0.119, +0.063] |
| 3e-5 | +0.044 | [-0.043, +0.129] |

**No trend.** All three intervals contain zero and the sign is inconsistent. At 1e-5
the advantage is in fact largest on the *easiest* problems (+0.058 at level 1).

This rules out the difficulty explanation and is worth reporting as a control.

## 20c. How much of this is just run-to-run luck?

The single most important control. Same configuration (plain OPD, 3e-6, 120 steps),
run three times with different random seeds. The seed changes both which problems are
sampled and the training randomness — i.e. the variance of *repeating the experiment*.

| seed | MATH500 |
|---|---|
| 0 | 0.7565 |
| 1 | 0.7665 |
| 2 | 0.7510 |
| **mean** | **0.7580** |
| **standard deviation** | **0.0079** |

Smallest difference detectable at 80% power:

| seeds per arm | detectable difference |
|---|---|
| **3** | **0.018** |
| 5 | 0.014 |
| 8 | 0.011 |
| 12 | 0.009 |

The effects we measured (0.028, 0.033) are **about 4x the seed standard deviation**
and above the 3-seed threshold. **They are not run-to-run noise.**

Two things this also settles:
- The tie at 3e-6 (-0.0045) sits well inside one standard deviation. It is a **real
  null**, not an undertrained baseline.
- **Best against best is confirmed a tie.** Plain OPD at its best rate averages
  0.7580 over three seeds; v0's best is 0.7590. A difference of 0.001.

Caveat: accuracy is much more stable across seeds than training metrics. Clipped-ratio
has a seed SD of 0.029 — *larger than the whole effect* — so single-seed differences
in answer length are not evidence of anything, which weakens the length explanation
offered earlier.

---

## 21. Practical findings worth keeping

- **Splitting one model across 4 GPUs is 3x slower than using 1 GPU** for a 1.7B model
  (1,201 vs 3,630 tokens/s). Run independent copies instead: 2.86x on 4 GPUs.
- **A silent failure that would have invalidated everything.** Qwen3 defaults to a
  "thinking" mode that writes long reasoning preambles. In that mode the mask selected
  24% of tokens instead of 63%. Caught only because we compared the live number
  against the pre-measured one.
- **A real bug in v1**: it was keeping 51% of the tokens it was designed to drop,
  because removing a character range left the whitespace in front of it and the
  tokenizer glues that space to the next token. Fixed: 49% -> 82% correctly dropped.
- Cost after tuning: **~4 GPU-hours per arm**, so the full 8-arm x 3-seed design is
  ~98 GPU-hours (was 441-680).

---

## 22. What is NOT established

- **One seed per point.** The design calls for three. Both significant results
  (p=0.001, p=0.002) rest on a single run each.
- **Two arms of eight evaluated.** `v1`, `v0ms`, `v2`, `random`, `entropy`, `kl` all
  run and are budget-matched, but only `vanilla` and `v0` have been trained and scored.
- **One benchmark carries the result.** MATH500 (500 problems) resolves the effect;
  AMC23 (40) and AIME (30) do not.
- **The mechanism is unexplained.** The stated hypothesis was falsified; the
  replacement explains part of the effect, not all of it.
- Narrow teacher-student gap (4B -> 1.7B). Strong-to-weak is untested.

**Next step:** the 8-arm x 3-seed design at 3e-6 to 1e-5, ~98 GPU-hours.

---

## 23. Summary

1. **The premise fails.** The mask built to select mathematically important tokens
   captures 0.55 mass/token — **0.63x a budget-matched random mask**, i.e. worse than
   chance.
2. **Inverting it maximises the criterion.** Prose captures 1.77x; sentence-initial
   words 3.62x from 2.6% of tokens, reaching 75% of *paid* entropy selection at only
   0.10 overlap with it — a different high-signal set, found by regex.
3. **And the inverted mask trains worst.** 0.7072 vs the maths mask's 0.7580
   (p = 0.022), and indistinguishable from no training at all (p = 0.570).
4. **So gradient-mass coverage anti-predicts accuracy** across three arms. The metric
   the field selects masks by would have picked the worst of the three.
5. Supporting: the learning rate moves accuracy 0.039 while no mask difference
   exceeds 0.022; the teacher-side compute saving is capped at 11.6%; and every
   single-seed effect shrank when seeds were added.

**The contribution is a warning about the metric, not a better mask.**
