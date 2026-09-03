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

## 17. Result 4 — the headline comparison

v0 vs plain OPD, **matched** on learning rate, batch size, schedule and data order.
120 steps. Significance by resampling problems (paired bootstrap).

MATH500, 500 problems, 4 samples each:

| learning rate | plain OPD | v0 | difference | p |
|---|---|---|---|---|
| 3e-6 | 0.7565 | 0.7520 | -0.0045 | 0.683 |
| 1e-5 | 0.7260 | **0.7590** | **+0.0330** | **0.001** |
| 3e-5 | 0.7100 | **0.7380** | **+0.0280** | **0.002** |

**v0 wins at two of three learning rates** — the opposite of what the gradient-mass
measurement predicted.

But: **best against best is a tie.** v0's best (0.7590) vs plain OPD's best (0.7565):
p = 0.805.

---

## 18. What that actually licenses us to claim

**Supported:** the mask makes training **less sensitive to the learning rate**. At
rates where plain OPD degrades, v0 degrades less.

**Not supported:** that the mask reaches a *higher* ceiling. Tuned against tuned, they
are indistinguishable.

**Partial explanation.** Plain OPD's accuracy falls as its answers get longer and hit
the cap. v0 inflates less (54% -> 44% cut off at 3e-6). Withholding gradient from prose
seems to damp the verbosity.

**The explanation is incomplete.** At 3e-5 both arms cut off 61% of answers —
identical — and v0 still wins by 0.028. Something else is contributing and we have not
identified it.

---

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

## 20. Result 6 — harder benchmarks (partial)

AIME 2024/2025/2026, 30 problems each, 32 samples per problem.

| model | AIME24 | AIME25 | AIME26 |
|---|---|---|---|
| untrained | 0.107 | 0.085 | — |
| plain @3e-6 | 0.184 | — | — |
| v0 @3e-6 | 0.196 | — | — |
| plain @1e-5 | 0.173 | 0.185 | 0.149 |
| v0 @1e-5 | 0.164 | 0.164 | — |

**Underpowered, as expected.** With 30 problems, a 95% interval is roughly +/-0.10 —
wide enough to contain both zero and the effect MATH500 resolves. Training clearly
helps (0.107 -> 0.16-0.20); the v0-vs-plain ordering flips between years and **should
not be read as signal**.

Runs still in progress; table will be completed.

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

1. A regex-chosen token subset **does** match, and at two of three learning rates beat,
   learning from all tokens — on the one benchmark with enough problems to tell.
2. It does so **despite** concentrating on *less*-informative tokens than a random
   subset. The stated hypothesis was wrong, and the working replacement is incomplete.
3. Best-tuned against best-tuned, the two are **tied**. The honest claim is robustness
   to learning rate, not a higher ceiling.
4. The compute advantage is real and **capped at 11.6%**.
5. All of it is one seed. Replication is the next expense, and it is now affordable.
