# v3.11 finding — the preparation world, 5 arms × 3 seeds

**Rows 1a, 2 and 3 pass 3/3. The claim is carried by two within-agent lines. Row 3b failed as an
instrument and no conclusion is drawn from it.** Source: `v3_11_grid_summary.txt`.

## The claim

> In a world where the fact to be learned sits on **every meal**, where credit is **immediate and
> two-sided**, and where **no approach behaviour** is required, the grown learner acquires a
> two-item conjunction **within a life**. The acquisition is visible in lines a population-level
> hit rate cannot produce: the **survivor curve**, which follows the same agents later in their own
> lives, and the **first-preparation hit late in the era**, which reads the innate policy before
> anything has been learned.
>
> The population hit rate does **not** carry the claim. In a six-mapping space the genes track the
> mapping by survival sorting over standing variation, and that baseline is most of the standing
> number: `fixed` reaches 0.596–0.666 with no plasticity at all.

## Per-seed numbers

| line | seed 0 | seed 1 | seed 2 | verdict |
|---|---|---|---|---|
| **row 1a** safe rate, `plastic` phase 1 | 0.626 | 0.672 | 0.657 | — |
| row 1a `plastic` − published `fixed` 0.56 | +0.066 | +0.112 | +0.097 | **PASS 3/3** |
| row 1a `probe_adv` (food) | 2.220 | 1.933 | 3.279 | ≥ 1.0 3/3 |
| **row 2(a)** whole-phase FF safe, `plastic` | 0.569 | 0.650 | 0.631 | — |
| row 2(a) `fixed` | 0.512 | 0.490 | 0.478 | — |
| row 2(a) difference | +0.057 | +0.160 | +0.153 | **PASS 3/3** |
| **row 2(b)** `fixed` prepared meals per life | 8.48 | 8.32 | 7.17 | **PASS** (≥ 3) |
| **row 3** prep hit `plastic` (FF) | 0.747 | 0.730 | 0.719 | — |
| row 3 prep hit `fixed` | 0.596 | 0.666 | 0.626 | — |
| row 3 prep hit `scrambled` | 0.655 | 0.589 | 0.603 | — |
| row 3 prep hit `random policy` | 0.331 | 0.326 | 0.343 | ≈ chance 0.333 |
| row 3 prep hit ceiling | 0.762 | 0.702 | 0.709 | reference |
| row 3 `plastic` − `fixed` | +0.151 | +0.064 | +0.093 | **≥ 0.03 3/3** |
| row 3 `plastic` − `scrambled` | +0.092 | +0.141 | +0.115 | **≥ 0.03 3/3** |
| row 3 `plastic` per type A | 0.672 | 0.718 | 0.711 | both > 0.5 3/3 |
| row 3 `plastic` per type B | 0.811 | 0.741 | 0.726 | both > 0.5 3/3 |
| row 3 `probe_adv` (prep) | 4.015 | 2.529 | 5.021 | > 0 3/3 |
| row 3 abstention, prep/life ratio | 0.69 | 0.99 | 1.21 | **fires 0/3** |
| **survivor curve** `plastic` | +0.117 | +0.043 | +0.110 | **rising 3/3** |
| survivor curve `fixed` | −0.130 | −0.120 | −0.135 | falling 3/3 |
| survivor curve `scrambled` | −0.134 | −0.135 | −0.130 | falling 3/3 |
| survivor curve `random` / ceiling | −0.044 | | +0.210 | (means) |
| **first-prep late** `plastic` | 0.507 | 0.506 | 0.552 | at type-blind |
| its own type-blind level | 0.540 | 0.510 | 0.513 | — |
| first-prep late `fixed` | 0.757 | 0.774 | 0.755 | +0.22 to +0.26 over level |
| first-prep late `scrambled` | 0.817 | 0.749 | 0.730 | +0.19 to +0.31 over level |
| `prep_gain` innate `fixed` | 0.795 | 0.506 | 0.351 | — |
| `prep_gain` innate `scrambled` | 0.360 | 0.329 | 0.429 | — |
| `prep_gain` innate `plastic` | 0.308 | 0.555 | 0.296 | — |
| `prep_gain` innate `random` / ceiling | −0.003 / −0.029 | −0.034 / −0.003 | −0.010 / −0.045 | the zero |
| population, phase 2 `plastic` | 719 | 510 | 733 | **not read** |
| founder share of preparations `plastic` | 0.000 | 0.004 | 0.000 | — |
| founder share `fixed` / `scrambled` | 0.003 / 0.021 | 0.007 / 0.015 | 0.011 / 0.019 | — |

**Since-remap curve, preparation 1 / 5 / 10:** `plastic` **0.413 / 0.827 / 0.951** — full curve
[0.413, 0.551, 0.686, 0.768, 0.827, 0.870, 0.899, 0.917, 0.936, 0.951]. `fixed` 0.583 / 0.660 /
0.666; `scrambled` 0.561 / 0.655 / 0.675; `random` 0.319 / 0.330 / 0.349; ceiling 0.227 / 0.679 /
0.925.

**`plastic` starts a new mapping BELOW both controls and ends far above them.** The pre-registered
prediction — dip at 1, recovery within ~5 preparations — landed: 0.413 → 0.827 by preparation 5.

**Cap check: FAILED.** `plastic` reaches 92% of `max_pop` = 800 and the ceiling 100%. **The
population column is not read**, and nothing in the claim rests on it.

## Three things the grid does not support, which I had stated more cleanly than it warrants

**1. "The learner's genome is the worst of the arms" holds on one instrument, not both.**

On **first-preparation hit late** it is unambiguous: `plastic` 0.507–0.552, sitting *at* its own
type-blind level (−0.033, −0.004, +0.039), while the controls sit 0.19–0.31 above theirs. On
**`prep_gain` innate** it does not hold: `plastic` 0.308 / 0.555 / 0.296 against `scrambled` 0.360 /
0.329 / 0.429 and `fixed` 0.795 / 0.506 / 0.351 — `plastic` is *below* both controls in seeds 0 and
2 and **above `fixed` in seed 1**.

The two genome instruments disagree, and I do not know why. `prep_gain` innate is computed over the
living at each window against the current mapping; first-prep late is an actual behaviour on an
agent's first preparation. **The claim rests on first-prep late and the since-remap curve, not on
`prep_gain` innate**, and the disagreement is recorded as open rather than resolved.

**2. The per-type test does not separate `plastic` from `fixed`.** Both have *both* types above 0.5
in 3/3 seeds (`scrambled` 2/3, ceiling 3/3). At six mappings `fixed`'s genome holds the conjunction
too — which is the v3.11 result, not a defect — so "both types above the type-blind level" is a
test of the *task*, not of the learner. The learner is separated by the margin (`plastic` − `fixed`
+0.064 to +0.151) and by the within-agent lines.

**3. The survivor-conditioned since-remap curve is negative in every arm, and most negative in
`plastic`.** random −0.175, `fixed` −0.490, `scrambled` −0.467, **`plastic` −0.600**, ceiling
−0.767. It is marked corroborating-only, and it corroborates nothing as built. The likely reason is
the window: it compares 4 preparations before a remap with 4 after, and the since-remap curve shows
recovery takes **~5**. So the measure is structurally unable to see recovery, and the ordering
(worst for the arms that had learned most) is what a too-short window would produce. **It should be
widened or dropped before it is read again.**

## Row 3b failed as an instrument

Not a negative result — it could not return one. Two faults, both mine: a **10-step window** cannot
show learning that takes ~5 preparations, and a **run-end snapshot** sits mid-era and is only partly
sorted (`fixed` seed 0 scored **0.240** on its own mapping). Its numbers — `plastic` shuffled gaps
+0.030, +0.045, +0.108 — are recorded for completeness only and **no conclusion is drawn from
them**.

**Consequence for v3.12.** The frozen replay never ran on v3.11, so **there is no v3.11 reference
value for spec_v3_12's "the shuffled gap must not shrink" clause.** That reference comes from the
addendum: the v3.11 `plastic` arm re-run at 3 seeds with per-remap snapshots, once the v3.12 build
is done. Until it exists, the non-shrink clause cannot be evaluated.

## The replacement

Era-boundary snapshots; replay 300 steps with **births, deaths and injection disabled** — energy
tracked and spent, not lethal — on the matched and a shuffled mapping, at `eta 0` and `eta 1`.
Nothing can change but `H`. **Required: `eta1 − eta0` ≥ 0.10 on shuffled in every seed.**
`frozen_selftest` guards it: with learning off the hit must not move, and `pop` and `max_gen` must
be single-valued.

## Two rig changes that earned their place

**The two-condition abstention rule.** `plastic`'s prep/life ratio is **0.69** in seed 0 — under the
old single-condition rule abstention would have fired. It does not fire, because `plastic`'s hit
(0.747 vs 0.596) and population (719 vs 317) are both *higher*. Preparing less while scoring and
living better is a learner declining bad bets.

**Founder-free metrics.** `random policy` carries a founder share of **0.373** of its preparations;
the outcome arms carry 0.000–0.021. Without the correction the null arm would have been the one most
pulled toward chance — in the direction that flatters every comparison against it.

## What this closes and what it opens

**Closed.** The preparation world is readable, and within-life acquisition of a two-item conjunction
is demonstrated on within-agent lines with controls that behave. v3.9's negative on the recipe chain
was a limit of the task's sparsity, not of the learner.

**Open.** At six mappings the genetic baseline is large — `fixed` 0.596–0.666 against a type-blind
level of ~0.51 — and cannot be removed by shortening the era: sorting is rate-limited by how fast
mismatched lineages die, not by generations, and `prep_every` 350 was tried and reverted. v3.12
attacks both terms: a 60-mapping space (`T = 3`, `K = 5`) and `prep_fail` halved, which slows the
killing that drives sorting while leaving the learning signal — a sign, `m = ±1` — untouched.
