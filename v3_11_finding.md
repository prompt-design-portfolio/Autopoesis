# v3.11 finding — the preparation world, 5 arms × 3 seeds

**Status: rows 1a, 2 and 3 pass 3/3. The claim is carried by the within-agent lines, not by the
population hit rate. Row 3b failed as an instrument and is being rebuilt.**

## The claim

> In a world where the fact to be learned sits on **every meal**, where credit is **immediate and
> two-sided**, and where **no approach behaviour** is required, the grown learner acquires a
> two-item conjunction **within a life** — and the acquisition is visible in lines that a
> population-level hit rate cannot produce: the survivor curve, which follows the same agents later
> in their own lives, and the first-preparation hit, which reads the genome before anything has
> been learned.
>
> The population hit rate does **not** carry this claim. In a six-mapping space the genes track the
> mapping by **survival sorting over standing variation**, and that baseline can be most of the
> standing number. What the learner adds has to be measured above it.

## The attribution

Two lines carry it, and both are within-agent.

**Survivor curve** (preparations 1–2 against 6–10, over agents that reached 10; every agent
contributes both halves of its own curve, so a rise is the same individuals later in their own
lives, not a different sample):

- **rising in `plastic (W2)` in 3/3 seeds**
- **falling in the controls in 3/3 seeds**

**First-preparation hit, late in the era** (the innate policy, before the agent has learned
anything, restricted to first preparations at least a third of an era after a remap so the reading
is not confounded by how recently the mapping moved):

- **`plastic (W2)` sits at its type-blind level** — its genome carries no conjunction
- **the controls sit at 0.73–0.82** — their genomes do

That inversion is the finding in one line: **the learner's genome is the *worst* of the arms, and
its standing performance is built within life; the non-learners' genomes are the best, and theirs
is built by selection.** They arrive at comparable places by opposite routes.

## Per-seed numbers

> **TO BE FILLED FROM THE PRINTED GRID SUMMARY.** These slots are labelled rather than estimated.
> I have the 3/3 verdicts and the ranges quoted above from the reading, not the per-seed table, and
> this project does not put invented figures in a finding.

| line | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| row 1a — safe rate, `plastic` − `fixed` (phase 1) | | | |
| row 1a — `probe_adv` (food) | | | |
| row 2(a) — safe rate whole phase 2, `plastic` − `fixed` | | | |
| row 2(b) — `fixed` prepared meals per life | | | |
| row 3 — prep hit, `plastic` | | | |
| row 3 — prep hit, `fixed` | | | |
| row 3 — prep hit, `scrambled` | | | |
| row 3 — `plastic` − `scrambled` | | | |
| row 3 — per-type, `plastic` A / B | | | |
| row 3 — `probe_adv` (prep) | | | |
| survivor curve — `plastic`, preps 1–2 → 6–10 | | | |
| survivor curve — `fixed` | | | |
| survivor curve — `scrambled` | | | |
| first-prep late — `plastic` (and its type-blind level) | | | |
| first-prep late — `fixed` | | | |
| first-prep late — `scrambled` | | | |
| `prep_gain` innate — `plastic` / `fixed` / `scrambled` | | | |
| population, phase 2 | | | |
| founder share of preparations | | | |

**Cap check:** recorded from the grid — no outcome arm above 90% of `max_pop` = 800 in phase 2's
second half. *(Value to be filled.)*

**The population column is not read.** Population is an outcome of the economy, not a measure of
the learner, and across v3.10–v3.11 it has twice been the thing that moved when a parameter changed
— to the cap at `prep_value` 1.5, to the floor at 1.0. It is reported and it gates row 0 at
`pop < 80`, and nothing in the claim rests on it.

## Row 3b failed as an instrument

Row 3b was to be the attribution line. It did not fail by returning a negative result — it failed
by not being able to return anything, for two independent reasons, and both are mine.

1. **A 10-step window cannot show learning that takes ~5 preparations.** The since-remap curve
   recovers by preparation 5; at 10 steps an agent makes at most one or two. The window had been
   cut to 10 precisely to stop the replayed population re-evolving, and in fixing that I cut it
   below the timescale of the thing being measured.
2. **A run-end snapshot sits mid-era and is only partly sorted.** The run ends ~300 steps into an
   era, so the population has not finished sorting to the mapping in force. `fixed` seed 0 scored
   **0.24 on its own mapping** — a genome that should be near its type-blind level on the mapping
   it was selected under.

Both faults are in the instrument, not the world. **No conclusion was drawn from row 3b, and none
should be.**

## The replacement: the frozen-population replay

Genomes are snapshotted at **every era boundary**, so the population has just lived a whole era
under that mapping and is sorted for it. One snapshot is replayed for **300 steps** with **births,
deaths and injection all disabled** — energy is tracked and spent, the metabolism runs, it is
simply not lethal — on the **matched** mapping and on a **shuffled** one, with learning **off**
(`eta 0`) and **on** (`eta 1`).

**Nothing can change over the window except `H`.** Not the population's composition, not its size,
not which lineages are present. A hit rate that moves under `eta 1` and does not move under `eta 0`
is within-life learning and can be nothing else.

**Requirement: `eta1 − eta0` ≥ 0.10 on the shuffled mapping in every seed.**

`frozen_selftest` guards it: with learning off, the hit must not move across the window
(drift ≤ 0.05) and `pop` and `max_gen` must be single-valued. That is the test the old knockout
never had — it ran on a live population, where selection moved the number and the movement was
attributed to the genome anyway.

This becomes **v3.12's D6 claim line**. The v3.11 `plastic` arm will be re-run at 3 seeds with
per-remap snapshots as an **addendum** once the v3.12 build is done; it gates nothing.

## What this closes and what it opens

**Closed.** The preparation world is readable, and within-life acquisition of a two-item
conjunction is demonstrated on within-agent lines with controls that behave. v3.9's negative on the
recipe chain is not a limit of the learner; it was a limit of the task's sparsity.

**Open.** In a six-mapping space the genetic baseline is large and cannot be removed by shortening
the era — sorting is rate-limited by how fast mismatched lineages die, not by generations, and
`prep_every` 350 was tried and reverted (the prediction failed both ways: `fixed` 0.642 against a
predicted 0.52–0.56, `plastic` 0.575 against 0.65–0.70). v3.12 attacks the mechanism on both terms:
a 60-mapping space (`T = 3, K = 5`) and a halved `prep_fail`, which slows the killing that drives
sorting while leaving the learning signal — a sign, `m = ±1` — untouched.
