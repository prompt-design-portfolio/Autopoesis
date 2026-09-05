## 8. Reading it

**Rows 1a and 2 are stop conditions.**

1. **Row 1a** — phase 1 must be v3.1. If not, nothing below is readable.
2. **Row 1b** — `fixed` must not hold the *conjunction*. A type-blind 0.5 is allowed; read the
   per-type split to tell them apart.
3. **Row 2** — food learning in the first mapping era **read on safe rate** (`plastic` − `fixed`
   ≥ 0.03), with the corrected `probe_adv` (food) in the first 500 steps corroborating; then
   prepared meals per life ≥ 3 in `fixed`, and P(prep | on food) against the null. The
   **eat → prep shift timing** and **prep share by era** are printed as diagnostics, not gates.
   A probe that decays *after* the first era is expected: once the mapping is known, preparation
   pays on any food and the safe/poison fact stops mattering.
4. **Row 3** — the abstention line first, then the two result lines, then **the per-type split**:
   a conjunction is *both* types above 0.5, not one at 1.0 and one at 0.
5. **Rows 3a and 3b — survivorship.** `plastic` − `scrambled` on hit rate is a *contaminated*
   contrast: agents whose random `H` happens to help live longer, enriching the standing
   population with nothing learned. It stays required, but the attribution is carried by the
   within-agent lines. The **survivor curve** (preps 1–5 vs 6–10 over agents that reached 10)
   must **rise in `plastic` and stay flat in `scrambled`** — every agent counted contributes both
   halves of its own curve, so a rise is the same individuals later in their own lives. The
   **knockout** replays late genomes with `eta_scale = 0` in a fresh world: if the advantage lives
   in `H`, both arms fall to the type-blind floor. Note `probe_adv` is computed over the *living*
   and so is itself partly survivorship-selected; the survivor curve is not. The pre-checks had
   `scrambled` at 0.722 on hit rate but only 0.342 on the probe, against plastic's 5.624 — that
   gap is what these rows are here to adjudicate.
6. **Row 5** — a null here, with rows 1–2 clean, is the first earned statement about the learner's
   limit: no approach behaviour required, credit immediate and two-sided, opportunity on every meal.

---

## 9. What the v3.10 acceptance run established (2 seeds, full phases)

These two are settled, and both are recorded here because they change what the controls mean.

### Row 1b fired, and the cause is genetic era-tracking — not luck, not a broken floor

`fixed` prep hit was 0.625 and 0.699 against a gate of 0.55. Type encounters are balanced
(share of preparations on type A: 0.499 / 0.498 in `random policy`, 0.435–0.541 elsewhere), so
the 0.5 type-blind floor constant is right and the excess is not a rig asymmetry. The elevation
is arithmetic on the per-type split:

    seed 0:  0.625 = 0.5 x 0.872 + 0.5 x 0.377

and the second term is the finding. A pure type-blind genome — "always prep_k" — scores **0** on
the wrong type. 0.377 is not 0. So the genome is not type-blind: it holds a **partial genetic
conjunction**, and the phase-half aggregate hid it because it spans two mapping eras with
different mappings, averaging a *switch* into a false one-high-one-low reading. Per era:

    fixed seed 0  (A,B)  [(0.72, 0.93), (0.56, 0.96), (0.95, 0.34), (0.77, 0.43)]
                  both types above 0.5 in 2 of 4 eras

`prep_gain innate` — the genome's preference for the correct preparation on a synthetic
observation, with no gating in it — confirms it directly: `fixed` **0.995, 1.611**, against
`random policy`'s **−0.102, −0.128**. At `prep_every` = 2000 there are ~12 generations per era
and selection uses them. Hence the change to 700.

### `scrambled`'s elevation is the same mechanism, not lucky H

`scrambled` reached 0.746 / 0.575 on prep hit, which invites reading it as survivorship — agents
whose random `H` happens to help living longer. It is not. Three lines say so:

- **Survivor halves match `fixed` to three decimals.** `scrambled` 0.815 → 0.724, `fixed`
  0.812 → 0.721. An arm whose advantage came from lucky `H` would not track the arm that has no
  `H` at all.
- **First-preparation hit is 0.821 / 0.607** — high, and that measurement is taken before the
  agent has learned anything, so it reads the genome and nothing else.
- **`prep_gain` innate is 0.859 / 0.446**, close to `fixed`'s and far above `random policy`'s
  zero, while its learned `probe_adv` is **−0.008 / 0.114** — essentially nil.

So `scrambled` is a `fixed`-like genome carrying a scrambled, useless `H`. That is exactly what
the control is supposed to be, and it means `plastic − scrambled` is a cleaner contrast than the
survivorship worry suggested — but it also means `scrambled` inherits `fixed`'s era-tracking, so
it is subject to the same gate 1b.

### The route that separates the arms

`plastic` and `fixed` reached comparable hit rates by **opposite routes**, and this is what the
whole rig is now built to read:

| | plastic | fixed |
|---|---|---|
| first-preparation hit (genome) | 0.458, 0.256 | 0.648, 0.761 |
| `prep_gain` innate (genome) | 0.324, −0.149 | 0.995, 1.611 |
| knockout, first window (genome) | 0.485, 0.646 | — |
| `probe_adv` (prep), learned | 6.52, 6.84 | n/a |
| live phase-2 hit | 0.810, 0.715 | 0.625, 0.699 |

`plastic` learns it within life on a genome that is **worse** than `fixed`'s; `fixed` evolves it
into the genome. Four instruments, two of them within-agent, all agreeing.

---

## 10. Founder dilution, and why a floor population is not a broken arm

When a population falls to `min_pop` the world injects fresh **random** genomes to hold it off the
floor. Those agents forage and prepare like anyone else, and their events land in the same
event-weighted totals as everyone's — so an arm that needs injecting has its metrics pulled toward
chance **in proportion to how badly it is doing**. A non-learner therefore reads as *more random*
the worse it does, which is a metric artifact, not a fact about the arm. That is founder dilution,
and it is what made phase-2 `fixed` look uninterpretable in the v3.11 pre-check.

The fix is at the metric, not the world. Injected agents carry `injected=True`; **their children do
not** — a founder's descendants are ordinary selected lineages and count from the first generation.
Every event-weighted number is reported **founder-free**, with the injected agents' own events
excluded: prep hit, per-type hit, survivor curve, since-remap curve, first-preparation hit and safe
rate. The **founder share of events** prints per arm and per phase so the size of the removed
dilution is visible, and the all-agents version prints alongside for this build.

**Row 0 excludes on `pop < 80` over the half and nothing else. Injections are reported, not
exclusionary.**

### The verdict this makes readable

A non-learning population sits at the floor in this world **because value comes only through
knowledge.** With `prep_value` 1.0 against `prep_fail` 0.5 the chance EV of a preparation is exactly
zero, and raw eating pays +0.10 at chance against +0.70 knowing the flip. An arm that learns
neither fact has no income to grow on. That is not a rig failure — it is **the same verdict v3.1
gave**, arrived at again in a world where the fact to be learned sits on every meal. The v3.10
`prep_value` of 1.5 had been concealing it by paying a *chance* preparation +0.167, which let
`fixed` grow to the population cap on knowledge it did not have.

The **row-0 fallback for phase 1 stays as is**: where phase-1 `fixed` is excluded on population,
that seed's row 1a reads against v3.1's published range, conservative end 0.56.

### One consequence for reading the gates

Once encounters are skewed, `max(share_A, share_B)` — not 0.5 — is the level a type-blind policy
reaches, because "always prep_k" for the commoner type beats 0.5 with no type knowledge at all.
Each arm's own type-blind level now prints beside its hit rate, and gate 1b is read against it.

---

## 11. Why gate 1b fired at `prep_every` = 700: standing polymorphism

The v3.11 acceptance had **every within-life line positive in 2/2** and gate 1b firing anyway. The
mechanism is not mutation and not within-life learning in `fixed`. It is **standing polymorphism**.

There are only **six** distinct mappings of two food types onto three preparations. A population of
several hundred carries genotypes for several of them **at the same time**. So when the mapping is
redrawn, nothing has to be invented: **lineage selection promotes whichever genotype already
matches**, and at 700 steps an era is long enough — more than a generation — for it to do so.

Two signatures, both in the printed output:

- **The per-era `(A, B)` pairs flip between eras** rather than drifting. A genome slowly acquiring
  a conjunction would improve monotonically; a population switching between standing genotypes
  shows the high type jumping from A to B and back as the mapping moves.
- **First-preparation hit of 0.67–0.72.** That is measured on an agent's very first preparation,
  before it has learned anything, so it reads the innate policy the standing population carries —
  and it is well above the type-blind level.

This is why `prep_every` goes to **350**: below a generation, so a matching lineage cannot be
selected up inside an era. It is also deliberately **not a multiple of `flip_every` = 300**, so the
fast fact (which type is safe) and the slow fact (which preparation goes with which type) do not
come into phase and cannot be tracked by a single periodic cue.

**If the gate still fires at 350, the next change is K = 4 preparations.** That takes the number of
distinct mappings from 6 to 12, which makes carrying standing genotypes for all of them
substantially more expensive — it attacks the mechanism directly rather than the time available to
it.

---

## 12. The v3.11 acceptance at `prep_every` 700, and why 350 was reverted

**Every within-life line was positive in 2/2, and gate 1b fired anyway.** The mechanism is now
measured rather than inferred, and it is not one a shorter era can beat.

### Genes track the mapping by survival sorting over standing variation

There are only **six** distinct mappings. A population of several hundred carries genotypes for
several of them at once, so a remap requires no mutation and no adaptation: **survival sorting
promotes whichever genotype already matches**, and it completes well inside a third of an era.

| measurement | value |
|---|---|
| `fixed` first-preparation hit, **late in the era** | **0.82** |
| genome-only hit (`eta = 0`), **matched** mapping | **0.844** (A 0.899, B 0.798) |
| genome-only hit (`eta = 0`), **shuffled** mapping | **0.296** (A 0.546, B 0.093) |

The pair is the whole story: the genome holds the conjunction **for the mapping it was sorted
under, and only for that one** — below chance on the swap.

### 350 shortened the learner's payoff window without touching the sorting

The pre-registered prediction failed in **both** directions:

| | predicted at 350 | observed |
|---|---|---|
| `fixed` | 0.52–0.56 | **0.642** |
| `plastic (W2)` | 0.65–0.70 | **0.575** |

So the learner lost to the non-learner, and `prep_gain innate` for `fixed` went *up* over the
change (0.149 at 700 → 0.379 at 350). Sorting is not rate-limited by generations — only by how
fast the mismatched fraction dies, which is fast. **`prep_every` is back to 700**, and the fix is
the size of the mapping space, not the speed of the world: see `spec_v3_12.md`.

### Gate 1b is now a measured genetic baseline, not a stop

The row no longer halts the reading. It reports how much of the standing hit rate the genome
already carries — `fixed`'s late first-preparation hit, the per-era A+B sum, and row 3b's
matched/shuffled genome hit with learning off — and **the learner's contribution is read above
it**.

### Row 3b is the attribution line

For each `plastic` seed, the late genomes are replayed on the **matched** mapping and on a
**shuffled** one, with learning **off** (`eta 0`) and **on** (`eta 1`), in a single 10-step window,
with `pop` and `max_gen` beside every number.

**The shuffled pair carries the claim.** On a mapping no genotype was sorted for, learning-off is
the genetic floor and learning-on is what the rule adds within a life.
**Required: `eta 1` − `eta 0` ≥ 0.10 on the shuffled mapping in every seed.**

The window is 10 steps because 50 was not short enough — `max_gen` reached 4.0 inside it, which is
four generations of selection on the pinned mapping, i.e. sorting rather than the genome.
`knockout_window_selftest` holds the line: a **non-plastic** genome's per-type hits must swap when
the mapping swaps, which they do not if the window allows re-evolution.

---

## 13. The v3.11 grid, read — and row 3b's failure as an instrument

**Rows 1a, 2 and 3 pass 3/3.** The full write-up with per-seed numbers is `v3_11_finding.md`.

### What carries the attribution

Not the population hit rate. Two within-agent lines:

- **Survivor curve** (preparations 1–2 vs 6–10, over agents that reached 10; every agent
  contributes both halves of its own curve): **rising in `plastic (W2)` 3/3, falling in the
  controls 3/3.**
- **First-preparation hit, late in the era** (the genome, before anything is learned):
  **`plastic` sits at its type-blind level; the controls sit at 0.73–0.82.**

The inversion is the result: **the learner's genome is the worst of the arms and its standing
performance is built within life; the non-learners' genomes are the best and theirs is built by
selection.** Comparable places, opposite routes.

### Row 3b failed as an instrument — no conclusion was drawn from it

It did not return a negative. It could not return anything, for two independent reasons, both mine:

1. **A 10-step window cannot show learning that takes ~5 preparations.** The since-remap curve
   recovers by preparation 5; in 10 steps an agent makes one or two. The window had been cut to 10
   to stop the replayed population re-evolving — and in fixing that, I cut it below the timescale
   of the thing being measured.
2. **A run-end snapshot sits mid-era and is only partly sorted.** The run ends ~300 steps into an
   era. `fixed` seed 0 scored **0.24 on its own mapping**, when a genome selected under that
   mapping should be near its type-blind level.

### The cap check, and the population column

The cap check is recorded: no outcome arm above 90% of `max_pop` = 800 in phase 2's second half.

**The population column is not read.** Population is an outcome of the economy, not a measure of
the learner, and across v3.10–v3.11 it was twice the thing that moved when a parameter changed — to
the cap at `prep_value` 1.5, to the floor at 1.0. It is reported, it gates row 0 at `pop < 80`, and
nothing in the claim rests on it.

## 14. Row 3b rebuilt: the frozen-population replay

Genomes are snapshotted at **every era boundary**, so the population has just lived a whole era
under that mapping and is sorted for it. One snapshot is replayed for **300 steps** with **births,
deaths and injection all disabled** — energy is tracked and spent, the metabolism runs, it is
simply not lethal — on the **matched** mapping and on a **shuffled** one, with learning **off**
(`eta 0`) and **on** (`eta 1`).

**Nothing can change over the window except `H`.** Not the population's composition, not its size,
not which lineages are present. A hit rate that moves under `eta 1` and does not move under `eta 0`
is within-life learning and can be nothing else — not sorting, not survivorship, not founder
replacement.

**Requirement: `eta1 − eta0` ≥ 0.10 on the shuffled mapping in every seed.**

`analysis.frozen_selftest` guards it, and it is the test the old knockout never had. Measured on a
`plastic` run at the last era boundary:

```
eta 0: hit 0.688 -> 0.678  (drift +0.009)   pop [300]   max_gen [0]
eta 1: hit 0.773 -> 0.892  (drift +0.118)   pop [300]   max_gen [0]
```

`pop` and `max_gen` are single-valued across the window: no agent was born, none died. With
learning off the hit does not move; with it on, it climbs.

This is **v3.12's D6 claim line**. The v3.11 `plastic` arm will be re-run at 3 seeds with per-remap
snapshots as an **addendum** once the v3.12 build is done. It gates nothing.
