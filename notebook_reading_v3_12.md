## Reading v3.12

**The world changed to break one mechanism, and the pre-check says it did.**

In v3.10–v3.11 the mapping space held **six** mappings. A population of several hundred carried
genotypes for several at once, so a remap needed no adaptation: **survival sorting promoted a
matching genotype**, well inside a third of an era. Shortening the era did not help — sorting is
rate-limited by how fast mismatched lineages die, not by generations — and `prep_every` 350 was
tried and reverted.

v3.12 attacks both terms of that mechanism at once:

- **`T = 3`, `K = 5` → 60 mappings.** Ten times the space.
- **`prep_fail` 0.5 → 0.25** with `prep_value = (K−1)·prep_fail = 1.0`, so the chance EV of a
  preparation is still exactly zero and income to a knowing agent is unchanged at +1.00 a meal.
  **Sorting runs on energy** — a mismatched genotype pays `prep_fail` on 4 preparations in 5 — but
  **the learning signal is a sign**, `m = ±1` as literals, independent of both parameters. So this
  slows selection and leaves learning untouched.

### Read in this order

1. **Row 0** — `pop < 80` only. Injections are reported, not exclusionary; founder dilution is
   handled at the metric.
2. **Row 1a** — the phase-1 gate against v3.1. Phase 1 is unchanged: two types, five actions. The
   third food type arrives at the switch and is **inedible raw**.
3. **Row 1b — a measured genetic baseline, not a stop.** How much of the standing hit rate the
   genome already carries. The learner's contribution is read *above* it.
4. **Row 1c — standing variation (D2).** The premise of this world, measured rather than argued:
   how many of the 60 mappings the population would score above type-blind on. **A handful means
   the space exceeds standing variation. A large number means v3.12 has not achieved what it was
   built for, and that is the finding whatever row 3b says.**
5. **Row 2** — rig checks. 2(a) on whole-phase founder-free safe rate.
6. **Row 3** — the outcome rows, then the survivorship diagnostics.
7. **Row 3b — the claim line.** The frozen-population replay.

### Row 3b is the claim

Genomes snapshotted at an **era boundary**, so the population has just lived a whole era under that
mapping. Replayed 300 steps with **births, deaths and injection disabled** — energy tracked and
spent, simply not lethal — on the **matched** mapping and a **shuffled** one (with `T = 3` the
shuffle is a derangement), at `eta 0` and `eta 1`. **Nothing can change but `H`.**

**Required: `eta1 − eta0` ≥ 0.10 on the shuffled mapping in every seed, and the gap must not shrink
from v3.11's measured value.**

`frozen_selftest` holds it: with learning off the hit must not move across the window, and `pop` and
`max_gen` must be single-valued.

### What the pre-check already established (1 seed × 3000-step phases)

| | v3.11 (6 mappings) | v3.12 (60 mappings) |
|---|---|---|
| `fixed` prep hit | 0.625–0.699 | **0.361** |
| its own type-blind level | 0.563 | 0.350 |
| gap above type-blind | +0.06 to +0.14 | **+0.011** |
| per-era per-type sum, `fixed` | ~0.99–1.36 | **0.79** |

**The genetic baseline has collapsed.** Standing variation covers only **6.8–11.1 of 60** mappings,
with essentially none held by ≥20 carriers.

Two pre-check adjustments, both judged on `fixed` only:

- **`spawn_per_patch` 3.0 → 6.0.** D5's readability criterion (≥ 3 preparations per *type* per era)
  fails at 3.0 once the food is split three ways. Raised by density, never by era length.
- **`type_spawn_w` left unused.** Type C's *standing* share does run to ~0.50 — it has one
  consumption route where A and B have two — but the quantity the experiment reads is the
  **preparation share per type**, which is balanced at equal spawn (2.9 / 3.2 / 3.1), and both
  weighted settings made it worse. The type-blind level is computed from preparation shares.
