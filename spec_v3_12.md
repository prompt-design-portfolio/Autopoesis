# v3.12 — the preparation world with a mapping space that exceeds standing variation

**Spec only. No code. DECISION points are flagged and none is settled here.**

## Why the world has to change

v3.11 established the mechanism that has been defeating gate 1b, and it is not one a shorter era
can beat.

In a **six-mapping** space, a population of several hundred carries genotypes for several mappings
at once. A remap therefore requires no mutation and no adaptation: **survival sorting over standing
variation promotes whichever genotype already matches**, and it completes well inside a third of an
era. Three measurements say so:

| measurement | value | what it says |
|---|---|---|
| `fixed` first-preparation hit, late in era | **0.82** | the innate policy of the standing population already tracks the mapping |
| genome-only hit, matched mapping (`eta = 0`) | **0.844** (A 0.899, B 0.798) | the genome holds the conjunction *for the mapping it was sorted under* |
| genome-only hit, shuffled mapping (`eta = 0`) | **0.296** (A 0.546, B 0.093) | and only for that one — below chance on the swap |

Shortening the era to 350 was tried and **reverted**. It shortened the *learner's* payoff window
without touching the sorting at all, and the pre-registered prediction failed in both directions:

| | predicted | observed |
|---|---|---|
| `fixed` | 0.52–0.56 | **0.642** |
| `plastic (W2)` | 0.65–0.70 | **0.575** |

So the learner lost to the non-learner. `prep_gain innate` for `fixed` went *up* over the change
(0.149 at 700 → 0.379 at 350). There is no era length this world supports that is slow for
selection and fast for learning, because sorting standing variation is not rate-limited by
generations — it is rate-limited only by how quickly the mismatched fraction dies, which is fast.

**The fix is the size of the space, not the speed of the world.** If the number of possible
mappings substantially exceeds the number a population can hold in standing variation, then at a
remap there is usually **no matching genotype to promote**, and the only route to the new mapping
is within a life.

## The parameter that decides it

With `T` food types and `K` preparations, and mappings required to send distinct types to distinct
preparations, the space is `P(K, T) = K! / (K−T)!`.

| T | K | mappings | chance `1/K` | type-blind `1/T` | `prep_value` = (K−1)·`prep_fail` |
|---|---|---|---|---|---|
| 2 | 3 | **6** (v3.10–v3.11) | 0.333 | 0.500 | 1.00 |
| 2 | 4 | 12 | 0.250 | 0.500 | 1.50 |
| 2 | 6 | 30 | 0.167 | 0.500 | 2.50 |
| 3 | 4 | 24 | 0.250 | 0.333 | 1.50 |
| **3** | **5** | **60** | **0.200** | **0.333** | **2.00** |
| 3 | 6 | 120 | 0.167 | 0.333 | 2.50 |
| 4 | 5 | 120 | 0.200 | 0.250 | 2.00 |

`prep_value = (K−1) × prep_fail` holds the chance EV of a preparation at exactly zero:
`(1/K)·(K−1)f − ((K−1)/K)·f = 0`. Value continues to come only through knowledge of the mapping.

### How big does the space have to be?

Standing variation is not the population size. It is the number of distinct mapping-genotypes the
population holds **with enough copies of each to survive the era in which they are useless**. A
genotype that is wrong for the current mapping earns nothing from preparation, so its lineage
shrinks; to still be present at the next remap it needs enough carriers now.

At a standing population of ~600 and, say, ~20–30 carriers needed per surviving genotype, the
population can hold on the order of **20–30** mappings. A six-mapping space is therefore covered
several times over — which is exactly what was measured. A **60-mapping** space is roughly 2–3×
what the population can hold; a **120-mapping** space is 4–6×.

> **DECISION 1 — T and K.** My recommendation is **T = 3, K = 5**: 60 mappings, chance 0.200,
> type-blind 0.333, and the three levels (chance / type-blind / full) are cleanly separated. It
> raises both parameters as the ruling requires, without the costs D3 and D4 describe becoming
> severe. The alternative worth considering is **T = 3, K = 6** (120 mappings) if the sorting
> survives 60. **T = 2, K = 6** is available but raises only K, and leaves the type-blind level at
> 0.5 where it has been hardest to read against.

> **DECISION 2 — is the coverage claim measured or assumed?** The "20–30 mappings" figure above is
> an estimate, not a measurement. I propose the world carry a **standing-variation probe**: at each
> remap, record how many of the `P(K, T)` mappings the living population would score above the
> type-blind level on, using the existing within-agent `prep_gain` machinery on synthetic
> observations. That number, printed per era, is the direct test of whether the space exceeds
> standing variation — and it makes the v3.12 claim checkable rather than argued. It costs one
> probe sweep per era.

## What else moves, and what must not

### Phase 1 with more than two food types

Phase 1 is v3.1's flip world: two types, one safe and one poison, flipping every 300 steps. It
anchors row 1a and has been unchanged since v3.1.

> **DECISION 3 — phase 1 under T > 2.** Three options, in my order of preference:
> **(a) leave phase 1 at two types** and introduce the extra type(s) only when preparations switch
> on. Row 1a keeps reading against v3.1's published range, which is its whole value, and the
> staging stays "one change at the boundary". The cost is that the type-3 channel is dead in phase
> 1.
> **(b) one safe type of T**, flipping which one. Chance safe rate becomes `1/T`, so v3.1's
> published 0.51–0.56 / 0.60–0.66 range no longer applies and row 1a loses its anchor.
> **(c) T types, half safe**, keeping the 0.5 chance safe rate. Readable, but it is a different
> fast fact from v3.1's and the anchor is again gone.
> I recommend (a). Row 1a is the only stop row that has held across five versions.

### The economy, and the population cap

At `K = 5`, `prep_value` = 2.0. A chance preparation is still worth exactly 0, but a *knowing*
agent now earns **+2.00 per meal against +0.70** for eating raw with the flip known — a wider gap
than v3.11's +1.00 against +0.70.

> **DECISION 4 — the cap.** v3.11 already showed the failure mode in both directions: at
> `prep_value` 1.5 every arm sat at 675–799 against a cap of 800, and at 1.0 the non-learners fell
> to the floor. A learner that can reach the mapping within a life will earn 2.0 per meal and
> should be expected to hit the cap. I propose `max_pop` **1200** from the start, with the existing
> 90%-of-cap check as the gate, and the runtime re-estimated against it (cost scales roughly with
> standing population: v3.11 measured ~11–15 min per 16000-step run at `max_pop` 800).

### Density and readability

With `T = 3` the same food density is split three ways, so preparations per type per era fall by a
third relative to `T = 2` — and the era must still hold enough events per type to read a per-type
hit rate.

> **DECISION 5 — density and era length.** `prep_every` stays at **700** unless the readability
> criterion fails. The criterion, as in v3.9 and v3.10, is read on **`fixed`** only: **≥ 3
> preparations per type per era** in `fixed`, measured in a pre-check before anything else runs. If
> it fails, raise spawn density rather than lengthening the era — a longer era gives sorting more
> time, which is the thing being removed.

## The claim this world is built to settle

> **DECISION 6 — what counts as the record result.** I propose the claim be stated on **row 3b's
> shuffled pair**, not on the population hit rate:
>
> **On a mapping no genotype in the population was sorted for, learning-on exceeds learning-off by
> ≥ 0.10 in every seed, and the gap does not shrink as the mapping space grows.**
>
> The population hit rate cannot carry it — it mixes the genetic baseline with the learner's
> contribution, and v3.11 showed the baseline can be most of it. The shuffled knockout is the only
> line that isolates what the rule adds within a life, on a mapping selection cannot have supplied.
> The genetic baseline is then *reported* alongside as a measured quantity rather than gated
> against, which is how row 1b is already restated in v3.11.

## Carried forward from v3.11 unchanged

Founder-free metrics as the primary reading, with the founder share printed per arm and phase ·
row 0 excluding on `pop < 80` only · the mapping **pinned** in every replay, with
`replay_mapping_selftest` guarding it · the knockout window at one log bin, with
`knockout_window_selftest` guarding it and `pop`/`max_gen` printed beside every number · each arm's
own type-blind level `max(share)` printed beside its hit rate · the world-semantics and
learning-rule self-tests, which must be extended to `T` types and `K` preparations before any
v3.12 run is read.

> **DECISION 7 — the semantics test is not optional here.** The existing test walks
> 4 actions × 3 cell states × 6 mappings = 72 rows plus poison, move, masking and distinctness
> rows. At `T = 3, K = 5` that becomes 6 actions × 4 cell states × 60 mappings. It should be
> written to enumerate from `T` and `K` rather than from literals, and it is the first thing to
> build — a mapping applied to the wrong food type is precisely the bug this world would hide.
