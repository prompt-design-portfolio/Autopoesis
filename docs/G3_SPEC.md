# G3 spec — the store outlives the run

> **This is the goal.** G0–G2 are scaffolding. B§2:
>
> > A record left by one population raises the competence of a population that never met it, and
> > the gain compounds.

A3's row:

| milestone | gate | number it must move |
|---|---|---|
| G3 | the store outlives the run; population B born into population A's record | the G3 claim line |

---

## 1. The mechanism

Population **A** is a `collective` run. When it ends, `final_store` is what it left: the marks and
π as they stood mid-era, live, the record a living population was actually using.

Population **B** is a *new* run — fresh founders, no genomes crossing, no `H` crossing — born into
A's record via `init_store`, and started in A's era via `init_mapping`.

Nothing about A survives except what was **externalised**. That is the whole claim, and it is
enforced rather than asserted: `snapshot()` saves genomes only (*"Nothing learned (H) is saved"*),
B's founders are constructed fresh by `run()` and never restored from A, and
`b_founders_carry_no_H` checks it.

### 1.1 B's arms — the only thing that varies is the store

| arm | store handed to B | what it removes |
|---|---|---|
| `fresh store` | none | the reference cell: B alone |
| `inherited store` | A's `final_store` | nothing — the treatment |
| `inherited scrambled` | A's store, `ScrambleMode.PER_CELL` | the label→preparation association, keeping density and the multiset of signs exactly |
| `inherited gain-zero` | A's store, `sym_gain_lock=True` | B's ability to *read*, keeping the store live in the world |

All four use **A's final mapping** as their starting era and the **same seed**. The store is the
only difference. `sym_gain_lock` consumes no RNG and `init_mapping` overwrites a draw rather than
skipping one, so the four arms are matched step for step.

The last two are A2.3's separation done twice over. `inherited scrambled` asks whether the *content*
mattered; `inherited gain-zero` asks whether the store's mere *presence* — channels, cell state,
decay — mattered. A gain that survives both is not a gain from information passing.

---

## 2. Two rulings the mechanism forces

### G3-D1 — B is not staged identically, and it cannot be

B§5.2 says "fresh founders, **staged identically**". Measured, that erases the thing being tested:

| phase-1 length | fraction of A's marks surviving |
|---|---|
| 100 steps | 0.670 |
| 173 steps (one half-life) | 0.500 |
| 700 steps (one era) | 0.060 |
| **3000 steps (A's phase 1)** | **0.0000060** |

A record injected at B's step 0 and then left through an identical 3000-step food-only phase is
gone six orders of magnitude before B's first preparation. `inherited store` would read exactly
like `fresh store`, and the null would be about `mark_decay`, not about transmission.

**Ruled: B runs chain-on from step 0.** No food-only phase.

The cost is real and worth naming: B's founders are unsorted, so their food instinct — which is
genetic (`nav_dir`, `nav_here`, wired by `wire_nav`) and which A's phase 1 selected for — is
random. B is therefore worse at everything than A was.

That cost is **matched**. All four arms share it exactly: same fresh founders, same seed, same
absent phase 1. The claim is a *between-arm* difference on B's first-ever preparations, and a
handicap common to every arm cancels in that difference. The frozen assay already replays
chain-only for the same reason.

What this ruling forfeits is any comparison of B's *absolute* competence to A's. That comparison is
not the claim and is not reported.

### G3-D2 — era-clock alignment (flagged by B§5.2)

A mark says *preparation k succeeded on type f at this cell*, encoded as label π(k). That is true
only of the era it was written in. If B draws its own mapping, every one of A's positive marks
points at a preparation that is now wrong — the record is not merely stale, it is **inverted**.

**Ruled: run both, and they answer different questions.**

* **aligned** — B starts at A's final mapping. This is the claim line. It is the only arrangement
  in which A's record is *true* in B's world, and therefore the only one where "raises the
  competence" is even a coherent thing to measure.
* **misaligned** — B draws its own mapping. B§5.2 pre-registers the reading: *a null under a
  misaligned clock is stale culture, not absent culture, and is reported as such.* Note it need not
  be a null: the v3.13 pre-check found arms with a stale-mark ratio **below** 1, and recorded why
  that is still evidence of reading — *"ratio > 1 = FOLLOWS a mark it should not; ratio < 1 =
  AVOIDS it. Either way the label was read: you cannot avoid what you cannot see."* A B that learns
  to avoid an inverted record has read it.

Alignment is a recorded parameter of every G3 run, never an assumption.

---

## 3. The spec, in A5's form

Cells and co-occupancy, actions, the action × state table, the modulator event list, the
observation layout, the standing densities and the costs and values are **unchanged** from
`docs/G0_G1.md` §3. **No mechanic is added at G3** — the ratchet starts at G4, and a world that
moved here would make the claim line incomparable to G2's.

Three things are new, and all three are about *initial conditions* rather than about the world:

| | |
|---|---|
| **phases** | A: two phases, food-only then chain, as at G1/G2. B: **one phase, chain-on** (G3-D1) |
| **initial store** | B's `world.marks` and `world.pi` are written from A's `final_store` before step 0 |
| **initial era** | B's `world.mapping` is overwritten with A's final mapping (aligned) or left as drawn (misaligned) |

---

## 4. Claim lines

All on **B's first-ever preparations only**. B§5.2 is explicit about why that window and not a
pooled one: an agent making its first preparation has learned nothing and written nothing, and by
the no-self-echo property the mark it reads cannot be its own. A pooled line mixes transmission
with an agent's own within-life binding.

| line | statistic | null |
|---|---|---|
| **stale-mark ratio** | `follow_split_newborn`: P(chose the endorsed preparation \| a mark is present and endorses a preparation that is **wrong** for this type now) | A2.2's matched null, `(1 − hit) / (K − 1)` from the same arm — never 1/K |
| **preparations-to-first-correct** | `nfc`, founder-free, over agents that reached `nfc_max = 5` | `fresh store` |
| **frozen assay on B** | A2.1's twelve cells at B's first era boundary | `eta_scale` 0 against 1, on shuffled |
| **Gate R** | on the inherited marks under **B's** π-epochs | matched permutation, z ≤ 2.0 |

**Acceptance:** the first two against `fresh store` in **3/3 seeds**, with `inherited scrambled`
and `inherited gain-zero` **flat**. Anything less is reported as what it is.

Gate R is the one that can invalidate the rest. B inherits A's π and then redraws it on B's own
clock, so the inherited marks span π-epochs that are not A's. If pooled MI over B's epochs exceeds
the matched permutation null, the label→preparation binding survived the epoch rotation and
selection could have reached it — and then a gain is not evidence of transmission through a
record whose meaning was not inherited.

---

## 5. What would make this claim wrong, and how each is caught

A1.2: *passing by construction is failing.* The ways this could produce a positive number that
means nothing:

| failure | what catches it |
|---|---|
| the store helps because it *exists* (channels, cell state, decay), not because it says anything | `inherited gain-zero` — store live, reading off |
| the store helps because of its *statistics* (density, sign balance), not its labels | `inherited scrambled` — density and signs preserved exactly, labels destroyed per cell |
| B's advantage is really A's genes leaking through | `b_founders_carry_no_H`; fresh founders; `snapshot()` saves no `H` |
| the "transmission" is an agent reading its own mark | no-self-echo: a preparation consumes the cell, verified in `record_semantics_selftest` and `check_no_self_echo` |
| the binding was inheritable after all, so selection could find it | Gate R under B's π-epochs |
| the capture handed B a record its own π does not decode | `store_decodes_selftest` — this one already fired once, on my own code |
| the arms differ in something other than the store | `init_mapping` and `sym_gain_lock` are RNG-neutral; measured |

---

## 6. Order of work

1. `b_founders_carry_no_H` — before any B is run, not after.
2. A single A run, its `final_store` persisted, hashed, and reproducible from the database.
3. B's four arms at one seed, chain-on, aligned — a pre-check. Files come to you at this stage.
4. The misaligned arm beside it.
5. Gate R on the inherited marks; the frozen assay on B.
6. Only then, three seeds.

**Nothing at step 3 or beyond is a claim until step 6.** One seed is a pre-check, and B§5.2 asks
for 3/3.
