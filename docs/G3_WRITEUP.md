# G3 — the store outlives the run

> **This is the goal.** G0–G2 were scaffolding.
>
> A record left by one population raises the competence of a population that never met it, and the
> gain compounds. (B§2)

G3 tests the first half. The second half is G4. The spec is `docs/G3_SPEC.md`.

---

## 1. The mechanism

Population **A** is an ordinary `collective` run. When it ends, `final_store` is what it left —
marks and π as they stood mid-era, live, the record a living population was actually using.

Population **B** is a *new* run: fresh founders, no genomes and no `H` crossing, born into A's
record through `init_store` and started in A's era through `init_mapping`.

Nothing of A survives except what was externalised, and that is enforced rather than asserted:
`snapshot()` saves twelve genome fields and no learned state, the G-lineage engine wrapper
**refuses** `init_genomes` outright so a B cannot be seeded from A even by accident, and a fresh
agent's `H` is exactly zero. `b_founders_carry_no_H` checks all three before any B number is read.

---

## 2. Two rulings the mechanism forced

### G3-D1 — B is not staged identically, and it cannot be

B§5.2 says "fresh founders, **staged identically**". Measured, that erases the thing being tested:

| B's food-only phase | fraction of A's marks surviving |
|---|---|
| 173 steps (one half-life) | 0.500 |
| 700 steps (one era) | 0.060 |
| **3000 steps (A's own phase 1)** | **0.0000060** |

A record injected at B's step 0 and left through an identical food-only phase is gone six orders
of magnitude before B's first preparation. `inherited store` would read exactly like `fresh store`
and the null would be about `mark_decay`.

**Ruled: B runs chain-on from step 0.** The cost is real — B's founders are unsorted, so their
food instinct, which is genetic and which A's phase 1 selected for, is random. That cost is
**matched**: all four arms share the same founders, seed and absent phase 1, and the claim is a
between-arm difference. What it forfeits is any comparison of B's *absolute* competence to A's,
which is not the claim and is not reported.

### G3-D2 — era-clock alignment

A mark says *preparation k succeeded on type f*, and that is true only of the era it was written
in. If B draws its own mapping, A's positive marks are not merely stale but **inverted**.

**Ruled: run both.** Aligned is the claim line — the only arrangement in which A's record is true
in B's world. Misaligned carries B§5.2's pre-registered reading: *a null under a misaligned clock
is stale culture, not absent culture.*

That turned out to be the most informative comparison in the milestone. See §4.2.

---

## 3. The four arms are a 2×2

B§5.2's wording — "both against `fresh store`, with the scrambled and gain-zero arms flat" —
describes one comparison and three checks. The arms actually form a factorial:

| | labels meaningful | labels unusable |
|---|---|---|
| **can read** | `inherited store` | `inherited scrambled` |
| **cannot read** | `inherited gain-zero` | (`fresh store`: no marks at all) |

which gives **two one-factor contrasts** instead of one confounded comparison:

* **content** = `inherited store` − `inherited scrambled`. Marks, channels and decay held; only
  what the labels *say* varies.
* **reading** = `inherited store` − `inherited gain-zero`. Marks and labels held; only whether the
  agent can *see* them varies.

And a check that is in no part of B§5.2 and is the load-bearing one:

* **the controls must agree.** `inherited scrambled` and `inherited gain-zero` remove the same
  thing by different routes, so a design measuring what it thinks it is must land them together —
  on *both* statistics. If they diverge, something other than label information is moving the
  metric and neither contrast means what it says.

`fresh store` is reported as the **total** effect and is not a claim baseline: it differs from the
treatment in the presence of marks *and* their content at once, and presence turned out to be
about a third of it.

Everything but the store is matched across the four arms, including π — `fresh store` is handed a
*hidden* store (same π, zero marks) rather than nothing, because with `init_store=None` the world
would draw its own π and the baseline would differ in the label permutation as well as in the
marks.

---

## 4. Results

Read on B's first-ever preparations in an early window (700 steps, one era) while A's record is
still live; Gate R read over B's whole run (2100 steps, three π-epochs). Those pull opposite ways
and reading both on one window would have meant choosing which to make meaningless.

### 4.1 Per seed

*(seeds 1 and 2 pending)*

| seed | alignment | content | reading | controls differ (nfc / ratio) | Gate R |
|---|---|---|---|---|---|
| 0 | aligned | **−0.131** | **−0.119** | 0.012 / 0.04 | PASS (z −0.68, 3 epochs) |
| 0 | misaligned | +0.018 | +0.012 | 0.006 / **0.76** | PASS (z 0.82, 3 epochs) |

Seed 0 aligned, in full:

```
  arm                      stale    null   ratio       n     nfc   nfc n     hit    gain  density
  fresh store                nan   0.131     nan       0   2.035    1270   0.474  0.0528   0.0000
  inherited store          0.209   0.117    1.79     393   1.841    1435   0.533  0.0362   0.3950
  inherited scrambled      0.186   0.129    1.44     334   1.972    1232   0.483  0.0554   0.3950
  inherited gain-zero      0.196   0.132    1.48     383   1.960    1310   0.472  0.0000   0.3950
```

### 4.2 The alignment contrast is the sharpest internal check

| | content | reading |
|---|---|---|
| **aligned** — A's record is true in B's world | **−0.131** | **−0.119** |
| **misaligned** — A's record is inverted in B's world | +0.018 | +0.012 |

The effect is present when the record is *true* of B's world and absent when it is not. No control
arm can show that: every arm in the 2×2 varies what the store contains or whether it can be read,
and this varies whether what it says is **correct**. A mechanism that helped in both alignments
would have been helping for some reason other than what the labels say.

It is also what B§5.2 pre-registered, in the direction it predicted.

### 4.3 What the misaligned arm cost me

Its nfc controls agreed to 0.006 while its stale ratios sat **0.76 apart** (1.87 against 1.11). The
coherence check at the time looked only at nfc and would have called that design coherent while one
of its two instruments was not measuring label information at all. The check now covers both.

---

## 4.4 The acceptance basis

**G3 is accepted at one seed, by the project owner's decision, against B§5.2's stated bar of 3/3.**

That is recorded here rather than folded into a verdict because it is the single most important
caveat on everything downstream. A5 assigns acceptance to the project owner, so the decision is
theirs to make and is not a defect. But a reader comparing this to B§5.2 will find a different
number, and the difference should not have to be reconstructed.

`acceptance()` takes `min_seeds` and `min_seeds_basis` as parameters and reports both, so any
result carries the bar it was judged against.

What one seed does and does not buy:

* it buys the **direction** of two independent one-factor contrasts, agreeing with each other;
* it buys the **alignment contrast** (§4.2), which is a within-seed comparison and therefore the
  strongest thing here;
* it does **not** buy an error bar, a variance estimate, or evidence that the effect survives a
  different world draw. 3/3 would have been a sign check; 1/1 is not even that.

A three-seed run is in flight as corroboration. If it disagrees with seed 0, that will be reported
as a disagreement, not averaged away.

---

## 5. What this does not show

* **One seed, not three** (§4.4).
* **Nothing about compounding.** That is G4, and it is a *difference from* this number.
* **Nothing about B's absolute competence.** G3-D1 forfeited that deliberately.
* **Nothing from the misaligned arm about transmission.** A null there is stale culture, not absent
  culture, and it is reported as such.
* **No result about the research lineage's own v3.13 question.** The directive is explicit that the
  transmission experiment running elsewhere is not an input here and its outcome is not assumed.

---

## 6. Defects this milestone found, all mine

Four, and the order matters: each was found before it could contaminate a number.

1. **The stored π did not decode its own marks.** `new_recipe()` redraws π *before* the era-boundary
   snapshot, so the capture recorded the π of the era about to start. Every type of every snapshot
   decoded to the wrong preparation. A record like that looks intact — right density, right signs,
   a valid permutation — and would have made the transmission arm read as a plausible null.
   Caught by writing `store_decodes_selftest`, which is now permanent.
2. **A degenerate Gate R reported as a pass.** One π-epoch means nothing to permute across, so the
   null equalled the observation, `sd` was exactly zero and `z` came out 0.00 — reported as PASS.
   That is an absent measurement reported as a passed gate, in the place meant to catch exactly
   that.
3. **`fresh store` drew its own π**, so the baseline differed from the treatment in the label
   permutation as well as in the marks.
4. **The coherence check looked at one statistic** (§4.3).

Two more were found earlier and are recorded in `docs/G2_WRITEUP.md`: the engine could neither hand
out the store nor take one in, and `assay_selftest` — cited by A2.1 as an existing reference — did
not exist.
