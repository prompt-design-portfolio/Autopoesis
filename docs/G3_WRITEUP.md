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


---

## 5.0 Where this stands (read this before the subsections)

§5 was written in layers as a campaign came in, and two of its conclusions were later corrected —
in both directions. This is the current position; the subsections keep their original wording so
the corrections are visible rather than tidied away.

**The claim looks real. The apparatus built to certify it has two defects.**

| | state |
|---|---|
| content contrast, aligned | **−0.058 ± 0.041 (se), t = −1.42 on 5 df, negative in 5 of 6 seeds — NOT significant.** At n = 5 this read −0.092 ± 0.027, t = −3.38. Seed 5 came in at **+0.113**, the only positive aligned seed, and it both moved the mean and doubled the spread. The n = 5 figure is superseded, not an alternative. |
| content contrast, misaligned | +0.054 ± 0.049, signs mixed — the record does not help a world it is not true of |
| out-of-sample seeds (3, 4, 5) | −0.141, −0.135, **+0.113**. Two large negatives and one clear positive: the out-of-sample seeds are the most variable of the set, which is the opposite of corroboration. |
| **alignment contrast** | **−0.100 ± 0.066, t = −1.51 on 5 df, negative in 5 of 6 — NOT significant.** Recomputed once seed 5's misaligned run landed; at n = 5 it read −0.145 ± 0.058, t = −2.51. It is still the sharpest check the design contains — same A, same record, same draws, differing only in whether the record is *true* of B's world, which no control arm can supply — but it no longer separates from zero. |
| `g3.acceptance` verdict | **false** — and the reason is the coherence criterion, not the claim |
| coherence defect 1 (§5.6) | the tolerance scales with the effect, so a seed with no effect cannot pass however well its controls agree |
| coherence defect 2 (§5.6a) | the two arms it asks to agree **provably do not remove the same thing** — a scramble preserves the per-cell multiset exactly, so `inherited scrambled` keeps presence and `inherited gain-zero` does not |

**The noise floor, measured (8 replicates, same A, same record, same arm, only B's RNG seed):**

| arm | sd of `nfc_mean` | se of an unpaired 2-run difference |
|---|---:|---:|
| `inherited store` | **0.144** | 0.204 |
| `inherited gain-zero` | **0.088** | 0.125 |

Two things follow. First, the arm that can read the record is **1.6× as variable** as the arm
that cannot (variance ratio 2.65 on 7 and 7 df, F ≈ 0.11 — suggestive, not established). If that
holds up it is a result in its own right and not one the claim line asks for: a record does not
only shift competence, it disperses it. Second, and immediately: these are *unpaired* figures and
the G3 contrast is paired, so neither is the yardstick for it — see below.

**The error term matters and is easy to get wrong.** The four arms of a succession share one
seed, so they consume the same draws in the same order and differ only in the store — common
random numbers. One arm replicated over B's RNG seed has sd **0.144**; the paired contrast across
seeds has sd **0.061**. Both are measured. Reading the first as the noise floor for the second
understates the design about twofold and would reject a real effect, which is a mistake this
write-up made before catching it (§5.4, and `campaign.paired_test` carries the warning at the
point of use).

**The other two statistics B§5.2 names show nothing, and could not have.** The claim line asks for
the stale-mark ratio *and* preparations-to-first-correct. Only the second moves at all:

| statistic | effect (aligned) | se | t | an effect of the same *relative* size (3.4%) would be |
|---|---:|---:|---:|---|
| preparations-to-first-correct | −0.0581 | 0.0409 | −1.42 | — |
| stale-mark ratio | +0.0890 | 0.1317 | +0.68 | +0.052 = **0.39 se**, below its resolution |
| preparation hit vs fresh | +0.0125 | 0.0197 | +0.63 | +0.019 = **0.94 se**, below its resolution |

So the two null results are *not* evidence against the claim — neither statistic has the power to
see an effect the size the third one reports. But it does mean the claim rests on **one** of the
three statistics its own claim line names, and that one is currently at t = −1.42.

**What is claimed, as of six balanced seeds: nothing reaches significance.** Every test the design
supports now sits between t = −1.5 and t = +1.3:

| test | n | estimate | se | t |
|---|---:|---:|---:|---:|
| content, aligned | 6 | −0.058 | 0.041 | −1.42 |
| content, misaligned | 6 | +0.042 | 0.033 | +1.25 |
| **alignment contrast** | 6 | −0.100 | 0.066 | −1.51 |
| stale-mark ratio | 6 | +0.089 | 0.132 | +0.68 |
| preparation hit | 6 | +0.013 | 0.020 | +0.63 |

The alignment contrast was recomputed once seed 5's misaligned run landed, exactly as §5.0 said it
must be: it fell from −0.145 ± 0.058 (t = −2.51) to −0.100 ± 0.066 (t = −1.51). The warning was
written before the number was known, and the number went the way the warning allowed for.

**The signs are still consistent and that is the whole of what is left.** Aligned content is
negative in 5 of 6, the alignment contrast in 5 of 6, and misaligned content is positive in 4 of
6. Consistency at n = 6 reaches p = 0.016 at best by a sign test, and only if unanimous, which
none of these are. B§5.2's acceptance returns false, and while §5.6 and §5.6a show its coherence
criterion is ill-posed, that no longer rescues anything: the claim statistic does not reach
significance on its own terms.

---

## 5. Corroboration at seed 1: **seed 0 does not reproduce**

Added after §4.4's one-seed acceptance, from the resumable campaign in `var/g3/`. It is recorded
here rather than folded into §1's numbers, because §4.4's acceptance was read on seed 0 and a
second seed does not get to quietly replace the number a gate was read on.

| run | content | reading | total | stale content | stale reading | controls agree | Gate R |
|---|---:|---:|---:|---:|---:|---|---|
| seed 0 aligned | **−0.131** | −0.119 | −0.194 | +0.35 | +0.31 | yes | PASS |
| seed 1 aligned | **−0.005** | −0.008 | −0.033 | **−0.08** | **−0.22** | **no** | PASS |
| seed 0 misaligned | +0.018 | +0.011 | +0.009 | −0.46 | +0.30 | no | PASS |
| seed 1 misaligned | +0.018 | +0.132 | −0.027 | +0.17 | −0.24 | no | PASS |

**Aligned, 1 of 2 seeds passes.** Seed 1 fails on two of the four criteria: the stale-mark ratio
moves the *wrong way* on both contrasts, and the two information-removing controls do not agree —
which is the load-bearing check, the one that says the four arms are a design rather than four
numbers. Its content effect is −0.005: not a smaller effect, no effect.

`acceptance` returns **`accepted: false`**, and it does so at *any* `min_seeds`. The rule is
`n >= min_seeds and len(passing) == n`: the bar is a floor on how many seeds must be **run**, not
a licence to drop the ones that disagree. Relaxing the bar to one seed, which is on record as the
project owner's decision, does not turn a disagreeing second seed into an absent one.

What holds up and what does not:

* **The alignment contrast holds.** Both misaligned seeds fail, and both put content the wrong way
  (+0.018 each). The record does not help a population whose world it is not true of — which was
  the sharpest internal check in the milestone and is the one thing here that reproduces.
* **The magnitude does not.** Seed 0's −0.131 is the number every downstream comparison has been
  built on, G4's compounding baseline included, and seed 1 says the expected value of that number
  across seeds is much closer to zero.
* **Gate R passes everywhere, including where nothing else does**, so it is not carrying the
  claim and never was.

### 5.1 What this does to G4

G4 mechanic 1's compounding number is `content(hardened) − content(baseline)` with the baseline
being seed 0's −0.131. If the baseline's own reproducibility is this weak, **+0.166 is not a
measurement of hardening**; it is mostly the distance between one draw and zero.

This does not change mechanic 1's *diagnosis* — `fresh store` is better in the harder world
(hit 0.474 → 0.544 against a chance that fell 0.200 → 0.143), and that is a within-run comparison
that does not depend on the baseline at all. It does mean the compounding number should not be
quoted as a magnitude, and `docs/G4_WRITEUP.md` §1 should be read with this section next to it.

### 5.2 Seed 2 landed: **1 of 3. The gate is not met.**

§5.2 pre-registered three readings before seed 2 ran. The second one is what happened: *seed 2
fails → 1 of 3, and seed 0 is the outlier. G3's gate is not met, and the milestone should be
reopened rather than carrying an acceptance its own campaign contradicts.* That is the finding.

| seed | content | reading | controls gap (nfc) | controls gap (stale) | content right way | stale right way | controls agree |
|---|---:|---:|---:|---:|---|---|---|
| 0 | −0.131 | −0.119 | +0.011 | −0.047 | yes | yes | **yes** |
| 1 | −0.005 | −0.008 | −0.003 | −0.138 | yes | **no** | **no** |
| 2 | −0.051 | −0.119 | −0.068 | +0.279 | yes | yes | **no** |

`acceptance` returns **`accepted: false`** at B§5.2's own bar of 3, and would at any bar. G3's
gate is **not met**.

### 5.3 What the three seeds do show, stated at the strength the evidence supports

The failure is not on the content direction. It is on the coherence criteria — the two
information-removing controls landing together, and the stale-mark ratio's direction. Content
moved the right way in **3 of 3** aligned seeds:

| | n | mean | sd | se | range |
|---|---:|---:|---:|---:|---|
| aligned content | 3 | **−0.062** | 0.064 | 0.037 | [−0.131, −0.005] |
| misaligned content | 3 | **+0.005** | 0.022 | 0.013 | [−0.021, +0.018] |

* **The alignment contrast is the result that survives.** Aligned is negative in all three seeds
  and averages −0.062; misaligned is centred on zero. Separation −0.067 ± 0.039 (se), about 1.7
  standard errors. The record helps only when it is true of B's world.
* **It is not significant and is not claimed to be.** Three seeds, all negative, is p = 0.125 by
  a sign test — the smallest p three seeds can produce. The direction is consistent; the effect
  is not established.
* **Seed 0 is high, not aberrant.** −0.131 against a mean of −0.062 and sd 0.064 is about 1.1 sd.
  It is the largest of three draws from a distribution that is plausibly centred below zero, not
  an outlier from a distribution centred at zero.
* **The magnitude is unresolved.** A one-seed estimate carries a standard error of ~0.064; the
  spread across seeds is the same size as the effect being measured.

### 5.4 Correcting §5.1

§5.1 said mechanic 1's +0.166 "is not a measurement of hardening; it is mostly the distance
between one draw and zero", and `docs/G4_WRITEUP.md` §2.5 put it more strongly still. With three
seeds in hand that is **too strong**. The standard error of a one-seed paired difference, given
sd = 0.064, is 0.090, so +0.166 is about **1.8 se** — suggestive, unresolved, not noise. The
correct statement is that mechanic 1 is a one-seed measurement of a quantity whose seed-to-seed
spread is comparable to the effect, so it cannot be resolved either way, and needs seeds at K=7
before it means anything. It is not established that the number *is* seed variance.

### 5.6 The coherence criterion has a defect, and it changes how the 1-of-3 reads

Diagnosing *why* seeds 1 and 2 fail turns up a structural problem in the criterion itself.

`controls_agree` requires `|scrambled − gain-zero| ≤ 0.35 × max(|content|, |reading|)`. The
tolerance is a **fraction of the effect**, so it shrinks as the effect shrinks:

| seed | control gap | tolerance applied | gap ÷ tolerance | verdict |
|---|---:|---:|---:|---|
| 0 | **+0.0111** | 0.0457 | 0.24 | agree |
| 1 | **−0.0034** | 0.0029 | 1.18 | disagree |
| 2 | −0.0677 | 0.0415 | 1.63 | disagree |

**Seed 1's controls are the closest of the three — 0.0034 apart, three times closer than seed 0's
0.0111, which passes.** It fails because 0.35 of its own 0.005 effect is 0.0018, and its gap,
though tiny, is larger than that. A seed with no effect cannot pass this check however well its
controls agree, because the bar goes to zero with the effect.

So `controls_agree` is not independent of `content_moves_the_right_way`: the acceptance counts
effect size twice, once as the claim and once as the coherence check. That is a defect in the
instrument, not a property of the world.

It splits the two failures apart:

* **Seed 2 fails for a real reason.** Its controls sit 0.068 apart while its whole effect is
  0.051 — the two ways of removing the information disagree by more than the information is
  worth. That is exactly what the check exists to catch.
* **Seed 1 fails for an artifact.** Its controls agree to 0.003, better than the seed that
  passed. What it lacks is an effect, and the claim line already measures that separately.

**The criterion is left exactly as it is.** Changing an acceptance rule after seeing which seeds
it rejects is how a gate gets fitted to a result, and the fix would flip a seed from fail to pass,
which is the worst possible provenance for a change. What has been added is *reporting*: the
absolute gap and the tolerance it was measured against now travel with every seed, so the defect
is visible in the output rather than only in this document.

**A corrected criterion is the project owner's decision, and it needs a scale that does not come
from the effect.** The candidates, in the order I would argue for them:

1. **The sampling noise of the statistic.** `nfc_n` is ~1400 preparations per arm; the standard
   error of each arm's `nfc_mean` is estimable from the run, and two controls agree if their
   difference is within a few of those. This is the principled answer and it is the one that
   makes the check independent of the effect.
2. **An absolute tolerance in preparations**, pre-registered before the seeds are read — simple,
   defensible, and arbitrary in a way a reader can see and argue with.
3. **Between-seed spread**, once there are enough seeds to estimate it. Currently sd = 0.064 on
   three seeds, which is too few to set a bar with.

Under **none** of these does G3 pass as it stands: seed 2's failure is real under all three. The
defect changes the 1-of-3 from *"two seeds show the design is incoherent"* to *"one seed shows the
design is incoherent and one shows no effect"* — a different and more tractable problem, but not
a passing one.

### 5.6a Why the controls disagree: they do not remove the same thing

§5.6 shows the criterion is defective. It does not explain why seeds 2 and 3 disagree for real.
This does, and its premise is now checked rather than argued
(`tests_g/test_store.py::test_a_scramble_preserves_the_per_cell_multiset_exactly`):

**A scramble permutes the K channels within each cell, so at every `(ftype, y, x)` the sorted
vector of K values is unchanged.** An agent reads the K channels *at its own cell*, so everything
it could compute from that vector without the labels — is there a mark here, how many, how
strong, what signs — survives a scramble exactly. Only which label carries which value moves.

So the two arms called controls remove different things:

| arm | content | presence, as the agent can read it |
|---|---|---|
| `inherited scrambled` | **removed** | **kept, exactly** |
| `inherited gain-zero` | removed | **removed** — every channel is zero |

`store_for_arm`'s docstring says gain-zero is A2.3's presence-versus-content separation because
"the store still changes the world by existing". That is true *of the world* and not of the
agent: with `sym_gain = 0` the agent reads zeros, so presence is gone from its observations
entirely. The scrambled arm keeps it. **They were never two routes to the same removal**, and
`controls_agree` — which the acceptance calls load-bearing — asks two arms to land together that
the design gives no reason to.

The supporting evidence is suggestive and no more. Across four aligned seeds the control gap
tracks the scrambled arm's evolved `sym_gain` at r = −0.97 — which is what the mechanism predicts,
since `sym_gain` is the gate on how much of that preserved presence the population lets in, it is
heritable, and it evolves to a different value in every seed. At n = 4 that correlation is worth
one sentence and not a conclusion, and the direction was predicted before it was computed rather
than found by looking.

**What follows, and what does not.** The two claim contrasts are unaffected: content
(`store − scrambled`) and reading (`store − gain-zero`) each still hold everything but one factor.
What is affected is the coherence check built on top of them. If this is right, the honest repairs
are to drop `controls_agree` as specified, or to add the arm that would make it meaningful — a
store scrambled *and* gain-zeroed is just gain-zero, so the missing arm is the reverse: presence
kept, content removed, and reading disabled has no such arm. Either way it is a design change and
the project owner's call.

### 5.6b What this design can detect, which is barely this effect

The per-seed spread of the paired contrast is sd ≈ 0.100. That fixes what the design can see:

| seeds | se | minimum detectable effect (t = 2) |
|---:|---:|---:|
| 6 | 0.041 | 0.082 |
| **12** | 0.029 | **0.058** |
| 24 | 0.020 | 0.041 |
| 48 | 0.014 | 0.029 |
| 96 | 0.010 | 0.020 |

**The observed effect is −0.058 and twelve seeds detect 0.058.** The campaign is powered to
almost exactly the size of the thing it is measuring, which is the worst place to be: it will
come out marginal either way, and whether it lands at t = 1.8 or t = 2.2 will be close to a coin
flip. That is not a fact about this campaign's luck. It is a property of the design that was
knowable before any seed was run, and was not computed until now.

**The pairing is not the problem — it is already doing real work.** If the four arms did not share
a seed, a two-run difference would have sd = √2 × 0.144 = **0.204**. The observed per-seed
contrast sd is **0.100**, so common random numbers cut the sd twofold and the variance fourfold.
Without it this campaign would need four times as many seeds.

**What is left after pairing is seed-level variation** — a different A, a different record, a
different world draw. Replicating B within a seed cannot touch it, because it is not B's noise.
Only more seeds reduce it, and only as √n: going from 12 to 48 seeds buys a factor of two in the
detectable effect and costs four times the compute.

So the lever that matters is **not more seeds**. It is reducing the per-seed spread — a longer B,
a larger population, a wider claim window — each of which changes what is being measured and needs
its own justification. Anyone continuing this should decide that before buying another twelve
seeds at √n.

### 5.7 What would settle it

Seed 2 is running. Three outcomes and what each means:

(Written before seed 2 ran; seed 2 failed, so the second reading applies.)

* **seed 2 passes** → 2 of 3, and the honest statement is a positive effect at roughly two thirds
  of draws, with the magnitude unresolved. Still not B§5.2's 3/3.
* **seed 2 fails** → 1 of 3, and seed 0 is the outlier. G3's gate is not met, and the milestone
  should be reopened rather than carrying an acceptance its own campaign contradicts.
* **seed 2 passes with a small effect** → the direction is real and the magnitude is seed 0's
  alone, which is the most likely reading of the two seeds already in hand.

None of these is decided by averaging. The per-seed table above is the result.
