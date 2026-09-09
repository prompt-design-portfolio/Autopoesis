# G4 write-up — mechanic 1 (K raised): **the number measured the economics, not K**

> **Read §2.3 first.** This document was written when mechanic 1 reported and its §1 and §2 were
> correct about what the run produced and wrong about what it meant. The economics diagnostic and
> the seed-1 corroboration both landed afterwards and together they withdraw the finding: changing
> nothing but the seed reproduces most of the effect this milestone measured. Then a third seed
> landed and pulled that correction back part way — see §2.5, which is the version to trust. The
> original sections are kept, not rewritten, because a write-up that quietly became right is
> worth less than one that shows where it was wrong.

    A record left by one population raises the competence of a population that never met it,
    **and the gain compounds.** (B§2)

G3 measured the first half and got it *at one seed, which §5 of its own write-up now qualifies*.
This is the second half, and mechanic 1's run answers **no** — but §2.3 shows the answer is not
about hardening.

Everything below is **one seed, paired**. A direction check, not a statistical test, and
`G4Result.verdict()` says so in its own output rather than leaving it to the prose.

---

## 1. The number

Seed 0, aligned. Baseline is the stored G3 seed-0 result (`var/g3/seed0_aligned.json`), reloaded
rather than re-run, so both sides are differences against the same draw of the same world.

|  | baseline (K=5) | hardened (K=7) | compounding |
|---|---:|---:|---:|
| **content** | −0.131 | **+0.035** | **+0.166** |
| reading | −0.119 | +0.053 | +0.173 |
| total | −0.194 | +0.008 | +0.202 |

`content = inherited store − inherited scrambled` on preparations-to-first-correct, where **lower
is better**. So −0.131 at K=5 is a record saving about an eighth of a preparation; +0.035 at K=7
is a record worth nothing, very slightly the wrong way.

    content effect grew:        False
    content carries the growth: True
    controls still agree:       True
    -> compounds: False

Two of the three falsifications pass, and they are the ones that would have made the number
*uninterpretable*. The controls still agree, so the design is coherent at K=7; and the content
line carries what movement there is, so this is not the total effect drifting while content sits
still. The claim itself is what fails.

### 1.1 The arms at K=7

| arm | stale | null | ratio | n | nfc | hit | gain | density |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fresh store | — | 0.076 | — | 0 | 1.678 | 0.544 | 0.0516 | 0.0000 |
| inherited store | 0.116 | 0.076 | 1.53 | 293 | 1.686 | 0.544 | 0.0511 | 0.3684 |
| inherited scrambled | 0.128 | 0.073 | 1.76 | 288 | 1.651 | 0.562 | 0.0455 | 0.3684 |
| inherited gain-zero | 0.134 | 0.071 | 1.89 | 291 | 1.633 | 0.575 | 0.0000 | 0.3684 |

The matched null is 0.076 = (1 − 0.544)/6, computed at the run's own K. That it is K-aware at all
is recent: A2.2's null was one of four places where K was read from the module constant, which at
K = 5 gives the right answer and on this run would not have.

---

## 2. Why: the mechanic did not harden the world

Read `fresh store` across the two worlds — the arm with no record at all, which is the population's
purely genetic and within-life competence:

| | K=5 | K=7 |
|---|---:|---:|
| chance hit | 0.200 | 0.143 |
| `fresh store` hit | 0.474 | **0.544** |
| ratio to chance | 2.4× | **3.8×** |
| `fresh store` nfc | 2.035 | **1.678** |

**The population is markedly better in the "harder" world.** It reaches its first correct
preparation faster (1.678 against 2.035) and sits nearly four times chance instead of a little
over twice. A world with three and a half times the mapping space should not be easier. It is not
that the record failed to help in a hard world; the world was not hard.

### 2.1 The confound, which is structural and was in the mechanic from the start

`world_at_k` holds the chance EV of a preparation at zero, which it must — a K change that left
the economics alone would make a chance preparation worth −0.250 at K=7 and every arm would get
worse together for a reason unrelated to the mapping space. The invariant is
`prep_value = (K − 1) · prep_fail`, so:

    K: 5 -> 7      prep_value: 1.0 -> 1.5      prep_fail: 0.25 (unchanged)

**A correct preparation became 50% more valuable in the same step that the mapping space grew.**
Those cannot be separated inside the mechanic; the EV constraint welds them together.

The modulator did not move — that is measured, and `modulator_is_independent_of_economics` holds
the ±1 table identical under a 4× change in the economics. What moved is **energy**, and energy is
survival and reproduction. A larger payoff for being right feeds the *genetic* channel, and v3.11
already found that channel dominant:

> At six mappings the genetic baseline is large ... and cannot be removed by shortening the era.

The G4 spec cited v3.11 and v3.12 as the reason K was the right first lever. The citation was
right about the lever and missed that EV-neutrality reintroduces the very baseline those findings
warned about, through a different parameter.

### 2.2 The diagnostic, and what it can and cannot settle

`world_with_economics_scaled(1.5)` — K held at 5, `prep_value` raised to mechanic 1's 1.5 with
`prep_fail` to 0.375, chance EV still exactly zero. It is registered in `HARDENING_PARAMETERS` as
`economics-diagnostic` and **it is not a mechanic and has no claim line.**

* content collapses here too → the collapse is the payoff scale, and **mechanic 1 never tested K
  at all**;
* content survives here → the economics is not the story and K itself is what removed the effect,
  which would be a much more interesting and much worse result for B§2.

What it cannot settle: holding EV at zero at fixed K forces `prep_value / prep_fail = K − 1`, so
scaling one scales the other. It separates *the scale of the preparation economics* from *K*. It
does not separate reward from penalty, and no EV-neutral world can.

---

## 2.3 The diagnostic ran, and it refutes §2.1 rather than confirming it

`world_with_economics_scaled(1.5)` completed at seed 0. Content collapsed there too — from
−0.131 to **−0.009**, without K moving at all. Read alone that is §2.1 confirmed: 74% of mechanic
1's content shift reproduced with K held at 5.

It does not survive the control that matters, which is the one that changes **nothing**:

| seed 0, aligned | content | shift from baseline | what moved |
|---|---:|---:|---|
| baseline, K=5 | −0.131 | — | — |
| mechanic 1, K=7 | +0.035 | **+0.166** | K and the payoff |
| diagnostic, econ ×1.5, K=5 | −0.009 | **+0.122** | the payoff only |
| **seed 1, baseline, K=5** | −0.005 | **+0.126** | **the seed** |

**Changing nothing but the seed reproduces +0.126 of mechanic 1's +0.166.** The diagnostic's
+0.122 is indistinguishable from it. So the diagnostic cannot attribute the collapse to the
economics: its null — same world, different draw — produces the same shift.

The correct conclusion is not §2.1's. It is that **seed 0's −0.131 is the outlier, and every
quantity in this write-up measured as a difference from it is measuring that outlier's distance
from zero.** That includes mechanic 1's +0.166 and the diagnostic's +0.122 alike.

### 2.4 §2 is withdrawn as a finding and kept as a hypothesis

The claim in §2 — that the population is *better* in the harder world, so the mechanic did not
harden it — rests on `fresh store`, which I took to be a within-run comparison immune to the
baseline problem. It is not immune to the seed problem:

| world | seed | `fresh store` hit | chance | ratio | `fresh store` nfc |
|---|---|---:|---:|---:|---:|
| K=5 baseline | 0 | 0.474 | 0.200 | 2.4× | 2.035 |
| K=5 baseline | **1** | **0.568** | 0.200 | 2.8× | 1.809 |
| K=5 econ ×1.5 | 0 | 0.545 | 0.200 | 2.7× | 1.625 |
| K=7 mechanic 1 | 0 | 0.544 | 0.143 | 3.8× | 1.678 |

`fresh store` hit varies **0.474 → 0.568 at a fixed world across two seeds** — a spread wider than
the 0.474 → 0.544 difference between the two *worlds* that §2 built its argument on. So "the
population is markedly better in the harder world" is not established. It is one seed at each of
two worlds, and the seed-to-seed spread swamps the world-to-world difference.

The EV-neutrality confound is still a real property of the mechanic — `prep_value` genuinely does
move from 1.0 to 1.5 when K goes 5 → 7, and that genuinely does feed the genetic channel. What
this run does not show is that it *did*. It stays a hypothesis, and §2.2's diagnostic is not the
instrument that can test it.

### 2.5 The actual finding — restated once three seeds were in

This section first said the number *is* seed variance, on two seeds. Three seeds say that is too
strong, and the correction matters more than the original claim.

The aligned content effect now runs to five seeds: −0.131, −0.005, −0.051, −0.141, −0.135, mean
**−0.092**, sd **0.061**. A difference between two one-seed contrasts therefore carries a standard
error of about `0.061 * sqrt(2)` = **0.086**, so mechanic 1's +0.166 is about **1.9 se**.
Suggestive and unresolved — not noise, and not a measurement either.

The figure to use is the **paired** contrast's between-seed spread (0.061), not the spread of a
single arm over RNG seeds (0.144, measured). The four arms share a seed, so common random numbers
cancel most of the single-arm noise; using the larger number here would understate the design.
`civitas_g.campaign.paired_test` carries the explanation.

**One seed at K=7 cannot resolve this.** What is missing is not a better statistic but seeds in
the hardened world: mechanic 1 has n = 1 there, so its side of the difference has no error
estimate at all. The baseline now has five seeds and a usable one. **The next run G4 needs is
mechanic 1 at seeds 1-4**, which turns +0.166 from a direction check into a paired comparison
with a spread. Before any
difference of differences means anything, the design needs an estimate of that spread —
several seeds per condition, reported with their spread, not a paired direction check.

That is a finding about the instrument rather than the world, and it invalidates the *numbers*
in §1 as measurements of hardening while leaving the machinery that produced them intact and
correct. It also applies backwards: `docs/G3_WRITEUP.md` §5 reaches the same conclusion from the
seed-1 corroboration, independently.

## 2.6 Four seeds at K=7: the effect goes away, and the *difference* is what is consistent

`hold="fail"`, seeds 0-3, each paired against its own K=5 baseline:

| seed | baseline (K=5) | hardened (K=7) | compounding |
|---|---:|---:|---:|
| 0 | −0.1305 | +0.0350 | +0.1656 |
| 1 | −0.0049 | +0.1943 | +0.1992 |
| 2 | −0.0510 | +0.1012 | +0.1522 |
| 3 | −0.1406 | **−0.0065** | +0.1341 |

| | n | mean | sd | positive | exact p |
|---|---:|---:|---:|---:|---:|
| baseline K=5 | 4 | −0.082 | 0.065 | 0 of 4 | 0.0625 (the floor) |
| hardened K=7 | 4 | **+0.081** | 0.088 | 3 of 4 | 0.1250 |
| **compounding** | 4 | **+0.163** | **0.028** | **4 of 4** | 0.0625 (the floor) |

**A correction to what this section said at three seeds.** It read "positive in 3 of 3 hardened
worlds … the effect changes sign". Seed 3's hardened value is −0.0065, so it is 3 of 4, and the
hardened mean of +0.081 has sd 0.088 and does not separate from zero (exact p = 0.125). The
defensible statement is weaker and different: at K = 7 the content effect is **gone** — centred
near zero with a spread that covers it — not reversed.

**What is consistent is the difference, not either level.** Compounding is +0.163 with sd 0.028
across four seeds, positive in 4 of 4: a signal-to-noise of six, where the K=5 baseline it is
measured from has sd 0.065 and the K=7 side sd 0.088. The paired difference is much tighter than
either thing being differenced, which is what common random numbers are for and is the strongest
evidence in this project that the pairing works.

**And it is still p = 0.0625.** The t is +11.84. The exact paired permutation test over all
2⁴ = 16 sign assignments gives 0.0625 one-tailed, **which is the floor at n = 4** — four
observations cannot produce a smaller number however consistent they are. The gap between
t = 11.84 and p = 0.0625 is the clearest illustration in this project of why the exact test was
added, and a t of 12 invites exactly the reporting error this write-up made at n = 4, 5 and 6.

### 2.7 What it would mean, if it survives more seeds

Read against §2.1 and G4-D1, there are two live readings and this run cannot separate them:

1. **A larger mapping space removes the record's value.** At K = 7 a stale mark endorses one of
   seven preparations rather than one of five, so following it costs more when the mapping has
   moved and there are more ways to be wrong. Note this is now "removes" and not "reverses":
   the hardened level does not separate from zero (§2.6).
2. **The record is fine and the economics did it.** `hold="fail"` raises `prep_value` from 1.0 to
   1.5 alongside K, and §2 already showed the K=7 population is *better* without any record at
   all. A population that has solved the world genetically has nothing to gain from a record and
   something to lose from attending to it.

**G4-D1 is the experiment that separates them**, and it now has a specific prediction to test
rather than a general worry: `hold="value"` holds `prep_value` at 1.0 and lowers `prep_fail` to
0.167 instead. If the sign flip survives there, reading 1 is live. If it disappears, the flip was
the economics and mechanic 1 never tested K at all.

That run is the next thing G4 needs, ahead of mechanics 2 and 3 — a mechanic whose own result
cannot be attributed is not a foundation for designing two more.

## 2.8 G4-D1 ran. **Mechanic 1 measured the economics, not K.**

The variant specified in §2.7 and G4-D1 is done: `hold="value"` holds `prep_value` at 1.0 and
lowers `prep_fail` to 0.167 instead. Same K. Same mapping space, P(7,3) = 210. Same chance hit
rate, 0.143. EV exactly zero in both. The **only** difference is which economic term absorbs the
EV constraint.

| | n | hardened content | compounding |
|---|---:|---:|---:|
| `hold="fail"` — reward rises to 1.5 | 5 | +0.037 (3/5 positive) | **+0.129** (4/5 positive) |
| `hold="value"` — penalty falls to 0.167 | 4 | **−0.119** (1/4 positive) | −0.037 (1/4 positive) |
| K=5 baseline, for reference | 5 | −0.082 | — |

Paired on the four seeds both were run at:

| seed | `hold="fail"` | `hold="value"` | difference |
|---|---:|---:|---:|
| 0 | +0.0350 | −0.2044 | +0.2395 |
| 1 | +0.1943 | −0.2337 | +0.4279 |
| 2 | +0.1012 | −0.0707 | +0.1719 |
| 3 | −0.0065 | +0.0338 | −0.0403 |

mean difference **+0.200**, positive in 3 of 4 (exact p = 0.125, the floor at n = 4 being 0.0625).

**The content effect changes sign between two parameterisations of the same world.** Under
`hold="value"` the K=7 content effect is **−0.119**, which is not smaller than the K=5 baseline of
−0.082 — it is slightly larger. Holding the reward fixed, there is no hardening effect at all.

So §2.6's compounding is an artifact of the parameterisation. Mechanic 1 raised K and the reward
together, and **the reward was doing the work**. §2.1 named this as a worry, §2.3 wrongly withdrew
it after a diagnostic that could not test it, and G4-D1 is the test that could.

### 2.9 What G4 has established

* **Mechanic 1 does not test what it was designed to test.** Its compounding number tracks
  `prep_value`. Any restatement of it as a fact about mapping-space size is unsupported.
* **`hold="value"` is the mechanic that isolates K**, and at four seeds it shows **no effect of
  hardening on the record's value** — the K=7 content effect sits where the K=5 baseline sits.
  That is a null result, not a positive one, and at n = 4 it cannot be significant anyway.
* **B§2's second half — that the gain compounds — is not supported by any run in this project**,
  and the run that appeared to support it has been withdrawn by its own control.
* **Mechanics 2 and 3 must not be designed against §2.6's numbers.** `docs/G4_SPEC.md` §2–3 said
  to settle their design questions against mechanic 1's results; those results are now known to
  measure the economics, and building two more mechanics on them would propagate the error.

The one methodological point worth carrying forward: the confound was found by asking what *else*
satisfies the invariant. `prep_value = (K−1) × prep_fail` constrains a ratio, and treating it as
though it named one term is what welded K to the reward. Any future mechanic should be checked the
same way — what else would satisfy the constraint, and does the result survive that too.

## 3. What this does not license

* **It is not a null result about B§2.** It is one seed, one mechanic, and a mechanic with a
  confound in it. But nor is it evidence *for* B§2's second half, and after §2.3 it is not
  evidence about hardening at all — the compounding number is dominated by seed variance.
* **It is not a measured magnitude.** §1's table stands as what the run produced and must be read
  with §2.3 and §2.5 beside it. `+0.166` is a one-seed difference at about 1.8 standard errors of
  the seed-to-seed spread: not resolvable either way without seeds at K=7.
* **It is not a reason to redesign mechanic 1 until it passes.** The number is recorded as it
  came. A corrected mechanic — one that enlarges the mapping space without enlarging the payoff —
  is a *different* mechanic and needs its own spec and its own claim line under B§5.3, written
  before it is run.
* **It is not a reason to average in a second seed.** One seed is what the acceptance basis says,
  and a second seed that disagreed would be a disagreement to report, not a number to blend.

## 4. What mechanics 2 and 3 inherit from this

`docs/G4_SPEC.md` deliberately left both unspecified until mechanic 1 reported. It reports one
thing they must both answer before they are built:

**Does the mechanic change the difficulty of the world without changing what survival pays?** A
second era clock (mechanic 2) plausibly does — it changes how often the fact moves, not what
being right is worth. A compositional preparation (mechanic 3) plausibly does not, since a
preparation that takes two steps has a different cost structure by construction, and that has to
be priced deliberately rather than fall out of an invariant.

Mechanic 1's failure mode is now a check either of them can be held to: report `fresh store`'s hit
and nfc in both worlds, and if the population is *better* in the harder world, the mechanic did
not harden it and its claim line means nothing.
