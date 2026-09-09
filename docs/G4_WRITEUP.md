# G4 write-up — mechanic 1 (K raised): **the effect does not compound. It disappears.**

    A record left by one population raises the competence of a population that never met it,
    **and the gain compounds.** (B§2)

G3 measured the first half and got it. This is the second half, and mechanic 1 answers **no** —
not "no growth", but no effect at all in the harder world.

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

## 3. What this does not license

* **It is not a null result about B§2.** It is one seed, one mechanic, and a mechanic with a
  confound in it. G3's transmission effect is unaffected and was measured in the world it was
  measured in.
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
