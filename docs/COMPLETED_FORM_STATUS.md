# The completed form, against what exists

`COMPLETED_FORM.md` (8 September 2026) specifies the project at completion. This document maps it
onto the tree as built, and reports one finding **about the specification itself** that the G3
campaign turned up.

Written to be read beside §8's path table and §6's proofs table. Nothing here argues with the
vision; §8 already says *"a null redirects the path; it does not shorten it."*

---

## 1. The finding: §6's `transmits` line is passed by an arm that cannot read

§6 states the claim line for **transmits** as:

> newborn stale-mark ratio above its matched null; cross-population transmission

Taken literally that line is **met, strongly**, on thirteen seeds of the G3 succession:

| | mean ratio | sd | above 1 |
|---|---:|---:|---:|
| `inherited store` | **1.548** | 0.330 | 11 of 13 |

ratio − 1 = **+0.548 ± 0.092**, t = +5.99, exact permutation **p = 0.0005**.

It is also met by the arms that **cannot read the record at all**:

| arm | mean ratio | sd | above 1 |
|---|---:|---:|---:|
| `inherited store` | 1.548 | 0.330 | 11 of 13 |
| `inherited scrambled` (labels destroyed) | 1.464 | 0.253 | **13 of 13** |
| `inherited gain-zero` (**every read channel is exactly zero**) | **1.602** | 0.218 | **13 of 13** |

`inherited gain-zero` has `sym_gain = 0`, so `sym_gain × marks` is zero at every cell, every step.
It cannot read one mark. **Its ratio is higher than the treatment's**, and it clears the matched
null in every seed.

So the stale-mark ratio against its own matched null is **not a test of transmission**. A
population that shares innate action preferences follows stale marks at above-null rates without
reading anything, because "the mark endorses what I was going to do anyway" and "I read the mark"
are indistinguishable in that statistic. `civitas_g/g3.py` already says this in its docstring and
compares against `inherited gain-zero` for exactly this reason — the finding here is that
**§6 still states the weaker line**, and the weaker line can be passed by a control.

This matters more than a normal null result. §6 is the table that protects the vision from being
faked, and §9 commits to *"every claim on a frozen line with the control that would kill it."* This
line has a control that would kill it, the control was run, and the control passes the line.

**Suggested amendment**, in the form §6 already uses elsewhere:

| claim | line | controls |
|---|---|---|
| transmits | newborn stale-mark ratio **above the same statistic in an arm handed the same record with reading disabled**; cross-population transmission | scrambled record; **gain forced to zero, which is the line's baseline and not merely a control**; frozen assay |

Under that line, thirteen seeds give: treatment 1.548, gain-zero 1.602, difference **−0.054** —
flat, and pointing the wrong way. The claim is **not met**, which is the honest state of it.

## 2. What the thirteen seeds do support

The statistic that moves is preparations-to-first-correct, which §6 does not name:

| test | n | estimate | exact p |
|---|---:|---:|---:|
| content (`store` − `scrambled`) | 13 | **−0.057 ± 0.020** | 0.0089 |
| alignment contrast (aligned − misaligned) | 13 | −0.091 ± 0.042 | — (t = −2.17) |
| misaligned content | 13 | +0.034 ± 0.026 | — (flat, as designed) |

Twelve of thirteen seeds negative; ten of the thirteen were fixed before being read. This is a
real, small effect on a statistic the specification does not currently claim. Whether to add it
to §6 is a decision for the project owner, and it should be made knowing the effect is ~0.06
preparations and sits near what the design can resolve (`docs/G3_WRITEUP.md` §5.6b).

## 3. The path, §8, against the tree

| build | establishes | state |
|---|---|---|
| **Civitas-G G1** | reproduction | **met** — 182/182 fields, both backends |
| **Civitas-G G2** | the record as artifact store | **met** — byte-identical at reference length |
| **Civitas-G G3** | the record outlives the run | **mechanism built and measured; gate not met** |
| v3.14 | the record outlives the run | **this is G3**, and the same state applies |
| v3.15 | writing as a policy | **not built.** `write_mark` is called `AUTOMATIC, costless` in the engine; there is no write decision to measure |
| v4.0 | first compositional fact; architecture genes | **not built.** `GENOME` is 12 weight/rate/gain fields; `hidden` is a fixed `Config` value of 24, not a gene. §4.1's "architecture under selection" does not exist yet |
| v4.x | credit reaches chain links | not built |
| v5.0 | two-link chains; first posed task | not built |
| v6.0 | meaning without an imposed π | not built — π is drawn by the world every era |
| v7.0 | marks referencing marks; abstraction tree | not built |
| v8.0 | the open tree; frontier; problem lab | not built — the world has 3 fixed food types and K preparations, no tiers |
| v9.0–v12.0 | humans in the world; language; dialogue | not built |
| G-observatory / oracle / lab / speech | the human surfaces | not built |

**Two of thirteen path rows are met**, both on the platform side. §7's platform is the part that
is furthest along: persistence, measurement, manifests, reproduction-before-acceptance, the record
as an artifact store with provenance, and a versioned engine seam are all real and tested.

## 4. §6's proofs, one line at a time

| claim | apparatus | result |
|---|---|---|
| **grown** | `frozen_knockout` exists; architecture ablation **cannot** exist without architecture genes | half the line is buildable today, half is not |
| **transmits** | fully built — succession, scrambled, gain-zero, frozen assay | **see §1: the line as written is passed by a control** |
| **writes by choice** | no write policy exists | not testable yet |
| **means** | Gate R built, with a NOT AVAILABLE verdict for the degenerate case | runs; passes; not load-bearing on its own |
| **accumulates** | no tree, no tiers, no frontier | not testable yet |
| **solves** | no posed tasks | not testable yet |
| **understands human language** | no humans in the world | not testable yet |
| **speaks** | no rendering | not testable yet |

One of eight lines has been built and run end to end. Its result is the subject of §1.

## 5. What this says about §10

> The project is complete when a human poses a problem in their own language to a civilization
> that grew from metabolism and selection…

Nothing measured so far bears on that. What the work to date establishes is narrower and worth
stating exactly: **a platform that can hold a claim still**, and one claim run through it honestly
enough that its own control was able to withdraw it. §9's last line is *"retract when a closer look
breaks a result"* — this document is that, applied to the specification rather than to a build.
