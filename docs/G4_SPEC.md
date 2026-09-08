# G4 spec — a world that hardens, one mechanic per milestone

A3's row:

| milestone | gate | number it must move |
|---|---|---|
| G4 | a world that hardens, one mechanic per milestone | G3 claim line, per mechanic |

B§5.3 fixes the order and the form:

> Each mechanic added to the G3 world is its own spec with its own claim line on B's newborns.
> Order: K raised; a second era clock; the first compositional fact — a preparation whose correct
> choice depends on a prior outcome — which is the first fact no single life can assemble.

---

## 0. What "the gain compounds" means, and what would falsify it

B§2's claim has two halves and G3 tested only the first:

> A record left by one population raises the competence of a population that never met it, **and
> the gain compounds.**

G4 is the second half. Compounding is not "the effect stays positive as the world gets harder" —
that would follow from the record simply continuing to work. It is **the effect getting larger as
the world gets harder**, because a record's value is the trial-and-error it saves, and a harder
world makes trial-and-error cost more.

So the G4 claim line, for every mechanic, is the **G3 content contrast measured at the harder
setting against the same contrast at the setting before it**:

```
compounding(m) = content_effect(mechanic m) − content_effect(mechanic m−1)
```

with `content_effect` exactly as G3 defines it — `inherited store` minus `inherited scrambled` on
preparations-to-first-correct, everything but the labels' meaning held.

**What falsifies it.** Three things, and each is a real possible outcome that would be reported
rather than explained away:

1. **The content effect shrinks as the world hardens.** The record's value would then be an
   artifact of an easy world, not a mechanism that scales.
2. **The content effect stays flat while the total effect grows.** That would mean the harder world
   makes a store more valuable *by existing* — more marks, more cells covered — without its labels
   mattering more. G3's 2×2 separates those, which is why it is stated in those terms.
3. **The controls stop agreeing.** `inherited scrambled` and `inherited gain-zero` must keep
   landing together. If a mechanic pulls them apart, the mechanic has introduced something other
   than label information into the metric and its claim line means nothing until that is found.

---

## 1. Mechanic 1 — K raised

### 1.1 What changes

| | G3 | G4-K |
|---|---|---|
| `N_PREPS` (K) | 5 | **7** |
| `N_ACTIONS` | 10 | **12** |
| `N_IN` | 31 | **33** |
| mapping space `P(K, T)` | 60 | **210** |
| chance hit | 0.200 | **0.143** |
| type-blind | 0.333 | 0.333 |
| `prep_value`, to keep chance EV at zero | 1.0 | **1.5** = (K−1)·`prep_fail` |

Everything else is held: `prep_fail` 0.25, `prep_every` 700, the three era clocks, the metabolism,
the staging, and B's chain-on single phase.

### 1.2 Why this is the right first mechanic

It hardens the world along the one axis the research lineage already knows is load-bearing.
v3.11's finding: *"At six mappings the genetic baseline is large — `fixed` 0.596–0.666 against a
type-blind level of ~0.51 — and cannot be removed by shortening the era."* v3.12's answer was to
enlarge the mapping space, and it worked. K = 7 takes the same lever further: **210 mappings
against 60**, so survival sorting over standing variation has three and a half times as much to
sort through, and within-life learning — and therefore an inherited record — is worth
correspondingly more.

It is also the mechanic with the least new machinery. No new fact, no new clock, no new
dependency: the same world with a bigger answer space. If compounding is not visible here it is
unlikely to be visible in the harder two, and that is worth knowing before building them.

### 1.3 The invariant that must survive

`prep_value = (K−1)·prep_fail`, so the expected value of a chance preparation stays **exactly
zero** and a positive return is still knowledge rather than income. At K = 7 that means
`prep_value` 1.5. `check_chance_ev_is_zero` already enforces it and will refuse the world if the
arithmetic is not done.

**This is the one place K can be raised wrongly and produce a plausible number.** Leaving
`prep_value` at 1.0 makes a chance preparation worth −0.25 — a harder world for a reason that has
nothing to do with the mapping space — and every arm would get worse together while the contrast
between them meant something different.

### 1.4 Engine change

`N_PREPS` is a module constant, not a `Config` field, and array shapes, the action count and the
observation width all derive from it. Raising it is a versioned engine change with its own gate,
recorded in `ENGINE_VERSIONS` like the three before it, and it **invalidates the G1 reproduction**
— `precheck_v3_13.txt` is a K = 5 artifact and cannot be reproduced by a K = 7 engine.

That is not a regression and must not be papered over. The G1 reproduction stays the gate for the
K = 5 engine, which stays in `ENGINE_VERSIONS` and stays runnable; G4 runs a *different world* and
says so. A milestone that quietly broke the reproduction and reported a new number would have lost
the only thing tying this platform to the research lineage.

**Ruled (G4-D1):** K is a `Config` field with a default of 5, not a changed constant. Then one
engine serves both worlds, the G1 reproduction keeps passing at the default, and G4's world is a
parameter rather than a fork. The cost is that every module-level array sized by `N_PREPS` becomes
sized per-run; that cost is paid once, here, rather than at every mechanic.

### 1.5 Claim line

The G3 four-arm succession, re-run at K = 7, 3/3 seeds:

* **content** = `inherited store` − `inherited scrambled` on preparations-to-first-correct;
* **reading** = `inherited store` − `inherited gain-zero`;
* **controls agree**: |scrambled − gain-zero| small against both;
* **compounding** = content(K=7) − content(K=5), which is the G4 number.

Reported beside: the total effect against `fresh store`, and the presence half, so that a
compounding effect that turns out to be presence rather than content is visible as that.

---

## 2. Mechanic 2 — a second era clock

*Specified after mechanic 1 reports.* The shape: a second slow fact on a period that is not a
multiple of `prep_every`, so the two cannot come into phase and a genome cannot track their
conjunction. The design question it raises — whether the second fact is a second mapping, a
rotating forbidden preparation, or a rotating type-weighting — is a DECISION that should be made
against mechanic 1's numbers rather than in advance of them.

## 3. Mechanic 3 — the first compositional fact

*Specified after mechanic 2 reports.* B§5.3 names it precisely: *"a preparation whose correct
choice depends on a prior outcome — which is the first fact no single life can assemble."*

That last clause is the whole point and it is a quantitative claim, not a rhetorical one. The fact
must be large enough that the number of instances an agent can encounter in one bounded life is
smaller than the number needed to determine it — so that a population that only ever learns
within life cannot get there, and one that reads an accumulated record can. **Sizing it is the
central design problem of mechanic 3**, and it has to be done from the measured distribution of
preparations per life, not chosen.

If mechanic 3 works, it is the first result in this project that no single life could have
produced. That is the claim B§2 has been building toward, and it is the one worth being slowest
about.
