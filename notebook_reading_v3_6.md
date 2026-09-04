## 7. Result (recorded after the run — 6 conditions × 3 seeds, 8000 steps)

**Rows 3 and 13 are null. Rows 7 and 10b fired.** No rule-form change is spent.

| row | outcome |
|---|---|
| 3 — acquisition | **null.** `plastic (W2)` does not beat `fixed` or `scrambled` on recipe hit. |
| 13 — elimination | **null.** An immediate `m = −1` at the station does not rescue it either. |
| 7 — learner broken in this world | **fired.** Safe rate is 0.50 in *every* condition, `fixed` included: the food scaffold removed the positive control entirely. |
| 10b — plasticity selected off | **fired.** `eta2` is selected off and `lam2` selected short — **more so with an informative modulator than with a scrambled one.** |

### What this says, and what it does not

The learner grown in the flip world **does not transfer into the scaffolded recipe world.** That is
the finding. It is *not* a finding that the local rule cannot bridge station → nut, because rows 7
and 10b fired: the learner was not working in this world to begin with, and its plasticity was
being selected away before the recipe question was reached. Row 7 in the pre-registered table says
exactly this — *fix that first, no rule-form change, and no claim about delayed credit* — so the
one agreed rule-form change is **not** spent.

The `lam2` detail is the sharpest result and the one that was not anticipated. Selection shortens
the eligibility trace **more** when the modulator carries information than when it is scrambled.
A trace that reaches back to the station is worse than useless in this world: it lands credit on
the navigation that the scaffold was driving anyway. The scrambled control is what makes that
readable — without it, a short trace would just look like plasticity being neglected.

Safe rate at 0.50 in `fixed` as well is the diagnostic for row 7. The pre-run checks had already
flagged the positive control as failing (`probe_adv (food)` 0.068/0.155 against an acceptance of
≥ 0.5; safe rate −0.002/+0.007). Four tuning passes did not fix it. The mechanism identified there
holds up: the instinct's interact push is ~5 logits, a learned preference of ~0.4 logits cannot
move the argmax, so plasticity does not pay, so `eta2` sits at drift, so `H` stays small. The grid
confirms it at n=3 and adds that the same scaffold flattens `fixed` too.

**Per-seed numbers to paste in.** The block below is empty because the grid was run outside this
session and its printed summary has not been pasted back. Fill it from the run's own output —
`summary()` prints every one of these per seed:

```
row 3   plastic - fixed        [ , , ]        plastic - scrambled   [ , , ]
row 13  plastic+fail - fixed+fail  [ , , ]    fixed+fail - fixed    [ , , ]
row 7   safe_rate  fixed [ , , ]  plastic [ , , ]  scrambled [ , , ]
        probe_adv (food) plastic [ , , ]
row 10b eta2  fixed [ , , ]  plastic [ , , ]  scrambled [ , , ]
        lam2  fixed [ , , ]  plastic [ , , ]  scrambled [ , , ]
        (the claim "more so with an informative modulator" is plastic's lam2 BELOW scrambled's)
row 2   ceiling - fixed        [ , , ]        (pre-run at n=2: 0.082, 0.065)
```

### Alternative explanations, before they are asked for

- **The world was tuned nine times.** Every pass was judged against `fixed`, the ceiling, or the
  positive control, never against a plastic condition's recipe hit — but that is a discipline, not
  a proof, and the ceiling never reached the 0.10 headroom the test needed (best viable 0.082 /
  0.065). A null on row 3 is therefore partly a statement about the test's power.
- **`lam2` shorter with information than with noise** could be selection acting on something the
  trace length co-varies with rather than on credit assignment directly. The clean follow-up is
  the knockout (`eta_scale = 0`) on saved genomes, which v3.2 already has the machinery for.
- **Safe rate 0.50 in `fixed`** could be depletion rather than an inability to discriminate:
  `crop_safe` was ~0.5 in v3.6, not the 0.34 of the v3.2 learner worlds, which argues against
  depletion — but it is worth reading off the run.

### Qualification added after the v3.8 acceptance checks

The attribution above — *the food scaffold removed the positive control* — does not survive
unqualified. v3.8's phase 1 has **no scaffold of any kind** and reproduces v3.1 (safe rate 0.627 /
0.641 for `plastic` against 0.517 / 0.559 for `fixed`; `probe_adv` (food) 1.33 / 2.10). But when
the chain switches on for that same population, safe rate falls to **0.500** — with no scaffold
present. So the chain collapses food discrimination on its own, and the scaffold is not established
as the cause of v3.6's flat positive control. It may have contributed; it is not needed to explain
it. The candidates are separated in v3.8: `nut_share`, `crop_safe`, and time spent on the chain
rather than eating. (n = 2, from an acceptance check, not a result.)

### What follows

Row 7's remedy is not a rule change: it is to stop handing the learner an instinct that overrides
it. That is v3.8 — grow the learner in the world that made it (v3.1's flip world, no scaffold at
all), then switch the chain on for the same population. Whether the recipe is learnable is not
asked again until the learner is demonstrably working in the world where it is asked.
