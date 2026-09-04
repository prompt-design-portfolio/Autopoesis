## 8. Result (recorded after the run — 5 conditions × 3 seeds, 8000-step phases)

**Rows 1, 2 and 5 positive. Row 4 null. The oracle arm collapsed.**

| row | outcome |
|---|---|
| 1 — phase-1 gate | **positive.** v3.1 is reproduced at 60 inputs and 24 hidden. The observation size is not the problem. |
| 2 — transition | **positive.** The grown population survives the chain and keeps `eta2` and `lam2` through it. In v3.6 both were selected away; here they are not — so **the scaffold did that, not the chain.** |
| 5 — staged vs scratch | **positive.** From-scratch collapses where staged survives. **Staging is the method**, and that is the transferable result of this build. |
| 4 — delayed credit | **null.** No recipe acquisition against `scrambled`. |
| 7 — oracle arm | **collapsed** on a −1 at the station. |

### What this is worth, and what it is not

Rows 1, 2 and 5 are the first clean positives on this line since v3.2, and row 5 is a method result:
**the learner has to be grown in the world that made it and handed the new mechanic afterwards.**
Row 2 also retires the v3.6 reading — `eta2` off and `lam2` short were the *scaffold*, not the
chain, because with no scaffold anywhere both survive the same transition.

Row 4 is **not** a result about the learner. The oracle collapsed, and the audit
(`build_plan_v3_9_onward.md` §1) found why: one action did attempt / crack / pickup / eat in
priority order, so a −1 for a wrong attempt fell on the shared interact action and suppressed
eating with it. On a rig where declining is not a policy and the chain is ambient, a null on row 4
carries no information about the rule. v3.9 rebuilds the rig; v3.8's row 4 is void as a test.

**Per-seed numbers to paste in.** Empty because the grid was run outside this session and its
printed summary has not been pasted back. `summary()` prints every one of these per seed:

```
row 1   safe_rate  plastic [ , , ]  fixed [ , , ]   probe_adv (food) [ , , ]
        (and which seeds used the row-0 published-range fallback)
row 2   pop phase 2  plastic [ , , ]   eta2 P1->P2 [ , , ] -> [ , , ]   lam2 [ , , ] -> [ , , ]
        first-bin safe-rate drop [ , , ]   h_norm P1->P2   nut_share   crop_safe
row 4   plastic - scrambled [ , , ]   probe_adv (recipe) [ , , ]   trace_recency [ , , ]
row 5   staged vs scratch: pop [ , , ] vs [ , , ]   attempts/1k [ , , ] vs [ , , ]
row 7   plastic+fail - fixed [ , , ]   e_fail/1k [ , , ]   pop [ , , ]
```

### The four-candidate reading of the safe-rate drop

The transition table's first-bin drop and `h_norm` decide between the basis shift (immediate,
`h_norm` unchanged), modulator swamping, the time budget, and depletion. Read it off the run and
record which one it was — it is the same question v3.9's rig fix A is aimed at.
