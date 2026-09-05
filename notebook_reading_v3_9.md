## 8. Result — the stop condition was invoked

**The recipe world is closed for individual-learner tests.** The full write-up is
`v3_9_finding.md`; the short form:

A multi-step chain whose payoff arrives only at the end is **not discoverable by an individual
learner under autopoietic economics without an instinct**. At ~0.5 attempts per life, an attempt at
chance is worth ≈ **+0.37 energy** against a lifetime income of several meals — about **5%** — and a
four-step approach behaviour cannot be selected out of that differential in twenty generations.

| pass | `fixed` att/life | `random` att/life | P(interact \| at station) vs null |
|---|---|---|---|
| 1 — payoff size | 0.21 | 0.16 | P(int \| on item) 0.124 vs 0.165 |
| 2 — journey length | 0.56 | 0.96 | — |
| 3 — co-location | 0.38 | 0.58 | 0.135 vs 0.156 |

Station density at 3× the agreed band moved the null only 0.96 → 2.16; `tool_value` 1.5 → 8.0
*lowered* it. Co-location halved item supply without creating a gradient.

**Correction to the pre-check reading:** plastic's phase-2 population of 45 was a window transient,
not co-location taxing foraging — a full run at seed 0 held 107–209 with pickups ~1/1k. The chain
costs almost nothing because almost nobody touches it.

**What the rig fixes established.** Fix A restored food learning across the chain: v3.8's 0.65 → 0.50
decay in every arm did **not** recur (`plastic` holds 0.622 and 0.579 in phase 2, `probe_adv` (food)
above 1.0). The one agreed rule-form change is **not** spent — the learner was never tested on a rig
that could show it.

## 8b. Pre-check note — the abstention observed under amendment 2

*An observation from a 1-seed pre-check, not a claim.* Before the chain was co-located, with
`tool_value` 1.5 and a carry tax, `plastic (W2)` **declined 99.7% of station opportunities** and
made **0.006 attempts per life** — 160× below the random-walk null — while being the **only arm not
at the population floor** (153 against 41–52, zero injections) and holding the best safe rate
(0.622). Given a genuine choice, the best learner rejected the chain and specialised in eating.
That reading is what amendment 3 responds to: the failure was opportunity cost, not payoff size.

## 8b. Reading it

In the order of the decision table. **Rows 1 and 2 are stop conditions.**

1. **Row 1, the phase-1 gate.** If v3.1 is not reproduced, the finding is about observation size
   and nothing below is readable.
2. **Row 2, the rig checks — this build exists for these.** (a) does food learning *survive*
   phase 2 (v3.8 saw safe rate fall to 0.500 in every arm), (b) does the oracle survive, (c) are
   contact rates above the `random policy` null. If (a) or (b) fails, **the rig is still broken;
   report and stop, do not tune past it.**
3. **Row 3, and the abstention check inside it comes first.** Attempts below 0.8× `fixed + oracle`
   with `declined` rising is abstention, not knowledge. The prediction is that a two-sided signal
   should not produce abstention; if it does, that is a finding about the rule under mixed signals.
4. **Row 4** is the delayed-credit question, read against `scrambled`, in a world silent on both
   attempt outcomes.
5. **Row 6** is what a null on row 3 buys: with rows 1–2 clean, it is the first earned statement
   about the learner's limit, and the one rule-form change is spent there.

`random policy` is never an outcome. It is the null for contact rates, and it is exempt from
row 0's exclusion — a random walker belongs at the population floor.
