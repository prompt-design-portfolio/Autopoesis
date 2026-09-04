## 8. Pre-check note — the abstention observed under amendment 2

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
