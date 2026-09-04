## 7. Reading it

In the order of the decision table.

1. **Row 1 first, and it is a stop condition.** If phase 1 does not reproduce v3.1 — safe rate
   `plastic` − `fixed` ≥ 0.03, `probe_adv` (food) ≥ 1.0, `eta2` above `fixed`'s — then the learner
   is not working even in the world that made it, the only deviation left is 24 hidden units
   against v3.1's 16, and that is a finding about observation size. Nothing in phase 2 is readable.
2. **The transition table** (section 4). Population, safe rate, `eta2`, `lam2`, `h_norm` and
   attempts/1k in 500-step bins across 2000 steps either side of the switch. This is where a
   collapse, if there is one, becomes visible — and where `eta2`/`lam2` either hold or start
   falling as they did in v3.6.
3. **Row 3 before row 4.** Attempts/1k flat at zero means the chain was never found and the recipe
   rows are vacuous. Row 4's hit rate is only meaningful over a non-trivial number of attempts.
4. **`scrambled`, not `fixed`, carries row 4.** It has the same plasticity, the same H magnitudes
   and the same power to change the policy — and no information in the modulator.
5. **Row 5 last.** If from-scratch matches staged, staging was unnecessary and the v3.6 failure was
   the scaffold on its own. If from-scratch collapses, staging is the method, and that is the
   transferable claim regardless of what row 4 says.
6. **Row 6 as the sanity check on every gene claim.** `nav_dir` and `nav_here` are wired to nothing
   here; their spread is what a heritable scalar does under drift alone in this many generations.
