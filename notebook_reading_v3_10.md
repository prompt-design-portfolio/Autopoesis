## 8. Reading it

**Rows 1a and 2 are stop conditions.**

1. **Row 1a** — phase 1 must be v3.1. If not, nothing below is readable.
2. **Row 1b** — `fixed` must not hold the *conjunction*. A type-blind 0.5 is allowed; read the
   per-type split to tell them apart.
3. **Row 2** — food learning in the first mapping era **read on safe rate** (`plastic` − `fixed`
   ≥ 0.03), with the corrected `probe_adv` (food) in the first 500 steps corroborating; then
   prepared meals per life ≥ 3 in `fixed`, and P(prep | on food) against the null. The
   **eat → prep shift timing** and **prep share by era** are printed as diagnostics, not gates.
   A probe that decays *after* the first era is expected: once the mapping is known, preparation
   pays on any food and the safe/poison fact stops mattering.
4. **Row 3** — the abstention line first, then the two result lines, then **the per-type split**:
   a conjunction is *both* types above 0.5, not one at 1.0 and one at 0.
5. **Rows 3a and 3b — survivorship.** `plastic` − `scrambled` on hit rate is a *contaminated*
   contrast: agents whose random `H` happens to help live longer, enriching the standing
   population with nothing learned. It stays required, but the attribution is carried by the
   within-agent lines. The **survivor curve** (preps 1–5 vs 6–10 over agents that reached 10)
   must **rise in `plastic` and stay flat in `scrambled`** — every agent counted contributes both
   halves of its own curve, so a rise is the same individuals later in their own lives. The
   **knockout** replays late genomes with `eta_scale = 0` in a fresh world: if the advantage lives
   in `H`, both arms fall to the type-blind floor. Note `probe_adv` is computed over the *living*
   and so is itself partly survivorship-selected; the survivor curve is not. The pre-checks had
   `scrambled` at 0.722 on hit rate but only 0.342 on the probe, against plastic's 5.624 — that
   gap is what these rows are here to adjudicate.
6. **Row 5** — a null here, with rows 1–2 clean, is the first earned statement about the learner's
   limit: no approach behaviour required, credit immediate and two-sided, opportunity on every meal.
