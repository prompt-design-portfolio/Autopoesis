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

---

## 9. What the v3.10 acceptance run established (2 seeds, full phases)

These two are settled, and both are recorded here because they change what the controls mean.

### Row 1b fired, and the cause is genetic era-tracking — not luck, not a broken floor

`fixed` prep hit was 0.625 and 0.699 against a gate of 0.55. Type encounters are balanced
(share of preparations on type A: 0.499 / 0.498 in `random policy`, 0.435–0.541 elsewhere), so
the 0.5 type-blind floor constant is right and the excess is not a rig asymmetry. The elevation
is arithmetic on the per-type split:

    seed 0:  0.625 = 0.5 x 0.872 + 0.5 x 0.377

and the second term is the finding. A pure type-blind genome — "always prep_k" — scores **0** on
the wrong type. 0.377 is not 0. So the genome is not type-blind: it holds a **partial genetic
conjunction**, and the phase-half aggregate hid it because it spans two mapping eras with
different mappings, averaging a *switch* into a false one-high-one-low reading. Per era:

    fixed seed 0  (A,B)  [(0.72, 0.93), (0.56, 0.96), (0.95, 0.34), (0.77, 0.43)]
                  both types above 0.5 in 2 of 4 eras

`prep_gain innate` — the genome's preference for the correct preparation on a synthetic
observation, with no gating in it — confirms it directly: `fixed` **0.995, 1.611**, against
`random policy`'s **−0.102, −0.128**. At `prep_every` = 2000 there are ~12 generations per era
and selection uses them. Hence the change to 700.

### `scrambled`'s elevation is the same mechanism, not lucky H

`scrambled` reached 0.746 / 0.575 on prep hit, which invites reading it as survivorship — agents
whose random `H` happens to help living longer. It is not. Three lines say so:

- **Survivor halves match `fixed` to three decimals.** `scrambled` 0.815 → 0.724, `fixed`
  0.812 → 0.721. An arm whose advantage came from lucky `H` would not track the arm that has no
  `H` at all.
- **First-preparation hit is 0.821 / 0.607** — high, and that measurement is taken before the
  agent has learned anything, so it reads the genome and nothing else.
- **`prep_gain` innate is 0.859 / 0.446**, close to `fixed`'s and far above `random policy`'s
  zero, while its learned `probe_adv` is **−0.008 / 0.114** — essentially nil.

So `scrambled` is a `fixed`-like genome carrying a scrambled, useless `H`. That is exactly what
the control is supposed to be, and it means `plastic − scrambled` is a cleaner contrast than the
survivorship worry suggested — but it also means `scrambled` inherits `fixed`'s era-tracking, so
it is subject to the same gate 1b.

### The route that separates the arms

`plastic` and `fixed` reached comparable hit rates by **opposite routes**, and this is what the
whole rig is now built to read:

| | plastic | fixed |
|---|---|---|
| first-preparation hit (genome) | 0.458, 0.256 | 0.648, 0.761 |
| `prep_gain` innate (genome) | 0.324, −0.149 | 0.995, 1.611 |
| knockout, first window (genome) | 0.485, 0.646 | — |
| `probe_adv` (prep), learned | 6.52, 6.84 | n/a |
| live phase-2 hit | 0.810, 0.715 | 0.625, 0.699 |

`plastic` learns it within life on a genome that is **worse** than `fixed`'s; `fixed` evolves it
into the genome. Four instruments, two of them within-agent, all agreeing.
