## Reading v3.13

**The question:** can an agent use, within its life, a mark whose meaning it cannot have inherited?

Writing is **automatic, costless and universal** — every preparation writes `(label, sign)` for its
food type, with no mark action and no writer gene, so there is no public good and no
write-probability to collapse. Labels are a permutation `π` **redrawn at every remap**, so no genome
can carry what a label means. Reading is an **observation**, so a null cannot be ambiguous between
"cannot read" and "does not bother".

### Read in this order

1. **Gate R** — the matched permutation null, grouped by **π-epoch**. `z ≤ 2.0` means the observed
   cross-era association is no stronger than chance given the epoch count: `π` is doing its job.
   Note the grouping is by epoch, not era, or the slow arm would fire on its own definition.
2. **`sym_gain`** — |gain| in a record arm **minus the no-record arm's**. |gain| rises everywhere on
   drift, so the no-record arm is the baseline.
3. **(i) — a DENSITY CHECK, not transmission.** First-ever preparations split by whether a
   positive mark was present. Confounded: a mark exists only where someone recently *succeeded*.
4. **(ii) — THE BINDING LINE.** Following, split by whether the mark endorses the correct
   preparation, read on the **stale** cell against `(1 − hit)/(K − 1)`.
5. **(ii-newborn) — THE TRANSMISSION LINE.** The same stale-mark ratio over **first-ever
   preparations only**, kept separate from (ii): (ii) pools over a life and so mixes transmission
   with an agent's own within-life binding; a first-ever preparation cannot.
6. `sym_gain` — **reported only**, licensing nothing.
7. Preparations-to-first-correct, corroborating, over agents that reached 5.

### What the pre-check established about the instruments

Two of the specified lines were confounded, and the controls caught both.

**Line (i) is confounded and should be read with that in mind.** A positive mark exists only where
someone recently *succeeded*, so it marks places and times where success is common. The gaps are
positive in every record arm — and **largest in `noise`** (+0.241 against `record`'s +0.120), which
is the proof: the noise arm's labels carry nothing, so the gap cannot be about mark content.

**Line (ii) needed two corrections.** Splitting by whether the mark endorses the correct
preparation separates "read the mark" from "was simply right". And `1/K` is the wrong null for the
stale cell: an agent that knows the answer never agrees with a stale mark, whatever it reads. The
null is `(1 − hit)/(K − 1)`.

Read that way, the pre-check (1 seed, 3000-step phases — **not a result**) shows:

| arm | stale follow | null | ratio |
|---|---|---|---|
| `plastic + record` | 0.041 | 0.124 | **0.33** |
| `plastic + noise` | 0.096 | 0.123 | 0.78 |
| `fixed + record` | 0.051 | 0.147 | 0.35 |
| `plastic + record (slow)` | 0.353 | 0.122 | **2.89** |

Ratio above 1 means the agent **follows** a mark it should not; below 1 means it **avoids** it.
Either way the label was read — you cannot avoid what you cannot see. The noise arm at 0.78 is the
reference for how far from 1 an unread channel sits.

### If (ii-newborn) prints "NOT AVAILABLE"

The counter is accumulated inside the sim at the preparation event and is **not derivable from the
log** — the same class of thing as `final_mapping` in v3.11 and `sr_w` in v3.12. A checkpoint
written before it exists cannot be re-analysed for it; the line prints `NOT AVAILABLE` and names
the arms, rather than guessing. Re-run those arms with the current `sim_v3_13.py`.

Everything else — Gate R, the binding line (ii), the density check, `sym_gain`, population, the
transition table — reads correctly off a checkpoint without it.
