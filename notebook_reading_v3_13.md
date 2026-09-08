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

---

## The grid, and what it is required to show

The v3.13 acceptance did **not** support a transmission claim and nothing was written from it.
Gate R passed both arms (z −0.33, −0.80) and the stale-mark ratio was ~2× its null in both record
arms including first-ever preparations (2.02 / 2.04) — but **preparations-to-first-correct did not
improve** (1.96 / 2.04 against 1.92 with no record) and the population hit gain was +0.04 / +0.01.

**Following stale marks while getting no benefit from correct ones is a confound signature, not
reading.** Four instrument changes follow from it.

1. **The matched null.** `(1 − hit)/(K − 1)` assumes wrong choices are uniform. They are not — a
   sorted genome concentrates them, and in the slow arm on exactly the preparation a stale mark
   endorses. Replaced by: for the same endorsed preparation *k′* on the same food type, the rate
   at which this arm chooses *k′* **when no mark is present**, per arm and per seed. The old null
   prints beside it for this build.
2. **`store_gain` is scaled by each agent's own `sym_gain`.** Unscaled it injected +1 while mean
   `sym_gain` was **negative** — an input of the wrong sign and of a magnitude no agent ever sees.
   Re-measured on the acceptance genomes, **innate moved from −0.2016 to −0.0103: the −0.19 was
   the sign.** (Those are restored genomes, so `H` is zero and only the innate side is settled by
   that check; the learned side needs live agents.)
3. **Per-seed on every line.**
4. **The frozen record assay**, below.

### The frozen record assay — the attribution row

Era-boundary snapshot; population frozen exactly as row 3b; the **store carried in as it stood**;
a **fresh independent `π`**, so every mark already present endorses a preparation unrelated to what
it was written about and the binding has to be rebuilt inside the window. 300 steps, four cells,
stale-mark ratio on the matched null over experience in 50-step bins:

| cell | marks | plasticity |
|---|---|---|
| hidden / on | written, read channels zeroed | on |
| visible / on | as they are | on |
| permuted / on | present, label axis permuted | on |
| visible / off | as they are | off |

**Required: rising in visible/on; flat in hidden, permuted and visible/off.**
**Pre-registered: visible/off rising means shared genome + shared context, not reading.**

`assay_selftest` holds the plumbing: hidden/on and visible/off must not differ in the first bin —
before learning, seeing the marks and not seeing them should be the same. Measured 0.869 against
0.976, |diff| 0.107.

### The pre-registered reading — all four required

1. `plastic + noise` and `fixed + record` at a matched-null ratio **≈ 1**;
2. the record arms **above** it;
3. **preparations-to-first-correct lower with a record than without**;
4. the assay pattern above.

**Any one failing means it is not transmission** — and the reading says which confound it was.
