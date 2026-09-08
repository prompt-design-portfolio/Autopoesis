# G2 — the record as the artifact store

A4.3's form: the spec, the DECISIONs and how each was ruled, the pre-check numbers, the gate
numbers per seed, and the effect on the G1 and G3 numbers. The spec is `docs/G2_SPEC.md`; the
audit it rests on is `docs/G0_G1.md`.

---

## 1. What G2 turned out to be about

The gate reads *"the record as the artifact store; Gate R, stale-mark lines, frozen assay reproduce
through the store"*. Two of those three were already reproducing at G1 — Gate R and the stale-mark
lines come from log rows, and G1 recomputed all 182 fields from the database. The third was not
merely unbuilt. It was **unrunnable**, and had been since the frozen replay was written:

* `run()` never returned the store. Its result is `log / flips / recipe_changes / chain_start /
  phase_bounds / n_steps / final_mapping / era_snaps / cfg / final`, and `era_snaps` entries carry
  genomes only.
* `World.__init__` zeroed the marks and redrew π unconditionally; `run()` had no parameter that
  took a store in.
* So `frozen_replay`, which calls `run()` with `init_genomes`, replayed the population that wrote
  the record into a world that had none of it.

A2.1 specifies `store visible / hidden / label-permuted` as arms of that replay, and B§4's
`collective_frozen` names them. **They had never run.** G3's "population B born into A's record" was
blocked by the same fact.

It could not be worked around from the Civitas side. Marks **overwrite**, so the surviving array is
not a function of any aggregate the log records; re-deriving it would have meant re-implementing
`run()`, which A1.8 forbids and which would be a second engine to keep correct.

---

## 2. The engine change, and how A1.8 survives it

A1.8: *"The compute engine is untouched. `sim_v3_13.py` runs byte-identical; its hash is in every
manifest."* That is a claim about a **named** engine, not a claim that it can never change — the
directive's own escape hatch is that an instrument change *"arrives as a versioned change with its
own gate"*. So the engine is now versioned rather than pinned:

| version | sha256 | change | equivalence to its predecessor |
|---|---|---|---|
| `reference` | `fa577b9c…` | the blob that produced `precheck_v3_13.txt` | — |
| `G0` | `3b51b191…` | the four `(ii-newborn)` counters (`41c100d`) | `engine_drift_selftest`: **129 shared log fields identical** across a mapping remap |
| `G2-store` | `d5bb11c8…` | capture and injection | `engine_store_selftest`: **bit-identical to `G0` with the flags off, and still identical with capture on** |

`verify_engine()` names the version on disk and **fails on an unrecorded hash** — which is the
failure a plain pin could not distinguish from a deliberate change, and the one that actually
matters: a stray edit no manifest can name. `engine_identity()` carries `engine_version` into every
manifest beside the hash.

The change itself is four additions and nothing removed: `Config.store_snaps` (default `False`),
`run(..., init_store=None)`, an era-boundary copy of `world.marks` and π beside the genome
snapshot, and `store_snaps` in the returned dict. `analysis_v3_13.py` is untouched (G2-D3), so
`frozen_replay` and `frozen_knockout` still work exactly as they did — which is what keeps the G1
reproduction a valid regression test for the engine change.

---

## 3. The assay, running for the first time

Twelve cells: `mapping ∈ {matched, shuffled} × eta_scale ∈ {0, 1} × store ∈ {visible, hidden,
label_permuted}`.

Three matching decisions, each of which could quietly have turned a control into something else:

1. **The store conditions differ only in what the population starts with.** All three keep
   `record = "real"`, so agents go on writing during the window exactly as in a live run. Setting
   `record = "none"` for `hidden` was the obvious alternative and is wrong: it removes the writing
   as well as the content, so the arm would differ from `visible` in two ways and a gap could be
   either.
2. **`label_permuted` is a GLOBAL permutation.** The assay asks whether a population that already
   bound labels *inside its own life* was reading those particular labels; a global permutation
   breaks that binding while preserving density, signs and cross-cell consistency. This is the
   opposite of what B§5.2's inherited-scrambled control needs, and swapping them would break
   whichever it was used for.
3. **`sym_gain` cannot be zeroed with `cfg.sym_gain_lock` in a replay.** The lock is applied in
   `Agent.__init__` and at reproduction; `restore()` then writes every genome field back over the
   top, and `sym_gain` is in `GENOME`. A replay that set the lock would have run silently at the
   snapshot's own gain. It is zeroed in the genome copy instead, which is the only place that
   survives `restore`.

### Pre-check numbers — mechanism, not result

One seed, a 120-step window on an 800-step run. **Nothing here is a claim**; it is the twelve cells
proving they can be filled.

```
  store           mapping     eta      hit    first     last     pop       n
  visible         matched       0    0.373    0.364    0.373     300    1587
  visible         matched       1    0.512    0.407    0.607     300    1426
  visible         shuffled      0    0.274    0.308    0.292     300    1610
  visible         shuffled      1    0.445    0.334    0.533     300    1478
  hidden          matched       0    0.379    0.348    0.436     300    1539
  hidden          matched       1    0.544    0.440    0.670     300    1436
  hidden          shuffled      0    0.284    0.307    0.248     300    1575
  hidden          shuffled      1    0.460    0.332    0.526     300    1391
  label_permuted  matched       0    0.361    0.372    0.389     300    1582
  label_permuted  matched       1    0.550    0.412    0.622     300    1472
  label_permuted  shuffled      0    0.261    0.290    0.225     300    1633
  label_permuted  shuffled      1    0.460    0.301    0.602     300    1445
```

* **claim line** (`eta1 − eta0`, shuffled, visible store): **+0.171**, against v3.11's
  pre-registered ≥ 0.10. On one seed and a 120-step window, which is a fifth of A2.1's 300.
* **store effect** at shuffled/eta1: visible 0.000, hidden **+0.016**, label_permuted **+0.016**.

The store effect is the honest part to read carefully. It is **flat, and slightly negative for the
real store** — the two arms with no usable label information score marginally *higher*. That is
what a population with `sym_gain` near zero should produce: it is not reading the record, so
handing it one changes nothing, and the ±0.016 is noise on 1400 preparations. The instrument is
working; the population is not yet using the channel. Reading that gap as a null about
*transmission* would be reading a 120-step pre-check as a result.

The population is constant at 300 across all twelve cells, which is the frozen replay doing its
job: births, deaths and injection off, energy pinned by identity, so nothing but `H` can move.

---

## 4. Reproduction through the store (gate clause 4)

G2-D1's recommendation made the ruling conditional on the G1 reproduction being re-run against the
patched engine and still giving 182/182. That re-run is in flight; what is already measured is
stronger than the smoke test the spec proposed, so it is recorded here first.

### 4.1 Row-identity at full reference length

Rather than an 800-step probe, the same arm's **stored rows** were compared between the two
databases: the G1 campaign (engine `G0`, capture **off**) and the G2 campaign (engine `G2-store`,
capture **on**), same seed, same 3000-step phases.

```
log rows:            120 vs 120
shared log fields:   133   differing: []
final_mapping:       (3, 2, 4) vs (3, 2, 4)
```

**Row-identical.** Not "the summary statistics agree" — every field of every era row, at the length
the reference itself was produced at, with capture running. That is the trajectory-neutrality claim
at full strength, on real data rather than on a probe.

It is worth being precise about what this does and does not show. It shows that turning capture on
does not perturb the run. It does **not** by itself show that the reading still reproduces the
reference — identical rows must produce identical fields, but the field-level diff is what the gate
asks for, and it is what the re-run is for.

### 4.2 The field-level diff

*(pending — the five-arm campaign)*

---

## 5. DECISIONs, and how each was ruled

| | decision | ruled |
|---|---|---|
| G2-D1 | apply the engine patch? | **applied**, under all four conditions. The engine is versioned, not re-pinned: `verify_engine()` names the version on disk and fails on an unrecorded hash |
| G2-D2 | which scramble for the inherited-store control? | **per-cell**. A global permutation is isomorphic to the real store, so the arm would read as a null while information had passed. Global is the assay's arm, where the question is different |
| G2-D3 | where does the assay live? | **in Civitas**, calling `sim_v3_13.run` directly. Exactly one research file changed, so `frozen_replay` and `frozen_knockout` are untouched and the G1 reproduction stays a valid regression test for the engine change |
| G2-D4 | how many store snapshots? | **the same rolling window of 2** as the genomes, so a record and the genomes at one boundary are always both present or both absent |
| G2-D5 | per-mark provenance? | **no, not at G2.** It would triple the store's memory and serve no claim line here or at G3. Provenance is at the artifact level and the gap is recorded rather than papered over |

---

## 6. What G2 leaves for G3

The engine can now hand a record to a population that did not write it. That is precisely what
B§5.2 needs — *"Population B — fresh founders, staged identically, no genomes crossing — is born
into A's record"* — and the pieces are in place: `init_store` on the engine, `ScrambleMode.PER_CELL`
for the `inherited scrambled` arm, `sym_gain_lock` for the gain-zero arm, and the store as a
durable artifact with its own hash.

Two things G3 will need that do not exist yet:

* **`b_founders_carry_no_H`** (B§6). At G1 the engine refuses `init_genomes` outright; G3 must
  allow it for population B while proving that nothing learned crossed. `snapshot()` already saves
  genomes only — *"Nothing learned (H) is saved"* — so the self-test is a check on that promise
  rather than a new mechanism.
* **Era-clock alignment between A and B** is flagged in B§5.2 as a DECISION, with a pre-registered
  reading attached: *a null under a misaligned clock is stale culture, not absent culture, and is
  reported as such.* That has to be settled before B's arms are run, not after.
