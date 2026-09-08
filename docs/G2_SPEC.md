# G2 spec — the record as the artifact store

> **Status: G2-D1 ruled, and the engine change applied.** The engine is now versioned rather than
> pinned (`civitas_g.manifest.ENGINE_VERSIONS`): `reference` → `G0` → `G2-store`, each with its
> hash, what changed, and how equivalence to its predecessor was checked. `engine_store_selftest`
> keeps measuring the equivalence now that the change is in. §1 and §2 below are kept as written,
> because they are the argument the ruling was made on, and a spec that quietly rewrote its own
> premises after the fact would be worth nothing.
>
> What the ruling unblocked, and what it did not: A2.1's twelve cells now run for the first time,
> and `assay_selftest` exists. G3's population B is no longer blocked by the engine.

A3's row:

| milestone | gate | number it must move |
|---|---|---|
| G2 | the record as the artifact store; Gate R, stale-mark lines, frozen assay reproduce through the store | reproduction diff = 0 |

A5's form follows in §3. Read §1 first: one clause of that gate cannot be met by any amount of
Civitas-side work, and the reason is a property of the engine rather than a gap in the platform.

---

## 1. The blocker: the engine can neither hand out the store nor take one in

Three facts, each checked against `sim_v3_13.py` at `63d42f6`:

1. **`run()` never returns the store.** Its result is
   `dict(log, flips, recipe_changes, chain_start, phase_bounds, n_steps, final_mapping, era_snaps,
   cfg, final)`. `era_snaps` entries are `dict(t, mapping, n_pop, genomes=snapshot(sample))` —
   genomes only. `world.marks` is never copied out. The log carries `mark_density`,
   `mark_mean_abs`, `pi` and `mi_counts`, and none of those can reconstruct a `(T, K, g, g)` array
   of signed marks.
2. **`run()` has no parameter that takes a store in.** `World.__init__` does
   `self.marks = np.zeros((N_TYPES, N_PREPS, g, g))` and `self.pi = self._draw_pi()`,
   unconditionally.
3. **Therefore `frozen_replay` has never seen a store.** It calls `sim_v3_13.run` with
   `init_genomes` and `force_mapping`, which builds a *fresh* `World` — empty marks, freshly drawn
   π. The population it replays is the one that wrote the record; the world it replays in has none
   of it.

The consequence is larger than G2. **A2.1's `store visible / hidden / label-permuted` arms have
never been runnable**, and B§4's `collective_frozen` names them as if they were. G3's claim —
"population B, fresh founders, no genomes crossing, **born into A's record**" — is blocked by the
same fact.

This is not a Civitas-side problem with a Civitas-side fix. The store is created inside `World`,
mutated in place every step, and dropped when `run()` returns. Reconstructing it from the log is
impossible: marks **overwrite**, so the surviving array is not a function of any aggregate the log
records. Re-deriving it would mean re-implementing `run()`, which A1.8 forbids and which would
be a second engine to keep correct.

**So G2 requires a versioned engine change.** The directive anticipates exactly this: *"If v3.13's
result later changes an instrument, that arrives as a versioned change with its own gate."* The
change is specified, built and measured in §2, and is **not applied** pending DECISION G2-D1.

### A related, smaller finding

`Config.frozen`'s comment says energy "is still tracked and still spent — the metabolism runs — it
is simply not lethal." That prose is stale. The code at `sim_v3_13.py:983` and `:1222` pins each
agent's energy to its snapshot value, keyed by identity because `rng.shuffle` reorders the list
every step — which is what A2.1 asks for ("energy pinned by identity") and what makes the
attribution literally true, since energy is an observation channel and would otherwise drift the
policy on its own. **The code is right and the comment is wrong**; no change is proposed, and it is
recorded so the next reader does not trust the comment over the code.

---

## 2. The proposed engine change

`docs/patches/engine-store-capture-injection-and-init-mapping.diff`, against `sim_v3_13.py` @
`3b51b191d953a71691831095e1f631eb69ab07d1d17b180376547ea4fceb84f5`. **Unapplied.**

Four additions, and nothing removed:

| # | change | why |
|---|---|---|
| 1 | `Config.store_snaps: bool = False` | capture, off by default |
| 2 | `run(..., init_store=None)` | injection |
| 3 | at the era boundary, beside the genome snapshot: append `dict(t, mapping, pi, marks=world.marks.copy())` when `cfg.store_snaps` | the record at the same moment as the genomes — they are two different things, and G3 turns on being able to hand the second to a population that never had the first |
| 4 | `store_snaps` in the returned dict | so it outlives the run |

**Why it is safe, measured rather than argued.** The injection writes an array; it draws nothing.
The capture copies an array; it draws nothing. Neither changes a branch when the flag is off. So a
run with `store_snaps=False, init_store=None` must be bit-identical to the unpatched engine — and
that is checked the same way `engine_drift_selftest` already checks the `3d7b228 → HEAD` drift, by
running both engines at one seed across a mapping remap and comparing every shared log field.

> **Measured** — `civitas_g.selftests.store_patch_selftest`, at seed 0 over 800-step phases so the
> run crosses a mapping remap. The patch is applied to a temporary copy, so the working tree stays
> byte-identical while G2-D1 is open.
>
> | claim | result |
> |---|---|
> | both flags off | **32 log rows, 133 shared fields, zero differing, no log fields added, same `final_mapping`** |
> | capture on | **zero differing fields**; one store snapshot at `t = 1500`, `pi = (1, 3, 0, 2, 4)`, density **0.129167** |
> | injection | a world at a different seed reads density **0.152488** with the record against **0.025926** without |
>
> The "without" figure is not zero because that probe writes its own marks over its 50 steps; the
> difference between the two is the injected record. A probe that had been forced to zero would
> have been a weaker check, not a stronger one — it would not have shown the injected store
> surviving alongside new writes.

`analysis_v3_13.py` is **not** touched. Civitas runs the assay itself, calling `sim_v3_13.run`
directly with `init_store`, `init_genomes`, `frozen=True` and `force_mapping` — so `frozen_replay`
and `frozen_knockout` keep working exactly as they do, and the G1 reproduction is unaffected.

---

## 3. The spec, in A5's form

### 3.1 Cells and co-occupancy — unchanged

48 × 48 torus, 2304 cells, wrap on both axes. Agents per cell unbounded (clipped at 3 for the
observation); food exclusive across types, one item per cell; marks `T × K = 15` signed floats per
cell, always present as storage. See `docs/G0_G1.md` §3.1.

**What G2 adds:** the marks array becomes an addressable artifact. A `Record` is
`(marks, pi, provenance)`, where provenance is `(run_seed, arm, t, era_index, engine_sha256,
cfg_digest, mapping, prev_mapping)`.

> **Provenance is at the artifact level, not per mark, and that is a limitation.** §A1.2 (kept)
> asks for provenance on every mark. The engine's marks carry none — a write is
> `marks[ftype, label, y, x] = ±1`, overwriting whatever was there, with no writer identity and no
> timestamp. Per-mark provenance would be a much larger engine change (a parallel `(T,K,g,g)`
> writer-id and timestamp array, tripling the store's memory) and it is **not** proposed here. The
> gap is stated rather than papered over.

### 3.2 Actions — unchanged

Ten. Four moves, `eat`, five preparations (phase 2 only). Reading the record is **not** an action:
it is an observation, so a null cannot be ambiguous between "cannot read" and "did not bother".

### 3.3 Action × state, with the modulator event — unchanged

See `docs/G0_G1.md` §3.3. **No mechanic is added at G2.** That is deliberate: G2's gate is
`reproduction diff = 0`, so anything that moved a number would make the gate unmeetable by
construction. The ratchet starts at G4.

### 3.4 The complete modulator event list — unchanged

Seven events; four signed. `eat_safe +1`, `eat_poison −1`, `prep_ok +1`, `prep_bad −1`, and
`move`, `noop`, `eat_inedible` at 0. The modulator is this table, not the energy delta, and
`modulator_selftest` measures it.

### 3.5 Observation layout — unchanged

31 inputs; indices 26–30 are the K read channels, `sym_gain * marks[type underfoot, :, y, x]`,
here-only, zero with no record or no food underfoot. Five inputs (16–19, 24) remain structurally
dead and are not touched — changing the layout changes every genome and voids the reproduction.

**What G2 adds:** three *store conditions* that feed the same channels.

| condition | store handed to the replay | what it removes |
|---|---|---|
| `visible` | the record as captured | nothing — the reference cell |
| `hidden` | the same array, all marks zeroed | the content, keeping the shape: the channels exist and are zero, exactly as in `memory_reset` |
| `label_permuted` | one permutation of the label axis, applied globally | the *learned binding*, keeping density, signs and cross-cell consistency |

### 3.6 Standing densities as cover fractions — unchanged, and one gap closes

Grid 2304 cells; 8 patches of Chebyshev radius 6 cover 0.587 of it if disjoint; 48 spawn attempts
per step against rot 0.005; mark half-life 173 steps.

Standing **food cover** stays `NOT AVAILABLE` (D10) — it is not in the engine's log and the
proposed patch does not add it, because it is not needed for G2's gate and A5 says one change per
experiment.

**Mark density and mean |mark| are now measured from the artifact** rather than only read from the
log, and by construction agree with it: `Record.density()` uses the engine's own `|mark| > 1e-3`
threshold.

### 3.7 Phases — unchanged for a run; new for a replay

A run is two phases of `PHASE_STEPS`, food-only then chain, on one population.

A **replay** is one phase, `FROZEN_STEPS = 300`, `chain=True`, `frozen=True`: births, deaths and
injection disabled, energy pinned by identity, mapping pinned by `force_mapping`. Nothing can
change but `H`.

### 3.8 Costs and values — unchanged

`analysis_v3_13.WORLD`. `prep_value 1.0`, `prep_fail 0.25`, chance EV **0.000**; `food_value 0.7`,
`poison_value 0.5`; `base_cost 0.006`, `move_cost` = `noop_cost` = 0.002; `prep_every 700`,
`flip_every 300`, `patch_drift_every 400`; `mark_decay 0.004`.

### 3.9 The frozen assay, fully specified (A2.1)

**Twelve cells** per snapshot: `mapping ∈ {matched, shuffled} × eta_scale ∈ {0, 1} × store ∈
{visible, hidden, label_permuted}`.

* snapshot at an era boundary, the last one kept (`era_snap_keep = 2`);
* `shuffled` is a derangement of the matched mapping, so no genotype sorted for it retains any
  advantage;
* the reported quantity is the founder-free preparation hit over the window, plus the per-bin
  curve;
* the claim line is `eta1 − eta0` on **shuffled**, which v3.11's finding pre-registered at ≥ 0.10.

**Nulls are matched (A2.2).** For any stale-mark line computed inside the assay, the null is
`(1 − hit) / (K − 1)` from the same arm, never `1/K`.

**Presence versus content (A2.3).** `hidden` is the presence control for the assay, as
`plastic + noise` is for the run.

### 3.10 The G2 gate

1. `assay_selftest` green — its two clauses (§4).
2. `store_round_trip` and `scrambled_load` green (already are — §4).
3. `store_patch_selftest` green: the patched engine is bit-identical with the flags off.
4. **Reproduction diff = 0, through the store:** the G1 reproduction re-run with
   `store_snaps=True`, giving the same 182/182 fields. If capturing the store moved a number, the
   capture is not a capture.
5. Gate R and the stale-mark lines recomputed **from stored records plus stored rows**, matching
   the G1 values.
6. The frozen assay runs its twelve cells and reports them. **Not a claim** — one seed, and G2's
   number is the reproduction diff.

---

## 4. Self-tests at G2

| self-test | state |
|---|---|
| `store_round_trip` | **built, green** — save, load, byte-identical marks and π |
| `scrambled_load` | **built, green** — density and signs preserved, labels destroyed, and the per-cell scramble proved distinct from a global one |
| `assay_preconditions` | **built, green** — the two clauses of `assay_selftest` that need no injection |
| `assay_selftest` | **built** (D5) — a world with `record="none"` cannot be moved by what it is handed, and a reader with `sym_gain = 0` reads nothing whatever the labels say |
| `engine_store_selftest` | **built** — the store change's trajectory-neutrality, measured against the engine as it was at `G0` |
| `b_founders_carry_no_h` | G3 |

---

## 5. DECISIONs

### G2-D1 — apply the engine patch?

The gate cannot be met without it, and A1.8 says the engine is untouched and its hash is in every
manifest.

*Recommendation:* **apply it**, as a versioned change with its own gate, on four conditions:

1. the bit-identity proof of §2 is green and is registered as a permanent self-test, so a later
   edit cannot quietly break it;
2. the new hash is pinned in `civitas_g.manifest` beside the old one, and every manifest names
   which engine produced each number;
3. the G1 reproduction is re-run against the patched engine and still gives 182/182 — the
   reproduction is the regression test for the patch;
4. `analysis_v3_13.py` stays untouched, so `frozen_replay`, `frozen_knockout` and the committed
   readings keep working unchanged.

*Alternative considered and rejected:* fork the engine as `sim_v3_14.py`. That doubles the thing
A1.8 exists to protect — there would be two engines, both claiming to be the world, and the
reproduction would have to say which. A default-off flag on one engine is strictly less risk than
two engines.

### G2-D2 — which scramble does the inherited-store control use?

*Recommendation:* **per-cell**, and this one is not close. A **global** permutation is *isomorphic*
to the real store: it says "label σ(j) marks what label j marked" everywhere, consistently. A
population born into it faces exactly the learning problem it would face with the real store,
because π is redrawn every era and no genome ever inherits the binding. The arm would read as a
null while information had in fact passed — a broken control of precisely the kind A1.2 says makes
a level wrong. Per-cell independent permutation preserves density and the multiset of signs exactly
(a permutation is a bijection) and destroys the cross-cell consistency a reader could exploit.

Global is still needed — it is the assay's `label_permuted` arm, where the question is different:
whether a population that has *already* bound labels within its own life was reading those
particular labels. Both are implemented; `scrambled_load` checks they are distinct.

### G2-D3 — where does the assay live?

*Recommendation:* **in Civitas, calling `sim_v3_13.run` directly.** `frozen_replay` would need
`init_store` threading through it, which means changing `analysis_v3_13.py` too. Keeping the assay
on the Civitas side means exactly one research file changes, the committed readings are untouched,
and the G1 reproduction stays a valid regression test for the patch.

### G2-D4 — how many store snapshots are kept?

`era_snap_keep = 2` currently bounds the genome snapshots. A `(3, 5, 48, 48)` float64 store is
276 KB raw, ~5.7 KB compressed at the densities measured.

*Recommendation:* **keep the same rolling window of 2**, matching the genomes, so a record and the
genomes at the same boundary are always both present or both absent. G3 needs the *last* record of
population A, which a window of 2 supplies. Raising it is cheap and can be done when a milestone
needs a longer history.

### G2-D5 — per-mark provenance?

*Recommendation:* **no, not at G2.** It would triple the store's memory, require a second and third
parallel array in the engine, and serve no claim line in G2 or G3 — the claims are about what a
population reads, not about who wrote what. Record the gap (§3.1) and revisit if a G4 mechanic
needs mark-level attribution.

---

**G2-D1 was ruled and the change applied.** The four conditions the recommendation was
conditional on:

| condition | state |
|---|---|
| the bit-identity proof is a permanent self-test | `engine_store_selftest`, comparing the current engine against the blob at `G0` |
| the new hash pinned beside the old one, and every manifest names which engine produced each number | `ENGINE_VERSIONS`; `engine_identity()` carries `engine_version` into every manifest |
| the G1 reproduction re-run against the patched engine, still 182/182 | **G2 gate clause 4 — see the write-up** |
| `analysis_v3_13.py` untouched | pinned and checked (`test_analysis_v3_13_was_not_touched`) |
