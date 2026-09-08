# G1 — population provider, and the reproduction gate

A4.3 asks a milestone write-up to state the spec, the DECISIONs and how each was ruled, the
pre-check numbers, the gate numbers per seed, and the effect on the G1 and G3 numbers. This is
that, for G1. The G0 audit it was built on is `docs/G0_G1.md`; the numbers section is filled from
the run, not from expectation.

---

## 1. What was built

`civitas_g/`, a rebuild. There is no agent in it, and that absence is the design:

> **The learner is the only thing that thinks.** Civitas provides persistence, measurement and the
> environment; it never provides cognition. (A1.1)

| module | what it is |
|---|---|
| `world/spec.py` | the world of record, pinned and checked; the chance-EV invariant; the era clocks; the derivable half of the standing densities |
| `world/adapter.py` | the vector domain adapter — ten actions, the seven-event modulator table, the 31-input layout, §47's leak rule and no-self-echo as measurements |
| `world/arms.py` | B§4's seven arms mapped to the research names, plus the slow-label arm the reference summary carries |
| `world/engine.py` | the whole of Civitas's contact with `sim_v3_13.py` |
| `persistence/` | campaigns, runs, era rows, era snapshots, reproductions, on SQLite and PostgreSQL |
| `provider/population.py` | the population provider |
| `reading/` | recompute the reading from stored rows; parse a reference summary; diff the two |
| `selftests.py` | B§6's registry, gaps included |
| `g1.py`, `cli.py` | the milestone and the command line |

### The provider is narrower than §19's, on purpose

A §19 provider supplies cognition to an episode. This one supplies a *population*: it starts the
engine and writes down what the engine produced. It holds no policy, no belief state, no LLM and no
per-step surface. `touches_world_only_at_era_boundaries()` checks the last of those as a property
rather than asserting it in prose — `run()` takes no callback, the engine's `resolve_action`,
`World`, `Agent` and `run` are unreplaced, and nothing in the module subclasses them.

---

## 2. Three findings from G0, built in as mechanisms

Prose in a spec does not survive contact with a refactor. Each of these is a check that runs.

### 2.1 The world of record is `analysis_v3_13.WORLD`, not `Config()` (F7)

`sim_v3_13.Config`'s defaults are v3.10 leftovers. Building a bare `Config()` builds a different
experiment, and the difference is exactly the one the world exists to exclude:

| | `Config()` default | `WORLD` |
|---|---|---|
| `prep_value` / `prep_fail` | 1.5 / 0.5 | **1.0 / 0.25** |
| chance EV of a preparation | **−0.100** | **0.000** |
| `prep_every` | 2000 | **700** |
| `spawn_per_patch` | 3.0 | **6.0** |
| `max_pop` | 400 | **1000** |

`build_run_spec` calls `check_world` and `check_chance_ev_is_zero` before it constructs anything,
and a world that has moved raises `WorldMismatch` rather than producing numbers. The reasoning is
in the exception text: a number produced against an unexpected world is not a number that failed a
gate, it is a number about a different experiment, and the only safe thing to do with it is not to
produce it.

### 2.2 The modulator is a table, not the energy delta (F8, D7)

The directive says "the agent's own energy change as the modulator". `resolve_action` says the
opposite in terms — *"The modulator is NOT the agent's own energy change in general; it is this
table"* — and that independence is load-bearing: it is why v3.12 could halve `prep_fail`, slowing
the survival sorting that was swamping the learner, without touching the learning signal.

Ruled: correct the directive, and check it. `modulator_is_independent_of_economics` runs the same
world at two `(prep_value, prep_fail)` pairs a factor of four apart and compares the modulator
sequence step by step.

**Measured: 400 steps, modulator identical under a 4× change in the economics.**

This matters for G4, not for G1. The ratchet adds one mechanic at a time and each has its own claim
line; if Civitas had recorded the modulator as the energy delta, the first economics change would
have moved the learner silently and every per-mechanic claim would have been confounded from the
start.

### 2.3 HEAD's engine reproduces the reference blob's trajectory (F3)

`precheck_v3_13.txt` was produced by `sim_v3_13.py` at `3d7b228`; HEAD's differs. Reading the diff
says the difference is four added counters, two added log fields, and one `else:` rewritten as
`elif True:` — no RNG consumed, no branch changed. The entire reproduction rests on that reading
being right, so it is checked rather than trusted: `engine_drift_selftest` extracts the old blob,
runs both at one seed across a mapping remap, and compares every shared field.

**Measured: 129 shared log fields identical between `3d7b228` and HEAD; the only difference is the
four counters `41c100d` added.**

---

## 3. Two disciplines that are easy to lose

### 3.1 NaN is a measurement, and storage is where it gets destroyed

PostgreSQL's `JSONB` rejects NaN and Python's `json` writes a bare `NaN` token that is not valid
JSON. Both obvious fixes are wrong in the way A5 names: dropping the key makes an absent
measurement look like a field the sim never recorded, and coercing to `0.0` makes it look like a
measured zero. So non-finite doubles are encoded as tagged strings and decoded back to exactly the
float they were, and the round-trip is tested on real log rows rather than on a synthetic dict —
the fields that break a round-trip are the ones nobody thinks to synthesise.

### 3.2 The reading is recomputed from the rows, not from memory (D12)

Otherwise "both backends" is vacuous. The numbers come from numpy either way, so running the
simulation twice against two databases would prove nothing about either. Each `(arm, seed)` is run
**once** and its result written to every backend; each backend's rows are then read back and the
reading recomputed from what came out. Any disagreement is a round-trip defect, which is the only
thing a backend can be wrong about here.

`analysis_v3_13`'s readers take a `log` of dicts plus `phase_bounds`, `n_steps`, `chain_start` and
`cfg` — so a faithful round-trip of those dicts makes the *entire* reading recomputable from the
database.

---

## 4. The gate

Six clauses. The numeric one is the least interesting.

1. **Every field matches at the width the reference printed it.** A summary records only what it
   printed; comparing a stored `0.05074321` against a printed `0.5074` at any tolerance tests the
   print width, not the number. Columns are compared at their own width — `z` at two decimals,
   `mark_dens` at four — because using three for both would be wrong in opposite directions.
2. **On the reference's own seeds.** Recovered, not recorded — §5.
3. **From the persisted rows on both backends, which must agree** at float noise, not at three
   decimals.
4. **Every self-test green first.** A reproduction computed after a failed self-test is a number
   from an instrument known to be broken.
5. **The code hashes in the manifest**, naming which blob each number came from.
6. **Absent is never zero.** A field on one side only is `missing` and fails; NaN against NaN is an
   agreement about an absence; NaN against a number is the worst kind of mismatch, never a skip;
   and an empty comparison does not pass — `all([])` is `True`, and an acceptance report over zero
   fields reporting success is precisely the defect this rule exists to prevent.

---

## 5. The reference invocation, recovered rather than recorded (F5, D4)

`precheck_v3_13.txt` records neither its seeds nor its configuration. The `sr_w` lesson — *every
parameter that a reader needs is recorded with the data* — fails on the one artifact G1 exists to
reproduce. What is recoverable from the file, and on what basis:

| | value | basis |
|---|---|---|
| arms | all five of `analysis_v3_13.VARIANTS` | every table prints all five |
| `phase_steps` | **3000** | every transition table is headed "switch at t = 3000" |
| `prep_every` | **700**, not overridden | Gate R reports 5 π-epochs for the fast arms and 2 for the slow arm, whose `label_every` is 3 × 700 |
| seeds | **one** | no per-seed table anywhere, and `plastic` and `plastic + record (slow)` print identical phase-1 bins, which they can only do on one seed |
| seed value | **NOT RECOVERABLE** | almost certainly 0, since every committed stage starts there — but that is inference, and it is recorded as inference |

This configuration matches **no committed `MODE`** in `make_notebook_v3_13.py` (`quick` is 3 arms
at 1500, `acceptance` 3 arms at 8000, `grid` 5 arms at 8000). The producing invocation is not
committed either.

**Ruled (D4):** try seeds 0, 1, 2 in order and record which matched. If none does, declare the
artifact unreproducible and fall back to `v3_11_grid_summary.txt`. Do **not** vary the
configuration until the numbers agree — A1.2 calls that failing.

---

## 6. Self-tests

Eleven available, all green. Four are new to this build; each is attached to something the audit
found.

| self-test | source | result |
|---|---|---|
| `world` | `civitas_g.world.spec` | **PASS** — world matches `EXPECTED_WORLD`; chance EV +0.0000; 31-input layout intact; leak rule holds |
| `world_semantics` | `sim_v3_13` | **PASS** |
| `record_semantics` | `sim_v3_13` | **PASS** |
| `founder_tag` | `sim_v3_13` | **PASS** |
| `replay_mapping` | `sim_v3_13` | **PASS** |
| `learning_rule` | `sim_v3_13` | **PASS** |
| `frozen` | `analysis_v3_13` | **PASS** |
| `no_self_echo` | new | **PASS** — 15 (type, preparation) pairs; the cell is consumed every time |
| `modulator` | new | **PASS** — 400 steps, identical under a 4× change in the economics |
| `engine_drift` | new | **PASS** — 129 shared fields identical between `3d7b228` and HEAD |
| `row_round_trip` | new | **PASS** — real log rows round-trip exactly, `mi_counts` and NaNs included |

Four are not available, and the reason differs in a way that matters:

| self-test | milestone | why |
|---|---|---|
| **`assay_selftest`** | **G2** | **Named by A2.1 as an *existing* reference and by B§6 as a gate. It is nowhere in the repository (F6).** Registered as a named gap so it appears in every report rather than silently not running. |
| `store_round_trip` | G2 | B§6's form needs the store as an artifact store. The row round-trip is the G1 half of it and does run. |
| `scrambled_load` | G3 | There is no load until a store outlives a run. |
| `b_founders_carry_no_h` | G3 | There is no population B; at G1 the engine refuses `init_genomes`. |

An unavailable **G1** test halts a read. An unavailable G2 or G3 test does not — it is a milestone
that has not happened, not an instrument that is broken.

---

## 7. The twelve DECISIONs, and how each was ruled

| | decision | ruled |
|---|---|---|
| D1 | which summaries are the reference | `precheck_v3_13.txt` primary, `v3_11_grid_summary.txt` secondary; `precheck_v3_12.txt` **recorded, never gated** — its producing code does not exist at the commit that added it (F4) |
| D2 | which blob is "the producing code" | the blob at the summary's own commit. Both hashes recorded; the equivalence to HEAD is measured, not assumed (§2.3) |
| D3 | text diff or numeric diff | numeric, on a parsed `(table, arm, column)` field set at each column's printed width. The text diff is produced beside the gate and is not part of it |
| D4 | the unrecorded invocation | recovered with a stated basis per element; seeds tried in order and the match recorded as *recovered*. No configuration search (§5) |
| D5 | `assay_selftest` missing | specified now, built at G2; registered as a named gap so it cannot be quietly skipped |
| D6 | `v3_12_finding.md` missing | proceed on `spec_v3_12.md`, `precheck_v3_12.txt` and the `WORLD` comments; the absence is recorded in the manifest, not reconstructed |
| D7 | the modulator's definition | corrected, and checked by a self-test (§2.2) |
| D8 | the `ExperimentArm` enum | **not carried into `civitas_g`.** The G package stores the arm as the manifest name and `get_arm` names a retired arm as retired rather than raising. No migration is needed because there is no deployed G database; the original package's enum is G0 cleanup, not a G1 dependency |
| D9 | five dead observation inputs, `type_spawn_w = None` | changed neither. Changing the layout changes every genome and voids the reproduction; both are recorded as measured facts and are G4 candidates |
| D10 | standing cover recorded nowhere | **partially unmet, and said so.** The provider records mark density and mean \|mark\| per era. Food cover and mark age are **not in the engine's log**, and A1.8 forbids adding them here, so both report `None` with the reason. Adding them is a versioned engine change with its own gate |
| D11 | the knowledge layer | dormant, not removed — G2 makes the record the artifact store and the provenance machinery is its likely home |
| D12 | what "both backends" means | each run executed once and written to both; the reading recomputed from each backend's rows; the two compared at float noise (§3.2) |

D10 is the one that came out differently from the recommendation. The audit asked the provider to
fix the missing cover measurement; it cannot, at G1, without touching the engine. Reporting it as
unavailable with the reason is the honest form of A5's rule, and it is worth being plain that this
leaves the `sr_w` lesson only half-applied until an engine change lands.

---

## 8. Reproduction

**Reproduction diff = 0. The G1 gate is met.**

| | |
|---|---|
| reference | `precheck_v3_13.txt` @ `89b493e8de42…` (commit `3d7b228`) |
| producing code | `sim_v3_13.py` @ `fa577b9c…`, `analysis_v3_13.py` @ `844b1487…`, both at `3d7b228` |
| run with | HEAD's engine @ `3b51b191…`, on the drift equivalence measured in §2.3 |
| invocation | 5 arms × 1 seed at 3000-step phases, `prep_every` 700 — **recovered, not recorded** |
| seed | **0**, recovered by search; the file records none |
| campaign wall clock | 17.3 min (3.8 + 4.2 + 4.0 + 1.3 + 4.0 per arm) |
| G0 hashes | 11/11 verified |

```
precheck_v3_13 on postgresql: 182/182 fields matched  ->  DIFF = 0
precheck_v3_13 on sqlite:     182/182 fields matched  ->  DIFF = 0
backends agree: yes
```

182 fields across nine tables and five arms — Gate R pooled and its matched permutation null,
`sym_gain` and its baseline-differenced form, `store_gain` learned against innate, the density
check, the stale-mark line with A2.2's matched null, preparations-to-first-correct, and the
population and store table. Every one recomputed from `g_era_rows` on each backend, and every one
equal to what the file prints at the width the file prints it.

The seed matters more than it looks. The reference records neither its seeds nor its
configuration, so "on their own seeds" had to be recovered from what the file happened to print —
"switch at t = 3000" in every transition table, five π-epochs against the slow arm's two,
identical phase-1 bins for two arms that differ only in phase 2. **That configuration appears
nowhere in the repository**, and it reproduces to the last decimal. The inference was right, and
it is recorded as an inference regardless.

### The two defects the first run found, both mine

The first gate run reported **144/225 matched, 0 differ, 81 missing**. Nothing disagreed; 81
fields were compared against nothing. Both causes were in the new code, and both are now
regression-tested:

1. **The research name must be the `VARIANTS` key, not the directive's description of it.** B§4's
   table calls the noise arm `plastic + noise record`; `analysis_v3_13.VARIANTS` — and therefore
   every summary ever printed — calls it `plastic + noise`. Taking the directive's phrasing split
   38 fields onto each side with every value identical underneath (`gap` 0.241, `n_stale` 22013,
   `pooled` 0.343, and so on). B§4's phrase now lives in `holds_or_removes`, where it is a
   description rather than a key.
2. **An arm with no record contributes no permutation row.** `v313_precheck` prints
   `plastic  --  (no record)` and moves on: there is no label axis to permute, so the row is an
   absence. The computed side was emitting five cells (`epochs` 0, the rest NaN) against a
   reference that prints none.

Both were caught only because the gate counts a field present on one side as **missing** rather
than comparing the intersection. A gate that compared the intersection would have reported
144/144 and passed.

---

## 9. Gate clauses, one by one

| A4 clause | state |
|---|---|
| full suite green on both backends | **144 G tests green on SQLite and PostgreSQL**; ruff clean |
| `nbcheck` green on every notebook | **green, 8/8** |
| the arm regression green | **green** — `tests_g/test_arms_regression.py`, 18 tests, including an empirical check that the noise arm preserves mark count and signs while destroying the labels |
| every self-test in B§6 green | **11 available, all green.** `assay_selftest` is a named gap (F6, D5); three more are G2/G3 mechanisms that do not exist yet |
| a write-up | this document |
| `HANDOFF.md` updated the same day | done |

---

## 10. Effect on the G1 and G3 numbers

**G1: reproduction diff = 0.** Met.

**G3: no effect, and that is the correct answer.** A1.7 says every feature is attached to a
number and a feature that moves neither is reported as such. G3's claim line — *a record left by
one population raises the competence of a population that never met it* — cannot move at G1,
because there is no population B: `run()` refuses `init_genomes` and says why. What G1 contributes
to G3 is the instrument, not a number: an engine that runs byte-identical with its hash recorded,
a reading recomputable from stored rows on either backend, a store of era-boundary snapshots to
start a replay from, and a gate that will not pass an empty comparison.

Three things now measured that G3 will rest on:

- the modulator does not track the economics, so G4's mechanics cannot move the learner by
  accident;
- HEAD's engine reproduces the reference blob's trajectory, so a G3 run and a G0 reference are
  numbers about the same world;
- a log row survives storage with its NaNs intact, so an absent measurement in population B will
  read as absent rather than as zero.

---

## 11. What is not done

- **G0's removal has not been carried out.** `docs/G0_G1.md` §1 enumerates it against the tree;
  `civitas/` still contains the policy agents, the two agent-facing domains and
  `acceptance.py`'s six levels. `civitas_g` does not import any of them — it takes only the kept
  dialect seam from `civitas.persistence.types` — so the removal is cleanup, not a blocker.
- **`assay_selftest` does not exist** and is specified rather than built (D5).
- **Standing food cover is still unmeasured** (D10), and will stay so until an engine change
  lands with its own gate.
- **One seed.** The reference is a one-seed pre-check and nothing here is a claim. `A5`: I run
  pre-checks; you run acceptance and grids.

G2 does not start until you have agreed its spec.
