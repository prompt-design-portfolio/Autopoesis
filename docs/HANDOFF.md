# Handoff

Read this first if you are resuming work (A5).

## Where the project is

Two lineages live in this repository and they are not the same project.

**Civitas-G (`civitas_g/`) is the live one.** It is a rebuild under
`CIVITAS_G_MASTER_BUILD_DIRECTIVE.md`, which supersedes the original combined prompt wherever the
two conflict. Its inversion is the whole point:

> The learner is the only thing that thinks. Civitas provides persistence, measurement and the
> environment; it never provides cognition. (A1.1)

There is no agent in `civitas_g/`. No LLM is a component of one, no hand-written policy stands in
for one, and nothing reads the store on a learner's behalf. The learner is the grown network of
`sim_v3_13.py`.

**Civitas M1–M13 (`civitas/`) is history, and partly on the removal list.** B§1 removes the
deterministic policy agents, the two agent-facing domains, `acceptance.py`'s six levels and the
`m4`–`m13` results as acceptance artifacts; makes §9–§18 and §26–§32 and §49 dormant; and keeps
§33–§46, §47 and §50–§58. The audit that enumerates all of that against the tree is
`docs/G0_G1.md` §1, and **the removal is carried out** — moved to `civitas/legacy/`,
`scripts/legacy/` and `results/legacy/` rather than deleted, because these modules produced the
results B§1 keeps as history.

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]" "psycopg[binary]"
bash scripts/start_test_postgres.sh /var/tmp/civitas-pg     # prints a URL

# the G suite, both backends
CIVITAS_G_TEST_POSTGRES_URL=postgresql+psycopg://civitas@127.0.0.1:55432/civitas_test \
  .venv/bin/python -m pytest tests_g/ -q

.venv/bin/python -m civitas_g world       # the world of record, arms, modulator table, layout
.venv/bin/python -m civitas_g pins        # the G0 hashes against the working tree
.venv/bin/python -m civitas_g selftest    # B§6; any failure halts a read
.venv/bin/python -m civitas_g store       # G2: the store, and the decision that blocks it
.venv/bin/python -m civitas_g reproduce --postgres-url "$PG"   # the G1 gate
```

`notebooks/Civitas_G_Colab.ipynb` runs the world check, the G0 hashes, the self-tests and the
reproduction on a Colab CPU without a local checkout — A5's "files come to me at the pre-check
stage". It clones the branch rather than asking for uploads, because `civitas_g` is a package and
not two loose files. Rebuild it with `make_notebook_civitas_g.py`; `nbcheck` compiles every cell
before the file is written, and `tests_g/test_notebook.py` resolves every symbol it imports —
which is the failure nbcheck structurally cannot catch, since a renamed symbol compiles fine and
fails on Colab in front of whoever is trying to reproduce a number.

The PostgreSQL cluster lives outside the scratchpad because the scratchpad's permissions are reset
periodically, which kills the server mid-run.

## Things a resumed session must know

1. **The world of record is `analysis_v3_13.WORLD`, not `sim_v3_13.Config()`.** The `Config`
   defaults are v3.10 leftovers: they put the chance EV of a preparation at **−0.100** and the era
   at 2000 steps. A provider that builds a bare `Config()` builds a different experiment. This is
   checked at construction (`check_world`, `check_chance_ev_is_zero`) and it is the easiest way for
   a G number to be silently wrong.
2. **The modulator is a table, not the energy delta.** `resolve_action` says so in terms. `m` is a
   literal ±1, independent of `prep_value` and `prep_fail`, which is why v3.12 could halve
   `prep_fail` without touching learning. G4 changes economics one mechanic at a time; if this were
   recorded wrong, the first change would confound every claim line. `modulator_selftest` measures
   it.
3. **Reproduce against the blob at the reference's own commit, not HEAD.** Every candidate
   reference summary in this repository was produced by code that changed afterwards.
   `precheck_v3_12.txt` is worse: its producing code does not exist at the commit that added it, so
   it is recorded and never gated on.
4. **`precheck_v3_13.txt` records neither its seeds nor its configuration.** The invocation is
   recovered by inference, with a stated basis per element, in `manifest.REFERENCE_INVOCATION`. The
   seed is *not* recoverable and is recorded as `None` rather than as a value. If a reproduction
   fails, D4 says try the candidate seeds and then declare the artifact unreproducible — it does
   **not** say vary the configuration until the numbers agree. A1.2 calls that failing.
5. **The reading is recomputed from the persisted rows, never from the result in memory.**
   Otherwise "both backends" is vacuous: the numbers come from numpy either way. Each run is
   executed once and written to every backend, so a disagreement is a round-trip defect.
6. **NaN is a measurement and storage is where it gets destroyed.** PostgreSQL's JSONB rejects it.
   Dropping the key makes an absent measurement look unrecorded; coercing to 0.0 makes it look
   measured. `persistence/encoding.py` tags it and decodes it back.
7. **Absent is not zero, anywhere.** The gate counts a field present on one side only as missing
   and fails; NaN against NaN is an agreement; NaN against a number is a mismatch, never a skip;
   and an empty comparison does not pass — `all([])` is `True`.
8. **`assay_selftest` does not exist.** A2.1 cites it as an existing reference. It is registered as
   a named gap so it shows up in every self-test report; D5 specifies it and G2 builds it.
9. **Reading must not write.** SQLite's `BEGIN IMMEDIATE` takes the *write* lock even to read, so
   every read path uses `read_only_session_factory`. Three separate M11 defects were this one fact
   wearing different hats.
10. **The engine is untouched (A1.8).** `civitas_g/world/engine.py` is the whole of Civitas's
    contact with it: build a `Config`, call `run()`, take what comes out. No callback, no subclass,
    no monkeypatch — and `touches_world_only_at_era_boundaries()` checks that rather than promising
    it. If you find yourself wanting a per-step hook, that is cognition arriving, and A1.1 forbids
    it.
11. **The engine cannot hand out the store or take one in, and that is G2's whole blocker.**
    `run()`'s result has `era_snaps` (genomes) and no marks; `World.__init__` always zeroes the
    marks and redraws π. It cannot be worked around from the Civitas side: marks **overwrite**, so
    the surviving array is not a function of anything the log records, and re-deriving it would
    mean re-implementing `run()`. Do not try. Rule G2-D1 instead.
12. **Two scrambles, and confusing them breaks a control.** `ScrambleMode.PER_CELL` is B§5.2's
    `inherited scrambled`; `ScrambleMode.GLOBAL` is A2.1's `label-permuted` assay arm. A global
    permutation is *isomorphic* to the real store — it says "label σ(j) marks what label j marked"
    everywhere — so a population born into it faces exactly the learning problem the real store
    poses, and the arm would read as a null while information had in fact passed. `scrambled_load`
    checks the two are distinct.
13. **B cannot be staged identically to A, and the reason is measured** (G3-D1). `mark_decay`
    0.004 over a 3000-step food-only phase leaves 6e-6 of A's marks: a record injected at B's step
    0 is gone six orders of magnitude before B's first preparation, so `inherited store` would read
    exactly like `fresh store` and the null would be about decay. B runs chain-on from step 0. The
    cost -- unsorted founders, so a random food instinct -- is matched across all four arms and
    cancels in a between-arm difference. It does forfeit any comparison of B's absolute competence
    to A's, which is not the claim and is not reported.
14. **Alignment is a recorded parameter of every G3 run, never an assumption** (G3-D2). A mark is
    true only of the era it was written in, so a misaligned B gets a record that is not stale but
    INVERTED. Both are run. A misaligned null is stale culture, not absent culture -- and it need
    not even be a null, because a ratio below 1 means B learned to avoid the record, and you
    cannot avoid what you cannot see.
15. **`type_spawn_w` is `None` and five observation inputs are dead.** Both are recorded facts, not
    oversights. Changing the layout changes every genome and voids the reproduction; changing the
    spawn weights changes the world. They are G4 candidates with their own specs.

## What is built

### Civitas-G

| milestone | gate | state |
|---|---|---|
| **G0** | audit and removal (B§1) | **MET.** The audit is `docs/G0_G1.md`; the removal is carried out — the removed surface is in `civitas/legacy/`, its scripts in `scripts/legacy/`, `m4`–`m13` in `results/legacy/`, its tests deleted, and `civitas acceptance` and `civitas benchmark` retired. Moved rather than deleted because these modules produced the results B§1 keeps as history, and a result whose producing code is gone is the very gap this audit found for `precheck_v3_12.txt`. |
| **G1** | reproduction diff = 0 | **MET.** `precheck_v3_13.txt` reproduced **182/182 fields on SQLite and on PostgreSQL**, backends agreeing, from rows recomputed out of the database. `docs/G1_WRITEUP.md` |
| **G2** | the record as the artifact store; Gate R, stale-mark lines, frozen assay reproduce through the store | **MET.** G2-D1 ruled and the engine change applied; the reproduction re-run against it gives **182/182 on both backends**, and the same arm's rows are **byte-identical** to the G1 run at full reference length. A2.1's twelve cells run for the first time; `assay_selftest` is built. `docs/G2_WRITEUP.md` |
| **G3** | the store outlives the run; population B born into population A's record | **MET at one seed**, on the project owner's explicit decision to relax B§5.2's 3/3 bar; `min_seeds` is a recorded parameter of `acceptance`, not a hard-coded constant, and `docs/G3_WRITEUP.md` §4.4 records the basis. Seed 0 aligned: content **−0.131**, reading **−0.119**, the two information-removing controls agreeing to 0.012 nfc and 0.04 on the stale ratio, Gate R PASS (z −0.68, 3 epochs). Seed 0 **misaligned: +0.018** — the record helps only when it is true of B's world, which is the sharpest internal check in the milestone and one no control arm can supply. |
| G4 | a world that hardens, one mechanic per milestone | **mechanic 1 built and running.** `world_at_k` + `check_hardened_world` (a hardened world may move only what a mechanic declares, and the chance EV of a preparation must still be zero); the engine is K-parameterised as version `G4-k` and is **bit-identical at K = 5** — 32 rows × 133 fields with zero differing, and a full-length succession reproducing seed 0 to the digit. Mechanics 2 and 3 are deliberately unspecified: `docs/G4_SPEC.md` §2–3 settle their design questions against mechanic 1's numbers rather than in advance of them. |
| G5 | reference arm: one frozen LLM against the same store | **step 1 done, step 2 specified.** The presentation is built with its leak rule tested by trying to break it (`civitas_g/g5/presentation.py`); G5-D6 records that the field names are more than the learner gets and rules that they stay, so the arm is generously provisioned and a poor number from it is the informative one. G5-D7 specifies the engine hook and the equivalence check that needs no model. **The provider cannot run in this environment** — see Open issues. Reported, never claimed. |

191 G tests green on both backends (176 in one 8m18s run plus the 15 notebook tests added
after it started); ruff clean; `nbcheck` green on all nine notebooks; fifteen
available self-tests green, with **no named gaps left at G1 or G2** — `assay_selftest` was the last
one and it is now a measurement. The five added at G2 are `store_round_trip`, `scrambled_load`,
`assay_preconditions`, `assay_selftest` and `engine_store`. The M1–M13 suite still passes untouched (778 passed, 4 skipped, both
backends), so the two lineages coexist without either disturbing the other — `civitas_g` imports
nothing from `civitas/` except the kept dialect seam in `persistence/types.py`.

**Three instances of one defect, worth knowing before adding a fourth.** K appears as a module
constant, and at the reference K = 5 a K-blind computation gives the right answer — so it passes
every test and goes wrong only on a hardened world. It was found in `check_chance_ev_is_zero`
(the guard meant to catch a badly-raised K computed the economics from the module's K),
`matched_stale_null`, `Record.__post_init__` (which made a hardened store unrepresentable — the
K = 7 smoke test could not build its own record), and A2.2's matched null in
`civitas_g.reading.compute`. Each now reads K off the run or the artifact. Anything new that
touches K should take it as a parameter, and `civitas_g/g5/policy.py`'s `check_action` does.

Three measurements the build produced, beyond the reproduction:

- **the modulator is identical under a 4× change in the economics** (400 steps) — so G4's
  per-mechanic claim lines are safe from their own economics changes;
- **129 shared log fields identical between `sim_v3_13.py` @ `3d7b228` and HEAD** across a mapping
  remap — so the reference and a run today are numbers about the same world;
- **the recovered invocation reproduces exactly**: 5 arms, seed 0, 3000-step phases, `prep_every`
  700 — a configuration that appears nowhere in the repository, inferred from what the summary
  happened to print.

### Civitas M1–M13 (history)

M1–M13 complete, **778 passed / 4 skipped** on both backends as of this commit, `results/m4…m13`. B§1 moves those results to
`results/legacy/` as history and takes `acceptance.py`'s six levels off the acceptance path. The
milestone notes are in `docs/milestones/`. Two limitations recorded at the time and now superseded
by A1.1: the agents were deterministic policies, not learners, and two domains is two.

## What remains

1. **G4 mechanic 1's number**, then its write-up. The run is against the *stored* G3 seed-0
   baseline (`var/g3/seed0_aligned.json`, reloaded by `G3Result.from_dict`) rather than a fresh
   draw: re-running the baseline per mechanic would compare each mechanic against a different draw
   of the same world, which is the one thing a difference of differences cannot survive.
2. **Mechanics 2 and 3**, each its own spec with its own claim line (B§5.3), designed against
   mechanic 1's numbers. `docs/G4_SPEC.md` §3 names mechanic 3's central design problem — how much
   of a compositional preparation's difficulty is the composition and how much is the longer
   horizon — and says it has to be settled from the measured distribution, not guessed.
3. **G5 step 2**: apply the `G5-policy` engine hook and run §7.1's equivalence check, which needs
   no provider. A `MirrorPolicy` run must be bit-identical to a run with no policy; if it is not,
   the hook is wrong and finding that out costs nothing.
4. **Three seeds for G3 as corroboration.** The resumable campaign writes per `(seed, alignment)`
   to `var/g3/`; seed 0 aligned has already reproduced to the digit. Report disagreement with
   seed 0 *as disagreement* — the one-seed acceptance is on record and averaging a second seed
   into it would quietly replace the number the gate was read on.
5. **Confirm the M-suite with a quiet full run.** The 6 failures and 30 errors seen after G0's
   removal were all PostgreSQL and all disappeared when the four affected files ran alone (164
   passed clean). CPU contention is the *attribution*, not yet a finding.

## Open issues

- **Standing food cover is measured nowhere** (D10). It is consumption-limited, not derivable from
  the parameters, and not in the engine's log — and A1.8 forbids adding it in Civitas. Until a
  versioned engine change lands, the provider reports it as `None` with the reason, and the `sr_w`
  lesson stays half-applied.
- **`v3_12_finding.md` is required reading and does not exist**; there is **no v3.12 grid** despite
  A3 naming one; and **`precheck_v3_12.txt` has no producing code at its own commit**, so it is
  recorded and never gated on.
- **Five of the 31 observation inputs are dead** and `type_spawn_w` is `None`, so type C
  accumulates to roughly half of standing food. Both are recorded facts and G4 candidates; neither
  can be changed without voiding the reproduction or changing the world.
- **The G1 result is one seed.** Nothing in it is a claim. Nor is G3's, which is accepted at one
  seed by decision rather than by evidence — the write-up says so where the number is stated.
- **G5's provider cannot run here.** No model API key is set in this environment, only a base URL.
  G5-D5 makes that a refusal at the start of the run rather than a silent fallback, so the arm
  stops at the equivalence check: the loop can be proven correct, and the table cannot be
  produced. That is a fact about the environment, not about the design, and it is recorded here
  rather than worked around.
