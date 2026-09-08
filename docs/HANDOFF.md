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
`docs/G0_G1.md` §1. **The removal itself has not been carried out** — see "What remains".

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
.venv/bin/python -m civitas_g reproduce --postgres-url "$PG"   # the G1 gate
```

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
13. **`type_spawn_w` is `None` and five observation inputs are dead.** Both are recorded facts, not
    oversights. Changing the layout changes every genome and voids the reproduction; changing the
    spawn weights changes the world. They are G4 candidates with their own specs.

## What is built

### Civitas-G

| milestone | gate | state |
|---|---|---|
| **G0** | audit and removal (B§1) | **audit delivered** (`docs/G0_G1.md`): the removal list against the tree, the reference hashes, nine findings. **The removal itself is not carried out.** |
| **G1** | reproduction diff = 0 | **MET.** `precheck_v3_13.txt` reproduced **182/182 fields on SQLite and on PostgreSQL**, backends agreeing, from rows recomputed out of the database. `docs/G1_WRITEUP.md` |
| **G2** | the record as the artifact store; Gate R, stale-mark lines, frozen assay reproduce through the store | **spec delivered** (`docs/G2_SPEC.md`); the store layer is **built and green**; the assay is **blocked on G2-D1** — see below |
| G3 | the store outlives the run | not started |
| G4 | a world that hardens | not started |
| G5 | frozen-LLM reference arm | not started |

174 G tests green on both backends; ruff clean; `nbcheck` green on all eight notebooks; eleven
available self-tests green. The M1–M13 suite still passes untouched (778 passed, 4 skipped, both
backends), so the two lineages coexist without either disturbing the other — `civitas_g` imports
nothing from `civitas/` except the kept dialect seam in `persistence/types.py`.

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

1. **Rule on G2-D1: apply the engine patch?** This is the blocking decision and everything else in
   G2 and G3 waits behind it. `run()` neither returns the store nor accepts one, so
   `frozen_replay` has **never seen a store** — A2.1's `store visible / hidden / label-permuted`
   arms have never been runnable, and G3's "population B born into A's record" is blocked by the
   same fact. The patch is written, applies cleanly, and is **measured** trajectory-neutral;
   it is deliberately **not applied**. `docs/patches/g2-store-capture-and-injection.diff`,
   `docs/G2_SPEC.md` §2 and §5.
2. **Carry out G0's removal.** `docs/G0_G1.md` §1 lists every path and line. One judgement call is
   already made and should be honoured: the four leak-check tests in `tests/test_domains.py` are
   the only executable statement of the §47 rule B§1 *keeps*, so they are re-homed to the vector
   adapter rather than deleted with the file.
3. **Finish `assay_selftest`** once G2-D1 is ruled. Two of its three clauses already run as
   `assay_preconditions`; the third needs a replay that can see a store.

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
- **The G1 result is one seed.** Nothing in it is a claim.
