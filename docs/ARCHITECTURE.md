# Civitas — Architecture and Repository Audit (M1)

Status: M1 complete. This document is the audit required by Part A §A5 and Part B §3/§62 Phase 1.
It maps what exists to Part B sections, names what is missing, and fixes the architecture the
later milestones build against.

---

## 1. Audit finding: there is no Civitas prototype

Part B §3 says an "existing Civitas prototype repository" is provided and must be audited before
being changed. **It does not exist.** The repository at `prompt-design-portfolio/Autopoesis` has a
single branch, `claude/autopoiesis-v3-6-grown-learner-vihil2`, whose 22 commits are the
*autopoiesis simulation lineage* (v3.6 → v3.13) and nothing else. There is no API, no database, no
agent runtime, no service code of any kind.

This is a material correction to the build directive's premises, and it changes M1's answer to the
eight audit questions of Part B §3:

| Question | Answer |
|---|---|
| 1. What already works | The research instrument: `sim_v3_*.py`, `analysis_v3_*.py`, the notebooks, and the delivery gate `nbcheck.py`. All of it is numpy-only and self-contained. |
| 2. What is partially implemented | Nothing on the platform side. |
| 3. What is missing | The entire platform — every one of Part B §6's twenty-five components. |
| 4. What will not scale | Not applicable to the platform. The simulation is a fixed-size numpy world and is not intended to scale; it is not being ported (§3 forbids porting it). |
| 5. What is insecure | Not applicable — no network surface exists yet. The security boundary is therefore built from M3 rather than retrofitted (Part A §A1.7). |
| 6. What should be retained | **The research files, unmodified, and the methodology they encode.** See §3 below. Also `nbcheck.py` and its delivery discipline, which the Colab notebook (§34) inherits. |
| 7. What requires migration | Nothing. There are no persisted structures to preserve, so M2 starts from a clean Alembic baseline rather than a migration of an existing schema. |
| 8. What should be redesigned | Not applicable. |

**Consequence for the build order.** Part A §A3 gives M1 as "repository + research-file audit". With
no engineering baseline, M1 collapses to this document plus the package skeleton, and M2 begins from
an empty schema. Nothing in A3 changes; the M1 gate is simply cheaper than it would otherwise be.

**Consequence for §3's instruction to preserve useful foundations.** The research files stay exactly
where they are, at the repository root, under their existing names. The platform is built alongside
them in `civitas/`. They are the scientific reference (§3), not a dependency of the platform, and
they are not imported by it.

---

## 2. What the simulation actually establishes (read from the executable code)

`sim_v3_13.py` and `analysis_v3_13.py` were read as code, not as comments, per §3. The mechanism, in
the terms that matter for the platform:

**The world.** A 48×48 toroidal grid; `T = 3` food types, `K = 5` preparations, and a *mapping*
sending each food type to exactly one correct preparation — `P(5,3) = 60` mappings. The mapping is
redrawn every `prep_every = 700` steps. A correct preparation pays `prep_value = 1.0`; a wrong one
costs `prep_fail = 0.25`, so `(K−1)·prep_fail = prep_value` and the expected value of a *chance*
preparation is exactly zero. That equality is deliberate: it means any positive return on
preparation is evidence of knowledge, not of volume.

**The agent.** A two-layer network whose synapse is `innate weight + learned component H`, with a
local rule, an eligibility trace, and the agent's own energy change as the modulator. Lifetimes are
bounded; `H` is **not inherited** — only the genome is. This is the simulation's form of Part B §4:
within-life state dies with the agent, and only what is externalised survives.

**The record (v3.13).** Every preparation writes `(label, sign)` to the cell for its food type —
automatically, at no cost, with no writer gene. `label = π(k)` where `π` is a permutation of the K
preparations **redrawn at every remap**. Reading is an *observation channel*, not an action. A
single heritable scalar, `sym_gain`, gates the whole read block and starts near zero.

**Why `π` is the load-bearing detail.** If a mark recorded the preparation index directly, its
meaning would be fixed by the world and a genome could evolve to read it — and the result would be
*inheritance*, not transmission. Redrawing `π` each era makes the label→preparation binding
worthless to selection. The population must establish it within lives or not at all.

**No self-echo.** A preparation consumes the food cell, so the mark it writes cannot be read for a
preparation until food respawns and the reader is whoever is then standing there. An agent can never
read its own mark about the food it just prepared. This is verified by a self-test, not argued.

### The three principles the platform generalises

Per §3, the simulation is **not ported**. Three principles are generalised into real agent
architecture, and each is named against the platform component that carries it:

1. **Bounded lifetime, non-inheritable internal state.** → `Episode` (Part B §8) with hard budgets
   and enumerated termination reasons; the state-isolation tests of §4 assert that no private
   scratchpad, chain-of-thought, or runtime state crosses an episode boundary.
2. **Externalised information persists and is exploitable by later agents.** → the artifact graph
   (§9–§13) and hybrid retrieval (§14).
3. **Changing semantics prevent trivial permanent encoding.** → this is the *scramble* control
   (§21 `collective_scrambled`) and, more sharply, the reason a benchmark task family must rotate
   its surface form between generations. A collective that has memorised one answer string is not
   the phenomenon; `π` is the simulation's proof that the distinction is measurable.

---

## 3. The methodology the benchmark framework inherits (Part B §3, §47, §48)

`analysis_v3_13.py` is the methodological reference. Eleven disciplines are carried into
`civitas.experiments`, and each becomes a code-level obligation rather than a convention:

| # | Discipline in the analysis | Obligation in Civitas |
|---|---|---|
| 1 | **Gates precede rows.** A failed prerequisite means the run is *not read*, not read-with-caveats. | `BenchmarkResult` carries gate outcomes; a failed gate makes the metric block unreadable by construction — the API returns the gate failure, never the number. |
| 2 | **Matched nulls.** Gate R permutes each π-epoch's label axis independently, preserving era count, sample size and within-era structure, destroying only cross-era consistency. The unmatched comparison (against the noise arm) was *shown wrong by the pre-check*. | Every ablation control must hold everything constant except the thing under test. `collective_scrambled` returns *comparable quantities* of artifacts with relevance destroyed — presence held, content destroyed. |
| 3 | **The null is derived, not assumed.** `(1 − hit)/(K − 1)`, not `1/K`, because an agent that knows the answer never agrees with a stale mark. | Baselines are computed from the arm's own conditional structure. No hard-coded chance levels. |
| 4 | **Presence vs. content controls.** `noise record` exists because a store *changes the world* — channels exist, cells carry state, decay runs — so an arm that improves because a store exists is not an arm that improved because information passed. | `collective_scrambled` is mandatory alongside `memory_reset`; the newcomer benchmark reports both. |
| 5 | **Founder-free metrics.** Injected agents' own events are excluded from event-weighted metrics; their children's are not. Founder share is printed. | Newcomer-benchmark episodes are excluded from the collective-maturity statistics they are measured against, and the exclusion is exercised by a test, not assumed. |
| 6 | **Learned vs innate decomposition.** `store_gain` learned-vs-innate is *the claim's instrument*: innate ≈ 0 says the genome cannot read it, learned > 0 says the agent bound it in its own life. | The equivalent split is arm `collective` vs `solo` under a **frozen model** (§20): any advantage that survives with the collective removed was never collective. |
| 7 | **Frozen replay as attribution.** Snapshot at an era boundary, disable births/deaths/injection, 300 steps, nothing can change but `H`. Turns a correlation into an attribution. | `collective_frozen` freezes collective state at a snapshot; the A2.2 experiment gate uses the same shape — identical configuration hash, one variable moved. |
| 8 | **Self-tests before results.** `world_semantics_selftest`, `founder_tag_selftest`, `replay_mapping_selftest`, `frozen_selftest`, `learning_rule_selftest`, `record_semantics_selftest` — each raises `SystemExit` before anything is read. | Part A §A4.2: arms and frozen mode are regression-tested every milestone; the benchmark refuses to run against a configuration whose self-tests fail. |
| 9 | **Unrecomputable fields print NOT AVAILABLE.** Counters accumulated inside the sim are not derivable from the log; a checkpoint written before a field existed prints `nan`, never a guess. | Manifests (§46) record the schema and code version; a metric that cannot be recomputed for a given manifest version is reported absent, never inferred. |
| 10 | **Survivorship conditioning.** Preparations-to-first-correct is read over agents that reached 5, because conditioning on reaching the window is what stops censoring being confounded by short lives. | Episode-level metrics condition on budget exhaustion explicitly; time-to-solution is never averaged over unterminated work. |
| 11 | **Pre-registration with an honest alternative.** The v3.13 spec states the prediction *and* the outcome that would refute it, and names a null that would be a real result. | Every milestone write-up (§A4.3) states the benchmark number it expected to move and reports it whether or not it moved. A feature that moves nothing is reported as such. |

The single most important inheritance is **#1 and #11 together**: this repository's history contains
a spec whose instrument was wrong and whose pre-check caught it (Gate R's pooled MI had a floor set
by era count; two of the ruled lines were confounded and both were fixed before any result was
claimed). The platform's benchmark framework must be able to fail in that same visible way.

---

## 4. Target architecture

### 4.1 Layering

```
                    ┌──────────────────────────────────────────┐
   HTTP / WS  ────► │  civitas.api          FastAPI, OpenAPI   │  §50 §51 §49
                    └───────────────┬──────────────────────────┘
                                    │
   ┌────────────────────────────────┼─────────────────────────────────────┐
   │                    civitas.services  (use cases)                     │
   │  episodes · retrieval · artifacts · tools · experiments · scheduling │
   │  projects and request decomposition                            §5 §32│
   └────────────────────────────────┬─────────────────────────────────────┘
   ┌────────────────────────────────┼─────────────────────────────────────┐
   │  civitas.domains  (adapters)                                      §5 │
   │  the four things the core cannot know: what a task says, which tools │
   │  act on it, who judges the answer, what the stages are called — plus │
   │  the §47 leak rule every domain must state and pass                  │
   └────────────────────────────────┬─────────────────────────────────────┘
                                    │
   ┌──────────────┬──────────────┬──┴───────────┬──────────────┬──────────┐
   │ runtime      │ knowledge    │ experiments  │ scheduler    │ institut.│
   │ §8 §18 §19   │ §9–§16       │ §20–§25 §46  │ §26–§28      │ §29–§31  │
   │ episodes     │ artifacts    │ arms         │ allocation   │ procedure│
   │ budgets      │ graph        │ manifests    │ decomposition│ policy   │
   │ providers    │ retrieval    │ benchmarks   │ specialization│ reputation│
   │ tools        │ consolidation│ credit (A2.1)│ roles        │ gate A2.2│
   │ sandbox      │ staleness    │ dupfail A2.3 │              │          │
   └──────────────┴──────────────┴──────┬───────┴──────────────┴──────────┘
                                        │
                    ┌───────────────────┴──────────────────────┐
                    │  civitas.persistence                     │  §7 §42–§45
                    │  SQLAlchemy 2.0 · Alembic · events · jobs│
                    └───────────────────┬──────────────────────┘
                                        │
                       PostgreSQL (production)  |  SQLite+WAL (Colab)
```

Dependencies point downward only. `knowledge`, `experiments`, `scheduler` and `institutions` never
import each other directly; they compose in `services`. This is what keeps the ablation arms
enforceable in one place instead of scattered through call sites.

### 4.2 The two-database rule (§6, §43)

One set of application logic, two backends, chosen by URL. Part B §6 forbids separate Colab logic.
The dialect differences are confined to three seams, each behind an interface:

- **JSON columns** — `JSONB` on PostgreSQL, `JSON` on SQLite, via a `JSONVariant` type decorator.
- **Full-text search** — `tsvector`/`websearch_to_tsquery` on PostgreSQL; a portable BM25
  implementation over a token index table on SQLite. Both satisfy the same `LexicalIndex` protocol,
  and the retrieval tests run against both.
- **Vector search** — `pgvector` where available; a numpy brute-force `VectorIndex` otherwise. Same
  protocol, same tests.

Job leasing (§42) uses `SELECT … FOR UPDATE SKIP LOCKED` on PostgreSQL and an atomic
compare-and-set `UPDATE … WHERE lease_expires_at < now` on SQLite. Both are exercised by the
worker-recovery tests.

### 4.3 The state boundary (§4) — enforced, not requested

The foundational constraint is that agents do not inherit hidden episode state. It is enforced three
ways, and each has a test:

1. **Construction.** An `Episode` is built by `EpisodeFactory` from an `EpisodeSpec` — profile,
   model config, prompts, budgets, policies, arm. The factory has no channel through which a prior
   episode's runtime object could reach it; episode-local state lives on the `EpisodeContext` object
   which is created inside `run()` and dropped at return.
2. **Serialization.** Nothing in `EpisodeContext` is persisted except through the artifact and event
   writers, which take structured, typed payloads. There is no "dump the scratchpad" path.
3. **Test.** `tests/test_state_isolation.py` runs two consecutive episodes with a scripted provider
   that emits a canary string into its scratchpad in episode 1, and asserts the canary appears in
   *no* input to episode 2 under every arm — including `collective`, where the artifact ecology is
   fully available. The artifacts episode 1 *chose to externalise* do reach episode 2; that is the
   point of the distinction and the test asserts both halves.

### 4.4 Where the three A2 mechanisms live

- **A2.1 outcome-linked credit** — `civitas.experiments.credit`. `ArtifactUsage` and `ToolRun` rows
  record what was *actually read or invoked*, distinct from what retrieval returned. On evaluation,
  credit propagates backward along usage links, decaying with graph distance through
  `derived_from`/`depends_on`/`uses`. Every update is an immutable event; `ArtifactUtilityMetric` and
  `ReputationMetric` are folds over those events and are never written directly.
- **A2.2 experiment gate** — `civitas.institutions.gate`. Versioned policy artifacts are inert until
  an `ExperimentRun` with a matched configuration hash records an improvement. The runtime's version
  loader filters on `approved_by_experiment_run_id IS NOT NULL`, so an unapproved version cannot be
  loaded even by a caller that wants it.
- **A2.3 duplicate-failure detection** — `civitas.knowledge.duplicate`. A pre-action check hashes
  (tool, canonicalised args) and (hypothesis, environment version); a match against a documented
  failure emits `duplicate_failure` and, in `collective`, injects the prior failure at elevated rank
  *before* the action executes.

### 4.5 Non-goals held explicitly (§61)

The architecture is checked against §61 at every milestone. Specifically: there is no shared
conversation object between episodes (not a group chat); consensus is never formed by counting
agents (not voting); no episode receives the full history (not one giant context); retrieval is
hybrid and logged with its features (not ordinary RAG or vector search alone); the scheduler
allocates against evidence gaps and uncertainty (not a static workflow DAG); and roles are
`AgentProfile` rows whose assignment strategies are *learned from measured performance by work type*
(not static manager/worker prompts).

---

## 5. Implementation plan

The milestone table is Part A §A3 and is not restated. What follows is the per-milestone entry
criterion, the mechanism test that gates it, and the benchmark number it must report against
(Part A §A4, §A1.5).

| M | Entry criterion | Gating mechanism test | Reports against |
|---|---|---|---|
| M1 | — | — | — (audit only) |
| M2 | M1 doc committed | Round-trip every §7 entity on both backends; events reject update/delete; lease expiry recovers a job | schema version in the manifest |
| M3 | M2 green | Two episodes, canary isolation under every arm; every termination reason reachable; sandbox denies network, filesystem escape and fork-bomb | episode cost/latency baselines |
| M4 | M3 green | All nine arms enforced at the policy layer (not the prompt); frozen mode byte-identical config hash across runs; credit propagation decays with distance; duplicate failure surfaces *before* execution | **the newcomer benchmark itself** |
| M5–M8 | prior green | per Part A §A3 | newcomer advantage, duplicate-failure rate, knowledge reuse, adoption/rollback |
| M9 | M8 green | distributed-knowledge task is unsolvable by any single episode's information set — asserted, not assumed | capability frontier |
| M10–M12 | prior green | per Part A §A3 | cost, latency, reliability |
| M13 | M12 green | §63 Colab acceptance automated; §65 levels 1–6 with negative controls committed under `results/` | all |

**M4 is the pivot.** Part A §A3 moves experimental infrastructure ahead of the knowledge, tool,
orchestration and institutional phases precisely so that M5 onward can be measured. The consequence
is that M4 must produce a *real* newcomer benchmark against a *real* task family before the
knowledge system exists — so M4 ships a deliberately minimal artifact store (create, read, retrieve
by recency and keyword) sufficient for the benchmark to discriminate, and M5 replaces it with the
full ecology and re-reports the same benchmark. The M5 write-up therefore carries a before/after on
the identical benchmark, which is the strongest available evidence that the knowledge system does
what it claims.

---

## 6. Open issues carried forward

1. **Task family for the newcomer benchmark.** Must be objectively evaluable (§47), must not be
   solvable by memorising one answer (the `π` lesson, §2 above), and must have enough surface
   variation that a mature collective's advantage is knowledge rather than recall. Resolved in M4.
2. **Credit decay constant.** The simulation's `mark_decay` was bracketed by two measured
   constraints — outlive the agent, do not outlive the era — and *reported*, not guessed. The credit
   decay through graph distance gets the same treatment: bracketed, measured, and reported in the M4
   write-up rather than chosen.
3. **Sandbox on Colab.** Containers are unavailable in a Colab runtime, so the hardened backend and
   a restricted-subprocess fallback both exist behind `SandboxBackend`. The fallback's weaker
   guarantees are stated in the manifest so a result run under it is never mistaken for one run
   under isolation.
4. **Does the newcomer advantage generalise?** A result measured on one task family is a result
   about that task family. Addressed in M10: §22's procedure was made domain-parameterised and run
   on a second domain — code repair, judged by executing the submission in the sandbox — at a
   matched naive ceiling. Advantage +0.200 against the device's +0.300, with reset and scramble
   each removing 100%. Two domains is two, not many; the layer that made it possible is the part
   that generalises.

---

## 7. Running the tests

```bash
python -m pytest tests/ -q                    # SQLite only
CIVITAS_TEST_POSTGRES_URL=postgresql+psycopg://user@host:5432/db \
  python -m pytest tests/ -q                  # both backends
```

The PostgreSQL parametrisation is not optional in CI. Part B §6 forbids separate application logic
for Colab, and the only way to know that holds is to run the same tests on both dialects — M2 found
a concurrency race that SQLite could not expose (see `docs/milestones/M02.md`).
