# Civitas

A persistent artificial civilization for cumulative collective intelligence.

> The collective becomes more intelligent over time even if the underlying individual foundation
> model remains unchanged.

Individual agents are bounded and temporary. What survives them — knowledge, tools, failures,
provenance, procedures, reputation — is the civilization, and the claim is that it becomes
measurably more capable while the model stays frozen.

This repository also contains the research lineage the platform generalises: `sim_v3_*.py`,
`analysis_v3_*.py` and the autopoiesis notebooks. They are the scientific reference
(`docs/ARCHITECTURE.md` §2–§3) and are retained unmodified; the platform does not import them.

---

## The measurement

The point of the platform is that its central claim is *checkable*. The newcomer-advantage
benchmark (Part B §22) freezes every individual-agent variable and varies only the collective
environment:

| arm | success rate | mean probes | what it is |
|---|---|---|---|
| `baseline_empty` | 0.467 | 4.60 | a fresh agent in an empty environment |
| `collective` | **0.767** | **0.27** | the same agent, after others have worked |
| `memory_reset` | 0.467 | 4.60 | the same, with accumulated state reset |
| `collective_scrambled` | 0.500 | 2.53 | same quantity of artifacts, relevance destroyed |

n = 30 probe episodes per arm, five seeds, one configuration hash across all four arms.

- **newcomer advantage: +0.300**
- **reset removes 100% of it**
- **scrambling removes 89% of it**

The two ablations are what separate *information passed* from *a store existed*. An advantage that
survived scrambling intact would never have been about content.

Full data, including the difficulty sweep that justifies the budget, is in
`results/m4_newcomer_benchmark.json`. Read the gates first: a failed gate withholds the numbers
rather than annotating them.

### The other three scientific criteria

| §65 level | measurement | result |
|---|---|---|
| 4 distributed cognition | a two-factor device split across populations, provably unsolvable alone | **+0.417** over the best single partition |
| 5 cumulative culture | a chain where each generation's question is set by the last one's answer | **4 generations** reached, vs **1** with the chain broken |
| 6 capability growth | frontier across difficulty, model held constant | collective **14**, solo **6**, independent **6** |

`independent` tracking `solo` exactly is the check that matters: repeated sampling without
transmission buys nothing, so the collective's gain is transmission. Data in
`results/m9_advanced_benchmarks.json`.

---

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
alembic upgrade head
pytest -q
```

Colab is a first-class runtime (Part B §33). `notebooks/Civitas_Colab.ipynb` runs the same package
— detects the runtime and GPU, mounts Drive, migrates, runs the benchmark, exports the manifest,
and reconnects to a civilization a previous runtime left behind.

### Two backends, one set of application logic

PostgreSQL is the production source of truth; SQLite with WAL is the Colab-compatible
implementation. Part B §6 forbids separate Colab logic, and the only way to know that holds is to
run the same tests on both:

```bash
pytest -q                                            # SQLite only
bash scripts/start_test_postgres.sh /var/tmp/civitas-pg
CIVITAS_TEST_POSTGRES_URL=postgresql+psycopg://civitas@127.0.0.1:55432/civitas_test pytest -q
```

That parametrisation is not optional in CI. It found a concurrency race in the event-sequence
allocator that SQLite structurally cannot expose (`docs/milestones/M02.md`).

---

## Layout

```
civitas/
  config.py         settings and secrets (§40)
  domain/           closed enumerations — arms, termination reasons, artifact and relation types
  persistence/      SQLAlchemy models, portable types, events, jobs, leases (§7, §42–§45)
  runtime/          episodes, budgets, providers, tools, sandbox (§8, §18, §19)
  knowledge/        arm policies, hybrid retrieval, duplicate-failure detection (§14, §21, §12)
  experiments/      manifests, credit assignment, benchmark, metrics (§20, §22, §46, §48)
  workers/          durable worker loop (§37, §42)
alembic/            migrations
docs/               ARCHITECTURE.md and per-milestone write-ups
notebooks/          Civitas_Colab.ipynb
results/            committed benchmark output
```

## Design commitments

These are load-bearing, and each has a test that drives the real mechanism:

- **Agents do not inherit hidden episode state (§4).** `EpisodeContext` is built inside `run()`
  and dropped at return; `Episode` has no column for reasoning; `ModelCall` stores accounting and
  a request hash, never prompt content. Two consecutive episodes are driven under every arm to
  prove a canary in the first reaches no input of the second — and that what the first
  *externalised* does reach it.
- **Arms are runtime policy, not prompt convention (§21).** An unknown arm raises rather than
  falling back; `collective_frozen` without a snapshot cut raises rather than silently becoming
  `collective`; every suppression is recorded on the `RetrievalDecision`.
- **Credit is outcome-linked, never self-reported (§A2.1).** It flows backward along what an
  episode *read* — not what retrieval returned — decaying with graph distance, as immutable
  events. `downstream_utility` is their fold and is rebuilt from them.
- **Nothing is hard-deleted (§A1.2).** Archival, supersession and retirement; the event stream and
  the credit events reject UPDATE and DELETE at flush.
- **A weaker guarantee is declared, not implied (§18, §46).** The Colab sandbox reports
  `isolation_level: process`, and it declares that `max_processes` is unenforceable under uid 0 —
  measured, not assumed. Both reach the manifest.
- **Absence is not zero (§48).** A metric that cannot be computed reports `None` with a reason. A
  dashboard rendering "not measurable" as "measured zero" manufactures a finding.

## Status

| milestone | state |
|---|---|
| M1 audit · M2 persistence · M3 runtime · M4 experiments · M5 knowledge · M6 tools · M7 orchestration · M8 institutions · M9 advanced benchmarks | complete |
| M10–M13 | see `docs/ARCHITECTURE.md` §5 |

Each milestone reports its effect on the M4 benchmark. A feature that moves no number is reported
as such rather than hidden (Part A §A1.5) — M5's hybrid retrieval moved it by exactly 0.000, and
`docs/milestones/M05.md` explains why the benchmark cannot see it and builds the instrument that
can.
