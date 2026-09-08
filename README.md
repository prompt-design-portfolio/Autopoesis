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

### Acceptance: all six levels, one campaign, one manifest

```bash
civitas acceptance          # §65's six levels with their negative controls; non-zero if any fail
civitas preflight           # §64's production checklist, executable
```

| §65 level | value | negative control |
|---|---|---|
| 1 external knowledge usefulness | **+0.300** | a blinded agent in the *same* matured workspace |
| 2 newcomer advantage | **+0.300** | the same agent in an empty workspace |
| 3 ablation causality | **100%** removed | `memory_reset` and `collective_scrambled` |
| 4 distributed cognition | **+0.350** | each partition alone, proved insufficient beforehand |
| 5 cumulative culture | **4** generations | the same chain with inheritance broken |
| 6 capability growth | **14** | `solo` and `independent` at 8 |

Levels 1 and 2 report the same number from *different* comparisons, and the agreement is the
point: a blinded agent in a rich workspace scores exactly what an agent in an empty one scores
(0.467 both), which is what arm enforcement working looks like. Data in
`results/m13_acceptance.json`.

### The other three scientific criteria

| §65 level | measurement | result |
|---|---|---|
| 4 distributed cognition | a two-factor device split across populations, provably unsolvable alone | **+0.417** over the best single partition |
| 5 cumulative culture | a chain where each generation's question is set by the last one's answer | **4 generations** reached, vs **1** with the chain broken |
| 6 capability growth | frontier across difficulty, model held constant | collective **14**, solo **6**, independent **6** |

`independent` tracking `solo` exactly is the check that matters: repeated sampling without
transmission buys nothing, so the collective's gain is transmission. Data in
`results/m9_advanced_benchmarks.json`.

### Does it transfer?

The same procedure, run on a second domain that shares nothing with the first but the core: the
answer is a Python function the agent writes, and the evaluator **executes it in the hardened
sandbox** against checks the agent never sees.

| | hidden rule | code repair |
|---|---|---|
| baseline | 0.467 | 0.550 |
| **collective** | **0.767** | **0.750** |
| **newcomer advantage** | **+0.300** | **+0.200** |
| reset removes | 100% | 100% |
| scramble removes | 100% | 100% |
| mean probes, baseline → collective | 4.60 → 0.33 | 2.75 → 0.30 |

Both budgets are set so the naive probe ceiling is 0.500 — the ceilings are matched, not the raw
budgets, so a difference between the columns is a domain effect rather than a difficulty one. Both
domains run through one implementation of §22 (`run_newcomer_procedure`); the refactor that made
that true was verified by diffing the device benchmark's metrics before and after, which are
identical. Data in `results/m10_domain_transfer.json`.

---

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
alembic upgrade head
pytest -q
```

### The command line

```bash
civitas init-db
civitas bootstrap --name "My Org" --slug my-org --email me@example.com   # prints one API key
civitas workspace --org my-org --name Research --slug research
civitas submit --workspace <id> "investigate the intermittent data corruption and fix it"
civitas status <project-id>          # rebuilt from the database, identical after a restart
civitas domains                      # what this process can actually run
civitas benchmark --org my-org       # prints the gates; withholds metrics if one failed
civitas serve                        # the API, OpenAPI at /api/v1/docs
civitas worker --count 2
```

### The web UI

`civitas serve`, then open <http://127.0.0.1:8000/ui/>. Fourteen screens over the same versioned
API — overview, projects, tasks, episodes, artifacts, retrieval, provenance, tools, benchmarks,
metrics, specialization, institutions, cost and events — with live updates streamed from the
append-only event log.

No build step, no framework and no CDN: the page is served by the process that serves the API, so
a deployment is one artifact and a Colab or air-gapped runtime renders exactly what production
does. Two rules are enforced on screen and tested in a real browser: a metric the platform reports
as `null` renders as "not available" with its reason, never as `0`, and a benchmark whose gates
failed shows the gates and withholds the numbers.

### Running it in GitHub

Three ways, and the first is why GitHub is a **stronger** runtime for this than Colab: an Actions
runner has Docker and PostgreSQL, so the sandbox gets container isolation and the two-backend
parametrisation actually runs.

| where | what it gives you |
|---|---|
| **Codespaces** (`.devcontainer/`) | a terminal in the browser with Docker-in-Docker, PostgreSQL, and a **non-root** user — the strongest configuration in this project. Port 8000 is forwarded, so `civitas serve` gives you the UI. |
| **CI** (`.github/workflows/ci.yml`) | on every push: lint, the suite on **both** backends, a migrations check, the browser tests, and an image build that asserts the container is not uid 0 |
| **Acceptance** (`.github/workflows/acceptance.yml`) | run §65's six levels from the Actions tab or weekly on a schedule; the result table lands in the job summary and `acceptance.json` is uploaded as an artifact |

Non-root matters in all three: `RLIMIT_NPROC` is **silently unenforced for uid 0** — measured in
M4, where a test spawned 5000 processes against a limit of 8 — so a run as root advertises a
sandbox limit it does not hold. Colab runs as root and cannot fix this; a Codespace can.

### Deployment

```bash
docker compose -f ops/docker-compose.yml up      # Postgres, migrations, API, worker
kubectl apply -f ops/k8s/civitas.yaml            # or Kubernetes
```

The container runs as a non-root user, and that is load-bearing rather than hygiene: `RLIMIT_NPROC`
is **silently unenforced for uid 0** — measured in M4, where a test spawned 5000 processes against
a limit of 8 — so an image running as root would advertise sandbox limits it does not hold. CI
asserts it.

Operational surface: `/healthz` reports what is actually in force (the sandbox backend and the
limits it *cannot* enforce, whether auth is on, whether tracing is configured against a package
that is installed), `/readyz` reports whether the database is reachable, and `/metrics` serves
Prometheus exposition with no tenant identifier in any label.

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
  cli.py            the `civitas` command line
  acceptance.py     §65's six levels in one campaign, each with its control (§65)
  observability.py  structured logs, Prometheus metrics, optional tracing (§53)
  preflight.py      the production checklist, executable (§64)
  quotas.py         organization resource limits over a rolling window (§58)
  web/              the UI (§49) — one HTML file, one script, one stylesheet, no build step
  config.py         settings and secrets (§40)
  domain/           closed enumerations — arms, termination reasons, artifact and relation types
  domains/          domain adapters: tasks, tools, agents, evaluators, and the §47 leak rule (§5)
  persistence/      SQLAlchemy models, portable types, events, jobs, leases (§7, §42–§45)
  runtime/          episodes, budgets, providers, tools, sandbox (§8, §18, §19)
  knowledge/        arm policies, hybrid retrieval, duplicate-failure detection (§14, §21, §12)
  experiments/      manifests, credit assignment, the §22 procedure, metrics (§20, §22, §46, §48)
  scheduler/        task allocation, roles, specialization (§26–§28)
  institutions/     the A2.2 gate, reputation, procedures (§29–§31)
  services/         long-horizon projects and request decomposition (§5, §32)
  api/              the versioned HTTP API and its security boundary (§50–§52)
  workers/          durable worker loop (§37, §42)
alembic/            migrations
ops/                Dockerfile, docker-compose, Kubernetes manifests (§57)
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
| M1 audit · M2 persistence · M3 runtime · M4 experiments · M5 knowledge · M6 tools · M7 orchestration · M8 institutions · M9 advanced benchmarks · M10 projects, domains, API, CLI · M11 web UI · M12 hardening | complete |
| M13 acceptance: §65's six levels under one manifest, §64 preflight, §63 survival | complete |

Each milestone reports its effect on the M4 benchmark. A feature that moves no number is reported
as such rather than hidden (Part A §A1.5) — M5's hybrid retrieval moved it by exactly 0.000, and
`docs/milestones/M05.md` explains why the benchmark cannot see it and builds the instrument that
can.
