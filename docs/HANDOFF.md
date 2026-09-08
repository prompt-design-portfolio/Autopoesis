# Handoff

Read this first if you are resuming work (Part A §A5).

## State

Branch `claude/master-prompt-init-ehmsbe`, pushed. **M1–M10 complete**, 634 tests green on both
SQLite and PostgreSQL, ruff clean.

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]" "psycopg[binary]"
bash scripts/start_test_postgres.sh /var/tmp/civitas-pg     # prints the URL to export
CIVITAS_TEST_POSTGRES_URL=postgresql+psycopg://civitas@127.0.0.1:55432/civitas_test \
  .venv/bin/python -m pytest tests/ -q
```

The two-backend run is not optional — M2 found a concurrency race SQLite structurally cannot
expose. The PostgreSQL cluster lives outside the scratchpad because the scratchpad's permissions
are reset periodically, which kills the server mid-run.

## What is built

| milestone | what | committed measurement |
|---|---|---|
| M1 | audit; **no Civitas prototype existed** — the repo was the research lineage only | — |
| M2 | 44 tables, two backends, immutable events, durable leases | — |
| M3 | bounded episodes, nine enforced arms, hardened sandbox, nine providers | — |
| M4 | frozen mode, manifests, credit assignment, newcomer benchmark | `results/m4_newcomer_benchmark.json` |
| M5 | hybrid retrieval, graph, provenance, versioning, staleness, consolidation | `results/m5_knowledge_system.json` |
| M6 | tool ecology: creation, sandboxed validation, versioning, reuse | `results/m6_tool_ecology.json` |
| M7 | scheduler, specialization, six §28 roles, hypothesis state machine | `results/m7_scheduler.json` |
| M8 | A2.2 gate, reputation, procedures, meta-learning | `results/m8_institutions.json` |
| M9 | distributed knowledge, cumulative culture, capability frontier | `results/m9_advanced_benchmarks.json` |
| M10 | long-horizon projects, request decomposition, domain adapters, versioned API, CLI | `results/m10_domain_transfer.json` |

## What remains

- **M11** — the web UI (§49): thirteen screens, live updates, provenance inspector, benchmark
  dashboard.
- **M12** — production hardening: auth/RBAC (§51), security (§52), observability (§53),
  reliability (§54), CI/CD (§56), deployment targets (§57), resource management (§58).
- **M13** — acceptance: the Colab workflow (§63), the production checklist (§64), and §65's six
  levels with negative controls committed under `results/`.

## Things a resumed session should know

1. **Every milestone reports its effect on the newcomer benchmark, including nulls.** M5, M6 and
   M7 each moved it by exactly 0.000, and each null was diagnosed and produced its own instrument
   rather than being hidden. Keep doing this.
2. **Gates precede rows.** `BenchmarkResult.metrics()` raises when a gate failed; `raw_metrics()`
   is the diagnostic escape hatch and calling it is a statement that you are reading an unlicensed
   number.
3. **A metric that cannot be computed reports `None` with a reason, never 0.0.** This is enforced
   throughout and is worth preserving; several tests assert it directly.
4. **The benchmarks are reproducible across processes.** Anything seeded uses `hashlib`, never
   `hash()`, which Python randomises per process.
5. **The arm regression (`tests/test_arms.py`) must stay green** — Part A §A4.2.
6. **`scripts/start_test_postgres.sh`** brings up the test cluster.
7. **There is one implementation of §22.** `run_newcomer_benchmark` and `run_repair_benchmark`
   both call `run_newcomer_procedure(domain=...)`. If you change it, verify with
   `scripts/newcomer_snapshot.py` on the tree before and after and diff — that is how the M10
   refactor was shown to leave the M4 result untouched.
8. **A new domain cannot be added without stating its §47 argument.** `check_no_leak` runs in a
   test parametrized over the registry, in `create_domain_task`, and in the CLI. It has already
   caught a real leak (a description ending "the corrected function source" — "the correct" is a
   marker word).

## Open issues

- `shared_memory` underperforms `solo` on the capability frontier, because it filters to validated
  artifacts and the accumulated findings are self-reported. This is a real property of §21's arm
  and is reported, not patched — but if a later milestone adds routine external validation, the
  arm should be re-measured.
- The four reasoned allocation strategies are indistinguishable on the current task pool, which is
  symmetric. Distinguishing them needs a heterogeneous pool (M9 built the families; the scheduler
  benchmark has not been re-run against them).
- The subprocess sandbox declares `max_processes` unenforceable under uid 0. Running workers as an
  unprivileged user restores it; the container backend does not have the problem.
