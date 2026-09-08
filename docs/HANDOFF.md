# Handoff

Read this first if you are resuming work (Part A §A5).

## State

Branch `claude/master-prompt-init-ehmsbe`, pushed. **M1–M13 complete**, 778 tests green on both
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
| M11 | the web UI: fourteen screens, live updates over the event log, browser-driven tests | — (a null on the newcomer benchmark, expected and explained in `docs/milestones/M11.md`) |
| M12 | hardening: CSP, rate limiting, redaction, metrics, tracing, quotas, CI, Docker, Kubernetes | `results/m12_hardening.json` |
| M13 | acceptance: §65's six levels in one campaign, §64 preflight, §63 survival | `results/m13_acceptance.json` |

## What remains

All thirteen milestones are complete. `civitas acceptance` re-runs §65's six levels and exits
non-zero if any fails; `civitas preflight` checks §64's production criteria against a deployment.

Two limitations to carry forward rather than rediscover: the agents are deterministic policies,
not language models — the strongest instrument for measuring the *platform's* contribution (§20),
and not a measurement of what a model would do; and two domains is two.

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
8. **Reading must not write.** The engine's `BEGIN IMMEDIATE` takes SQLite's *write* lock even to
   read, so every read path uses `read_only_session_factory` and API GETs get a read-only session.
   Three separate defects in M11 were this one fact wearing different hats. If you add a read path,
   do not give it a writing session.
9. **Four settings did nothing for eleven milestones.** `log_json`, `metrics_enabled`,
   `otel_endpoint` and `rate_limit_per_minute` were read from the environment, printed into the
   manifest, and honoured by no code until M12. When you add a setting, add the code that reads
   it in the same change — a control that exists only in configuration is worse than no control.
10. **The browser tests are not decoration.** `tests/test_web_ui.py` drives the real page in
   Chromium and found four defects the API tests structurally could not — they issue one request
   at a time, and the bugs only appear when three arrive at once. They skip with a stated reason
   when no Chromium is available; they never silently pass.
11. **A new domain cannot be added without stating its §47 argument.** `check_no_leak` runs in a
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
