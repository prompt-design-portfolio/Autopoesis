"""Generator for notebooks/Civitas_Colab.ipynb (Part B §34).

Cell source is held in RAW literals (r'''...'''), and the notebook is written through
`nbcheck.write_checked`, which compiles every code cell first. Both defences are required and both
are the repository's existing delivery discipline — see DELIVERY.md for the defect they exist to
stop, which shipped three times before the gate was added.
"""

import nbcheck


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


CELLS = []

CELLS.append(md(r'''# Civitas — persistent artificial civilization

This notebook runs the **full production package**, not a simplified Colab build. The same
`civitas` package runs here, under Docker, on a single server and in a cloud deployment
(Part B §33): what differs between them is configuration, never application logic.

## What it does

1. detects Colab, the runtime and the GPU;
2. mounts Drive and lays out a persistent civilization on it (§35);
3. installs the package and applies migrations;
4. starts the API and workers;
5. runs the **newcomer-advantage benchmark** end to end (§22);
6. exports the manifest and the results;
7. and — the part that matters — **reconnects to a civilization a previous runtime left behind.**

Colab runtimes are ephemeral. The civilization is not.

## Reading the benchmark

The benchmark answers one question: *does a fresh, unchanged agent do better inside a mature
collective environment, and does that advantage go away when the accumulated information is
removed or scrambled?*

Read the **gates first**. If a gate fails the run is not read — not read with caveats. That is
inherited from this repository's own research lineage, where a specification's stated instrument
was shown wrong by its own pre-check before any result was claimed.
'''))

CELLS.append(md('''## 1. Runtime detection (§33, §38)

Nothing here assumes a GPU. The device, dtype and VRAM are detected and reported, because a result
produced on one runtime should be readable against the runtime it came from.'''))

CELLS.append(code(r'''import os, sys, platform, subprocess

IN_COLAB = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ
print("Colab:          ", IN_COLAB)
print("Python:         ", sys.version.split()[0], platform.python_implementation())
print("Platform:       ", platform.platform())
print("CPUs:           ", os.cpu_count())

try:
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                          "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
    detected = out.stdout.strip() if out.returncode == 0 else "none detected"
    print("GPU:            ", detected)
except Exception:
    print("GPU:             none detected (no nvidia-smi)")
'''))

CELLS.append(md('''## 2. Drive, and the persistent civilization layout (§35)

`/MyDrive/Civitas/` holds the database, the object store, manifests, checkpoints, exports and
logs. A new runtime reconnects to whatever a previous one left there.

Off Colab this falls back to a local directory, so the notebook is runnable anywhere — which is
also how it is tested.'''))

CELLS.append(code(r'''from pathlib import Path

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    CIVITAS_ROOT = Path("/content/drive/MyDrive/Civitas")
else:
    CIVITAS_ROOT = Path(os.environ.get("CIVITAS_ROOT", "./civitas_data")).resolve()

for sub in ("databases", "object_store", "artifacts", "experiments", "manifests",
            "checkpoints", "exports", "logs", "models"):
    (CIVITAS_ROOT / sub).mkdir(parents=True, exist_ok=True)

DB_PATH = CIVITAS_ROOT / "databases" / "civitas.db"
EXISTING = DB_PATH.exists()
print("root:           ", CIVITAS_ROOT)
print("database:       ", DB_PATH)
print("pre-existing:   ", EXISTING,
      "(reconnecting to a prior civilization)" if EXISTING else "(new)")
'''))

CELLS.append(md('''## 3. Install (§34)

Editable install from the repository if it is present, otherwise from PyPI. Normal operation does
not require editing application source.'''))

CELLS.append(code(r'''REPO = Path.cwd()
if not (REPO / "civitas").is_dir():
    for candidate in (Path("/content/Autopoesis"), REPO.parent):
        if (candidate / "civitas").is_dir():
            REPO = candidate
            break

if (REPO / "civitas").is_dir():
    print("installing from", REPO)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO)], check=False)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
else:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "civitas"], check=False)

import civitas
print("civitas import ok")
'''))

CELLS.append(md('''## 4. Configuration and secrets (§36, §40)

Native mode: SQLite with WAL on Drive. External-services mode: set `CIVITAS_DATABASE_URL` to a
PostgreSQL URL and the same code runs against it — the domain semantics are identical.

Secrets come from Colab's secret store, the environment or `.env`. They are never printed, never
written into a manifest, and never reach a tool sandbox.'''))

CELLS.append(code(r'''os.environ.setdefault("CIVITAS_DATA_DIR", str(CIVITAS_ROOT / "databases"))
os.environ.setdefault("CIVITAS_DATABASE_URL", f"sqlite:///{DB_PATH}")
os.environ.setdefault("CIVITAS_ENVIRONMENT_VERSION", "colab-1")
# The Colab runtime has no container runtime, so tool execution uses process isolation. It is
# materially weaker than a container and is recorded as such in every manifest (§18, §46).
os.environ.setdefault("CIVITAS_SANDBOX_BACKEND", "subprocess")

from civitas.config import get_settings, reset_settings_cache
reset_settings_cache()
settings = get_settings()

print("runtime:        ", settings.runtime.value)
print("database:       ", settings.database_url.split("://")[0])
print("sandbox:        ", settings.sandbox_backend.value)
print("config hash:    ", settings.config_hash()[:16])
print("provider keys:  ", {
    "anthropic": settings.anthropic_api_key is not None,
    "openai": settings.openai_api_key is not None,
    "google": settings.google_api_key is not None,
})
'''))

CELLS.append(md('''## 5. Database and migrations (§34, §43)

Real Alembic migrations, the same ones production runs. Re-running this cell on an existing
database is a no-op.'''))

CELLS.append(code(r'''from alembic import command
from alembic.config import Config

cfg = Config(str(REPO / "alembic.ini"))
cfg.set_main_option("script_location", str(REPO / "alembic"))
cfg.set_main_option("sqlalchemy.url", settings.database_url)
command.upgrade(cfg, "head")

from civitas.persistence.engine import get_session_factory, healthcheck
print("health:", healthcheck(settings))
SessionFactory = get_session_factory(settings)
'''))

CELLS.append(md('''## 6. Organization and workspace

A workspace is the unit the experimental arms operate on. If one already exists on Drive, it is
reused — this is where reconnection to a prior civilization actually happens.'''))

CELLS.append(code(r'''import uuid
from sqlalchemy import select
from civitas.persistence.models import Organization, Workspace, Artifact, Episode

with SessionFactory() as db:
    org = db.execute(select(Organization).where(Organization.slug == "colab")).scalar_one_or_none()
    if org is None:
        org = Organization(name="Colab Research", slug="colab")
        db.add(org)
        db.commit()
    ORG_ID = org.id

    prior_artifacts = db.execute(select(Artifact)).scalars().all()
    prior_episodes = db.execute(select(Episode)).scalars().all()

print("organization:   ", ORG_ID)
print("artifacts on disk:", len(prior_artifacts))
print("episodes on disk: ", len(prior_episodes))
if prior_episodes:
    print()
    print("This runtime has reconnected to a civilization an earlier runtime left behind.")
'''))

CELLS.append(md('''## 7. Workers (§37)

No systemd, no Docker daemon, no Kubernetes, no privileged process. Bounded runs, so a notebook
cell finishes instead of holding the kernel forever.'''))

CELLS.append(code(r'''from civitas.workers import Worker, WorkerConfig, registered_kinds

worker = Worker(SessionFactory, WorkerConfig(worker_id="colab-0", idle_exit=True, max_jobs=50))
print("handlers:", registered_kinds() or "(none registered; the benchmark runs inline below)")
print("processed:", worker.run())
'''))

CELLS.append(md(r'''## 8. The newcomer-advantage benchmark (§22)

Four conditions, one frozen configuration:

| arm | what it is |
|---|---|
| `baseline_empty` | a fresh agent in an empty environment |
| `collective` | the same agent, after other agents have worked |
| `memory_reset` | the same, with the accumulated state reset |
| `collective_scrambled` | the same quantity of artifacts, relevance destroyed |

`memory_reset` and `collective_scrambled` are the controls that separate *information passed* from
*a store existed*. An advantage that survives scrambling was never about content.

Every agent is the same deterministic policy under the same budget, so nothing about the
individual varies between arms and any difference is caused by what the arm let the agent see.'''))

CELLS.append(code(r'''from civitas.experiments.benchmark import run_newcomer_benchmark

SEEDS = [0, 1, 2]
ACCUMULATION_PASSES = 3

with SessionFactory() as db:
    result = run_newcomer_benchmark(
        db, organization_id=ORG_ID, seeds=SEEDS, accumulation_passes=ACCUMULATION_PASSES,
    )

print("GATES — read these first")
for name, gate in result.gates.items():
    print(f"  {'PASS' if gate.passed else 'FAIL'}  {name}: {gate.detail}")
print()
if not result.gates_passed:
    print("NOT READ:", result.failed_gates())
'''))

CELLS.append(code(r'''import json

if result.gates_passed:
    m = result.metrics()
    print(f"chance level {m['chance_level']:.3f}   "
          f"naive probe ceiling {m['naive_probe_ceiling']:.3f}")
    print()
    header = f"{'arm':<24}{'n':>4}{'rate':>8}{'tools':>8}{'probes':>8}{'read':>7}{'stale':>7}"
    print(header)
    print("-" * len(header))
    for name, arm in m["arms"].items():
        print(f"{name:<24}{arm['n']:>4}{arm['success_rate']:>8.3f}"
              f"{arm['mean_tool_calls']:>8.2f}{arm['mean_probes']:>8.2f}"
              f"{arm['mean_artifacts_read']:>7.2f}{arm['stale_uses']:>7}")
    print()
    print(json.dumps(m["derived"], indent=2))
else:
    print("Gates failed; the numbers are withheld. Diagnose with result.raw_metrics().")
'''))

CELLS.append(md('''### Reading the derived block

- `newcomer_advantage` — `collective` minus `baseline_empty`. The §22 headline.
- `advantage_lost_to_reset` / `advantage_lost_to_scramble` — how much the controls remove.
- `*_removes_fraction` — the *proportion* of the advantage each control removes. This, not the raw
  gap, is what §22's criterion is about: an advantage that survives its ablations intact was never
  caused by what accumulated.'''))

CELLS.append(md(r'''## 8b. Does it transfer? (§5, §22)

A result measured on one task family is a result about that task family. The same procedure runs
over a second domain — **code repair**, where the answer is a function the agent writes and the
evaluator *executes it in the sandbox* against checks the agent never sees.

Both budgets are set so the naive probe ceiling is 0.500, so the two advantages are read against
the same reference: a difference between them is a domain effect, not a difficulty one.'''))

CELLS.append(code(r'''from civitas.domains import all_domains

for d in all_domains():
    print(f"{d.name:14s} v{d.version}  stages={[s.family for s in d.stages()]}  "
          f"evaluators={[e.kind for e in d.evaluators()]}")

# Every generated task is checked for a §47 leak before it can be used.
from civitas.domains import check_no_leak, get_domain

for d in all_domains():
    for t in d.generate(seed=0, count=4):
        check_no_leak(t)
print("\nall generated tasks pass the §47 leak check")'''))

CELLS.append(code(r'''from civitas.experiments.benchmark import run_repair_benchmark

# Small by default so the cell finishes on a free runtime. Raise seeds/passes for the real number.
with SessionFactory() as db:
    repair = run_repair_benchmark(
        db, organization_id=ORG_ID, seeds=[0, 1, 2], accumulation_passes=2,
    )

print("gates:")
for name, gate in repair.gates.items():
    print(f"  {'PASS' if gate.passed else 'FAIL'}  {name}: {gate.detail}")

try:
    m = repair.metrics()
except Exception as exc:
    print("\nnot readable:", exc)
else:
    for arm, summary in m["arms"].items():
        print(f"  {arm:22s} n={summary['n']:3d}  rate={summary['success_rate']:.4f}  "
              f"probes={summary['mean_probes']:.2f}")
    print("\nnewcomer advantage on code repair:", m["derived"]["newcomer_advantage"])'''))

CELLS.append(md('''## 9. Metrics (§48)

Metrics that cannot yet be computed report `None` with a reason, never `0.0`. A dashboard that
renders "not measurable" as "measured zero" manufactures a finding.'''))

CELLS.append(code(r'''from civitas.experiments.metrics import export
from sqlalchemy import select
from civitas.persistence.models import Workspace

with SessionFactory() as db:
    ws = db.execute(
        select(Workspace).where(Workspace.slug.like("bm-collective-%"))
        .order_by(Workspace.created_at.desc())
    ).scalars().first()
    payload = export(db, workspace_id=ws.id) if ws else None

if payload:
    print("workspace:", payload["workspace_id"])
    for name, metric in payload["metrics"].items():
        if metric["value"] is None:
            print(f"  {name:<32} —        ({metric['unavailable']})")
        else:
            value = metric["value"]
            shown = f"{value:.4f}" if isinstance(value, float) else str(value)
            print(f"  {name:<32} {shown:<9}{metric['unit']}")
'''))

CELLS.append(md('''## 10. Manifest and export (§41, §46)

The manifest records everything needed to reconstruct the run — commit, schema version,
dependencies, model, sampling, prompts, tools, budgets, environment, sandbox backend and
configuration hash — and no secrets.'''))

CELLS.append(code(r'''from datetime import datetime, timezone
from civitas.experiments.manifest import build_manifest, manifest_dict
from civitas.experiments.benchmark import frozen_config
from civitas.runtime.sandbox import best_available

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
config = frozen_config()

with SessionFactory() as db:
    manifest = build_manifest(
        db, experiment_run_id=None, arm="newcomer_benchmark",
        provider=config["provider"], model_name=config["model"],
        model_version=config["model_version"],
        sampling={"temperature": config["temperature"]},
        prompt_versions={"system": config["system_prompt_version"]},
        retrieval_policy={"version": config["retrieval_policy_version"]},
        budgets=config["budgets"], evaluator={"kind": "exact_match"},
        seeds={"seeds": SEEDS, "accumulation_passes": ACCUMULATION_PASSES},
        sandbox_backend=best_available().name,
        sandbox_unenforced_limits=list(best_available().unenforced_limits()),
        settings=settings,
    )
    db.commit()
    manifest_payload = manifest_dict(manifest)

manifest_path = CIVITAS_ROOT / "manifests" / f"newcomer_{stamp}.json"
manifest_path.write_text(json.dumps(manifest_payload, indent=2))

results_path = CIVITAS_ROOT / "exports" / f"newcomer_{stamp}.json"
results_path.write_text(json.dumps(result.raw_metrics(), indent=2))

print("manifest ->", manifest_path)
print("results  ->", results_path)
print("config hash:", manifest_payload["config_hash"][:16])
print("sandbox:    ", manifest_payload["sandbox_backend"])
print("unenforced: ", manifest_payload["environment"]["sandbox_unenforced_limits"] or "none")
'''))

CELLS.append(md(r'''## 11. The acceptance test (§63)

The workflow §63 requires, automated: create a workspace, run a bounded agent, persist artifacts,
run a second fresh agent, **drop every in-memory object and reopen the database from disk**, run a
third fresh agent, and confirm it retrieves what the first two left behind.

Step 15 of §63 is "terminate application/runtime". A notebook cell cannot kill its own kernel, so
this disposes the engine and rebuilds every connection from the file on Drive — which is the
property that actually matters, and it is checked rather than asserted.'''))

CELLS.append(code(r'''from civitas.persistence.engine import dispose_engines, get_session_factory
from civitas.experiments.tasks.hidden_rule import build_device, era_instances
from civitas.experiments.runner import run_benchmark_episode
from civitas.domain.enums import ExperimentArm

device = build_device(seed=99, era=1)
instance = era_instances(device)[0]

with SessionFactory() as db:
    ws = Workspace(organization_id=ORG_ID, name="acceptance",
                   slug=f"acceptance-{uuid.uuid4().hex[:8]}")
    db.add(ws)
    db.commit()
    ACCEPT_WS = ws.id
    for i in range(2):
        r = run_benchmark_episode(db, workspace_id=ACCEPT_WS, device=device, instance=instance,
                                  arm=ExperimentArm.COLLECTIVE, record_findings=True,
                                  probe_order_seed=i)
        print(f"  episode {i + 1}: success={r.succeeded} probes={r.probes}"
              f" read={r.artifacts_read}")
    db.commit()

# Drop every engine, pool and session. Everything below comes off the file on Drive.
dispose_engines()
reset_settings_cache()
print("\nengines disposed — reopening from disk")

settings = get_settings()
SessionFactory = get_session_factory(settings)

with SessionFactory() as db:
    restored = db.execute(
        select(Artifact).where(Artifact.workspace_id == ACCEPT_WS)
    ).scalars().all()
    print("artifacts restored from disk:", len(restored))
    third = run_benchmark_episode(db, workspace_id=ACCEPT_WS, device=device, instance=instance,
                                  arm=ExperimentArm.COLLECTIVE, record_findings=False,
                                  probe_order_seed=2)
    db.commit()

print()
print("third agent, after reconnect:")
print("  retrieved prior knowledge:", third.artifacts_read > 0)
print("  succeeded:                ", third.succeeded)
print("  probes needed:            ", third.probes)
assert third.artifacts_read > 0, "the reconnected agent did not inherit the collective"
print()
print("ACCEPTANCE PASSED — the civilization survived the loss of every runtime object.")
'''))

CELLS.append(md(r'''## 11b. The six scientific levels (§65)

Everything above measures one mechanism. This runs all six of §65's criteria **in one campaign
under one manifest**, each beside its own negative control, and gates each separately — six
results produced by six runs are six claims, and the thing being accepted is one platform.

A level whose gate fails withholds its number rather than annotating it, and one failing level
fails the run: the levels are separate claims and an average over them would be a number no claim
supports.

Small by default so the cell finishes on a free runtime. `results/m13_acceptance.json` in the
repository holds the full five-seed run.'''))

CELLS.append(code(r'''from civitas.acceptance import acceptance_report

with SessionFactory() as db:
    report = acceptance_report(db, seeds=[0, 1], accumulation_passes=2, include={1, 2, 3})

for level in report["levels"]:
    mark = "PASS" if level["passed"] else "FAIL"
    value = "withheld" if level["value"] is None else level["value"]
    print(f"  {mark}  L{level['level']} {level['name']:32s} {value}  ({level['unit']})")
    print(f"        control: {level['control']}")
    if level["withheld_reason"]:
        print(f"        withheld: {level['withheld_reason']}")

print()
print("accepted:", report["accepted"])
print("sandbox limits NOT enforced here:",
      report["manifest"]["sandbox_unenforced_limits"] or "none")'''))

CELLS.append(md('''## 12. Stopping and resuming

Nothing needs to be done to stop safely: every write is committed to the database on Drive as it
happens, and experiment runs are keyed so a resumed campaign skips completed work rather than
repeating it (§41).

Re-run this notebook from the top in a new runtime. Cell 6 will report the artifacts and episodes
it found on Drive, and the civilization continues from there.'''))

NB = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"provenance": [], "toc_visible": True},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

if __name__ == "__main__":
    import pathlib
    pathlib.Path("notebooks").mkdir(exist_ok=True)
    nbcheck.write_checked(NB, "notebooks/Civitas_Colab.ipynb")
