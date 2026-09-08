"""Experiment manifests (Part B §46).

Everything needed to reconstruct a run, and no secrets. The exclusion is by *type* — every
`SecretStr` is dropped whatever it is called — so a credential added later cannot leak by being
named something a denylist does not recognise (§40).

Two fields are load-bearing beyond bookkeeping:

* `config_hash` makes "identical configuration" checkable rather than asserted, which is what the
  A2.2 matched A/B gate rests on.
* `sandbox_backend` records the isolation actually in force, so a result produced under the
  process-isolation fallback is never read as one produced under a container.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import uuid
from functools import lru_cache
from typing import Any

from sqlalchemy.orm import Session

from civitas.config import Settings, get_settings
from civitas.persistence.models import ExperimentManifest

#: Packages whose version can change a result. A full `pip freeze` would make every manifest
#: differ on irrelevant tooling and destroy the usefulness of comparing two.
TRACKED_PACKAGES = (
    "sqlalchemy", "alembic", "pydantic", "fastapi", "httpx", "numpy",
    "anthropic", "openai", "transformers", "torch",
)


@lru_cache(maxsize=1)
def git_commit() -> str:
    """The commit a run was produced at. `unknown` when there is no repository — recorded as
    such rather than omitted, because a manifest that silently lacks a commit looks complete."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0:
            dirty = subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5
            )
            suffix = "-dirty" if dirty.stdout.strip() else ""
            return out.stdout.strip() + suffix
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


@lru_cache(maxsize=1)
def dependency_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {"python": sys.version.split()[0]}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
    return out


@lru_cache(maxsize=1)
def schema_version() -> str:
    """The Alembic head. A result read against a different schema may not mean what it says."""
    try:
        from pathlib import Path

        from alembic.config import Config
        from alembic.script import ScriptDirectory

        root = Path(__file__).resolve().parents[2]
        cfg = Config(str(root / "alembic.ini"))
        cfg.set_main_option("script_location", str(root / "alembic"))
        return ScriptDirectory.from_config(cfg).get_current_head() or "unknown"
    except Exception:
        return "unknown"


def environment_facts() -> dict[str, Any]:
    from civitas.runtime.providers.local_hf import detect_device

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python_implementation": platform.python_implementation(),
        "device": detect_device(),
    }


def build_manifest(
    session: Session,
    *,
    experiment_run_id: uuid.UUID | None,
    arm: str,
    provider: str,
    model_name: str,
    model_version: str = "",
    sampling: dict[str, Any] | None = None,
    prompt_versions: dict[str, str] | None = None,
    tool_versions: dict[str, str] | None = None,
    retrieval_policy: dict[str, Any] | None = None,
    budgets: dict[str, Any] | None = None,
    evaluator: dict[str, Any] | None = None,
    seeds: dict[str, Any] | None = None,
    workspace_snapshot_id: uuid.UUID | None = None,
    sandbox_backend: str = "",
    sandbox_unenforced_limits: list[str] | None = None,
    settings: Settings | None = None,
    extra: dict[str, Any] | None = None,
) -> ExperimentManifest:
    """Record a run's full provenance. Immutable once written (§46, and the M2 session guard)."""
    settings = settings or get_settings()

    #: The evaluator spec is summarised, never copied: a manifest that embedded expected answers
    #: would be a file that leaks the benchmark's ground truth to anything allowed to read
    #: manifests (§47).
    evaluator_summary = dict(evaluator or {})
    for leaking in ("expected", "expected_answer", "answers", "mapping"):
        evaluator_summary.pop(leaking, None)

    manifest = ExperimentManifest(
        experiment_run_id=experiment_run_id,
        git_commit=git_commit(),
        app_version=settings.app_version,
        schema_version=schema_version(),
        dependency_versions=dependency_versions(),
        provider=provider,
        model_name=model_name,
        model_version=model_version,
        sampling_parameters=sampling or {},
        prompt_versions=prompt_versions or {},
        tool_versions=tool_versions or {},
        retrieval_policy=retrieval_policy or {},
        experiment_arm=arm,
        budgets=budgets or {},
        evaluator=evaluator_summary,
        environment={
            **environment_facts(),
            "settings": settings.manifest_dict(),
            # A bound the run did not actually have must not be inferred from the backend name.
            "sandbox_unenforced_limits": sandbox_unenforced_limits or [],
            **(extra or {}),
        },
        workspace_snapshot_id=workspace_snapshot_id,
        seeds=seeds or {},
        sandbox_backend=sandbox_backend,
    )
    manifest.config_hash = manifest_config_hash(manifest)
    session.add(manifest)
    session.flush()
    return manifest


def manifest_config_hash(manifest: ExperimentManifest) -> str:
    """Hash of the fields that define the *configuration*, excluding the run's identity.

    `experiment_run_id` and the seed are deliberately outside the hash: two arms of one matched
    experiment must produce the *same* configuration hash while differing in arm and seed, or the
    A2.2 gate could never verify that a comparison was matched. The arm is excluded for the same
    reason — it is the variable under test, not part of the configuration it is tested under.
    """
    payload = {
        "git_commit": manifest.git_commit,
        "app_version": manifest.app_version,
        "schema_version": manifest.schema_version,
        "dependency_versions": manifest.dependency_versions,
        "provider": manifest.provider,
        "model_name": manifest.model_name,
        "model_version": manifest.model_version,
        "sampling_parameters": manifest.sampling_parameters,
        "prompt_versions": manifest.prompt_versions,
        "tool_versions": manifest.tool_versions,
        "retrieval_policy": manifest.retrieval_policy,
        "budgets": manifest.budgets,
        "evaluator": manifest.evaluator,
        "sandbox_backend": manifest.sandbox_backend,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def manifest_dict(manifest: ExperimentManifest) -> dict[str, Any]:
    """Export form (§46, §48 exportable metrics)."""
    return {
        "id": str(manifest.id),
        "experiment_run_id": (
            str(manifest.experiment_run_id) if manifest.experiment_run_id else None
        ),
        "git_commit": manifest.git_commit,
        "app_version": manifest.app_version,
        "schema_version": manifest.schema_version,
        "dependency_versions": manifest.dependency_versions,
        "provider": manifest.provider,
        "model_name": manifest.model_name,
        "model_version": manifest.model_version,
        "sampling_parameters": manifest.sampling_parameters,
        "prompt_versions": manifest.prompt_versions,
        "tool_versions": manifest.tool_versions,
        "retrieval_policy": manifest.retrieval_policy,
        "experiment_arm": manifest.experiment_arm,
        "budgets": manifest.budgets,
        "evaluator": manifest.evaluator,
        "environment": manifest.environment,
        "workspace_snapshot_id": (
            str(manifest.workspace_snapshot_id) if manifest.workspace_snapshot_id else None
        ),
        "seeds": manifest.seeds,
        "config_hash": manifest.config_hash,
        "sandbox_backend": manifest.sandbox_backend,
        "created_at": manifest.created_at.isoformat(),
    }
