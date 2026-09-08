"""Settings (§34, kept).

Deliberately small. The original package's settings carry providers, quotas, sandboxes and a web
UI, all of which B§1 makes dormant. What G1 needs is a database URL, a place to put artifacts, and
the two knobs the reproduction gate is argued about with.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    #: SQLite is the Colab-compatible backend (§43); PostgreSQL is the source of truth (§41).
    database_url: str = field(
        default_factory=lambda: os.environ.get(
            "CIVITAS_G_DATABASE_URL", f"sqlite:///{REPO_ROOT / 'var' / 'civitas_g.db'}"))
    artifact_dir: Path = field(
        default_factory=lambda: Path(os.environ.get(
            "CIVITAS_G_ARTIFACT_DIR", str(REPO_ROOT / "var" / "artifacts"))))

    #: "three decimals" (A3, B§5.1) as a number. A reference summary prints `%.3f`, so two values
    #: that round to the same three decimals can differ by just under 5e-4 before rounding.
    reproduction_tolerance: float = 5e-4

    #: The tolerance used when comparing two BACKENDS' readings of the same rows (D12). Tighter,
    #: because this is not a rounding comparison: both sides recompute from the same stored
    #: numbers, so anything above float noise means the round-trip lost precision.
    backend_agreement_tolerance: float = 1e-9

    def resolved(self) -> Settings:
        """Ensure the directories a run writes to exist."""
        if self.database_url.startswith("sqlite:///"):
            Path(self.database_url.split("///", 1)[1]).parent.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        return self


_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Settings().resolved()
    return _SETTINGS


def set_settings(settings: Settings | None) -> None:
    """For tests, which point the URL at a temporary file."""
    global _SETTINGS
    _SETTINGS = settings.resolved() if settings else None
