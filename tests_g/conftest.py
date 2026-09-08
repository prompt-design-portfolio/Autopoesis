"""Fixtures for the G tests.

A1.6: tests exercise the real mechanism, and the simulation is never mocked -- it *is* the
mechanism. So the fixtures here give a test a real database and a real (short) run of the engine,
never a stub of either. The only concession to time is phase length, and short phases are the same
code path the reference pre-check itself used.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import Engine

from civitas_g.persistence.engine import (
    create_all,
    create_db_engine,
    read_only_session_factory,
    session_factory,
)
from civitas_g.persistence.models import Base

POSTGRES_URL = os.environ.get("CIVITAS_G_TEST_POSTGRES_URL")

#: Every persistence test runs on both, when both are configured. §43 forbids separate application
#: logic for Colab, so a test that only ran on SQLite would leave the seam untested where it
#: matters -- PostgreSQL's JSONB is the half that rejects NaN.
BACKENDS = ["sqlite"] + (["postgresql"] if POSTGRES_URL else [])


@pytest.fixture(params=BACKENDS)
def backend(request) -> str:
    return request.param


@pytest.fixture
def engine(backend: str, tmp_path) -> Engine:
    if backend == "sqlite":
        eng = create_db_engine(url=f"sqlite:///{tmp_path / 'g.db'}")
    else:
        eng = create_db_engine(url=POSTGRES_URL)
        Base.metadata.drop_all(eng)
    create_all(eng)
    yield eng
    if backend == "postgresql":
        Base.metadata.drop_all(eng)
    eng.dispose()


@pytest.fixture
def sessions(engine: Engine):
    return session_factory(engine=engine)


@pytest.fixture
def readers(engine: Engine):
    return read_only_session_factory(engine=engine)


@pytest.fixture(scope="session")
def short_run():
    """One real run of the engine, short. Session-scoped: it is the slowest thing here."""
    from civitas_g.world.engine import build_run_spec
    from civitas_g.world.engine import run as run_engine

    return run_engine(build_run_spec("collective", seed=0, phase_steps=150))
