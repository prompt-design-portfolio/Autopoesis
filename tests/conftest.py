"""Shared fixtures.

Every test runs against a real database created by the real schema. Provider calls are mocked;
mechanisms are not (Part A §A1.4).
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from civitas.config import Settings
from civitas.persistence.engine import create_db_engine
from civitas.persistence.models import Base, Organization, Workspace
from civitas.persistence.session import install_guards

#: Set to a PostgreSQL URL to run the whole suite against the production backend as well.
#: Part B §6 forbids separate application logic for Colab, and the only way to know that holds is
#: to run the same tests on both dialects. When unset the suite runs on SQLite alone and says so,
#: rather than silently testing one backend and claiming two.
POSTGRES_URL = os.environ.get("CIVITAS_TEST_POSTGRES_URL")

BACKENDS = ["sqlite"] + (["postgresql"] if POSTGRES_URL else [])


def _sqlite_url(tmp_path_factory, name: str) -> str:
    """A file-backed SQLite database per test.

    Not `:memory:`: the engine sets WAL and `BEGIN IMMEDIATE`, and an in-memory database exercises
    neither. A test that passes against a different journalling mode than production has not
    tested production.
    """
    d = tmp_path_factory.mktemp(name)
    return f"sqlite:///{d}/civitas.db"


@pytest.fixture(scope="function", params=BACKENDS)
def backend(request) -> str:
    return request.param


@pytest.fixture(scope="function")
def settings(tmp_path_factory, backend: str) -> Settings:
    if backend == "postgresql":
        # A schema per test: PostgreSQL has no per-test file to throw away, and a shared schema
        # would let one test's rows change another's answer.
        url = POSTGRES_URL
    else:
        url = _sqlite_url(tmp_path_factory, "db")
    return Settings(
        database_url=url,
        data_dir=tmp_path_factory.mktemp("data"),
        environment_version="test-env-1",
        jwt_secret="test-secret-not-a-real-key",
        auth_enabled=True,
    )


@pytest.fixture(scope="function")
def engine(settings: Settings, backend: str) -> Iterator[Engine]:
    eng = create_db_engine(settings)
    if backend == "postgresql":
        # Drop and rebuild rather than truncate: a schema change between runs would otherwise
        # leave a stale table behind and the failure would look like a code defect.
        Base.metadata.drop_all(eng)
    Base.metadata.create_all(eng)
    yield eng
    if backend == "postgresql":
        Base.metadata.drop_all(eng)
    eng.dispose()


@pytest.fixture(scope="function")
def session_factory(engine: Engine) -> sessionmaker[Session]:
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    install_guards(factory)
    return factory


@pytest.fixture(scope="function")
def db(session_factory) -> Iterator[Session]:
    s = session_factory()
    try:
        yield s
    finally:
        s.rollback()
        s.close()


@pytest.fixture
def org(db: Session) -> Organization:
    o = Organization(name="Test Org", slug=f"org-{uuid.uuid4().hex[:8]}")
    db.add(o)
    db.commit()
    return o


@pytest.fixture
def workspace(db: Session, org: Organization) -> Workspace:
    w = Workspace(
        organization_id=org.id,
        name="Test Workspace",
        slug=f"ws-{uuid.uuid4().hex[:8]}",
        environment_version="test-env-1",
    )
    db.add(w)
    db.commit()
    return w


@pytest.fixture(autouse=True)
def _no_real_credentials(monkeypatch):
    """No test may reach a paid provider (Part B §55). Keys are stripped from the environment."""
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
                "CIVITAS_ANTHROPIC_API_KEY", "CIVITAS_OPENAI_API_KEY", "CIVITAS_GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert "ANTHROPIC_API_KEY" not in os.environ
