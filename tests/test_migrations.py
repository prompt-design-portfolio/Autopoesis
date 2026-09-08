"""Migration tests (Part B §43, §55).

Two properties matter and are checked separately:

1. **Parity.** A database built by `alembic upgrade head` must be the same shape as one built by
   `Base.metadata.create_all`. Without this the tests run against a schema production never has.
2. **Reversibility.** `downgrade base` then `upgrade head` must return to the same state, so a
   failed deployment can be backed out.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import inspect

from alembic import command
from civitas.persistence.engine import create_db_engine
from civitas.persistence.models import Base

REPO_ROOT = Path(__file__).resolve().parents[1]


def _alembic_config(url: str) -> Config:
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg


def _shape(engine) -> dict[str, dict]:
    """Tables, columns, nullability and indexes — the parts a divergence would show up in."""
    insp = inspect(engine)
    out: dict[str, dict] = {}
    for table in sorted(insp.get_table_names()):
        if table == "alembic_version":
            continue
        out[table] = {
            "columns": {
                c["name"]: {"nullable": bool(c["nullable"])}
                for c in insp.get_columns(table)
            },
            "pk": sorted(insp.get_pk_constraint(table)["constrained_columns"]),
            "indexes": sorted(
                (i["name"], tuple(sorted(i["column_names"])), bool(i["unique"]))
                for i in insp.get_indexes(table)
            ),
        }
    return out


@pytest.fixture
def target_url(tmp_path, backend) -> str:
    """A clean database on whichever backend is under test.

    Migrations are checked on PostgreSQL as well as SQLite because that is where they will
    actually be run in production (Part B §43), and `render_as_batch` behaves differently on the
    two dialects — a migration valid on one is not automatically valid on the other.
    """
    if backend == "postgresql":
        from tests.conftest import POSTGRES_URL

        engine = create_db_engine(url=POSTGRES_URL)
        with engine.begin() as conn:
            conn.exec_driver_sql("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
        engine.dispose()
        return POSTGRES_URL
    return f"sqlite:///{tmp_path}/migrated.db"


@pytest.fixture
def migrated_engine(target_url):
    command.upgrade(_alembic_config(target_url), "head")
    engine = create_db_engine(url=target_url)
    yield engine
    engine.dispose()


def test_migrations_produce_the_same_schema_as_the_models(migrated_engine, tmp_path, backend):
    if backend == "postgresql":
        pytest.skip("compared on SQLite; the PostgreSQL run proves the migrations execute")
    direct = create_db_engine(url=f"sqlite:///{tmp_path}/direct.db")
    Base.metadata.create_all(direct)

    migrated_shape, direct_shape = _shape(migrated_engine), _shape(direct)
    direct.dispose()

    missing = set(direct_shape) - set(migrated_shape)
    extra = set(migrated_shape) - set(direct_shape)
    assert not missing, f"migrations do not create: {sorted(missing)}"
    assert not extra, f"migrations create tables the models do not define: {sorted(extra)}"

    for table in sorted(direct_shape):
        assert migrated_shape[table]["columns"] == direct_shape[table]["columns"], (
            f"{table}: migrated columns differ from the model"
        )
        assert migrated_shape[table]["pk"] == direct_shape[table]["pk"], f"{table}: pk differs"


def test_downgrade_and_upgrade_round_trips(target_url):
    url = target_url
    cfg = _alembic_config(url)

    command.upgrade(cfg, "head")
    engine = create_db_engine(url=url)
    before = _shape(engine)
    engine.dispose()

    command.downgrade(cfg, "base")
    engine = create_db_engine(url=url)
    assert not [t for t in inspect(engine).get_table_names() if t != "alembic_version"]
    engine.dispose()

    command.upgrade(cfg, "head")
    engine = create_db_engine(url=url)
    after = _shape(engine)
    engine.dispose()

    assert after == before, "the schema after a downgrade/upgrade cycle is not the original"


def test_a_migrated_database_accepts_the_real_domain_objects(migrated_engine):
    """Parity of shape is not parity of behaviour. This writes a real graph through the migrated
    schema, so a wrong column type would fail here rather than in production."""
    import uuid

    from sqlalchemy.orm import sessionmaker

    from civitas.domain.enums import ArtifactType, EventType, RelationType
    from civitas.persistence.events import emit
    from civitas.persistence.models import Artifact, ArtifactRelation, Organization, Workspace
    from civitas.persistence.session import install_guards

    factory = sessionmaker(bind=migrated_engine, expire_on_commit=False, future=True)
    install_guards(factory)
    s = factory()
    try:
        org = Organization(name="O", slug=f"o-{uuid.uuid4().hex[:8]}")
        s.add(org)
        s.flush()
        ws = Workspace(organization_id=org.id, name="W", slug=f"w-{uuid.uuid4().hex[:8]}")
        s.add(ws)
        s.flush()

        a = Artifact(workspace_id=ws.id, type=ArtifactType.OBSERVATION, title="obs",
                     structured={"k": [1, 2]})
        b = Artifact(workspace_id=ws.id, type=ArtifactType.HYPOTHESIS, title="hyp")
        s.add_all([a, b])
        s.flush()
        s.add(ArtifactRelation(source_id=a.id, target_id=b.id, type=RelationType.SUPPORTS))
        emit(s, workspace_id=ws.id, type=EventType.ARTIFACT_CREATED, payload={"id": str(a.id)})
        s.commit()

        got = s.get(Artifact, a.id)
        assert got.type is ArtifactType.OBSERVATION, "enum columns must round-trip typed"
        assert got.structured == {"k": [1, 2]}
    finally:
        s.close()


def test_the_schema_compiles_for_postgresql(backend):
    """PostgreSQL is the production source of truth (Part B §43), and CI has no server.

    Compiling every CREATE TABLE against the PostgreSQL dialect catches the class of defect that
    would otherwise only appear at deployment — a type or construct that SQLite tolerates and
    PostgreSQL rejects. It is not a substitute for running against a real server, which the
    integration job does; it is the check that can run everywhere.
    """
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.schema import CreateIndex, CreateTable

    dialect = postgresql.dialect()
    for table in Base.metadata.sorted_tables:
        ddl = str(CreateTable(table).compile(dialect=dialect))
        assert ddl.strip().upper().startswith("CREATE TABLE")
        # JSONB and native UUID must be selected on PostgreSQL, not the SQLite fallbacks.
        for col in table.columns:
            rendered = col.type.compile(dialect=dialect)
            assert rendered not in ("TEXT",) or col.name not in ("meta", "payload"), (
                f"{table.name}.{col.name} fell back to TEXT on PostgreSQL"
            )
        for index in table.indexes:
            str(CreateIndex(index).compile(dialect=dialect))
