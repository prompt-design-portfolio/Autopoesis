"""Engines and session factories (§35, §41, §43).

Two things here are not defaults, and both were learned the expensive way in the original build.

**Reading must not write.** SQLAlchemy's deferred BEGIN takes SQLite's write lock only at the first
write, which turns a read-then-update into a race that SQLite resolves by raising SQLITE_BUSY on
the second writer, mid-transaction. So every transaction opens with `BEGIN IMMEDIATE` -- except a
transaction marked read-only, and that exemption is load-bearing rather than an optimisation. An
immediate BEGIN on a *reader* takes the write lock too, so one long read would block every writer
for as long as it ran. WAL exists precisely so readers do not block the writer; taking the write
lock to read throws that away. The reproduction gate is a long read over every era row of every
run, so it uses `read_only_session_factory` and nothing else.

**Foreign keys are off by default in SQLite.** Without `PRAGMA foreign_keys=ON` every `ondelete`
in the models is decoration, and referential integrity holds on PostgreSQL and not on the backend
Colab actually runs.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from civitas_g.config import Settings, get_settings
from civitas_g.persistence.models import Base

#: Execution-option key marking a transaction as a reader.
READ_ONLY_OPTION = "civitas_g_read_only"

_ENGINES: dict[str, Engine] = {}


def _configure_sqlite(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn, _record):  # pragma: no cover - driver callback
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA busy_timeout=10000")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()

    @event.listens_for(engine, "begin")
    def _on_begin(conn):  # pragma: no cover - driver callback
        if conn.get_execution_options().get(READ_ONLY_OPTION):
            return
        conn.exec_driver_sql("BEGIN IMMEDIATE")


def create_db_engine(settings: Settings | None = None, url: str | None = None) -> Engine:
    settings = (settings or get_settings())
    url = url or settings.database_url
    if url.startswith("sqlite"):
        engine = create_engine(url, future=True, pool_pre_ping=True)
        _configure_sqlite(engine)
        return engine
    return create_engine(url, future=True, pool_pre_ping=True, pool_size=5, max_overflow=10)


def get_engine(settings: Settings | None = None) -> Engine:
    settings = settings or get_settings()
    if settings.database_url not in _ENGINES:
        _ENGINES[settings.database_url] = create_db_engine(settings)
    return _ENGINES[settings.database_url]


def dialect_of(engine: Engine) -> str:
    """"sqlite" or "postgresql". Recorded on every `Reproduction` row (D12)."""
    return engine.dialect.name


def create_all(engine: Engine | None = None, settings: Settings | None = None) -> Engine:
    """Create the G tables. No Alembic at G1: there is no deployed G database to migrate yet, and
    a migration chain with one revision and no history is ceremony rather than safety."""
    engine = engine or get_engine(settings)
    Base.metadata.create_all(engine)
    return engine


def session_factory(settings: Settings | None = None,
                    engine: Engine | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=engine or get_engine(settings), expire_on_commit=False,
                        future=True)


def read_only_session_factory(settings: Settings | None = None,
                              engine: Engine | None = None) -> sessionmaker[Session]:
    """Sessions that take no write lock. Every read path uses this."""
    eng = engine or get_engine(settings)
    return sessionmaker(bind=eng.execution_options(**{READ_ONLY_OPTION: True}),
                        expire_on_commit=False, future=True)


@contextmanager
def writing(settings: Settings | None = None,
            engine: Engine | None = None) -> Iterator[Session]:
    """A write transaction, committed on success and rolled back on anything else."""
    with session_factory(settings, engine)() as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise


@contextmanager
def reading(settings: Settings | None = None,
            engine: Engine | None = None) -> Iterator[Session]:
    """A read transaction that takes no write lock. Never commits."""
    with read_only_session_factory(settings, engine)() as session:
        yield session


def health(engine: Engine | None = None) -> dict[str, str]:
    """Enough to tell a reader which backend produced a number, and whether it was configured."""
    engine = engine or get_engine()
    info = {"dialect": dialect_of(engine),
            "url": str(engine.url.render_as_string(hide_password=True))}
    with engine.connect() as conn:
        if engine.dialect.name == "sqlite":
            info["journal_mode"] = str(conn.exec_driver_sql("PRAGMA journal_mode").scalar())
            info["foreign_keys"] = str(conn.exec_driver_sql("PRAGMA foreign_keys").scalar())
        else:
            info["server_version"] = str(conn.execute(text("SHOW server_version")).scalar())
    return info
