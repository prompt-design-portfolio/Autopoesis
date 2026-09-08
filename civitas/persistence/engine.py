"""Engine and session construction — the same application logic on both backends (Part B §6, §43).

SQLite needs three things PostgreSQL gives for free, and all three are set here rather than being
remembered at call sites: WAL journalling so a reader does not block the writer (§35), foreign-key
enforcement, which SQLite has off by default and which would silently make referential integrity a
comment, and a busy timeout so concurrent workers wait instead of raising `database is locked`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from civitas.config import Settings, get_settings

log = logging.getLogger(__name__)

_ENGINES: dict[str, Engine] = {}
_SESSION_FACTORIES: dict[str, sessionmaker[Session]] = {}


def _configure_sqlite(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn, _record):  # pragma: no cover - driver callback
        cur = dbapi_conn.cursor()
        # WAL: readers do not block the writer. Required for Colab, where the API, the workers and
        # the notebook all hold the same file open (Part B §35, §37).
        cur.execute("PRAGMA journal_mode=WAL")
        # SQLite ships with foreign keys OFF. Without this, every ondelete clause in the models is
        # decoration and referential integrity (Part B §7) is not enforced on the Colab backend.
        cur.execute("PRAGMA foreign_keys=ON")
        # Wait rather than raise when another worker holds the write lock.
        cur.execute("PRAGMA busy_timeout=10000")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()

    @event.listens_for(engine, "begin")
    def _on_begin(conn):  # pragma: no cover - driver callback
        # SQLAlchemy's default deferred BEGIN takes the write lock only at first write, which
        # turns the job-lease read-then-update into a race that SQLite resolves by raising
        # SQLITE_BUSY on the *second* writer mid-transaction. An immediate BEGIN takes the lock up
        # front, so leasing serialises the same way SELECT FOR UPDATE does on PostgreSQL (§42).
        conn.exec_driver_sql("BEGIN IMMEDIATE")


def create_db_engine(settings: Settings | None = None, url: str | None = None) -> Engine:
    settings = settings or get_settings()
    url = url or settings.database_url

    if url.startswith("sqlite"):
        path = url.split("///", 1)[-1]
        if path and path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict = {"connect_args": {"check_same_thread": False}}
        if ":memory:" in url:
            # An in-memory database is per-connection; a pool would give each caller a different,
            # empty database. StaticPool makes the whole process share one.
            kwargs["poolclass"] = StaticPool
        engine = create_engine(url, future=True, **kwargs)
        _configure_sqlite(engine)
        return engine

    return create_engine(
        url,
        future=True,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=1800,
    )


def get_engine(settings: Settings | None = None) -> Engine:
    settings = settings or get_settings()
    if settings.database_url not in _ENGINES:
        _ENGINES[settings.database_url] = create_db_engine(settings)
    return _ENGINES[settings.database_url]


def get_session_factory(settings: Settings | None = None) -> sessionmaker[Session]:
    settings = settings or get_settings()
    key = settings.database_url
    if key not in _SESSION_FACTORIES:
        from civitas.persistence.session import install_guards

        engine = get_engine(settings)
        factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
        install_guards(factory)
        _SESSION_FACTORIES[key] = factory
    return _SESSION_FACTORIES[key]


@contextmanager
def session_scope(settings: Settings | None = None) -> Iterator[Session]:
    """Transaction boundary. Commits on success, rolls back on any exception."""
    factory = get_session_factory(settings)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def dispose_engines() -> None:
    """Drop cached engines. Colab reconnects and tests both reconfigure the process in place."""
    for engine in _ENGINES.values():
        engine.dispose()
    _ENGINES.clear()
    _SESSION_FACTORIES.clear()


def healthcheck(settings: Settings | None = None) -> dict:
    """Liveness for `/healthz` and the Colab notebook (Part B §34, §54)."""
    settings = settings or get_settings()
    engine = get_engine(settings)
    info: dict = {"dialect": engine.dialect.name, "ok": False}
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
        info["ok"] = True
        if engine.dialect.name == "sqlite":
            info["journal_mode"] = conn.exec_driver_sql("PRAGMA journal_mode").scalar()
            info["foreign_keys"] = bool(conn.exec_driver_sql("PRAGMA foreign_keys").scalar())
    return info
