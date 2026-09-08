"""Alembic environment (Part B §43).

The URL comes from `Settings`, never from `alembic.ini`, so migrations run against whichever
backend the process is configured for and Colab needs no separate configuration file.
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import pool

from civitas.config import get_settings
from civitas.persistence.engine import create_db_engine
from civitas.persistence.models import Base

config = context.config
target_metadata = Base.metadata


def _render_item(type_, obj, autogen_context):
    """Render `EnumType` as the VARCHAR it actually is.

    The enum coercion is application-side; the DDL type is a plain string column, and Alembic
    cannot reproduce the Python enum class in a migration script. Rendering the true DDL type
    keeps migrations self-contained and makes the schema identical whether it was built by
    `create_all` or by a migration — which is what the migration test checks.
    """
    from civitas.persistence.types import EnumType

    if type_ == "type" and isinstance(obj, EnumType):
        autogen_context.imports.add("import sqlalchemy as sa")
        return f"sa.String(length={obj.impl_instance.length})"
    return False


def _url() -> str:
    return config.get_main_option("sqlalchemy.url") or get_settings().database_url


def run_migrations_offline() -> None:
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        render_item=_render_item,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_db_engine(url=_url())
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # SQLite cannot ALTER most things in place. Batch mode rebuilds the table around the
            # change, so one migration script is valid on both backends (Part B §6).
            render_as_batch=True,
            render_item=_render_item,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
