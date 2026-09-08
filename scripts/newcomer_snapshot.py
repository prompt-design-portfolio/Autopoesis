"""Run the newcomer benchmark under one fixed configuration and print its metrics as JSON.

Used to check that a refactor of the measuring instrument leaves the instrument reading the same:
run it on the tree before the change and on the tree after, and diff. A refactor of a benchmark
that is not checked this way is a refactor of an unknown quantity.
"""
from __future__ import annotations

import json
import sys
import uuid

from sqlalchemy.orm import sessionmaker

from civitas.config import Settings
from civitas.experiments.benchmark import run_newcomer_benchmark
from civitas.persistence.engine import create_db_engine
from civitas.persistence.models import Base, Organization
from civitas.persistence.session import install_guards

url = sys.argv[1] if len(sys.argv) > 1 else "sqlite:////var/tmp/newcomer-snapshot.db"
settings = Settings(database_url=url, environment_version="test-env-1",
                    jwt_secret="not-a-real-key")
engine = create_db_engine(settings)
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)
factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
install_guards(factory)
session = factory()

org = Organization(name="snapshot", slug=f"snap-{uuid.uuid4().hex[:8]}")
session.add(org)
session.commit()

result = run_newcomer_benchmark(
    session, organization_id=org.id, seeds=[0, 1, 2, 3, 4],
    accumulation_passes=3, settings=settings,
    retrieval_options={"version": "retrieval/1.0-lexical", "semantic": False},
)
session.commit()
print(json.dumps(result.metrics(), indent=1, sort_keys=True, default=str))
