"""Run the code-repair newcomer benchmark and print its gated metrics as JSON."""
from __future__ import annotations

import json
import sys
import uuid

from sqlalchemy.orm import sessionmaker

from civitas.config import Settings
from civitas.legacy.benchmark import run_repair_benchmark
from civitas.persistence.engine import create_db_engine
from civitas.persistence.models import Base, Organization
from civitas.persistence.session import install_guards

url = sys.argv[1] if len(sys.argv) > 1 else "sqlite:////var/tmp/repair-snapshot.db"
seeds = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "0,1,2").split(",")]
passes = int(sys.argv[3]) if len(sys.argv) > 3 else 2

settings = Settings(database_url=url, environment_version="test-env-1",
                    jwt_secret="not-a-real-key")
engine = create_db_engine(settings)
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)
factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
install_guards(factory)
session = factory()

org = Organization(name="repair snapshot", slug=f"rs-{uuid.uuid4().hex[:8]}")
session.add(org)
session.commit()

result = run_repair_benchmark(
    session, organization_id=org.id, seeds=seeds, accumulation_passes=passes,
    settings=settings,
    retrieval_options={"version": "retrieval/1.0-lexical", "semantic": False},
)
session.commit()
try:
    print(json.dumps(result.metrics(), indent=1, sort_keys=True, default=str))
except Exception as exc:
    print("GATES FAILED:", exc)
    print(json.dumps(result.raw_metrics(), indent=1, sort_keys=True, default=str))
    sys.exit(1)
