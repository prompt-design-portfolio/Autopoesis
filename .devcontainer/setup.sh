#!/usr/bin/env bash
# Codespaces setup. Runs once, after the container is created.
#
# Brings up the *strongest* configuration this project has: PostgreSQL alongside SQLite so the
# two-backend parametrisation actually runs, Docker available so the sandbox can use container
# isolation, and a non-root user so `RLIMIT_NPROC` is enforced. None of those three hold in Colab.
set -euo pipefail

echo "--- installing civitas"
pip install --upgrade pip
pip install -e ".[dev,postgres,ui-tests]"

echo "--- starting PostgreSQL for the two-backend test run"
# A container rather than an apt install: it starts clean, it is the same version CI uses, and
# `docker rm` is a complete reset when a test leaves the cluster in a bad state.
docker run -d --name civitas-pg -p 5432:5432 \
  -e POSTGRES_USER=civitas -e POSTGRES_PASSWORD=civitas -e POSTGRES_DB=civitas_test \
  postgres:16-alpine >/dev/null 2>&1 || echo "    (already running)"

for _ in $(seq 1 40); do
  if docker exec civitas-pg pg_isready -U civitas -d civitas_test >/dev/null 2>&1; then break; fi
  sleep 1
done

echo "--- applying migrations"
mkdir -p /workspaces/civitas-data
alembic upgrade head

echo "--- what is actually in force here"
python -m civitas.cli preflight || true

cat <<'BANNER'

  Civitas is ready.

    pytest -q                      the full suite, both backends
    civitas acceptance --levels 1,2,3   the §65 levels (all six takes ~1h)
    civitas preflight              §64's production checklist
    civitas bootstrap --name Org --slug org --email you@example.com
    civitas serve --host 0.0.0.0   API + UI; open the forwarded port 8000

  The sandbox backend and the limits it cannot enforce are printed above. Every result carries
  them in its manifest, so a run here is never mistaken for one under weaker isolation.

BANNER
