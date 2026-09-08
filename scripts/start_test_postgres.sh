#!/usr/bin/env bash
# Start a throwaway PostgreSQL for the two-backend test run (docs/ARCHITECTURE.md §7).
set -euo pipefail
PGROOT="${1:-/tmp/civitas-pg}"
PGBIN=/usr/lib/postgresql/16/bin
mkdir -p "$PGROOT/data" "$PGROOT/run"
chown -R postgres:postgres "$PGROOT"
chmod 700 "$PGROOT/data"
[ -f "$PGROOT/data/PG_VERSION" ] || su postgres -s /bin/bash -c "$PGBIN/initdb -D $PGROOT/data -U civitas --auth=trust"
su postgres -s /bin/bash -c "$PGBIN/pg_ctl -D $PGROOT/data -o '-k $PGROOT/run -p 55432 -c listen_addresses=127.0.0.1' -l $PGROOT/pg.log start" || true
sleep 2
psql -h 127.0.0.1 -p 55432 -U civitas -d postgres -tc "SELECT 1 FROM pg_database WHERE datname='civitas_test'" | grep -q 1 \
  || psql -h 127.0.0.1 -p 55432 -U civitas -d postgres -c "CREATE DATABASE civitas_test"
echo "export CIVITAS_TEST_POSTGRES_URL=postgresql+psycopg://civitas@127.0.0.1:55432/civitas_test"
