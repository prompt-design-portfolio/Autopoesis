# Civitas (Part B §57).
#
# Two stages so the runtime image carries no build toolchain, and a non-root user because the
# sandbox's process limits are *not enforced for uid 0* — measured in M4, reported by
# `Sandbox.unenforced_limits()`, and the single strongest reason this image does not run as root.

FROM python:3.11-slim AS build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /build
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git \
 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY civitas ./civitas
COPY alembic ./alembic
COPY alembic.ini ./
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install ".[postgres]"

FROM python:3.11-slim AS runtime
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CIVITAS_LOG_JSON=true \
    CIVITAS_DATA_DIR=/var/lib/civitas
RUN apt-get update \
 && apt-get install -y --no-install-recommends libpq5 curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --create-home --uid 10001 civitas \
 && mkdir -p /var/lib/civitas && chown civitas:civitas /var/lib/civitas
COPY --from=build /opt/venv /opt/venv
COPY --chown=civitas:civitas civitas /app/civitas
COPY --chown=civitas:civitas alembic /app/alembic
COPY --chown=civitas:civitas alembic.ini /app/alembic.ini
WORKDIR /app
USER civitas
EXPOSE 8000

# `/readyz`, not `/healthz`: readiness reports whether the database is reachable, and a container
# that answers "the process is up" while its database is gone is a container that keeps taking
# traffic it cannot serve.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/readyz || exit 1

CMD ["civitas", "serve", "--host", "0.0.0.0", "--port", "8000"]
