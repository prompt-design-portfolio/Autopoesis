"""Configuration and secrets (Part B §40, §35, §36).

One settings object for every deployment target. Colab-native mode and production differ only in
the values here, never in application logic (Part B §6, §33).

Secrets are read from the environment, from a `.env` file, or from Colab's secret store, and are
held in `SecretStr` so they cannot be printed by accident. `Settings.manifest_dict()` is the only
sanctioned way to put configuration into an experiment manifest, and it excludes every secret by
construction (§40, §46).
"""

from __future__ import annotations

import hashlib
import json
import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Runtime(str, Enum):
    """Where the process is running. Detected, not configured, unless overridden."""

    COLAB = "colab"
    LOCAL = "local"
    CONTAINER = "container"
    CLOUD = "cloud"


class SandboxBackend(str, Enum):
    """Tool-execution isolation (Part B §18).

    `SUBPROCESS` is the Colab fallback and is materially weaker than the others. It is recorded in
    the manifest so a result run under it is never mistaken for one run under isolation.
    """

    DOCKER = "docker"
    GVISOR = "gvisor"
    SUBPROCESS = "subprocess"
    NONE = "none"


def detect_runtime() -> Runtime:
    if "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ:
        return Runtime.COLAB
    try:
        import google.colab  # noqa: F401

        return Runtime.COLAB
    except Exception:
        pass
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return Runtime.CLOUD
    if Path("/.dockerenv").exists():
        return Runtime.CONTAINER
    return Runtime.LOCAL


def _colab_secret(name: str) -> str | None:
    """Read a Colab userdata secret. Returns None off Colab or when unset (Part B §40)."""
    try:
        from google.colab import userdata  # type: ignore

        return userdata.get(name)
    except Exception:
        return None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CIVITAS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        secrets_dir=None,
    )

    # --- identity -------------------------------------------------------
    app_version: str = "0.1.0"
    environment_version: str = Field(
        default="dev",
        description="Opaque marker for the world an artifact was produced against (Part B §15). "
        "Artifacts record it so staleness is decidable.",
    )

    # --- storage --------------------------------------------------------
    database_url: str = ""
    data_dir: Path = Field(default=Path("./civitas_data"))
    object_store_url: str = ""

    # --- runtime --------------------------------------------------------
    runtime: Runtime | None = None
    sandbox_backend: SandboxBackend = SandboxBackend.SUBPROCESS
    worker_concurrency: int = 2

    # --- api ------------------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    auth_enabled: bool = True
    jwt_secret: SecretStr = SecretStr("")
    jwt_ttl_seconds: int = 3600
    rate_limit_per_minute: int = 600
    cors_origins: list[str] = Field(default_factory=list)

    # --- providers (Part B §19, §40) ------------------------------------
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    default_provider: str = "deterministic"
    default_model: str = "deterministic-v1"

    # --- observability --------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = True
    otel_endpoint: str | None = None
    metrics_enabled: bool = True

    @field_validator("data_dir", mode="before")
    @classmethod
    def _expand(cls, v: Any) -> Any:
        return Path(os.path.expanduser(str(v))) if v else v

    def model_post_init(self, _ctx: Any) -> None:
        if self.runtime is None:
            object.__setattr__(self, "runtime", detect_runtime())
        # Colab secrets are consulted only for keys not already supplied by the environment, so an
        # explicit env var always wins and the resolution order is stable across runtimes.
        for field, secret_name in (
            ("anthropic_api_key", "ANTHROPIC_API_KEY"),
            ("openai_api_key", "OPENAI_API_KEY"),
            ("google_api_key", "GOOGLE_API_KEY"),
        ):
            if getattr(self, field) is None:
                got = _colab_secret(secret_name)
                if got:
                    object.__setattr__(self, field, SecretStr(got))
        if not self.database_url:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            object.__setattr__(
                self, "database_url", f"sqlite:///{(self.data_dir / 'civitas.db').resolve()}"
            )

    # --- derived --------------------------------------------------------
    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith(("postgresql", "postgres://"))

    def manifest_dict(self) -> dict[str, Any]:
        """Configuration for an experiment manifest, with every secret excluded (§40, §46).

        Excludes by *type* — any `SecretStr` field is dropped whatever it is called — so a secret
        added later cannot leak by being named something this function does not recognise.
        """
        out: dict[str, Any] = {}
        for name, field in type(self).model_fields.items():
            value = getattr(self, name)
            if isinstance(value, SecretStr):
                continue
            ann = field.annotation
            if ann is not None and "SecretStr" in str(ann):
                continue
            if name == "database_url":
                # The URL can carry a password. Record the dialect only.
                out[name] = value.split("://", 1)[0] if value else ""
                continue
            out[name] = str(value) if isinstance(value, Path | Enum) else value
        return out

    def config_hash(self) -> str:
        """Stable hash of the non-secret configuration (Part B §8, §46).

        Two runs with the same hash had the same configuration. This is what makes the A2.2
        matched A/B comparison checkable rather than asserted.
        """
        blob = json.dumps(self.manifest_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Tests and the Colab notebook reconfigure the process in place."""
    get_settings.cache_clear()
