"""Model providers, configurations and calls (Part B §7, §19, §20)."""

from __future__ import annotations

import uuid

from sqlalchemy import Boolean, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from civitas.persistence.base import Base, Metadataed, SoftDeletable, Timestamped, UUIDPrimaryKey
from civitas.persistence.types import GUID, JSONVariant


class ModelProvider(Base, UUIDPrimaryKey, Timestamped, Metadataed, SoftDeletable):
    """A registered backend (Part B §19). Credentials are **not** here — they resolve from
    `Settings` at call time and never touch the database (§40)."""

    __tablename__ = "model_providers"
    __table_args__ = (UniqueConstraint("name", name="uq_provider_name"),)

    name: Mapped[str] = mapped_column(String(64), nullable=False)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    base_url: Mapped[str | None] = mapped_column(String(500), default=None)
    is_local: Mapped[bool] = mapped_column(Boolean(), nullable=False, default=False)
    supports_tools: Mapped[bool] = mapped_column(Boolean(), nullable=False, default=True)
    supports_streaming: Mapped[bool] = mapped_column(Boolean(), nullable=False, default=True)
    supports_structured_output: Mapped[bool] = mapped_column(
        Boolean(), nullable=False, default=True
    )
    is_deterministic: Mapped[bool] = mapped_column(Boolean(), nullable=False, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean(), nullable=False, default=True)


class ModelConfiguration(Base, UUIDPrimaryKey, Timestamped, Metadataed, SoftDeletable):
    """A frozen individual-agent configuration (Part B §20).

    Frozen-model mode pins one of these for a whole campaign, so the only thing that varies across
    a run is the persistent collective environment. `config_hash` is what makes "identical
    configuration" checkable rather than asserted.
    """

    __tablename__ = "model_configurations"
    __table_args__ = (UniqueConstraint("name", name="uq_modelconfig_name"),)

    name: Mapped[str] = mapped_column(String(200), nullable=False)
    provider_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(200), nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    temperature: Mapped[float] = mapped_column(Float(), nullable=False, default=0.0)
    top_p: Mapped[float] = mapped_column(Float(), nullable=False, default=1.0)
    max_tokens: Mapped[int] = mapped_column(Integer(), nullable=False, default=4096)
    seed: Mapped[int | None] = mapped_column(Integer(), default=None)
    extra_params: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)
    cost_per_1k_prompt_usd: Mapped[float] = mapped_column(Float(), nullable=False, default=0.0)
    cost_per_1k_completion_usd: Mapped[float] = mapped_column(Float(), nullable=False, default=0.0)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="", index=True)
    is_frozen: Mapped[bool] = mapped_column(Boolean(), nullable=False, default=False)


class ModelCall(Base, UUIDPrimaryKey, Timestamped, Metadataed):
    """One provider request (Part B §19, §53).

    Prompts are **not** stored by default: an episode's reasoning is temporary individual state
    (Part B §4), and persisting it would create exactly the hidden channel §4 forbids. What is
    stored is the accounting — tokens, latency, cost, errors — plus a hash of the request, so a
    call is identifiable and repeatable without its content becoming inheritable.
    """

    __tablename__ = "model_calls"
    __table_args__ = (Index("ix_modelcall_episode_created", "episode_id", "created_at"),)

    episode_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("episodes.id", ondelete="CASCADE"), default=None, index=True
    )
    provider: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(200), nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="", index=True)
    prompt_tokens: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    latency_ms: Mapped[float] = mapped_column(Float(), nullable=False, default=0.0)
    cost_usd: Mapped[float] = mapped_column(Float(), nullable=False, default=0.0)
    stop_reason: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    tool_calls: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    error: Mapped[str] = mapped_column(Text(), nullable=False, default="")
    retries: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
