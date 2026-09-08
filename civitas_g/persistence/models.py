"""What a run produced, persisted (§35-§46, kept).

Five tables, and the shape of them is decided by one requirement in A3's G1 row: the reading that
the gate compares against a reference summary must be recomputed **from these rows**, not from the
result object still sitting in memory. Otherwise "both backends" is vacuous -- the numbers come
from numpy either way, and running the same simulation twice against two databases proves nothing
about either (D12). Persisting a summary blob and diffing that would be the same mistake one level
up.

So `era_rows` stores the engine's log rows whole. The engine's reading functions take a list of
log dicts and nothing else, which means a faithful round-trip of those dicts is sufficient for the
entire reading to be recomputable from the database. Hot columns (`t`, `phase`, `chain`, `pop`)
are lifted out for indexing and ordering; `payload` is the row itself.

A1.4: nothing is hard-deleted. `Run.status` records a failed or abandoned run rather than removing
it, because a run that died is evidence about the world and an absent row is evidence about
nothing.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from civitas_g.persistence.base import Base, Timestamped, UUIDPrimaryKey
from civitas_g.persistence.types import GUID, JSONVariant, UTCDateTime


class Campaign(Base, UUIDPrimaryKey, Timestamped):
    """One campaign under one manifest.

    A4.3 wants a write-up per milestone stating the spec, the DECISIONs, the pre-check numbers and
    the gate numbers per seed. A campaign is the unit those are about: several arms, several
    seeds, one world, one engine hash.
    """

    __tablename__ = "g_campaigns"

    #: "reproduction", "precheck", "acceptance". Free text, because inventing an enum for a set
    #: that is still moving would make adding a kind a migration.
    kind: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    label: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    #: The full manifest (§46): engine hashes, world, references by hash, environment, notes.
    manifest: Mapped[dict[str, Any]] = mapped_column(JSONVariant(), nullable=False)
    phase_steps: Mapped[int | None] = mapped_column(Integer(), nullable=True)
    #: A1.8's clause, per campaign, so a reader never has to resolve it from a commit.
    engine_sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    runs: Mapped[list[Run]] = relationship(back_populates="campaign",
                                             cascade="all, delete-orphan")
    reproductions: Mapped[list[Reproduction]] = relationship(
        back_populates="campaign", cascade="all, delete-orphan")


class Run(Base, UUIDPrimaryKey, Timestamped):
    """One (arm, seed) run of the engine."""

    __tablename__ = "g_runs"
    __table_args__ = (
        UniqueConstraint("campaign_id", "arm", "seed", name="uq_g_run_campaign_arm_seed"),
        Index("ix_g_runs_arm_seed", "arm", "seed"),
    )

    campaign_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("g_campaigns.id", ondelete="CASCADE"), nullable=False, index=True)
    #: The manifest vocabulary (B§4), never the research name. One name in the database.
    arm: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    seed: Mapped[int] = mapped_column(Integer(), nullable=False)
    phase_steps: Mapped[int] = mapped_column(Integer(), nullable=False)

    #: "ok" | "failed" | "abandoned". Never deleted (A1.4).
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="ok", index=True)
    failure: Mapped[str | None] = mapped_column(Text(), nullable=True)

    #: `asdict(cfg)` -- every parameter a reader needs, with the data (the `sr_w` lesson).
    cfg: Mapped[dict[str, Any]] = mapped_column(JSONVariant(), nullable=False)
    #: The mapping the surviving genomes were last selected under. A replay that re-seeds the
    #: world does NOT get this mapping, so a genome test has to pin it.
    final_mapping: Mapped[list[int]] = mapped_column(JSONVariant(), nullable=False)
    chain_start: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    n_steps: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    phase_bounds: Mapped[list[Any]] = mapped_column(JSONVariant(), nullable=False, default=list)
    flips: Mapped[list[int]] = mapped_column(JSONVariant(), nullable=False, default=list)
    recipe_changes: Mapped[list[int]] = mapped_column(JSONVariant(), nullable=False, default=list)
    #: The population at the end, as `sim_v3_13.snapshot` produced it.
    final_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONVariant(), nullable=True)
    wall_seconds: Mapped[float] = mapped_column(Float(), nullable=False, default=0.0)
    engine_sha256: Mapped[str] = mapped_column(String(64), nullable=False)

    campaign: Mapped[Campaign] = relationship(back_populates="runs")
    era_rows: Mapped[list[EraRow]] = relationship(
        back_populates="run", cascade="all, delete-orphan",
        order_by="EraRow.ordinal")
    snapshots: Mapped[list[EraSnapshot]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="EraSnapshot.ordinal")


class EraRow(Base, UUIDPrimaryKey):
    """One row of the engine's log, stored whole.

    The engine appends one of these every `log_every` steps and clears its accumulators, which is
    what makes Civitas's contact with the world era-scoped rather than per agent-step (A1.8).

    `payload` is the row verbatim. Everything the reading needs is in there, so the reading is
    recomputable from the database alone -- which is the only sense in which "reproduced on both
    backends" says anything (D12).
    """

    __tablename__ = "g_era_rows"
    __table_args__ = (
        UniqueConstraint("run_id", "ordinal", name="uq_g_era_row_run_ordinal"),
        Index("ix_g_era_rows_run_t", "run_id", "t"),
    )

    run_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("g_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    #: Position in the log. Ordering by `t` alone is not enough: the two phases run back to back
    #: on one clock, but a caller that filtered by phase and re-sorted could lose the tie order.
    ordinal: Mapped[int] = mapped_column(Integer(), nullable=False)
    t: Mapped[int] = mapped_column(Integer(), nullable=False)
    phase: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    chain: Mapped[bool] = mapped_column(Boolean(), nullable=False, default=False)
    pop: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONVariant(), nullable=False)

    run: Mapped[Run] = relationship(back_populates="era_rows")


class EraSnapshot(Base, UUIDPrimaryKey):
    """Genomes at an era boundary, with the mapping just lived under.

    A2.1's frozen replay starts from one of these: births, deaths and injection disabled, energy
    pinned by identity, 300 steps, `eta_scale` 0 and 1. G1 does not run the assay -- that is G2's
    "reproduce through the store" -- but the engine produces these snapshots as part of a run, and
    discarding a run's output would make the run unauditable (A1.4).
    """

    __tablename__ = "g_era_snapshots"
    __table_args__ = (
        UniqueConstraint("run_id", "ordinal", name="uq_g_snapshot_run_ordinal"),
    )

    run_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("g_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    ordinal: Mapped[int] = mapped_column(Integer(), nullable=False)
    t: Mapped[int | None] = mapped_column(Integer(), nullable=True)
    mapping: Mapped[list[int] | None] = mapped_column(JSONVariant(), nullable=True)
    pi: Mapped[list[int] | None] = mapped_column(JSONVariant(), nullable=True)
    n_agents: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    #: Pickled genomes. Opaque here on purpose: their structure is the engine's, and a schema that
    #: mirrored it would be a second definition of the genome to keep in step.
    blob: Mapped[bytes] = mapped_column(LargeBinary(), nullable=False)
    blob_sha256: Mapped[str] = mapped_column(String(64), nullable=False)

    run: Mapped[Run] = relationship(back_populates="snapshots")


class Reproduction(Base, UUIDPrimaryKey, Timestamped):
    """One G1 gate result: a reading recomputed from the rows, against a reference summary.

    `diff` holds every compared field, not only the failing ones. A gate that recorded only its
    failures would make a pass unfalsifiable -- there would be no way to tell "matched on 240
    fields" from "compared nothing".
    """

    __tablename__ = "g_reproductions"
    __table_args__ = (
        Index("ix_g_repro_reference_backend", "reference_key", "backend"),
    )

    campaign_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("g_campaigns.id", ondelete="CASCADE"), nullable=False, index=True)
    #: Which pinned reference (`civitas_g.manifest.REFERENCES`).
    reference_key: Mapped[str] = mapped_column(String(40), nullable=False)
    reference_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    #: "sqlite" | "postgresql". The reading was recomputed from rows on this backend.
    backend: Mapped[str] = mapped_column(String(20), nullable=False)
    tolerance: Mapped[float] = mapped_column(Float(), nullable=False)
    n_fields: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    n_matched: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    n_missing: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    passed: Mapped[bool] = mapped_column(Boolean(), nullable=False, default=False)
    #: Set when the gate could not be evaluated at all. A withheld number is never zero (§A5).
    withheld_reason: Mapped[str | None] = mapped_column(Text(), nullable=True)
    diff: Mapped[list[dict[str, Any]]] = mapped_column(JSONVariant(), nullable=False,
                                                       default=list)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)

    campaign: Mapped[Campaign] = relationship(back_populates="reproductions")


__all__ = ["Base", "Campaign", "Run", "EraRow", "EraSnapshot", "Reproduction"]
