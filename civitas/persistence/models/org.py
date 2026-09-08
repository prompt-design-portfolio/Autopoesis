"""Organizations, identities, workspaces and projects (Part B §7, §51)."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from civitas.domain.enums import ActorKind, Role
from civitas.persistence.base import Base, Metadataed, SoftDeletable, Timestamped, UUIDPrimaryKey
from civitas.persistence.types import EnumType, GUID, JSONVariant, UTCDateTime


class Organization(Base, UUIDPrimaryKey, Timestamped, Metadataed, SoftDeletable):
    __tablename__ = "organizations"

    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)

    users: Mapped[list[User]] = relationship(back_populates="organization")
    workspaces: Mapped[list[Workspace]] = relationship(back_populates="organization")


class User(Base, UUIDPrimaryKey, Timestamped, Metadataed, SoftDeletable):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("organization_id", "email", name="uq_user_org_email"),)

    organization_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("organizations.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    display_name: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    role: Mapped[Role] = mapped_column(EnumType(Role, 32), nullable=False, default=Role.VIEWER)
    password_hash: Mapped[str | None] = mapped_column(Text(), default=None)
    is_active: Mapped[bool] = mapped_column(Boolean(), nullable=False, default=True)

    organization: Mapped[Organization] = relationship(back_populates="users")


class ServiceIdentity(Base, UUIDPrimaryKey, Timestamped, Metadataed, SoftDeletable):
    """Non-human principals — workers, agents, CI (Part B §51).

    Separate from `User` because least privilege needs the distinction: an agent identity must
    never be able to hold a human's role, and a shared table makes that a runtime check rather
    than a structural one.
    """

    __tablename__ = "service_identities"
    __table_args__ = (UniqueConstraint("organization_id", "name", name="uq_service_org_name"),)

    organization_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("organizations.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    kind: Mapped[ActorKind] = mapped_column(EnumType(ActorKind, 32), nullable=False, default=ActorKind.SERVICE)
    role: Mapped[Role] = mapped_column(EnumType(Role, 32), nullable=False, default=Role.AGENT)
    is_active: Mapped[bool] = mapped_column(Boolean(), nullable=False, default=True)


class ApiKey(Base, UUIDPrimaryKey, Timestamped, Metadataed):
    """Scoped API keys (Part B §51).

    Only the hash is stored. The plaintext is returned once, at creation, and never again — there
    is no code path that can read a key back out, which is what makes a database dump not a
    credential leak (§40, §52).
    """

    __tablename__ = "api_keys"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("users.id", ondelete="CASCADE"), default=None, index=True
    )
    service_identity_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("service_identities.id", ondelete="CASCADE"), default=None, index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    prefix: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    key_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    scopes: Mapped[list] = mapped_column(JSONVariant(), default=list, nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), default=None)
    revoked_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), default=None)
    last_used_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), default=None)


class Workspace(Base, UUIDPrimaryKey, Timestamped, Metadataed, SoftDeletable):
    """A persistent collective environment (Part B §4, §49).

    The workspace is the unit the experimental arms operate on: `memory_reset` resets a workspace,
    `collective_frozen` pins one to a snapshot. Everything an agent can inherit lives under one.
    """

    __tablename__ = "workspaces"
    __table_args__ = (UniqueConstraint("organization_id", "slug", name="uq_workspace_org_slug"),)

    organization_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("organizations.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(Text(), nullable=False, default="")
    environment_version: Mapped[str] = mapped_column(String(100), nullable=False, default="dev")

    organization: Mapped[Organization] = relationship(back_populates="workspaces")
    projects: Mapped[list[Project]] = relationship(back_populates="workspace")


class Project(Base, UUIDPrimaryKey, Timestamped, Metadataed, SoftDeletable):
    """A long-running goal, surviving thousands of episodes (Part B §32)."""

    __tablename__ = "projects"

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text(), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    priority: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)

    workspace: Mapped[Workspace] = relationship(back_populates="projects")
    objectives: Mapped[list[Objective]] = relationship(back_populates="project")


class Objective(Base, UUIDPrimaryKey, Timestamped, Metadataed, SoftDeletable):
    """What the project is trying to establish. Tasks hang off objectives (Part B §27)."""

    __tablename__ = "objectives"

    project_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str] = mapped_column(Text(), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="open")
    priority: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    success_criteria: Mapped[dict] = mapped_column(JSONVariant(), default=dict, nullable=False)

    project: Mapped[Project] = relationship(back_populates="objectives")


class WorkspaceSnapshot(Base, UUIDPrimaryKey, Timestamped, Metadataed):
    """A frozen view of collective state (Part B §7, §21 `collective_frozen`, §46).

    Stores the *cut*, not a copy: `as_of` plus the artifact-id set that existed at that moment.
    Copying would double the storage and, worse, would let the copy drift from the originals it
    claims to be. Retrieval under `collective_frozen` filters live artifacts through the cut.
    """

    __tablename__ = "workspace_snapshots"

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    label: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    as_of: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, index=True)
    artifact_count: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    tool_count: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    episode_count: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
