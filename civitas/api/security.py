"""Authentication and authorization (Part B §51, §52).

Built in from the start rather than added at hardening (Part A §A1.7). Three principals — users,
service identities and agents — and five roles, with least privilege enforced by a dependency
rather than by each endpoint remembering to check.

API keys are stored **hashed**. The plaintext is returned once at creation and never again; there
is no code path that reads one back, which is what makes a database dump not a credential leak.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from dataclasses import dataclass
from datetime import timedelta

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from civitas.domain.enums import ActorKind, Role
from civitas.persistence.models import ApiKey, Organization, ServiceIdentity, User
from civitas.persistence.types import utcnow

KEY_PREFIX = "civ"
PREFIX_LENGTH = 12

#: Role ordering for `require_role`. A role satisfies a requirement when it is at least as
#: privileged. `AGENT` is deliberately **not** on this ladder: an agent identity is not a weak
#: operator, it is a different kind of principal, and putting it on the ladder would let an agent
#: inherit a human's permissions by being "above viewer".
ROLE_RANK: dict[Role, int] = {
    Role.VIEWER: 1,
    Role.RESEARCHER: 2,
    Role.OPERATOR: 3,
    Role.ADMIN: 4,
}

#: What an agent principal may do, listed explicitly. Least privilege (§51): an agent can read and
#: write knowledge, and cannot touch organizations, users, keys or experiment approval.
AGENT_SCOPES = frozenset({
    "artifacts:read", "artifacts:write", "retrieval:read", "tools:read", "tools:write",
    "events:read", "tasks:read",
})


@dataclass(frozen=True)
class Principal:
    """Who is making a request."""

    kind: ActorKind
    id: uuid.UUID
    organization_id: uuid.UUID
    role: Role
    scopes: frozenset[str]
    display: str = ""

    def has_scope(self, scope: str) -> bool:
        if self.kind is ActorKind.AGENT:
            return scope in self.scopes
        return not self.scopes or scope in self.scopes


def generate_api_key() -> tuple[str, str, str]:
    """Return `(plaintext, prefix, hash)`.

    The prefix is stored in clear so a key can be *identified* in a listing and revoked without
    the plaintext ever being needed. The rest is hashed.
    """
    secret = secrets.token_urlsafe(32)
    plaintext = f"{KEY_PREFIX}_{secret}"
    return plaintext, plaintext[:PREFIX_LENGTH], hash_api_key(plaintext)


def hash_api_key(plaintext: str) -> str:
    """SHA-256 over the key.

    Not a password hash: an API key is 256 bits of entropy from a CSPRNG, so it is not
    brute-forceable and a slow KDF would only add latency to every request. A user-chosen
    *password* is a different problem and uses a different function.
    """
    return hashlib.sha256(plaintext.encode()).hexdigest()


def verify_api_key(session: Session, plaintext: str) -> ApiKey | None:
    """Look a key up by prefix and compare in constant time.

    Constant-time comparison because a byte-by-byte compare over a stored hash leaks how much of a
    guess was correct (§52).
    """
    if not plaintext or not plaintext.startswith(KEY_PREFIX):
        return None
    prefix = plaintext[:PREFIX_LENGTH]
    candidates = session.execute(
        select(ApiKey).where(ApiKey.prefix == prefix, ApiKey.revoked_at.is_(None))
    ).scalars()

    digest = hash_api_key(plaintext)
    for candidate in candidates:
        if not hmac.compare_digest(candidate.key_hash, digest):
            continue
        if candidate.expires_at is not None and candidate.expires_at < utcnow():
            return None
        candidate.last_used_at = utcnow()
        return candidate
    return None


def create_api_key(
    session: Session,
    *,
    organization_id: uuid.UUID,
    name: str,
    user_id: uuid.UUID | None = None,
    service_identity_id: uuid.UUID | None = None,
    scopes: list[str] | None = None,
    ttl_days: int | None = None,
) -> tuple[ApiKey, str]:
    """Create a key. The plaintext is returned **once** and is never recoverable."""
    if (user_id is None) == (service_identity_id is None):
        raise ValueError("an API key belongs to exactly one of a user or a service identity")

    plaintext, prefix, digest = generate_api_key()
    key = ApiKey(
        organization_id=organization_id, user_id=user_id,
        service_identity_id=service_identity_id, name=name, prefix=prefix, key_hash=digest,
        scopes=scopes or [],
        expires_at=utcnow() + timedelta(days=ttl_days) if ttl_days else None,
    )
    session.add(key)
    session.flush()
    return key, plaintext


def principal_for(session: Session, key: ApiKey) -> Principal:
    if key.user_id is not None:
        user = session.get(User, key.user_id)
        if user is None or not user.is_active:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "the key's user is inactive")
        return Principal(
            kind=ActorKind.USER, id=user.id, organization_id=user.organization_id,
            role=user.role, scopes=frozenset(key.scopes or []), display=user.email,
        )
    identity = session.get(ServiceIdentity, key.service_identity_id)
    if identity is None or not identity.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "the key's identity is inactive")
    scopes = frozenset(key.scopes or []) or (
        AGENT_SCOPES if identity.kind is ActorKind.AGENT else frozenset()
    )
    return Principal(
        kind=identity.kind, id=identity.id, organization_id=identity.organization_id,
        role=identity.role, scopes=scopes, display=identity.name,
    )


# --------------------------------------------------------------------------
# FastAPI dependencies
# --------------------------------------------------------------------------
def get_principal(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> Principal:  # pragma: no cover - replaced by the app's dependency_overrides in tests
    raise NotImplementedError("bound in civitas.api.app")


def require_role(minimum: Role):
    """Least privilege (§51). An agent principal never satisfies a human role requirement."""

    def dependency(principal: Principal = Depends(get_principal)) -> Principal:
        if principal.role not in ROLE_RANK:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"role {principal.role.value!r} cannot satisfy a {minimum.value} requirement",
            )
        if ROLE_RANK[principal.role] < ROLE_RANK[minimum]:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"this endpoint requires {minimum.value}; you have {principal.role.value}",
            )
        return principal

    return dependency


def require_scope(scope: str):
    def dependency(principal: Principal = Depends(get_principal)) -> Principal:
        if not principal.has_scope(scope):
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"scope {scope!r} is required")
        return principal

    return dependency


def bootstrap_organization(
    session: Session, *, name: str, slug: str, admin_email: str
) -> tuple[Organization, User, str]:
    """Create an organization with one admin and one key — the only unauthenticated path.

    Deliberately not an endpoint: it is a CLI and notebook operation. An HTTP route that can
    create an admin without authentication is a route that can create an admin without
    authentication.
    """
    organization = Organization(name=name, slug=slug)
    session.add(organization)
    session.flush()
    admin = User(
        organization_id=organization.id, email=admin_email, display_name=admin_email,
        role=Role.ADMIN,
    )
    session.add(admin)
    session.flush()
    _key, plaintext = create_api_key(
        session, organization_id=organization.id, name="bootstrap admin key", user_id=admin.id
    )
    return organization, admin, plaintext
