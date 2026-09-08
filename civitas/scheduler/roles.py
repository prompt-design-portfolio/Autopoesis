"""Deliberate disagreement (Part B §28) and subproblem decomposition (Part B §27).

> Do not let consensus form solely because many models copy the first answer.

The six roles §28 names are not prompt personas. Each is an `AgentProfile` whose *tool policy* and
*retrieval policy* differ, so what a critic can see and do differs from what an explorer can — and
the difference is enforced by the runtime, not requested in a system prompt a model may ignore.

The sharpest of these is the **replicator**, whose retrieval deliberately withholds the conclusion
it is asked to reproduce. An agent shown the answer it is meant to independently confirm is not
replicating; it is agreeing.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from civitas.domain.enums import AgentRole, ArtifactType, HypothesisState
from civitas.persistence.models import AgentProfile, Artifact, Subproblem


@dataclass(frozen=True)
class RoleSpec:
    """A role, as a runtime configuration rather than a persona.

    `name` and `role` are separate because §28's six roles do not map one-to-one onto §26's
    profile taxonomy: "replication agent" and "adversarial agent" are both experimenters by
    category but are pointed at different work and see different things. The name is what the
    workspace and the scheduler use; the enum is the broad category performance is aggregated
    under.
    """

    name: str
    role: AgentRole
    description: str
    #: Tools the role may use. `None` means all of them.
    allowed_tools: tuple[str, ...] | None = None
    #: Retrieval parameters. This is where a role's *epistemic position* lives.
    retrieval_options: dict[str, Any] = None  # type: ignore[assignment]

    def to_profile(self, workspace_id: uuid.UUID) -> AgentProfile:
        return AgentProfile(
            workspace_id=workspace_id,
            name=self.name,
            role=self.role,
            description=self.description,
            tool_policy={"allowed": list(self.allowed_tools)} if self.allowed_tools else {},
            meta={"retrieval_options": self.retrieval_options or {}},
        )


#: The six roles §28 requires, plus explorer and toolmaker as the ordinary producers.
ROLE_SPECS: tuple[RoleSpec, ...] = (
    RoleSpec(
        name="explorer",
        role=AgentRole.EXPLORER,
        description="Investigates open questions and records what it establishes.",
        retrieval_options={},
    ),
    RoleSpec(
        name="critic",
        role=AgentRole.CRITIC,
        description=(
            "Looks for what is wrong with existing claims. Reads the record adversarially and "
            "records contradictions rather than new conclusions."
        ),
        allowed_tools=(
            "search_knowledge", "read_artifact", "create_artifact", "link_artifacts",
            "record_failure", "submit_result",
        ),
        #: Weighted toward *weak* claims: a critic that is shown the best-supported artifacts
        #: first is being pointed away from the ones most likely to be wrong.
        retrieval_options={"weights": {"validation": -0.35, "provenance": -0.25,
                                       "confidence": -0.20, "negative_relevance": 0.8}},
    ),
    RoleSpec(
        name="verifier",
        role=AgentRole.VERIFIER,
        description="Checks whether a specific claim holds, using tools rather than argument.",
        retrieval_options={"weights": {"provenance": 0.6, "validation": 0.5}},
    ),
    RoleSpec(
        name="replicator",
        role=AgentRole.EXPERIMENTER,
        description=(
            "Independently reproduces a result. Does not see the conclusion it is reproducing."
        ),
        #: The conclusion is withheld. An agent shown the answer it is meant to independently
        #: confirm is agreeing, not replicating — and the resulting `reproduces` edge would be
        #: worthless while looking like the strongest evidence in the graph (§28).
        retrieval_options={"exclude_types": ["conclusion", "result", "answer"],
                           "expose_contradictions": False},
    ),
    RoleSpec(
        name="adversary",
        role=AgentRole.CRITIC,
        description=(
            "Attacks the strongest current account: assumes it is wrong and looks for the case "
            "that breaks it."
        ),
        retrieval_options={"weights": {"validation": 0.4, "provenance": 0.4},
                           "expose_contradictions": True},
    ),
    RoleSpec(
        name="alternative_hypothesis",
        role=AgentRole.RESEARCHER,
        description="Proposes alternative explanations for an observation already accounted for.",
        #: Contradiction exposure on, and diversity weighted up: the point is to find a *different*
        #: account, and a ranking that returns the consensus first works against that.
        retrieval_options={"expose_contradictions": True,
                           "weights": {"utility": -0.2, "recency": 0.4}},
    ),
    RoleSpec(
        name="synthesizer",
        role=AgentRole.SYNTHESIZER,
        description="Combines separate findings into one account, keeping the links to both.",
        retrieval_options={"weights": {"graph": 0.8, "utility": 0.6}},
    ),
    RoleSpec(
        name="toolmaker",
        role=AgentRole.TOOLMAKER,
        description="Builds reusable tools for work the collective repeats.",
        retrieval_options={},
    ),
)


def ensure_roles(session: Session, workspace_id: uuid.UUID) -> list[AgentProfile]:
    """Create the role profiles a workspace does not yet have.

    Idempotent, and it never overwrites `performance_by_work_type`: that column is a measurement
    (§26), and re-seeding roles must not erase what the collective has learned about them.
    """
    existing = {
        p.name: p
        for p in session.execute(
            select(AgentProfile).where(AgentProfile.workspace_id == workspace_id)
        ).scalars()
    }
    out: list[AgentProfile] = []
    for spec in ROLE_SPECS:
        profile = existing.get(spec.name)
        if profile is None:
            profile = spec.to_profile(workspace_id)
            session.add(profile)
        out.append(profile)
    session.flush()
    return out


def retrieval_options_for(profile: AgentProfile) -> dict[str, Any]:
    """The retrieval parameters a role runs under."""
    return dict((profile.meta or {}).get("retrieval_options") or {})


# --------------------------------------------------------------------------
# what each role should be pointed at (§27, §28)
# --------------------------------------------------------------------------
def claims_needing_challenge(
    session: Session, workspace_id: uuid.UUID, *, limit: int = 10
) -> list[Artifact]:
    """Claims that are supported but unchallenged — where a critic is worth spending.

    Unchallenged is not the same as correct. §28 exists because a claim nobody has argued with
    looks exactly like a claim nobody can argue with.
    """
    from civitas.domain.enums import RelationType
    from civitas.knowledge.graph import neighbours

    out: list[Artifact] = []
    candidates = session.execute(
        select(Artifact).where(
            Artifact.workspace_id == workspace_id,
            Artifact.type == ArtifactType.HYPOTHESIS,
            Artifact.archived_at.is_(None),
            Artifact.hypothesis_state.in_([
                HypothesisState.SUPPORTED.value, HypothesisState.TESTED.value,
            ]),
        ).limit(limit * 5)
    ).scalars()

    for artifact in candidates:
        opposing = neighbours(
            session, artifact.id,
            types=frozenset({RelationType.CONTRADICTS, RelationType.FALSIFIES}),
            outgoing=False,
        )
        if not opposing:
            out.append(artifact)
        if len(out) >= limit:
            break
    return out


def claims_needing_replication(
    session: Session, workspace_id: uuid.UUID, *, limit: int = 10
) -> list[Artifact]:
    """Supported claims with no *independent* reproduction (§28)."""
    from civitas.knowledge.hypothesis import assess

    out: list[Artifact] = []
    for artifact in session.execute(
        select(Artifact).where(
            Artifact.workspace_id == workspace_id,
            Artifact.type == ArtifactType.HYPOTHESIS,
            Artifact.archived_at.is_(None),
            Artifact.hypothesis_state == HypothesisState.SUPPORTED.value,
        ).limit(limit * 5)
    ).scalars():
        if assess(session, artifact).independent_replications == 0:
            out.append(artifact)
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------
# decomposition (§27)
# --------------------------------------------------------------------------
def decompose(
    session: Session,
    *,
    task_id: uuid.UUID,
    subproblems: list[dict[str, Any]],
    episode_id: uuid.UUID | None = None,
) -> list[Subproblem]:
    """Record subproblems for a task (§27).

    A subproblem is a *question the scheduler tracks*, not a child task. Making it a task would
    commit an episode to it before anyone has judged whether it is worth one; keeping it a
    subproblem lets the scheduler weigh it against everything else first.
    """
    out: list[Subproblem] = []
    for entry in subproblems:
        subproblem = Subproblem(
            task_id=task_id,
            title=str(entry.get("title", ""))[:500],
            description=str(entry.get("description", "")),
            uncertainty=float(entry.get("uncertainty", 1.0)),
            expected_information_gain=float(entry.get("expected_information_gain", 0.0)),
            created_by_episode_id=episode_id,
        )
        session.add(subproblem)
        out.append(subproblem)
    session.flush()
    return out


def resolve_subproblem(session: Session, subproblem: Subproblem, *, resolved: bool = True) -> None:
    subproblem.status = "resolved" if resolved else "abandoned"
    subproblem.uncertainty = 0.0 if resolved else subproblem.uncertainty
    session.flush()


def open_subproblems(session: Session, task_id: uuid.UUID) -> list[Subproblem]:
    return list(
        session.execute(
            select(Subproblem).where(
                Subproblem.task_id == task_id, Subproblem.status == "open"
            ).order_by(Subproblem.uncertainty.desc())
        ).scalars()
    )
