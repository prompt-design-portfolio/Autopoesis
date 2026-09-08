"""Machine-enforced experimental arms (Part B §21).

§21 is explicit that arms must be built into runtime policy, not prompt convention. This module is
where that enforcement lives: retrieval asks `arm_policy()` what it is allowed to return, and the
answer is a data structure, not an instruction a model could ignore.

Every suppression is *recorded* on the `RetrievalDecision`, not merely applied. An ablation whose
effect can only be inferred from an absence is not auditable, and §14 requires the decision to be
reconstructable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from civitas.domain.enums import NEGATIVE_TYPES, ArtifactType, ExperimentArm


@dataclass(frozen=True)
class ArmPolicy:
    """What one arm permits. Read by retrieval; never by a prompt."""

    arm: ExperimentArm

    #: No collective information at all. `solo` and `independent` differ in the *population*, not
    #: in what an episode sees — both are blind to prior episodes — so they share this flag and
    #: are distinguished by how the experiment runner schedules them.
    blind: bool = False

    #: Only curated, accepted results — the ordinary shared-memory/RAG condition. Excludes
    #: failures, open questions, contradictions and unvalidated work, which is precisely what
    #: makes it the right control for `collective` rather than a weaker version of it.
    curated_only: bool = False

    #: Return comparable *quantities* while destroying relevance. Presence held, content
    #: destroyed — the `noise record` lesson from the research lineage (ARCHITECTURE §3.4).
    scramble: bool = False

    #: Strip negative knowledge (Part B §21 `collective_no_negative`).
    drop_negative: bool = False

    #: Strip provenance and trust metadata (Part B §21 `collective_no_provenance`). The artifacts
    #: are the same; what is removed is the reader's ability to tell strong evidence from weak.
    drop_provenance: bool = False

    #: Only artifacts that existed at this cut (Part B §21 `collective_frozen`).
    as_of: datetime | None = None

    #: Whether a prior failure may be surfaced ahead of an action about to repeat it
    #: (Part A §A2.3). False wherever negative knowledge is unavailable, so the mechanism cannot
    #: leak information the arm is supposed to have removed.
    duplicate_warnings: bool = True

    excluded_types: frozenset[ArtifactType] = field(default_factory=frozenset)


_POLICIES: dict[ExperimentArm, ArmPolicy] = {
    ExperimentArm.SOLO: ArmPolicy(
        arm=ExperimentArm.SOLO, blind=True, duplicate_warnings=False
    ),
    ExperimentArm.INDEPENDENT: ArmPolicy(
        arm=ExperimentArm.INDEPENDENT, blind=True, duplicate_warnings=False
    ),
    ExperimentArm.SHARED_MEMORY: ArmPolicy(
        arm=ExperimentArm.SHARED_MEMORY,
        curated_only=True,
        drop_negative=True,
        duplicate_warnings=False,
        excluded_types=frozenset(NEGATIVE_TYPES | {ArtifactType.QUESTION,
                                                   ArtifactType.UNRESOLVED_ISSUE}),
    ),
    ExperimentArm.COLLECTIVE: ArmPolicy(arm=ExperimentArm.COLLECTIVE),
    ExperimentArm.COLLECTIVE_SCRAMBLED: ArmPolicy(
        arm=ExperimentArm.COLLECTIVE_SCRAMBLED,
        scramble=True,
        # The warning mechanism selects *by relevance*, so leaving it on would reintroduce exactly
        # the relevance this arm exists to destroy.
        duplicate_warnings=False,
    ),
    ExperimentArm.COLLECTIVE_NO_NEGATIVE: ArmPolicy(
        arm=ExperimentArm.COLLECTIVE_NO_NEGATIVE,
        drop_negative=True,
        duplicate_warnings=False,
        excluded_types=frozenset(NEGATIVE_TYPES),
    ),
    ExperimentArm.COLLECTIVE_NO_PROVENANCE: ArmPolicy(
        arm=ExperimentArm.COLLECTIVE_NO_PROVENANCE, drop_provenance=True
    ),
    ExperimentArm.COLLECTIVE_FROZEN: ArmPolicy(arm=ExperimentArm.COLLECTIVE_FROZEN),
    ExperimentArm.MEMORY_RESET: ArmPolicy(
        arm=ExperimentArm.MEMORY_RESET, blind=True, duplicate_warnings=False
    ),
}


def arm_policy(arm: ExperimentArm | str, *, as_of: datetime | None = None) -> ArmPolicy:
    """The policy for an arm.

    An unknown arm raises. Falling back to `collective` would silently turn a mis-specified
    experiment into an unablated one, and the result would look like evidence.
    """
    if isinstance(arm, str):
        try:
            arm = ExperimentArm(arm)
        except ValueError as exc:
            raise ValueError(
                f"unknown experiment arm {arm!r}; arms are machine-enforced (Part B §21) "
                f"and must be one of {[a.value for a in ExperimentArm]}"
            ) from exc
    policy = _POLICIES[arm]
    if arm is ExperimentArm.COLLECTIVE_FROZEN:
        if as_of is None:
            raise ValueError(
                "collective_frozen requires a snapshot cut (`as_of`); without one the arm is "
                "indistinguishable from `collective` and the control is not a control"
            )
        return ArmPolicy(**{**policy.__dict__, "as_of": as_of})
    return policy


def all_arms() -> list[ExperimentArm]:
    return list(ExperimentArm)
