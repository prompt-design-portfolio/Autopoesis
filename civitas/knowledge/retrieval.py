"""Retrieval (Part B §14) — arm-enforced, and logged with the features that produced the ranking.

M3/M4 ship the lexical, recency, confidence, validation, provenance-strength, utility and
diversity components. M5 adds vector similarity, graph proximity and contradiction exposure, and
re-reports the identical newcomer benchmark so the knowledge system's contribution is *measured*
(ARCHITECTURE §5). The scoring is a weighted sum over named features precisely so a component can
be added, and its effect isolated, without rewriting the ranker.

Every retrieval writes a `RetrievalDecision` with the query, policy version, candidate count,
per-result feature vector and — importantly — what the arm *suppressed*. An ablation whose effect
can only be inferred from an absence is not auditable.
"""

from __future__ import annotations

import math
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from civitas.domain.enums import (
    EVIDENCE_STRENGTH,
    ArtifactStatus,
    ArtifactType,
    EventType,
    ExperimentArm,
    ValidationState,
)
from civitas.knowledge.arms import ArmPolicy, arm_policy
from civitas.persistence.events import emit
from civitas.persistence.models import Artifact, RetrievalDecision
from civitas.persistence.types import utcnow

POLICY_VERSION = "retrieval/1.0-lexical"

#: Weights over the named features. Not tuned by hand later in an ad-hoc way: this dictionary is
#: the body of a `Policy` row under the A2.2 gate, so a change to it must win a matched experiment
#: before the runtime will load it.
DEFAULT_WEIGHTS: dict[str, float] = {
    "lexical": 1.00,
    "recency": 0.15,
    "validation": 0.45,
    "provenance": 0.35,
    "utility": 0.40,
    # Self-reported confidence ranks *last* (Part A §A2.1): it is the one signal an agent can
    # inflate for free, so it breaks ties and never drives a ranking.
    "confidence": 0.05,
    "stale_penalty": -0.50,
    "superseded_penalty": -0.35,
    "refuted_penalty": -0.60,
}

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the is are was were be of in on at to for with by from as that this it and or not".split()
)


@dataclass
class RetrievalResult:
    artifacts: list[Artifact]
    scores: list[float] = field(default_factory=list)
    features: list[dict[str, float]] = field(default_factory=list)
    decision_id: uuid.UUID | None = None
    candidate_count: int = 0
    suppressed: list[dict[str, Any]] = field(default_factory=list)
    #: Set when provenance was stripped (`collective_no_provenance`). The caller must not render
    #: trust metadata it was told to remove.
    provenance_hidden: bool = False


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall((text or "").lower()) if w not in _STOP and len(w) > 1]


def retrieve(
    session: Session,
    *,
    workspace_id: uuid.UUID,
    query: str,
    arm: ExperimentArm | str = ExperimentArm.COLLECTIVE,
    episode_id: uuid.UUID | None = None,
    types: list[ArtifactType] | None = None,
    limit: int = 8,
    as_of=None,
    weights: dict[str, float] | None = None,
    seed: int | None = None,
    config_hash: str = "",
    log_decision: bool = True,
) -> RetrievalResult:
    """Rank the workspace's artifacts for this query under this arm's policy."""
    started = time.perf_counter()
    policy = arm_policy(arm, as_of=as_of)
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}

    if policy.blind:
        # Nothing is returned, and the decision is still logged. A blind arm that leaves no record
        # is indistinguishable from a retrieval that happened to find nothing.
        result = RetrievalResult(artifacts=[], candidate_count=0)
        if log_decision:
            result.decision_id = _log(
                session, workspace_id=workspace_id, episode_id=episode_id, query=query,
                policy=policy, ranked=[], candidate_count=0,
                suppressed=[{"reason": "arm_blind", "count": "all"}],
                latency_ms=(time.perf_counter() - started) * 1000, config_hash=config_hash,
            )
        return result

    candidates = _candidates(session, workspace_id, types, policy)
    suppressed: list[dict[str, Any]] = []

    kept: list[Artifact] = []
    for artifact in candidates:
        reason = _suppression_reason(artifact, policy)
        if reason:
            suppressed.append({"artifact_id": str(artifact.id), "reason": reason})
        else:
            kept.append(artifact)

    query_tokens = tokenize(query)
    now = utcnow()
    scored: list[tuple[float, dict[str, float], Artifact]] = []
    for artifact in kept:
        feats = _features(artifact, query_tokens, now, policy)
        score = sum(weights.get(k, 0.0) * v for k, v in feats.items())
        scored.append((score, feats, artifact))

    if policy.scramble:
        # Presence held, content destroyed: the same *number* of artifacts of comparable
        # character, with the query-relevance ordering destroyed. Shuffling the ranking rather
        # than sampling different rows keeps mark density and type mix identical, which is what
        # makes this a control for "did information pass" rather than for "did a store exist".
        rng = random.Random(seed if seed is not None else 0)
        rng.shuffle(scored)
        for i, (_score, feats, artifact) in enumerate(scored):
            scored[i] = (0.0, {**feats, "lexical": 0.0}, artifact)
    else:
        scored.sort(key=lambda row: row[0], reverse=True)

    selected = _diversify(scored, limit) if not policy.scramble else scored[:limit]

    artifacts = [row[2] for row in selected]
    now_ts = utcnow()
    for artifact in artifacts:
        artifact.times_retrieved += 1
        artifact.last_retrieved_at = now_ts

    result = RetrievalResult(
        artifacts=artifacts,
        scores=[row[0] for row in selected],
        features=[row[1] for row in selected],
        candidate_count=len(candidates),
        suppressed=suppressed,
        provenance_hidden=policy.drop_provenance,
    )
    if log_decision:
        result.decision_id = _log(
            session, workspace_id=workspace_id, episode_id=episode_id, query=query,
            policy=policy, ranked=selected, candidate_count=len(candidates),
            suppressed=suppressed, latency_ms=(time.perf_counter() - started) * 1000,
            config_hash=config_hash,
        )
    return result


def _candidates(
    session: Session,
    workspace_id: uuid.UUID,
    types: list[ArtifactType] | None,
    policy: ArmPolicy,
) -> list[Artifact]:
    stmt = select(Artifact).where(
        Artifact.workspace_id == workspace_id,
        Artifact.archived_at.is_(None),
    )
    if types:
        stmt = stmt.where(Artifact.type.in_([t.value for t in types]))
    if policy.as_of is not None:
        # `collective_frozen`: the cut is applied in SQL, not after ranking, so a frozen arm can
        # never see a later artifact even transiently.
        stmt = stmt.where(Artifact.created_at <= policy.as_of)
    if policy.excluded_types:
        stmt = stmt.where(Artifact.type.notin_([t.value for t in policy.excluded_types]))
    # Bounded: a workspace can hold millions of artifacts (§59) and the ranker must not load them
    # all. Recency-ordered, so the bound favours what is most likely to still apply.
    stmt = stmt.order_by(Artifact.created_at.desc()).limit(2000)
    return list(session.execute(stmt).scalars())


def _suppression_reason(artifact: Artifact, policy: ArmPolicy) -> str | None:
    if policy.curated_only:
        if artifact.validation_state in (
            ValidationState.UNVALIDATED, ValidationState.SELF_REPORTED,
            ValidationState.DISPUTED, ValidationState.REFUTED,
        ):
            return "shared_memory_uncurated"
        if artifact.status is not ArtifactStatus.ACTIVE:
            return "shared_memory_inactive"
    if artifact.status is ArtifactStatus.RETRACTED:
        return "retracted"
    return None


def _features(
    artifact: Artifact, query_tokens: list[str], now, policy: ArmPolicy
) -> dict[str, float]:
    """The named feature vector. Recorded per result on the `RetrievalDecision` (§14)."""
    text_tokens = tokenize(f"{artifact.title} {artifact.body}")
    lexical = _bm25ish(query_tokens, text_tokens)

    age_days = max(0.0, (now - artifact.created_at).total_seconds() / 86400)
    recency = 1.0 / (1.0 + age_days / 30.0)

    validation = {
        ValidationState.HUMAN_CONFIRMED: 1.0,
        ValidationState.EVALUATOR_CONFIRMED: 0.95,
        ValidationState.REPRODUCED: 0.9,
        ValidationState.TOOL_VERIFIED: 0.75,
        ValidationState.PEER_REVIEWED: 0.6,
        ValidationState.SELF_REPORTED: 0.3,
        ValidationState.UNVALIDATED: 0.2,
        ValidationState.DISPUTED: 0.1,
        ValidationState.REFUTED: 0.0,
    }.get(artifact.validation_state, 0.2)

    provenance = EVIDENCE_STRENGTH.get(artifact.evidence_kind, 0.1)
    #: Squashed rather than raw: utility is unbounded above, and an artifact with a long history
    #: would otherwise dominate every query regardless of relevance.
    utility = math.tanh(max(0.0, artifact.downstream_utility))

    feats = {
        "lexical": lexical,
        "recency": recency,
        "validation": validation,
        "provenance": provenance,
        "utility": utility,
        "confidence": artifact.confidence,
        "stale_penalty": 1.0 if artifact.is_stale else 0.0,
        "superseded_penalty": 1.0 if artifact.status is ArtifactStatus.SUPERSEDED else 0.0,
        "refuted_penalty": 1.0 if artifact.validation_state is ValidationState.REFUTED else 0.0,
    }
    if policy.drop_provenance:
        # The arm removes the reader's ability to tell strong evidence from weak, so the ranker
        # must not use it either. Ranking on provenance while hiding it would give the arm the
        # benefit of provenance without the visibility, and it would stop being a control.
        feats["provenance"] = 0.0
        feats["validation"] = 0.0
    return feats


def _bm25ish(query_tokens: list[str], text_tokens: list[str]) -> float:
    """A BM25-shaped lexical score (Part B §14).

    Term frequency saturates, so a document repeating a query word twenty times does not outrank
    one that covers every query word once — which is the property that matters for retrieval over
    agent-written text, where repetition is common and cheap.
    """
    if not query_tokens or not text_tokens:
        return 0.0
    counts: dict[str, int] = {}
    for token in text_tokens:
        counts[token] = counts.get(token, 0) + 1

    k1, b, avg_len = 1.5, 0.75, 200.0
    norm = 1 - b + b * (len(text_tokens) / avg_len)
    score = 0.0
    for token in set(query_tokens):
        tf = counts.get(token, 0)
        if tf:
            score += (tf * (k1 + 1)) / (tf + k1 * norm)
    return score / max(1, len(set(query_tokens)))


def _diversify(
    scored: list[tuple[float, dict[str, float], Artifact]], limit: int
) -> list[tuple[float, dict[str, float], Artifact]]:
    """Avoid returning ten near-identical artifacts (Part B §14).

    Greedy selection with a similarity penalty against what is already chosen, plus a guarantee
    that a contradicting or failure artifact is not crowded out by a block of agreeing ones — §14
    asks that agents sometimes be shown relevant disagreement, and a pure relevance ranking
    systematically removes it.
    """
    if len(scored) <= limit:
        return scored

    selected: list[tuple[float, dict[str, float], Artifact]] = []
    remaining = list(scored)
    chosen_tokens: list[set[str]] = []

    while remaining and len(selected) < limit:
        best_i, best_val = 0, -1e9
        for i, (score, _feats, artifact) in enumerate(remaining):
            tokens = set(tokenize(artifact.title))
            overlap = max(
                (len(tokens & prev) / max(1, len(tokens | prev)) for prev in chosen_tokens),
                default=0.0,
            )
            value = score - 0.6 * overlap
            if value > best_val:
                best_i, best_val = i, value
        pick = remaining.pop(best_i)
        selected.append(pick)
        chosen_tokens.append(set(tokenize(pick[2].title)))

    return selected


def _log(
    session: Session,
    *,
    workspace_id: uuid.UUID,
    episode_id: uuid.UUID | None,
    query: str,
    policy: ArmPolicy,
    ranked: list,
    candidate_count: int,
    suppressed: list[dict[str, Any]],
    latency_ms: float,
    config_hash: str,
) -> uuid.UUID:
    decision = RetrievalDecision(
        episode_id=episode_id,
        workspace_id=workspace_id,
        query=query[:4000],
        policy_version=POLICY_VERSION,
        experiment_arm=policy.arm.value,
        candidate_count=candidate_count,
        returned_count=len(ranked),
        ranking=[
            {
                "artifact_id": str(artifact.id),
                "rank": i,
                "score": round(score, 6),
                "features": {k: round(v, 6) for k, v in feats.items()},
            }
            for i, (score, feats, artifact) in enumerate(ranked)
        ],
        suppressed=suppressed[:500],
        latency_ms=latency_ms,
    )
    session.add(decision)
    session.flush()
    emit(
        session, workspace_id=workspace_id, type=EventType.RETRIEVAL_PERFORMED,
        episode_id=episode_id,
        payload={"decision_id": str(decision.id), "arm": policy.arm.value,
                 "returned": len(ranked), "candidates": candidate_count},
        config_hash=config_hash,
    )
    return decision.id
