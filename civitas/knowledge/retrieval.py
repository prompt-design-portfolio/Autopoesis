"""Retrieval (Part B §14) — arm-enforced, and logged with the features that produced the ranking.

All of §14's components are present: vector similarity, full-text and BM25-shaped lexical
ranking, graph proximity, task relevance, recency, confidence, validation, provenance strength,
environment applicability, negative-knowledge relevance, historical utility, diversity and
contradiction exposure.

The scoring is a weighted sum over *named* features so a component can be added and its effect
isolated without rewriting the ranker — which is what let M5 add three components and re-report
the identical M4 benchmark to measure what they were worth.

Two of §14's requirements are easy to state and easy to leave as prose, so they are mechanisms
here: a result set must not be ten near-identical artifacts (`_diversify`), and an agent must
sometimes be shown relevant *disagreement* (`_expose_contradictions`) — a pure relevance ranking
systematically removes the thing most likely to correct it.

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

from sqlalchemy import or_, select
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

POLICY_VERSION = "retrieval/2.0-hybrid"

#: Weights over the named features. Not tuned by hand later in an ad-hoc way: this dictionary is
#: the body of a `Policy` row under the A2.2 gate, so a change to it must win a matched experiment
#: before the runtime will load it.
DEFAULT_WEIGHTS: dict[str, float] = {
    "lexical": 1.00,
    #: Semantic similarity. Weighted *below* lexical on purpose: for agent-written artifacts an
    #: exact term match is the stronger signal, and §14 forbids relying on embeddings alone.
    "vector": 0.70,
    #: Proximity in the artifact graph to something already relevant. This is what lets a chain
    #: of evidence be retrieved together rather than only its best-matching link.
    "graph": 0.30,
    #: The task the episode is on. Cheap, and it separates two artifacts that match a query
    #: equally well but belong to different work.
    "task_relevance": 0.25,
    "recency": 0.15,
    "validation": 0.45,
    "provenance": 0.35,
    "utility": 0.40,
    # Self-reported confidence ranks *last* (Part A §A2.1): it is the one signal an agent can
    # inflate for free, so it breaks ties and never drives a ranking.
    "confidence": 0.05,
    #: §14 requires prior failures to gain relevance when an agent is about to repeat a known
    #: failed method. Positive, and applied only when the query overlaps the failure — a blanket
    #: boost would flood every result with unrelated failures.
    "negative_relevance": 0.55,
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
    task_id: uuid.UUID | None = None,
    use_vector: bool = True,
    expose_contradictions: bool = True,
    exclude_types: list[ArtifactType] | None = None,
    embedder=None,
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

    # A *role* may withhold types from itself, on top of whatever the arm withholds. This is how
    # the replicator works (§28): an agent shown the conclusion it is asked to independently
    # reproduce is agreeing, not replicating, and the `reproduces` edge it then writes would be
    # worthless while looking like the strongest evidence in the graph.
    if exclude_types:
        excluded_set = {t for t in exclude_types}
        removed = [a for a in candidates if a.type in excluded_set]
        if removed:
            candidates = [a for a in candidates if a.type not in excluded_set]
            suppressed.append({
                "reason": "role_excluded_type",
                "types": sorted(t.value for t in excluded_set),
                "count": len(removed),
            })

    # Exclusions applied in SQL are still exclusions. Recording them as an aggregate keeps the
    # ablation auditable without loading rows the arm has already ruled out — an arm whose effect
    # can only be inferred from an absence is not auditable (§14), and "it was never in the
    # candidate set" is exactly such an absence.
    if policy.excluded_types:
        removed = _count_excluded(session, workspace_id, types, policy)
        if removed:
            suppressed.append({
                "reason": "arm_excluded_type",
                "types": sorted(t.value for t in policy.excluded_types),
                "count": removed,
                "applied": "sql",
            })

    kept: list[Artifact] = []
    for artifact in candidates:
        reason = _suppression_reason(artifact, policy)
        if reason:
            suppressed.append({"artifact_id": str(artifact.id), "reason": reason})
        else:
            kept.append(artifact)

    query_tokens = tokenize(query)
    now = utcnow()

    # Semantic similarity, restricted to the candidates the arm already allowed. The vector index
    # must never be a way around an ablation (§21).
    vector_scores: dict[uuid.UUID, float] = {}
    if use_vector and kept:
        from civitas.knowledge.embeddings import search as vector_search

        try:
            hits = vector_search(
                session, workspace_id=workspace_id, query=query, embedder=embedder,
                limit=len(kept), candidate_ids={a.id for a in kept},
            )
            vector_scores = {h.artifact_id: h.score for h in hits}
        except Exception:  # pragma: no cover - a missing index must not fail retrieval
            vector_scores = {}

    graph_scores = _graph_proximity(session, kept, query_tokens, vector_scores)

    scored: list[tuple[float, dict[str, float], Artifact]] = []
    for artifact in kept:
        feats = _features(artifact, query_tokens, now, policy)
        feats["vector"] = vector_scores.get(artifact.id, 0.0)
        feats["graph"] = graph_scores.get(artifact.id, 0.0)
        feats["task_relevance"] = (
            1.0 if task_id is not None and artifact.task_id == task_id else 0.0
        )
        feats["negative_relevance"] = _negative_relevance(artifact, query_tokens, feats)
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

    if policy.scramble:
        selected = scored[:limit]
    else:
        selected = _diversify(scored, limit)
        if expose_contradictions:
            selected = _expose_contradictions(session, selected, scored, limit)

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


def _count_excluded(
    session: Session,
    workspace_id: uuid.UUID,
    types: list[ArtifactType] | None,
    policy: ArmPolicy,
) -> int:
    """How many candidates the arm's type exclusion removed, for the audit record."""
    from sqlalchemy import func

    stmt = select(func.count(Artifact.id)).where(
        Artifact.workspace_id == workspace_id,
        Artifact.archived_at.is_(None),
        Artifact.type.in_([t.value for t in policy.excluded_types]),
    )
    if types:
        stmt = stmt.where(Artifact.type.in_([t.value for t in types]))
    if policy.as_of is not None:
        stmt = stmt.where(Artifact.created_at <= policy.as_of)
    return int(session.execute(stmt).scalar_one() or 0)


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


#: Above this overlap two candidates are treated as the same result, and the second is skipped
#: rather than merely penalised. A penalty alone cannot work for exact duplicates: it is
#: subtracted equally from every one of them, so their relative order is unchanged and all of them
#: are still selected — measured on a corpus of twelve identical artifacts, which came back as
#: five identical results (§14 forbids exactly that).
NEAR_IDENTICAL = 0.85


def _signature(artifact: Artifact) -> set[str]:
    """Tokens used to judge whether two results say the same thing.

    Title *and* the opening of the body. Titles alone are not enough: agents write templated
    titles, so two artifacts recording different findings can share one verbatim.
    """
    return set(tokenize(f"{artifact.title} {artifact.body[:240]}"))


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _diversify(
    scored: list[tuple[float, dict[str, float], Artifact]], limit: int
) -> list[tuple[float, dict[str, float], Artifact]]:
    """Avoid returning ten near-identical artifacts (Part B §14).

    Two mechanisms, because one is not enough:

    * a **hard skip** for anything near-identical to something already chosen — this is what
      actually removes duplicates, since a soft penalty applies equally to all of them;
    * a **soft penalty** on partial overlap, which trades a little relevance for coverage among
      results that are related but not the same.

    Near-identical candidates are held back rather than discarded: if the limit cannot be filled
    from distinct results, they are appended. Returning fewer results than asked for would be a
    worse failure than returning a near-duplicate, because the caller cannot tell the difference
    between "nothing else matched" and "the ranker suppressed it".
    """
    if len(scored) <= 1:
        return scored

    selected: list[tuple[float, dict[str, float], Artifact]] = []
    held_back: list[tuple[float, dict[str, float], Artifact]] = []
    remaining = list(scored)
    chosen: list[set[str]] = []

    while remaining and len(selected) < limit:
        best_i, best_val = -1, -1e9
        for i, (score, _feats, artifact) in enumerate(remaining):
            signature = _signature(artifact)
            overlap = max((_overlap(signature, prev) for prev in chosen), default=0.0)
            if overlap >= NEAR_IDENTICAL:
                continue
            value = score - 0.6 * overlap
            if value > best_val:
                best_i, best_val = i, value
        if best_i < 0:
            break  # everything left duplicates something already chosen
        pick = remaining.pop(best_i)
        selected.append(pick)
        chosen.append(_signature(pick[2]))

    if len(selected) < limit:
        held_back = [row for row in remaining if row not in selected]
        selected.extend(held_back[: limit - len(selected)])
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


def _graph_proximity(
    session: Session,
    candidates: list[Artifact],
    query_tokens: list[str],
    vector_scores: dict[uuid.UUID, float],
) -> dict[uuid.UUID, float]:
    """How close each candidate is to the ones the query already matches strongly (§14).

    Seeded from the best directly-matching artifacts, then one hop out. An artifact adjacent to
    strong evidence is more likely to be part of the same chain — which is how a *chain* gets
    retrieved rather than only its best-matching link.

    One hop, not many: proximity decays fast in a dense graph, and a multi-hop walk over every
    candidate on every retrieval is the kind of cost that makes a ranker unusable at §59's scale.
    """
    if not candidates:
        return {}

    seeds: list[tuple[float, uuid.UUID]] = []
    for artifact in candidates:
        lexical = _bm25ish(query_tokens, tokenize(f"{artifact.title} {artifact.body}"))
        strength = max(lexical, vector_scores.get(artifact.id, 0.0))
        if strength > 0.15:
            seeds.append((strength, artifact.id))
    if not seeds:
        return {}
    seeds.sort(reverse=True)
    seed_ids = {aid for _s, aid in seeds[:8]}
    candidate_ids = {a.id for a in candidates}

    from civitas.persistence.models import ArtifactRelation

    scores: dict[uuid.UUID, float] = {}
    relations = session.execute(
        select(ArtifactRelation).where(
            or_(
                ArtifactRelation.source_id.in_(seed_ids),
                ArtifactRelation.target_id.in_(seed_ids),
            )
        )
    ).scalars()
    for relation in relations:
        for near, far in ((relation.source_id, relation.target_id),
                          (relation.target_id, relation.source_id)):
            if near in seed_ids and far in candidate_ids and far not in seed_ids:
                scores[far] = max(scores.get(far, 0.0), 0.6)
    return scores


def _negative_relevance(
    artifact: Artifact, query_tokens: list[str], feats: dict[str, float]
) -> float:
    """Boost a prior failure that is actually about what is being asked (§14).

    > If an agent is about to repeat a known failed method, prior failure artifacts should receive
    > increased relevance.

    Gated on the artifact already matching the query. A blanket boost for negative types would
    flood every result with unrelated failures, which is a good way to make agents ignore them.
    """
    from civitas.domain.enums import NEGATIVE_TYPES

    if artifact.type not in NEGATIVE_TYPES:
        return 0.0

    # Lexical evidence is what "about the same method" means here: a shared term. Semantic
    # similarity gets a much higher bar because a hashing embedder assigns moderate similarity to
    # any two pieces of English prose — measured, a failure about cache eviction scored 0.3
    # against a query about lock retries, which is noise, not aboutness.
    lexical = feats.get("lexical", 0.0)
    vector = feats.get("vector", 0.0)
    if lexical > 0.05:
        return max(lexical, vector)
    return vector if vector >= 0.55 else 0.0


def _expose_contradictions(
    session: Session,
    selected: list[tuple[float, dict[str, float], Artifact]],
    scored: list[tuple[float, dict[str, float], Artifact]],
    limit: int,
) -> list[tuple[float, dict[str, float], Artifact]]:
    """Ensure relevant disagreement survives the ranking (§14).

    > Agents should sometimes be deliberately shown relevant disagreement.

    A pure relevance ranking systematically removes the artifact most likely to correct the top
    result, because a contradiction of a strong match is usually a weaker match itself. Where the
    top result is contradicted by something in the candidate set, the contradiction displaces the
    *lowest-ranked* selection — the set size is preserved, so this changes what an agent sees
    without changing how much it sees.
    """
    if not selected:
        return selected

    from civitas.domain.enums import RelationType
    from civitas.persistence.models import ArtifactRelation

    chosen = {row[2].id for row in selected}
    top_ids = [row[2].id for row in selected[: min(3, len(selected))]]

    opposing = session.execute(
        select(ArtifactRelation).where(
            or_(
                ArtifactRelation.source_id.in_(top_ids),
                ArtifactRelation.target_id.in_(top_ids),
            ),
            ArtifactRelation.type.in_([
                RelationType.CONTRADICTS.value, RelationType.FALSIFIES.value,
                RelationType.INVALIDATES.value,
            ]),
        )
    ).scalars()

    wanted: set[uuid.UUID] = set()
    for relation in opposing:
        for other in (relation.source_id, relation.target_id):
            if other not in chosen and other not in top_ids:
                wanted.add(other)
    if not wanted:
        return selected

    available = {row[2].id: row for row in scored}
    additions = [available[aid] for aid in wanted if aid in available][:2]
    if not additions:
        return selected

    keep = selected[: max(1, limit - len(additions))]
    return keep + additions
