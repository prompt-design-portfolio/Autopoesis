"""A retrieval-quality benchmark (Part B §14, §48, §55 retrieval tests).

**Why this exists.** M5 added vector similarity, graph proximity and contradiction exposure, and
re-ran the M4 newcomer benchmark as a matched A/B. The result was a clean null: advantage 0.300
under both retrieval policies, identical to three decimal places.

That is not evidence the components do nothing. It is evidence the *newcomer benchmark cannot see
them*. Its corpus is a few dozen artifacts, and its queries contain the exact rare token that
identifies the target — so BM25 alone already ranks the right artifact first, and there is no rank
left to improve. An instrument that cannot resolve an effect must not be reported as having found
its absence (ARCHITECTURE §3.1, §3.9).

So this measures retrieval directly, under the two conditions that separate a lexical ranker from
a hybrid one:

* **exact** — the query shares the target's rare terms. What the newcomer benchmark tests.
* **paraphrase** — the query shares *no* rare term with the target, only meaning. What an agent
  actually produces when it does not already know the answer's vocabulary.

Both run against a corpus padded with distractors, because ranking only matters when the right
artifact has to be found among the wrong ones.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from civitas.domain.enums import ArtifactType, EvidenceKind, ExperimentArm
from civitas.knowledge.embeddings import index_missing
from civitas.knowledge.retrieval import retrieve
from civitas.persistence.models import Artifact, Workspace

#: (statement, exact query, paraphrase query). The paraphrase deliberately shares no content word
#: with its target — that is the whole point, and a paraphrase that reused a rare term would make
#: the condition indistinguishable from `exact`.
PROBES: list[tuple[str, str, str]] = [
    (
        "The writer retains its mutex across a retry, so a second attempt deadlocks",
        "writer retains mutex across retry deadlocks",
        "lock is not released before the operation is attempted again",
    ),
    (
        "Cache eviction under memory pressure discards entries that are still referenced",
        "cache eviction memory pressure discards referenced entries",
        "records held by active readers are dropped when storage runs low",
    ),
    (
        "The scheduler starves low-priority tasks when the queue depth exceeds sixteen",
        "scheduler starves low priority tasks queue depth sixteen",
        "work with little urgency never runs once the backlog grows",
    ),
    (
        "Serialisation of nested structures loses ordering on the second level",
        "serialisation nested structures loses ordering second level",
        "encoding a structure inside another structure scrambles its sequence",
    ),
    (
        "Connection pooling reuses a socket after the peer has already closed it",
        "connection pooling reuses socket peer closed",
        "a network handle is recycled once the far end went away",
    ),
    (
        "Batch inserts silently truncate values wider than the declared column",
        "batch inserts truncate values wider than column",
        "bulk writing quietly shortens data that exceeds the field size",
    ),
]

DISTRACTOR_TOPICS = [
    "telemetry sampling", "index rebuild", "certificate rotation", "log compaction",
    "shard rebalancing", "quota accounting", "retry backoff", "schema migration",
    "checksum verification", "tenant isolation", "clock skew", "message ordering",
]


@dataclass
class PolicyScore:
    label: str
    condition: str
    n: int = 0
    #: Mean reciprocal rank. 1.0 means the target was always first; 0.0 means never returned.
    mrr: float = 0.0
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_8: float = 0.0
    #: Queries where the target was not returned at all. Reported separately, because a target
    #: that is absent and one that is ranked last are different failures.
    misses: int = 0

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class RetrievalBenchmarkResult:
    corpus_size: int
    scores: dict[str, PolicyScore] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "corpus_size": self.corpus_size,
            "scores": {k: v.as_dict() for k, v in self.scores.items()},
            "notes": self.notes,
        }


def build_corpus(
    session: Session,
    *,
    organization_id: uuid.UUID,
    distractors: int = 300,
    seed: int = 0,
) -> tuple[Workspace, list[Artifact]]:
    """A workspace holding the targets plus a realistic amount of unrelated knowledge."""
    workspace = Workspace(
        organization_id=organization_id,
        name="retrieval benchmark",
        slug=f"retr-{uuid.uuid4().hex[:8]}",
        environment_version="retrieval-benchmark",
    )
    session.add(workspace)
    session.flush()

    targets: list[Artifact] = []
    for statement, _exact, _para in PROBES:
        artifact = Artifact(
            workspace_id=workspace.id, type=ArtifactType.EVIDENCE,
            title=statement, body=statement,
            evidence_kind=EvidenceKind.TOOL_OUTPUT,
            environment_version="retrieval-benchmark",
        )
        session.add(artifact)
        targets.append(artifact)

    rng = random.Random(f"distractors|{seed}")
    for i in range(distractors):
        topic = rng.choice(DISTRACTOR_TOPICS)
        session.add(Artifact(
            workspace_id=workspace.id, type=ArtifactType.OBSERVATION,
            title=f"{topic} behaviour observed in case {i}",
            body=f"During {topic}, the system exhibited variant {i} of the documented behaviour.",
            environment_version="retrieval-benchmark",
        ))
    session.flush()
    index_missing(session, workspace.id)
    session.flush()
    return workspace, targets


def score_policy(
    session: Session,
    *,
    workspace: Workspace,
    targets: list[Artifact],
    label: str,
    condition: str,
    retrieval_options: dict[str, Any],
    limit: int = 8,
) -> PolicyScore:
    """Rank quality for one policy under one query condition."""
    score = PolicyScore(label=label, condition=condition, n=len(targets))
    reciprocal, at1, at3, at8 = 0.0, 0, 0, 0

    for artifact, (_statement, exact, paraphrase) in zip(targets, PROBES, strict=True):
        query = exact if condition == "exact" else paraphrase
        result = retrieve(
            session,
            workspace_id=workspace.id,
            query=query,
            arm=ExperimentArm.COLLECTIVE,
            limit=limit,
            use_vector=retrieval_options.get("use_vector", True),
            expose_contradictions=retrieval_options.get("expose_contradictions", True),
            weights=retrieval_options.get("weights"),
            log_decision=False,
        )
        ids = [a.id for a in result.artifacts]
        if artifact.id not in ids:
            score.misses += 1
            continue
        rank = ids.index(artifact.id) + 1
        reciprocal += 1.0 / rank
        at1 += rank <= 1
        at3 += rank <= 3
        at8 += rank <= 8

    n = max(1, score.n)
    score.mrr = round(reciprocal / n, 4)
    score.recall_at_1 = round(at1 / n, 4)
    score.recall_at_3 = round(at3 / n, 4)
    score.recall_at_8 = round(at8 / n, 4)
    return score


def run_retrieval_benchmark(
    session: Session,
    *,
    organization_id: uuid.UUID,
    distractors: int = 300,
    seed: int = 0,
) -> RetrievalBenchmarkResult:
    """Both policies, both query conditions, one corpus."""
    workspace, targets = build_corpus(
        session, organization_id=organization_id, distractors=distractors, seed=seed
    )
    session.flush()

    policies = {
        "lexical": {"use_vector": False, "expose_contradictions": False},
        "hybrid": {"use_vector": True, "expose_contradictions": True},
    }
    result = RetrievalBenchmarkResult(corpus_size=len(targets) + distractors)
    for label, options in policies.items():
        for condition in ("exact", "paraphrase"):
            result.scores[f"{label}/{condition}"] = score_policy(
                session, workspace=workspace, targets=targets,
                label=label, condition=condition, retrieval_options=options,
            )

    result.notes.append(
        "The paraphrase queries share no content word with their target, which is what an agent "
        "produces when it does not already know the answer's vocabulary."
    )
    return result
