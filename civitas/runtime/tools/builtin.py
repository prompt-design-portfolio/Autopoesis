"""The tools through which an episode touches the collective (Part B §9, §12, §17).

These are the *only* way episode-local work becomes persistent collective state. That is the
mechanism behind Part B §4: an episode's reasoning disappears when it ends, and what survives is
what it deliberately externalised through one of these calls.

Retrieval here is intentionally minimal — recency and keyword. M5 replaces it with the full hybrid
ecology and re-reports the identical benchmark, so the knowledge system's contribution is
measured rather than assumed (ARCHITECTURE §5).
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select

from civitas.domain.enums import (
    ArtifactStatus,
    ArtifactType,
    EventType,
    EvidenceKind,
    RelationType,
    ValidationState,
)
from civitas.persistence.events import emit
from civitas.persistence.models import Artifact, ArtifactRelation
from civitas.runtime.tools.base import Tool, ToolContext, ToolResult

#: Types an agent may assert directly. `evidence`, `experiment_result` and the validated types are
#: excluded: they are produced by tool runs and evaluations, and letting an agent write one by
#: assertion would make provenance strength self-declared (Part B §13).
AGENT_WRITABLE = sorted(
    t.value for t in ArtifactType
    if t not in {ArtifactType.EXPERIMENT_RESULT, ArtifactType.BENCHMARK}
)


class CreateArtifactTool(Tool):
    name = "create_artifact"
    description = (
        "Record a finding in the collective knowledge base so later agents can use it. "
        "This is the only way anything you learn survives the end of your episode."
    )
    parameters = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": AGENT_WRITABLE},
            "title": {"type": "string"},
            "body": {"type": "string"},
            "confidence": {"type": "number"},
            "structured": {"type": "object"},
            "derived_from": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Artifact ids this finding rests on. Provenance, not decoration: "
                               "credit for a later success flows back along these links.",
            },
        },
        "required": ["type", "title", "body"],
    }

    def run(self, ctx: ToolContext, **kw: Any) -> ToolResult:
        try:
            artifact_type = ArtifactType(kw["type"])
        except ValueError:
            return ToolResult(ok=False, error=f"unknown artifact type {kw['type']!r}")

        confidence = float(kw.get("confidence", 0.5))
        if not 0.0 <= confidence <= 1.0:
            return ToolResult(ok=False, error="confidence must be between 0 and 1")

        artifact = Artifact(
            workspace_id=ctx.workspace_id,
            project_id=ctx.project_id,
            task_id=ctx.task_id,
            type=artifact_type,
            title=kw["title"][:500],
            body=kw["body"],
            confidence=confidence,
            structured=kw.get("structured") or {},
            creator_episode_id=ctx.episode_id,
            creator_kind="agent",
            # An agent's own claim is a model assertion until something external says otherwise
            # (Part B §11, §13). Nothing an agent can pass raises this.
            evidence_kind=EvidenceKind.MODEL_ASSERTION,
            validation_state=ValidationState.SELF_REPORTED,
            environment_version=ctx.environment_version,
        )
        ctx.session.add(artifact)
        ctx.session.flush()

        linked = 0
        for raw in kw.get("derived_from") or []:
            parent = _resolve(ctx, raw)
            if parent is None:
                continue
            ctx.session.add(
                ArtifactRelation(
                    source_id=artifact.id, target_id=parent.id,
                    type=RelationType.DERIVED_FROM,
                    creator_episode_id=ctx.episode_id, creator_kind="agent",
                )
            )
            linked += 1

        emit(
            ctx.session, workspace_id=ctx.workspace_id, type=EventType.ARTIFACT_CREATED,
            episode_id=ctx.episode_id, task_id=ctx.task_id, actor_kind="agent",
            payload={"artifact_id": str(artifact.id), "type": artifact_type.value,
                     "derived_from": linked},
            config_hash=ctx.config_hash,
        )
        ctx.made_progress = True
        return ToolResult(
            ok=True,
            content=f"Created {artifact_type.value} {artifact.id} ({linked} provenance links).",
            data={"artifact_id": str(artifact.id)},
        )


class ReadArtifactTool(Tool):
    name = "read_artifact"
    description = "Read one artifact in full, by id."
    read_only = True
    parameters = {
        "type": "object",
        "properties": {"artifact_id": {"type": "string"}},
        "required": ["artifact_id"],
    }

    def run(self, ctx: ToolContext, **kw: Any) -> ToolResult:
        artifact = _resolve(ctx, kw["artifact_id"])
        if artifact is None:
            return ToolResult(
                ok=False, error=f"no artifact {kw['artifact_id']!r} in this workspace"
            )

        # This is the read that earns credit (Part A §A2.1) — retrieval returning it does not.
        ctx.note_read(artifact.id)
        artifact.times_read += 1
        emit(
            ctx.session, workspace_id=ctx.workspace_id, type=EventType.ARTIFACT_READ,
            episode_id=ctx.episode_id, actor_kind="agent",
            payload={"artifact_id": str(artifact.id)}, config_hash=ctx.config_hash,
        )
        return ToolResult(ok=True, content=_render(artifact, full=True),
                          data={"artifact_id": str(artifact.id)})


class SearchKnowledgeTool(Tool):
    name = "search_knowledge"
    description = (
        "Search the collective knowledge base. Results depend on what earlier agents recorded; "
        "an empty result means nobody has worked on this yet, not that the question is new."
    )
    read_only = True
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "types": {"type": "array", "items": {"type": "string"}},
            "limit": {"type": "integer"},
        },
        "required": ["query"],
    }

    def run(self, ctx: ToolContext, **kw: Any) -> ToolResult:
        from civitas.knowledge.retrieval import retrieve

        limit = min(int(kw.get("limit", 8)), 25)
        types = None
        if kw.get("types"):
            types = []
            for raw in kw["types"]:
                try:
                    types.append(ArtifactType(raw))
                except ValueError:
                    return ToolResult(ok=False, error=f"unknown artifact type {raw!r}")

        results = retrieve(
            ctx.session,
            workspace_id=ctx.workspace_id,
            query=kw["query"],
            arm=ctx.experiment_arm,
            episode_id=ctx.episode_id,
            types=types,
            limit=limit,
            config_hash=ctx.config_hash,
        )
        if not results.artifacts:
            return ToolResult(ok=True, content="No matching artifacts.", data={"count": 0})

        lines = [f"{len(results.artifacts)} result(s):"]
        lines += [f"[{i + 1}] {_render(a)}" for i, a in enumerate(results.artifacts)]
        return ToolResult(
            ok=True,
            content="\n\n".join(lines),
            data={"count": len(results.artifacts),
                  "artifact_ids": [str(a.id) for a in results.artifacts]},
        )


class LinkArtifactsTool(Tool):
    name = "link_artifacts"
    description = (
        "Assert a typed relation between two artifacts — that one supports, contradicts, "
        "supersedes, tests or falsifies another."
    )
    parameters = {
        "type": "object",
        "properties": {
            "source_id": {"type": "string"},
            "target_id": {"type": "string"},
            "type": {"type": "string", "enum": [r.value for r in RelationType]},
            "confidence": {"type": "number"},
        },
        "required": ["source_id", "target_id", "type"],
    }

    def run(self, ctx: ToolContext, **kw: Any) -> ToolResult:
        source, target = _resolve(ctx, kw["source_id"]), _resolve(ctx, kw["target_id"])
        if source is None or target is None:
            return ToolResult(ok=False, error="both artifacts must exist in this workspace")
        if source.id == target.id:
            return ToolResult(ok=False, error="an artifact cannot relate to itself")
        try:
            relation = RelationType(kw["type"])
        except ValueError:
            return ToolResult(ok=False, error=f"unknown relation {kw['type']!r}")

        existing = ctx.session.execute(
            select(ArtifactRelation).where(
                ArtifactRelation.source_id == source.id,
                ArtifactRelation.target_id == target.id,
                ArtifactRelation.type == relation,
            )
        ).scalar_one_or_none()
        if existing is not None:
            return ToolResult(ok=True, content="That relation is already recorded.")

        ctx.session.add(
            ArtifactRelation(
                source_id=source.id, target_id=target.id, type=relation,
                confidence=float(kw.get("confidence", 1.0)),
                creator_episode_id=ctx.episode_id, creator_kind="agent",
            )
        )
        # A supersession changes the status of the artifact it replaces. Historical knowledge is
        # not overwritten (§9) — the superseded artifact stays readable and is visibly superseded.
        if relation is RelationType.SUPERSEDES:
            target.status = ArtifactStatus.SUPERSEDED
            target.superseded_by_id = source.id

        emit(
            ctx.session, workspace_id=ctx.workspace_id, type=EventType.ARTIFACT_LINKED,
            episode_id=ctx.episode_id, actor_kind="agent",
            payload={"source": str(source.id), "target": str(target.id), "type": relation.value},
            config_hash=ctx.config_hash,
        )
        ctx.made_progress = True
        return ToolResult(ok=True, content=f"Recorded: {source.id} {relation.value} {target.id}.")


class RecordFailureTool(Tool):
    name = "record_failure"
    description = (
        "Record an approach that did not work, and what it ruled out. Failures are first-class "
        "knowledge: a later agent that would have tried the same thing is shown this first."
    )
    parameters = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "approach": {"type": "string"},
            "what_happened": {"type": "string"},
            "ruled_out": {"type": "array", "items": {"type": "string"}},
            "hypothesis": {"type": "string"},
            "reproducible": {"type": "boolean"},
        },
        "required": ["title", "approach", "what_happened"],
    }

    def run(self, ctx: ToolContext, **kw: Any) -> ToolResult:
        from civitas.knowledge.duplicate import record_failure

        artifact = Artifact(
            workspace_id=ctx.workspace_id,
            project_id=ctx.project_id,
            task_id=ctx.task_id,
            type=ArtifactType.FAILURE,
            title=kw["title"][:500],
            body=kw["what_happened"],
            confidence=0.7,
            structured={
                "approach": kw["approach"],
                "ruled_out": kw.get("ruled_out") or [],
                "hypothesis": kw.get("hypothesis", ""),
                "reproducible": bool(kw.get("reproducible", False)),
            },
            creator_episode_id=ctx.episode_id,
            creator_kind="agent",
            evidence_kind=EvidenceKind.DIRECT_OBSERVATION,
            validation_state=ValidationState.SELF_REPORTED,
            environment_version=ctx.environment_version,
        )
        ctx.session.add(artifact)
        ctx.session.flush()

        record_failure(
            ctx.session,
            workspace_id=ctx.workspace_id,
            artifact_id=artifact.id,
            episode_id=ctx.episode_id,
            hypothesis=kw.get("hypothesis") or kw["approach"],
            environment_version=ctx.environment_version,
            summary=kw["what_happened"][:2000],
            ruled_out=kw.get("ruled_out") or [],
            reproducible=bool(kw.get("reproducible", False)),
        )
        emit(
            ctx.session, workspace_id=ctx.workspace_id, type=EventType.ARTIFACT_CREATED,
            episode_id=ctx.episode_id, actor_kind="agent",
            payload={"artifact_id": str(artifact.id), "type": "failure"},
            config_hash=ctx.config_hash,
        )
        ctx.made_progress = True
        return ToolResult(ok=True, content=f"Recorded failure {artifact.id}.",
                          data={"artifact_id": str(artifact.id)})


class SubmitResultTool(Tool):
    name = "submit_result"
    description = (
        "Submit your answer for this task. Ends your episode. You do not decide whether it is "
        "correct — an external evaluator does."
    )
    parameters = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "reasoning": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["answer"],
    }

    def run(self, ctx: ToolContext, **kw: Any) -> ToolResult:
        artifact = Artifact(
            workspace_id=ctx.workspace_id,
            project_id=ctx.project_id,
            task_id=ctx.task_id,
            type=ArtifactType.RESULT,
            title=f"Result for task {ctx.task_id}",
            body=kw["answer"],
            confidence=float(kw.get("confidence", 0.5)),
            structured={"reasoning": kw.get("reasoning", ""), "answer": kw["answer"]},
            creator_episode_id=ctx.episode_id,
            creator_kind="agent",
            evidence_kind=EvidenceKind.MODEL_ASSERTION,
            # Not `evaluator_confirmed`, whatever the agent claims: agents cannot self-certify
            # (Part B §47). The evaluator writes that state, and only after checking.
            validation_state=ValidationState.SELF_REPORTED,
            environment_version=ctx.environment_version,
        )
        ctx.session.add(artifact)
        ctx.session.flush()
        ctx.scratch["submitted_artifact_id"] = artifact.id
        ctx.scratch["submitted_answer"] = kw["answer"]
        ctx.made_progress = True
        emit(
            ctx.session, workspace_id=ctx.workspace_id, type=EventType.ARTIFACT_CREATED,
            episode_id=ctx.episode_id, task_id=ctx.task_id, actor_kind="agent",
            payload={"artifact_id": str(artifact.id), "type": "result"},
            config_hash=ctx.config_hash,
        )
        return ToolResult(ok=True, content="Result submitted.",
                          data={"artifact_id": str(artifact.id)})


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _resolve(ctx: ToolContext, raw: str) -> Artifact | None:
    """Look up an artifact by id, scoped to the episode's workspace.

    The workspace scoping is the security boundary, not a convenience: without it an agent that
    guessed or was told an id could read across workspaces, and every ablation arm would leak.
    """
    try:
        artifact_id = uuid.UUID(str(raw))
    except (ValueError, AttributeError, TypeError):
        return None
    artifact = ctx.session.get(Artifact, artifact_id)
    if artifact is None or artifact.workspace_id != ctx.workspace_id:
        return None
    return artifact


def _render(artifact: Artifact, *, full: bool = False) -> str:
    head = (
        f"id={artifact.id} type={artifact.type.value} "
        f"status={artifact.status.value} validation={artifact.validation_state.value} "
        f"confidence={artifact.confidence:.2f}"
    )
    if artifact.is_stale:
        head += " STALE"
    body = artifact.body if full else artifact.body[:400]
    return f"{head}\n{artifact.title}\n{body}"


def default_registry():
    """The tools every episode gets unless its profile's policy narrows the set."""
    from civitas.runtime.tools.base import ToolRegistry

    return ToolRegistry([
        SearchKnowledgeTool(),
        ReadArtifactTool(),
        CreateArtifactTool(),
        LinkArtifactsTool(),
        RecordFailureTool(),
        SubmitResultTool(),
    ])
