"""Request and response models for the API (Part B §50).

Pydantic models rather than raw ORM serialisation, for one reason that matters beyond tidiness:
the API is a *disclosure surface*. An evaluator's expected answer, a task's evaluator spec and a
policy's secret configuration must not leave the process merely because they are columns on a row
someone asked for (§47, §40). Explicit schemas make each field a decision.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# --------------------------------------------------------------------------
class OrganizationOut(ORMModel):
    id: uuid.UUID
    name: str
    slug: str
    created_at: datetime


class WorkspaceIn(BaseModel):
    name: str
    slug: str
    description: str = ""
    environment_version: str = "dev"


class WorkspaceOut(ORMModel):
    id: uuid.UUID
    organization_id: uuid.UUID
    name: str
    slug: str
    description: str
    environment_version: str
    created_at: datetime


class ProjectIn(BaseModel):
    name: str
    description: str = ""
    priority: int = 0


class ProjectOut(ORMModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    description: str
    status: str
    priority: int
    created_at: datetime


class RequestIn(BaseModel):
    """A high-level request (§5). The system answers it with a plan, not with an answer."""

    request: str = Field(min_length=8, max_length=8000)
    project_name: str | None = None


class PlanOut(BaseModel):
    project_id: uuid.UUID
    objective_id: uuid.UUID
    decomposer: str
    tasks: list[dict]


class ProjectStatusOut(BaseModel):
    """A project's state, reconstructed from rows (§32).

    `progress` is `None` rather than 0.0 for a project with no tasks: a project that has not been
    decomposed has no progress, and reporting zero would state that no work has been done when
    what is true is that there is nothing yet to do (ARCHITECTURE §3.9).
    """

    project_id: uuid.UUID
    name: str
    tasks_total: int
    tasks_completed: int
    tasks_blocked: int
    episodes: int
    episodes_succeeded: int
    artifacts: int
    validated_artifacts: int
    open_questions: int
    tokens_used: int
    cost_usd: float
    stages: dict[str, str]
    complete: bool
    progress: float | None


class OverviewOut(BaseModel):
    """The collective at a glance (§49 screen 1).

    Every count is of *live* rows. Archived artifacts are reported separately rather than folded
    in: `memory_reset` archives rather than deletes (Part A §A1.2), and a total that hid the
    distinction would make an arm that lost its memory look like one that never had any.
    """

    workspace_id: uuid.UUID
    name: str
    environment_version: str
    projects: int
    tasks: int
    tasks_ready: int
    tasks_blocked: int
    episodes: int
    episodes_succeeded: int
    agents: int
    artifacts: int
    artifacts_archived: int
    artifacts_stale: int
    tools: int
    events: int
    latest_sequence: int
    tokens_used: int
    cost_usd: float


class CostBucketOut(BaseModel):
    key: str
    episodes: int
    tokens: int
    cost_usd: float
    #: `None` where no episode in the bucket recorded a cost, rather than 0.0 — a provider with no
    #: priced calls has an unknown cost per episode, not a free one (ARCHITECTURE §3.9).
    mean_cost_usd: float | None


class CostOut(BaseModel):
    """§49's cost dashboard, and §39's budget accounting."""

    workspace_id: uuid.UUID
    total_tokens: int
    total_cost_usd: float
    episodes: int
    by_day: list[CostBucketOut]
    by_model: list[CostBucketOut]
    by_arm: list[CostBucketOut]
    #: Episodes whose provider reported no cost at all. Named so a dashboard can say how much of
    #: the total is unpriced rather than presenting a partial sum as the whole.
    unpriced_episodes: int


class EpisodeDetailOut(ORMModel):
    """One episode, as the API discloses it.

    There is no reasoning field, and there is none to add: §4 forbids an episode's private
    reasoning from being persisted at all, so the inspector shows what the episode *did* — its
    configuration, its budget, its tool runs, and what it wrote — never what it thought.
    """

    id: uuid.UUID
    workspace_id: uuid.UUID
    task_id: uuid.UUID | None
    project_id: uuid.UUID | None
    agent_profile_id: uuid.UUID
    model_provider: str
    model_name: str
    model_version: str
    system_prompt_version: str
    retrieval_policy_version: str
    experiment_arm: str
    environment_version: str
    config_hash: str
    is_benchmark_probe: bool
    termination_reason: str | None
    tokens_used: int
    tool_calls_used: int
    cost_usd: float
    duration_s: float | None
    artifacts_created: int
    artifacts_read: int
    duplicate_failures: int
    created_at: datetime
    tool_runs: list[dict] = Field(default_factory=list)
    artifacts: list[dict] = Field(default_factory=list)
    evaluations: list[dict] = Field(default_factory=list)


class SpecializationOut(BaseModel):
    """§49's specialization dashboard. `specialization_index` is `None` with a stated reason when
    there is not enough evidence to compute it — never 0.0, which would read as "no
    specialization" rather than "not measurable"."""

    workspace_id: uuid.UUID
    specialization_index: float | None
    index_unavailable_reason: str | None
    entries: list[dict]


class InstitutionsOut(BaseModel):
    """§29–§31: what the A2.2 gate has approved, and what each agent is trusted for."""

    workspace_id: uuid.UUID
    gate: dict
    reputations: list[dict]
    policies: list[dict]


class DomainOut(BaseModel):
    """A registered domain (§5). What this process can actually run, not a catalogue."""

    name: str
    version: str
    stages: list[str]
    tools: list[str]
    evaluators: list[str]
    artifact_vocabulary: list[str]


class TaskIn(BaseModel):
    title: str
    description: str = ""
    task_family: str = ""
    priority: int = 0
    difficulty: float = 0.0
    project_id: uuid.UUID | None = None


class TaskOut(ORMModel):
    """A task as the API discloses it.

    `evaluator_spec` is **deliberately absent**. It carries the expected answer, and §47 forbids
    an agent from reaching its own success criteria — an API that returns it makes the whole
    evaluation self-certifying for anyone who can call the endpoint.
    """

    id: uuid.UUID
    workspace_id: uuid.UUID
    project_id: uuid.UUID | None
    title: str
    description: str
    task_family: str
    status: str
    priority: int
    difficulty: float
    attempts: int
    max_attempts: int
    created_at: datetime
    completed_at: datetime | None


class EpisodeOut(ORMModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    task_id: uuid.UUID | None
    agent_profile_id: uuid.UUID
    model_provider: str
    model_name: str
    experiment_arm: str
    status: str
    termination_reason: str | None
    tokens_used: int
    tool_calls_used: int
    cost_usd: float
    artifacts_created: int
    artifacts_read: int
    duplicate_failures: int
    started_at: datetime | None
    ended_at: datetime | None
    config_hash: str
    is_benchmark_probe: bool


class ArtifactIn(BaseModel):
    type: str
    title: str
    body: str = ""
    confidence: float = 0.5
    structured: dict[str, Any] = Field(default_factory=dict)
    project_id: uuid.UUID | None = None
    task_id: uuid.UUID | None = None


class ArtifactOut(ORMModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    type: str
    title: str
    body: str
    confidence: float
    validation_state: str
    status: str
    evidence_kind: str
    version: int
    is_stale: bool
    environment_version: str
    downstream_utility: float
    times_retrieved: int
    times_read: int
    created_at: datetime
    creator_episode_id: uuid.UUID | None
    superseded_by_id: uuid.UUID | None


class RelationIn(BaseModel):
    source_id: uuid.UUID
    target_id: uuid.UUID
    type: str
    confidence: float = 1.0


class RelationOut(ORMModel):
    id: uuid.UUID
    source_id: uuid.UUID
    target_id: uuid.UUID
    type: str
    confidence: float
    evidence_kind: str
    created_at: datetime


class ProvenanceOut(BaseModel):
    root: dict[str, Any]
    depth: int
    grounding_score: float
    strongest_evidence: str | None
    truncated: bool
    nodes: list[dict[str, Any]]
    challenges: list[dict[str, Any]]


class RetrievalIn(BaseModel):
    query: str
    arm: str = "collective"
    limit: int = 8
    types: list[str] | None = None


class RetrievalHit(BaseModel):
    artifact: ArtifactOut
    score: float
    features: dict[str, float]


class RetrievalOut(BaseModel):
    decision_id: uuid.UUID | None
    candidate_count: int
    returned_count: int
    provenance_hidden: bool
    suppressed: list[dict[str, Any]]
    results: list[RetrievalHit]


class ToolOut(ORMModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    description: str
    kind: str
    times_used: int
    times_succeeded: int
    times_failed: int
    downstream_utility: float
    created_at: datetime


class EventOut(ORMModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    sequence: int
    type: str
    episode_id: uuid.UUID | None
    task_id: uuid.UUID | None
    actor_kind: str
    payload: dict[str, Any]
    created_at: datetime


class EvaluationOut(ORMModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    episode_id: uuid.UUID | None
    scope: str
    evaluator_kind: str
    succeeded: bool
    score: float
    max_score: float
    created_at: datetime


class JobOut(ORMModel):
    id: uuid.UUID
    kind: str
    status: str
    attempts: int
    max_attempts: int
    leased_by: str | None
    created_at: datetime
    completed_at: datetime | None
    error: str


class MetricsOut(BaseModel):
    workspace_id: uuid.UUID
    include_probes: bool
    metrics: dict[str, Any]
    unavailable: list[str]


class ExperimentOut(ORMModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    kind: str
    status: str
    hypothesis: str
    prediction: str
    created_at: datetime


class ExperimentRunOut(ORMModel):
    id: uuid.UUID
    experiment_id: uuid.UUID
    experiment_arm_id: uuid.UUID
    seed: int
    status: str
    config_hash: str
    gates: dict[str, Any]
    metrics: dict[str, Any]
    created_at: datetime


class BenchmarkResultOut(BaseModel):
    """A benchmark result, with the gate discipline preserved across the wire.

    When a gate fails the metrics are **absent**, not zeroed, and `gates_passed` says why. A
    client that renders a withheld number as zero would undo the whole point of the gate
    (ARCHITECTURE §3.1), so there is nothing there to render.
    """

    experiment: str
    config_hash: str
    gates_passed: bool
    gates: dict[str, Any]
    failed_gates: list[str]
    metrics: dict[str, Any] | None = None
    withheld_reason: str | None = None


class HealthOut(BaseModel):
    status: str
    dialect: str
    schema_version: str
    app_version: str
    sandbox_backend: str
    sandbox_unenforced_limits: list[str]
