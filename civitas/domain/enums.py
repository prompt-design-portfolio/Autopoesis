"""Domain enumerations (Part B §8, §9, §10, §21, §28, §45).

These are closed sets. A value not listed here is a bug, not an extension point: the experimental
arms and the termination reasons in particular are machine-enforced (§21), and an open set would
make that unenforceable.
"""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    def __str__(self) -> str:  # pragma: no cover - display only
        return self.value


# --------------------------------------------------------------------------
# episodes (Part B §8)
# --------------------------------------------------------------------------
class TerminationReason(StrEnum):
    SOLVED = "solved"
    EVALUATOR_SUCCESS = "evaluator_success"
    EVALUATOR_FAILURE = "evaluator_failure"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_PROGRESS = "no_progress"
    BLOCKED = "blocked"
    DELEGATED = "delegated"
    FAILED = "failed"
    PROVIDER_FAILURE = "provider_failure"
    TOOL_FAILURE = "tool_failure"
    WORKER_FAILURE = "worker_failure"
    POLICY_VIOLATION = "policy_violation"
    CANCELLED = "cancelled"


class EpisodeStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    TERMINATED = "terminated"


# --------------------------------------------------------------------------
# artifacts (Part B §9)
# --------------------------------------------------------------------------
class ArtifactType(StrEnum):
    HYPOTHESIS = "hypothesis"
    OBSERVATION = "observation"
    EVIDENCE = "evidence"
    EXPERIMENT = "experiment"
    EXPERIMENT_RESULT = "experiment_result"
    FAILURE = "failure"
    WARNING = "warning"
    QUESTION = "question"
    ANSWER = "answer"
    RESULT = "result"
    CONCLUSION = "conclusion"
    DECISION = "decision"
    PLAN = "plan"
    TASK = "task"
    SUBTASK = "subtask"
    SUMMARY = "summary"
    CONTRADICTION = "contradiction"
    CORRECTION = "correction"
    INSIGHT = "insight"
    ASSUMPTION = "assumption"
    THEORY = "theory"
    ABSTRACTION = "abstraction"
    TOOL = "tool"
    CODE_PATCH = "code_patch"
    DATASET = "dataset"
    BENCHMARK = "benchmark"
    DOCUMENT = "document"
    CITATION = "citation"
    PROCEDURE = "procedure"
    WORKFLOW = "workflow"
    POLICY = "policy"
    UNRESOLVED_ISSUE = "unresolved_issue"


#: Negative knowledge (Part B §12). Removed wholesale by the `collective_no_negative` arm, so the
#: set has to be nameable in one place rather than inferred at each call site.
NEGATIVE_TYPES: frozenset[ArtifactType] = frozenset(
    {
        ArtifactType.FAILURE,
        ArtifactType.WARNING,
        ArtifactType.CONTRADICTION,
        ArtifactType.CORRECTION,
        ArtifactType.UNRESOLVED_ISSUE,
    }
)


class ValidationState(StrEnum):
    """How much the claim has actually been checked (Part B §11).

    `UNVALIDATED` is the default and it matters: a statement created by a model is not verified
    merely because it exists.
    """

    UNVALIDATED = "unvalidated"
    SELF_REPORTED = "self_reported"
    PEER_REVIEWED = "peer_reviewed"
    TOOL_VERIFIED = "tool_verified"
    REPRODUCED = "reproduced"
    EVALUATOR_CONFIRMED = "evaluator_confirmed"
    HUMAN_CONFIRMED = "human_confirmed"
    DISPUTED = "disputed"
    REFUTED = "refuted"


class ArtifactStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    STALE = "stale"
    ARCHIVED = "archived"
    RETRACTED = "retracted"


class Visibility(StrEnum):
    WORKSPACE = "workspace"
    PROJECT = "project"
    ORGANIZATION = "organization"
    PRIVATE = "private"


class EvidenceKind(StrEnum):
    """Provenance strength (Part B §13). Ordered weakest to strongest; retrieval ranks on it."""

    MODEL_ASSERTION = "model_assertion"
    INFERENCE = "inference"
    EXTERNAL_CITATION = "external_citation"
    DIRECT_OBSERVATION = "direct_observation"
    TOOL_OUTPUT = "tool_output"
    REPRODUCED_RESULT = "reproduced_result"
    EVALUATOR_CONFIRMED = "evaluator_confirmed"
    HUMAN_CONFIRMED = "human_confirmed"


#: Numeric strength for ranking. The gaps are deliberate: a tool output outranks any amount of
#: model assertion, and no quantity of assertions sums to an observation.
EVIDENCE_STRENGTH: dict[EvidenceKind, float] = {
    EvidenceKind.MODEL_ASSERTION: 0.10,
    EvidenceKind.INFERENCE: 0.25,
    EvidenceKind.EXTERNAL_CITATION: 0.45,
    EvidenceKind.DIRECT_OBSERVATION: 0.60,
    EvidenceKind.TOOL_OUTPUT: 0.75,
    EvidenceKind.REPRODUCED_RESULT: 0.90,
    EvidenceKind.EVALUATOR_CONFIRMED: 0.95,
    EvidenceKind.HUMAN_CONFIRMED: 1.00,
}


# --------------------------------------------------------------------------
# artifact graph (Part B §10)
# --------------------------------------------------------------------------
class RelationType(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"
    DEPENDS_ON = "depends_on"
    DERIVED_FROM = "derived_from"
    TESTS = "tests"
    FALSIFIES = "falsifies"
    CONFIRMS = "confirms"
    CREATED_BY = "created_by"
    USES = "uses"
    ANSWERS = "answers"
    BLOCKS = "blocks"
    RESOLVES = "resolves"
    DUPLICATES = "duplicates"
    REFINES = "refines"
    GENERALIZES = "generalizes"
    SPECIALIZES = "specializes"
    CITES = "cites"
    REPRODUCES = "reproduces"
    FAILED_BECAUSE = "failed_because"
    ENABLED = "enabled"
    VALIDATES = "validates"
    INVALIDATES = "invalidates"


#: Edges credit propagates backward along (Part A §A2.1). Not every relation carries credit: an
#: artifact that merely *contradicts* a successful one did not contribute to the success.
CREDIT_EDGES: frozenset[RelationType] = frozenset(
    {
        RelationType.DERIVED_FROM,
        RelationType.DEPENDS_ON,
        RelationType.USES,
        RelationType.SUPPORTS,
        RelationType.ENABLED,
        RelationType.CITES,
        RelationType.REFINES,
    }
)


# --------------------------------------------------------------------------
# hypotheses (Part B §28)
# --------------------------------------------------------------------------
class HypothesisState(StrEnum):
    PROPOSED = "proposed"
    TESTED = "tested"
    SUPPORTED = "supported"
    CHALLENGED = "challenged"
    REPLICATED = "replicated"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


# --------------------------------------------------------------------------
# experimental arms (Part B §21) — machine-enforced, never prompt convention
# --------------------------------------------------------------------------
class ExperimentArm(StrEnum):
    SOLO = "solo"
    INDEPENDENT = "independent"
    SHARED_MEMORY = "shared_memory"
    COLLECTIVE = "collective"
    COLLECTIVE_SCRAMBLED = "collective_scrambled"
    COLLECTIVE_NO_NEGATIVE = "collective_no_negative"
    COLLECTIVE_NO_PROVENANCE = "collective_no_provenance"
    COLLECTIVE_FROZEN = "collective_frozen"
    MEMORY_RESET = "memory_reset"


# --------------------------------------------------------------------------
# tasks and scheduling (Part B §27)
# --------------------------------------------------------------------------
class TaskStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AllocationStrategy(StrEnum):
    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    PRIORITY = "priority"
    UNCERTAINTY = "uncertainty"
    INFORMATION_GAIN = "information_gain"
    SPECIALIZATION = "specialization"
    DIVERSITY = "diversity"
    ADVERSARIAL = "adversarial"
    COST_AWARE = "cost_aware"


class AgentRole(StrEnum):
    """Seed profiles (Part B §26). Not an exhaustive or permanent taxonomy — profiles are rows,
    and which role suits which work type is measured, not declared."""

    EXPLORER = "explorer"
    EXPERIMENTER = "experimenter"
    VERIFIER = "verifier"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"
    CODER = "coder"
    DEBUGGER = "debugger"
    MATHEMATICIAN = "mathematician"
    RESEARCHER = "researcher"
    PLANNER = "planner"
    TOOLMAKER = "toolmaker"
    EVALUATOR = "evaluator"
    COORDINATOR = "coordinator"


# --------------------------------------------------------------------------
# jobs (Part B §42)
# --------------------------------------------------------------------------
class JobStatus(StrEnum):
    QUEUED = "queued"
    LEASED = "leased"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"
    CANCELLED = "cancelled"


# --------------------------------------------------------------------------
# events (Part B §45)
# --------------------------------------------------------------------------
class EventType(StrEnum):
    TASK_CREATED = "task_created"
    TASK_COMPLETED = "task_completed"
    ASSIGNMENT_CREATED = "assignment_created"
    EPISODE_STARTED = "episode_started"
    EPISODE_TERMINATED = "episode_terminated"
    ARTIFACT_CREATED = "artifact_created"
    ARTIFACT_READ = "artifact_read"
    ARTIFACT_VERSIONED = "artifact_versioned"
    ARTIFACT_LINKED = "artifact_linked"
    ARTIFACT_ARCHIVED = "artifact_archived"
    RETRIEVAL_PERFORMED = "retrieval_performed"
    MODEL_CALLED = "model_called"
    TOOL_CALLED = "tool_called"
    TOOL_COMPLETED = "tool_completed"
    TOOL_CREATED = "tool_created"
    HYPOTHESIS_CHALLENGED = "hypothesis_challenged"
    EVALUATION_COMPLETED = "evaluation_completed"
    POLICY_CHANGED = "policy_changed"
    PROCEDURE_CHANGED = "procedure_changed"
    CREDIT_ASSIGNED = "credit_assigned"
    DUPLICATE_FAILURE = "duplicate_failure"
    REPUTATION_UPDATED = "reputation_updated"
    EXPERIMENT_STARTED = "experiment_started"
    EXPERIMENT_COMPLETED = "experiment_completed"
    VERSION_APPROVED = "version_approved"
    VERSION_ROLLED_BACK = "version_rolled_back"
    JOB_LEASED = "job_leased"
    JOB_COMPLETED = "job_completed"
    JOB_DEAD_LETTERED = "job_dead_lettered"
    CONSOLIDATION_PERFORMED = "consolidation_performed"
    STALENESS_MARKED = "staleness_marked"


# --------------------------------------------------------------------------
# actors and access (Part B §51)
# --------------------------------------------------------------------------
class ActorKind(StrEnum):
    USER = "user"
    SERVICE = "service"
    AGENT = "agent"
    SYSTEM = "system"


class Role(StrEnum):
    ADMIN = "admin"
    RESEARCHER = "researcher"
    OPERATOR = "operator"
    VIEWER = "viewer"
    AGENT = "agent"


class ToolValidationState(StrEnum):
    UNTESTED = "untested"
    TESTS_PASSING = "tests_passing"
    TESTS_FAILING = "tests_failing"
    VALIDATED = "validated"
    DEPRECATED = "deprecated"
    QUARANTINED = "quarantined"
