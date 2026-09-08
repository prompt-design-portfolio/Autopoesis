"""Every mapped class, imported so Alembic's autogenerate sees a complete metadata graph.

Part B §7 names 37 entities. The registry below is asserted against that list by
`tests/test_domain_model.py`, so an entity cannot be dropped without a test failing.
"""

from civitas.persistence.base import Base
from civitas.persistence.models.experiments import (
    Evaluation,
    Experiment,
    ExperimentArm,
    ExperimentManifest,
    ExperimentRun,
)
from civitas.persistence.models.infra import Event, EventSequence, Heartbeat, Job, JobLease
from civitas.persistence.models.institutions import (
    DuplicateFailureRecord,
    Policy,
    Procedure,
    PromptVersion,
    ReputationMetric,
)
from civitas.persistence.models.knowledge import (
    Artifact,
    ArtifactEmbedding,
    ArtifactRelation,
    ArtifactUsage,
    ArtifactUtilityMetric,
    ArtifactVersion,
    FileAsset,
    RetrievalDecision,
)
from civitas.persistence.models.org import (
    ApiKey,
    Objective,
    Organization,
    Project,
    ServiceIdentity,
    User,
    Workspace,
    WorkspaceSnapshot,
)
from civitas.persistence.models.providers import ModelCall, ModelConfiguration, ModelProvider
from civitas.persistence.models.tools import ToolDefinition, ToolRun, ToolVersion
from civitas.persistence.models.work import (
    AgentInstance,
    AgentProfile,
    Assignment,
    Episode,
    Subproblem,
    Task,
    TaskDependency,
)

__all__ = [
    "Base",
    "AgentInstance",
    "AgentProfile",
    "ApiKey",
    "Artifact",
    "ArtifactEmbedding",
    "ArtifactRelation",
    "ArtifactUsage",
    "ArtifactUtilityMetric",
    "ArtifactVersion",
    "Assignment",
    "DuplicateFailureRecord",
    "Episode",
    "Evaluation",
    "Event",
    "EventSequence",
    "Experiment",
    "ExperimentArm",
    "ExperimentManifest",
    "ExperimentRun",
    "FileAsset",
    "Heartbeat",
    "Job",
    "JobLease",
    "ModelCall",
    "ModelConfiguration",
    "ModelProvider",
    "Objective",
    "Organization",
    "Policy",
    "Procedure",
    "Project",
    "PromptVersion",
    "ReputationMetric",
    "RetrievalDecision",
    "ServiceIdentity",
    "Subproblem",
    "Task",
    "TaskDependency",
    "ToolDefinition",
    "ToolRun",
    "ToolVersion",
    "User",
    "Workspace",
    "WorkspaceSnapshot",
]
