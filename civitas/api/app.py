"""The Civitas API (Part B §50), with authentication (§51) and OpenAPI documentation.

Every route §50 requires, plus the ones the research surface needs — provenance, retrieval,
benchmark results, gate status. Three properties are enforced here rather than left to callers:

* **Workspace scoping is the security boundary.** Every read resolves its workspace against the
  principal's organization first. Without that, a guessed id reads across tenants and every
  ablation arm leaks.
* **The evaluator specification never leaves the process** (§47). `TaskOut` has no field for it.
* **A withheld benchmark number is absent, not zero** (ARCHITECTURE §3.1). There is nothing for a
  client to render as a result the run did not earn.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from civitas.api import schemas
from civitas.api.security import (
    Principal,
    principal_for,
    require_role,
    require_scope,
    verify_api_key,
)
from civitas.config import Settings, get_settings
from civitas.domain.enums import ArtifactType, EventType, ExperimentArm, RelationType, Role
from civitas.persistence.engine import get_session_factory, healthcheck
from civitas.persistence.events import emit
from civitas.persistence.models import (
    AgentProfile,
    Artifact,
    ArtifactRelation,
    Episode,
    Evaluation,
    Experiment,
    ExperimentRun,
    Job,
    Organization,
    Project,
    Task,
    ToolDefinition,
    Workspace,
)

API_PREFIX = "/api/v1"


def get_db(settings: Settings = Depends(get_settings)) -> Iterator[Session]:
    factory = get_session_factory(settings)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def resolve_principal(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> Principal:
    """Authenticate a request (§51).

    With `auth_enabled` false — the Colab and local-research configuration — an admin principal is
    synthesised. That is a *deployment choice*, and it is reported by `/healthz` so an
    unauthenticated deployment is visible rather than assumed.
    """
    if not settings.auth_enabled:
        org = db.execute(select(Organization).limit(1)).scalar_one_or_none()
        if org is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                                "no organization exists; bootstrap one first")
        from civitas.domain.enums import ActorKind

        return Principal(kind=ActorKind.SYSTEM, id=org.id, organization_id=org.id,
                         role=Role.ADMIN, scopes=frozenset(), display="auth-disabled")

    token = x_api_key
    if token is None and authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:]
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "an API key is required")

    key = verify_api_key(db, token)
    if key is None:
        # One message for "no such key" and "wrong key", so the response cannot be used to
        # enumerate valid prefixes (§52).
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid API key")
    return principal_for(db, key)


def scoped_workspace(
    workspace_id: uuid.UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(resolve_principal),
) -> Workspace:
    """Resolve a workspace **within the principal's organization**.

    A 404 rather than a 403 for a workspace in another organization: telling a caller that an id
    exists but is not theirs is itself a disclosure (§52).
    """
    workspace = db.get(Workspace, workspace_id)
    if workspace is None or workspace.organization_id != principal.organization_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such workspace")
    return workspace


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="Civitas",
        version=settings.app_version,
        description=(
            "A persistent artificial civilization for cumulative collective intelligence. "
            "Benchmark results follow the gate discipline: when a prerequisite gate fails the "
            "metrics are absent rather than zeroed."
        ),
        openapi_url=f"{API_PREFIX}/openapi.json",
        docs_url=f"{API_PREFIX}/docs",
    )
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=settings.cors_origins,
            allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
        )
    app.dependency_overrides[get_settings] = lambda: settings

    # ----------------------------------------------------------------------
    @app.get("/healthz", response_model=schemas.HealthOut, tags=["ops"])
    def healthz(settings: Settings = Depends(get_settings)) -> schemas.HealthOut:
        from civitas.experiments.manifest import schema_version
        from civitas.runtime.sandbox import best_available

        health = healthcheck(settings)
        sandbox = best_available()
        return schemas.HealthOut(
            status="ok" if health["ok"] else "degraded",
            dialect=health["dialect"],
            schema_version=schema_version(),
            app_version=settings.app_version,
            sandbox_backend=sandbox.name,
            sandbox_unenforced_limits=list(sandbox.unenforced_limits()),
        )

    @app.get("/readyz", tags=["ops"])
    def readyz(db: Session = Depends(get_db)) -> dict[str, Any]:
        """Readiness (§54). Distinct from liveness: this one touches the database."""
        db.execute(select(Organization).limit(1))
        return {"ready": True}

    # ----------------------------------------------------------------------
    @app.get(f"{API_PREFIX}/organizations", response_model=list[schemas.OrganizationOut],
             tags=["organizations"])
    def list_organizations(
        db: Session = Depends(get_db),
        principal: Principal = Depends(require_role(Role.VIEWER)),
    ):
        return db.execute(
            select(Organization).where(Organization.id == principal.organization_id)
        ).scalars().all()

    @app.get(f"{API_PREFIX}/workspaces", response_model=list[schemas.WorkspaceOut],
             tags=["workspaces"])
    def list_workspaces(
        db: Session = Depends(get_db),
        principal: Principal = Depends(require_role(Role.VIEWER)),
    ):
        return db.execute(
            select(Workspace).where(
                Workspace.organization_id == principal.organization_id,
                Workspace.archived_at.is_(None),
            )
        ).scalars().all()

    @app.post(f"{API_PREFIX}/workspaces", response_model=schemas.WorkspaceOut,
              status_code=201, tags=["workspaces"])
    def create_workspace(
        payload: schemas.WorkspaceIn,
        db: Session = Depends(get_db),
        principal: Principal = Depends(require_role(Role.RESEARCHER)),
    ):
        workspace = Workspace(
            organization_id=principal.organization_id, name=payload.name, slug=payload.slug,
            description=payload.description, environment_version=payload.environment_version,
        )
        db.add(workspace)
        db.flush()
        return workspace

    @app.get(f"{API_PREFIX}/workspaces/{{workspace_id}}", response_model=schemas.WorkspaceOut,
             tags=["workspaces"])
    def get_workspace(workspace: Workspace = Depends(scoped_workspace)):
        return workspace

    # ----------------------------------------------------------------------
    @app.get(f"{API_PREFIX}/workspaces/{{workspace_id}}/projects",
             response_model=list[schemas.ProjectOut], tags=["projects"])
    def list_projects(workspace: Workspace = Depends(scoped_workspace),
                      db: Session = Depends(get_db)):
        return db.execute(
            select(Project).where(Project.workspace_id == workspace.id,
                                  Project.archived_at.is_(None))
        ).scalars().all()

    @app.post(f"{API_PREFIX}/workspaces/{{workspace_id}}/projects",
              response_model=schemas.ProjectOut, status_code=201, tags=["projects"])
    def create_project(
        payload: schemas.ProjectIn,
        workspace: Workspace = Depends(scoped_workspace),
        db: Session = Depends(get_db),
        principal: Principal = Depends(require_role(Role.RESEARCHER)),
    ):
        project = Project(workspace_id=workspace.id, name=payload.name,
                          description=payload.description, priority=payload.priority)
        db.add(project)
        db.flush()
        return project

    @app.post(f"{API_PREFIX}/workspaces/{{workspace_id}}/requests",
              response_model=schemas.PlanOut, status_code=201, tags=["projects"])
    def submit_high_level_request(
        payload: schemas.RequestIn,
        workspace: Workspace = Depends(scoped_workspace),
        db: Session = Depends(get_db),
        principal: Principal = Depends(require_role(Role.RESEARCHER)),
        settings: Settings = Depends(get_settings),
    ):
        """§5: a request becomes a project and a chain of tasks, not an answer.

        Deliberately a separate route from `POST /projects`. Creating a project is a bookkeeping
        operation; submitting a request is the system committing to a decomposition, and the two
        returning the same shape would hide which of them happened.
        """
        from civitas.services.projects import submit_request

        plan = submit_request(
            db, workspace_id=workspace.id, request=payload.request,
            project_name=payload.project_name, config_hash=settings.config_hash(),
        )
        return schemas.PlanOut(**plan.as_dict())

    @app.get(f"{API_PREFIX}/workspaces/{{workspace_id}}/projects/{{project_id}}/status",
             response_model=schemas.ProjectStatusOut, tags=["projects"])
    def get_project_status(
        project_id: uuid.UUID,
        workspace: Workspace = Depends(scoped_workspace),
        db: Session = Depends(get_db),
    ):
        from civitas.services.projects import project_status

        project = db.get(Project, project_id)
        if project is None or project.workspace_id != workspace.id:
            raise HTTPException(status_code=404, detail="project not found")
        return schemas.ProjectStatusOut(**project_status(db, project_id).as_dict())

    @app.get(f"{API_PREFIX}/workspaces/{{workspace_id}}/projects/{{project_id}}/timeline",
             tags=["projects"])
    def get_project_timeline(
        project_id: uuid.UUID,
        workspace: Workspace = Depends(scoped_workspace),
        db: Session = Depends(get_db),
        limit: int = Query(default=200, le=1000),
    ):
        from civitas.services.projects import project_timeline

        project = db.get(Project, project_id)
        if project is None or project.workspace_id != workspace.id:
            raise HTTPException(status_code=404, detail="project not found")
        return project_timeline(db, project_id, limit=limit)

    # ----------------------------------------------------------------------
    @app.get(f"{API_PREFIX}/domains", response_model=list[schemas.DomainOut], tags=["domains"])
    def list_domains():
        """The domains this process can run (§5). Unscoped: a domain is code, not tenant data."""
        from civitas.domains import all_domains

        return [
            schemas.DomainOut(
                name=d.name, version=d.version,
                stages=[st.family for st in d.stages()],
                tools=list(d.tool_names()),
                evaluators=[e.kind for e in d.evaluators()],
                artifact_vocabulary=[a.value for a in d.artifact_vocabulary()],
            )
            for d in all_domains()
        ]

    # ----------------------------------------------------------------------
    @app.get(f"{API_PREFIX}/workspaces/{{workspace_id}}/tasks",
             response_model=list[schemas.TaskOut], tags=["tasks"])
    def list_tasks(
        workspace: Workspace = Depends(scoped_workspace),
        db: Session = Depends(get_db),
        status_filter: str | None = Query(default=None, alias="status"),
        limit: int = Query(default=100, le=500),
    ):
        stmt = select(Task).where(Task.workspace_id == workspace.id,
                                  Task.archived_at.is_(None))
        if status_filter:
            stmt = stmt.where(Task.status == status_filter)
        return db.execute(stmt.order_by(Task.created_at.desc()).limit(limit)).scalars().all()

    @app.post(f"{API_PREFIX}/workspaces/{{workspace_id}}/tasks",
              response_model=schemas.TaskOut, status_code=201, tags=["tasks"])
    def create_task(
        payload: schemas.TaskIn,
        workspace: Workspace = Depends(scoped_workspace),
        db: Session = Depends(get_db),
        principal: Principal = Depends(require_role(Role.RESEARCHER)),
    ):
        task = Task(
            workspace_id=workspace.id, project_id=payload.project_id, title=payload.title,
            description=payload.description, task_family=payload.task_family,
            priority=payload.priority, difficulty=payload.difficulty,
        )
        db.add(task)
        db.flush()
        emit(
            db, workspace_id=workspace.id, type=EventType.TASK_CREATED, task_id=task.id,
            actor_kind=principal.kind.value, actor_id=principal.id,
            payload={"task_id": str(task.id), "title": task.title, "via": "api"},
        )
        return task

    # ----------------------------------------------------------------------
    @app.get(f"{API_PREFIX}/workspaces/{{workspace_id}}/episodes",
             response_model=list[schemas.EpisodeOut], tags=["episodes"])
    def list_episodes(
        workspace: Workspace = Depends(scoped_workspace),
        db: Session = Depends(get_db),
        include_probes: bool = Query(default=False),
        limit: int = Query(default=100, le=500),
    ):
        stmt = select(Episode).where(Episode.workspace_id == workspace.id)
        if not include_probes:
            stmt = stmt.where(Episode.is_benchmark_probe.is_(False))
        return db.execute(
            stmt.order_by(Episode.created_at.desc()).limit(limit)
        ).scalars().all()

    @app.get(f"{API_PREFIX}/workspaces/{{workspace_id}}/agents", tags=["agents"])
    def list_agents(workspace: Workspace = Depends(scoped_workspace),
                    db: Session = Depends(get_db)) -> list[dict[str, Any]]:
        profiles = db.execute(
            select(AgentProfile).where(AgentProfile.workspace_id == workspace.id)
        ).scalars().all()
        return [
            {"id": str(p.id), "name": p.name, "role": p.role.value,
             "description": p.description,
             "performance_by_work_type": p.performance_by_work_type}
            for p in profiles
        ]

    # ----------------------------------------------------------------------
    @app.get(f"{API_PREFIX}/workspaces/{{workspace_id}}/artifacts",
             response_model=list[schemas.ArtifactOut], tags=["artifacts"])
    def list_artifacts(
        workspace: Workspace = Depends(scoped_workspace),
        db: Session = Depends(get_db),
        type_filter: str | None = Query(default=None, alias="type"),
        include_archived: bool = Query(default=False),
        limit: int = Query(default=100, le=500),
    ):
        stmt = select(Artifact).where(Artifact.workspace_id == workspace.id)
        if not include_archived:
            stmt = stmt.where(Artifact.archived_at.is_(None))
        if type_filter:
            stmt = stmt.where(Artifact.type == type_filter)
        return db.execute(
            stmt.order_by(Artifact.created_at.desc()).limit(limit)
        ).scalars().all()

    @app.post(f"{API_PREFIX}/workspaces/{{workspace_id}}/artifacts",
              response_model=schemas.ArtifactOut, status_code=201, tags=["artifacts"])
    def create_artifact(
        payload: schemas.ArtifactIn,
        workspace: Workspace = Depends(scoped_workspace),
        db: Session = Depends(get_db),
        principal: Principal = Depends(require_scope("artifacts:write")),
    ):
        try:
            artifact_type = ArtifactType(payload.type)
        except ValueError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                f"unknown artifact type {payload.type!r}") from exc
        artifact = Artifact(
            workspace_id=workspace.id, project_id=payload.project_id, task_id=payload.task_id,
            type=artifact_type, title=payload.title, body=payload.body,
            confidence=payload.confidence, structured=payload.structured,
            creator_kind=principal.kind.value, creator_id=principal.id,
            environment_version=workspace.environment_version,
        )
        db.add(artifact)
        db.flush()
        try:
            from civitas.knowledge.embeddings import index_artifact

            index_artifact(db, artifact)
        except Exception:  # pragma: no cover - an optional index must not lose a write
            pass
        # §45: every meaningful operation emits an immutable event. Creating an artifact through
        # the API is as meaningful as creating one through a tool, and a stream that recorded only
        # one of the two routes would not be a reconstructable history.
        emit(
            db, workspace_id=workspace.id, type=EventType.ARTIFACT_CREATED,
            actor_kind=principal.kind.value, actor_id=principal.id,
            payload={"artifact_id": str(artifact.id), "type": artifact_type.value,
                     "via": "api"},
        )
        return artifact

    @app.get(f"{API_PREFIX}/workspaces/{{workspace_id}}/artifacts/{{artifact_id}}",
             response_model=schemas.ArtifactOut, tags=["artifacts"])
    def get_artifact(artifact_id: uuid.UUID,
                     workspace: Workspace = Depends(scoped_workspace),
                     db: Session = Depends(get_db)):
        artifact = db.get(Artifact, artifact_id)
        if artifact is None or artifact.workspace_id != workspace.id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such artifact")
        return artifact

    @app.get(f"{API_PREFIX}/workspaces/{{workspace_id}}/artifacts/{{artifact_id}}/provenance",
             response_model=schemas.ProvenanceOut, tags=["artifacts"])
    def get_provenance(artifact_id: uuid.UUID,
                       workspace: Workspace = Depends(scoped_workspace),
                       db: Session = Depends(get_db)):
        """The provenance inspector's data (§13, §49)."""
        from civitas.knowledge.graph import provenance

        artifact = db.get(Artifact, artifact_id)
        if artifact is None or artifact.workspace_id != workspace.id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such artifact")
        chain = provenance(db, artifact_id)
        return schemas.ProvenanceOut(**chain.as_dict())

    @app.get(f"{API_PREFIX}/workspaces/{{workspace_id}}/artifact-relations",
             response_model=list[schemas.RelationOut], tags=["artifacts"])
    def list_relations(
        workspace: Workspace = Depends(scoped_workspace),
        db: Session = Depends(get_db),
        limit: int = Query(default=500, le=2000),
    ):
        return db.execute(
            select(ArtifactRelation)
            .join(Artifact, Artifact.id == ArtifactRelation.source_id)
            .where(Artifact.workspace_id == workspace.id)
            .limit(limit)
        ).scalars().all()

    @app.post(f"{API_PREFIX}/workspaces/{{workspace_id}}/artifact-relations",
              response_model=schemas.RelationOut, status_code=201, tags=["artifacts"])
    def create_relation(
        payload: schemas.RelationIn,
        workspace: Workspace = Depends(scoped_workspace),
        db: Session = Depends(get_db),
        principal: Principal = Depends(require_scope("artifacts:write")),
    ):
        for artifact_id in (payload.source_id, payload.target_id):
            artifact = db.get(Artifact, artifact_id)
            if artifact is None or artifact.workspace_id != workspace.id:
                raise HTTPException(status.HTTP_404_NOT_FOUND,
                                    f"no such artifact {artifact_id}")
        try:
            relation_type = RelationType(payload.type)
        except ValueError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                f"unknown relation {payload.type!r}") from exc
        relation = ArtifactRelation(
            source_id=payload.source_id, target_id=payload.target_id, type=relation_type,
            confidence=payload.confidence, creator_kind=principal.kind.value,
        )
        db.add(relation)
        db.flush()
        emit(
            db, workspace_id=workspace.id, type=EventType.ARTIFACT_LINKED,
            actor_kind=principal.kind.value, actor_id=principal.id,
            payload={"source": str(payload.source_id), "target": str(payload.target_id),
                     "type": relation_type.value, "via": "api"},
        )
        return relation

    # ----------------------------------------------------------------------
    @app.post(f"{API_PREFIX}/workspaces/{{workspace_id}}/retrieval",
              response_model=schemas.RetrievalOut, tags=["retrieval"])
    def perform_retrieval(
        payload: schemas.RetrievalIn,
        workspace: Workspace = Depends(scoped_workspace),
        db: Session = Depends(get_db),
        principal: Principal = Depends(require_scope("retrieval:read")),
    ):
        """Hybrid retrieval, with the arm enforced and the decision logged (§14, §21)."""
        from civitas.knowledge.retrieval import retrieve

        try:
            arm = ExperimentArm(payload.arm)
        except ValueError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                f"unknown experiment arm {payload.arm!r}") from exc
        types = None
        if payload.types:
            try:
                types = [ArtifactType(t) for t in payload.types]
            except ValueError as exc:
                raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                    "unknown artifact type") from exc
        try:
            result = retrieve(
                db, workspace_id=workspace.id, query=payload.query, arm=arm,
                types=types, limit=min(payload.limit, 50),
            )
        except ValueError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

        return schemas.RetrievalOut(
            decision_id=result.decision_id,
            candidate_count=result.candidate_count,
            returned_count=len(result.artifacts),
            provenance_hidden=result.provenance_hidden,
            suppressed=result.suppressed,
            results=[
                schemas.RetrievalHit(
                    artifact=schemas.ArtifactOut.model_validate(artifact),
                    score=score, features=features,
                )
                for artifact, score, features in zip(
                    result.artifacts, result.scores, result.features, strict=False
                )
            ],
        )

    # ----------------------------------------------------------------------
    @app.get(f"{API_PREFIX}/workspaces/{{workspace_id}}/tools",
             response_model=list[schemas.ToolOut], tags=["tools"])
    def list_tools(workspace: Workspace = Depends(scoped_workspace),
                   db: Session = Depends(get_db)):
        return db.execute(
            select(ToolDefinition).where(ToolDefinition.workspace_id == workspace.id,
                                         ToolDefinition.archived_at.is_(None))
        ).scalars().all()

    @app.get(f"{API_PREFIX}/workspaces/{{workspace_id}}/events",
             response_model=list[schemas.EventOut], tags=["events"])
    def list_events(
        workspace: Workspace = Depends(scoped_workspace),
        db: Session = Depends(get_db),
        after_sequence: int = Query(default=0, ge=0),
        limit: int = Query(default=200, le=1000),
    ):
        """The event stream, cursored by **sequence** (§45).

        Never by timestamp: two events in one millisecond are indistinguishable by clock, and a
        timestamp cursor silently skips or repeats rows.
        """
        from civitas.persistence.events import read_stream

        return read_stream(db, workspace.id, after_sequence=after_sequence, limit=limit)

    @app.get(f"{API_PREFIX}/workspaces/{{workspace_id}}/evaluations",
             response_model=list[schemas.EvaluationOut], tags=["evaluations"])
    def list_evaluations(workspace: Workspace = Depends(scoped_workspace),
                         db: Session = Depends(get_db),
                         limit: int = Query(default=100, le=500)):
        return db.execute(
            select(Evaluation).where(Evaluation.workspace_id == workspace.id)
            .order_by(Evaluation.created_at.desc()).limit(limit)
        ).scalars().all()

    @app.get(f"{API_PREFIX}/jobs", response_model=list[schemas.JobOut], tags=["jobs"])
    def list_jobs(
        db: Session = Depends(get_db),
        principal: Principal = Depends(require_role(Role.OPERATOR)),
        status_filter: str | None = Query(default=None, alias="status"),
        limit: int = Query(default=100, le=500),
    ):
        stmt = select(Job)
        if status_filter:
            stmt = stmt.where(Job.status == status_filter)
        return db.execute(stmt.order_by(Job.created_at.desc()).limit(limit)).scalars().all()

    # ----------------------------------------------------------------------
    @app.get(f"{API_PREFIX}/workspaces/{{workspace_id}}/metrics",
             response_model=schemas.MetricsOut, tags=["metrics"])
    def get_metrics(
        workspace: Workspace = Depends(scoped_workspace),
        db: Session = Depends(get_db),
        include_probes: bool = Query(default=False),
    ):
        """§48's metrics. A metric that cannot be computed reports `null` with a reason."""
        from civitas.experiments.metrics import export

        payload = export(db, workspace_id=workspace.id, include_probes=include_probes)
        return schemas.MetricsOut(**{**payload, "workspace_id": workspace.id})

    @app.get(f"{API_PREFIX}/workspaces/{{workspace_id}}/experiments",
             response_model=list[schemas.ExperimentOut], tags=["experiments"])
    def list_experiments(workspace: Workspace = Depends(scoped_workspace),
                         db: Session = Depends(get_db)):
        return db.execute(
            select(Experiment).where(Experiment.workspace_id == workspace.id)
        ).scalars().all()

    @app.get(f"{API_PREFIX}/experiments/{{experiment_id}}/runs",
             response_model=list[schemas.ExperimentRunOut], tags=["experiments"])
    def list_runs(
        experiment_id: uuid.UUID,
        db: Session = Depends(get_db),
        principal: Principal = Depends(require_role(Role.VIEWER)),
    ):
        experiment = db.get(Experiment, experiment_id)
        if experiment is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such experiment")
        workspace = db.get(Workspace, experiment.workspace_id)
        if workspace is None or workspace.organization_id != principal.organization_id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such experiment")
        return db.execute(
            select(ExperimentRun).where(ExperimentRun.experiment_id == experiment_id)
        ).scalars().all()

    @app.get(f"{API_PREFIX}/experiments/{{experiment_id}}/result",
             response_model=schemas.BenchmarkResultOut, tags=["experiments"])
    def experiment_result(
        experiment_id: uuid.UUID,
        db: Session = Depends(get_db),
        principal: Principal = Depends(require_role(Role.VIEWER)),
    ):
        """A run's result, with the gate discipline preserved across the wire.

        When a gate failed, `metrics` is **absent** — not zeroed, not partially filled. A client
        cannot render a number the run did not earn, because there is no number to render
        (ARCHITECTURE §3.1).
        """
        experiment = db.get(Experiment, experiment_id)
        if experiment is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such experiment")
        workspace = db.get(Workspace, experiment.workspace_id)
        if workspace is None or workspace.organization_id != principal.organization_id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such experiment")

        runs = db.execute(
            select(ExperimentRun).where(ExperimentRun.experiment_id == experiment_id)
        ).scalars().all()
        if not runs:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "this experiment has no runs")

        gates: dict[str, Any] = {}
        for run in runs:
            gates.update(run.gates or {})
        failed = [name for name, gate in gates.items() if not gate.get("passed")]
        passed = not failed

        return schemas.BenchmarkResultOut(
            experiment=experiment.name,
            config_hash=runs[0].config_hash,
            gates_passed=passed,
            gates=gates,
            failed_gates=failed,
            metrics=(
                {run.id.hex: run.metrics for run in runs} if passed else None
            ),
            withheld_reason=(
                None if passed
                else f"gates failed: {failed}. The run is not read (ARCHITECTURE §3.1)."
            ),
        )

    # ----------------------------------------------------------------------
    from civitas.api.security import get_principal

    app.dependency_overrides[get_principal] = resolve_principal

    @app.exception_handler(ValueError)
    def _value_error(_request: Request, exc: ValueError) -> JSONResponse:  # pragma: no cover
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    return app


app = None


def get_app() -> FastAPI:  # pragma: no cover - uvicorn entry point
    global app
    if app is None:
        app = create_app()
    return app
