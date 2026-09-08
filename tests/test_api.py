"""The API (Part B §50) and its security boundary (§51, §52).

Driven through a real TestClient against a real database, because an API test that mocks the
database tests the schemas rather than the endpoints.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from civitas.api.app import create_app, get_db
from civitas.api.security import bootstrap_organization, create_api_key
from civitas.domain.enums import ActorKind, Role
from civitas.persistence.models import Organization, ServiceIdentity, User, Workspace


@pytest.fixture
def api(settings, session_factory, db):
    app = create_app(settings)

    def _db_override():
        session = session_factory()
        try:
            yield session
            session.commit()
        finally:
            session.close()

    app.dependency_overrides[get_db] = _db_override
    return app


@pytest.fixture
def bootstrapped(db):
    org, admin, key = bootstrap_organization(
        db, name="Test Org", slug=f"org-{uuid.uuid4().hex[:8]}", admin_email="admin@example.com"
    )
    db.commit()
    return org, admin, key


@pytest.fixture
def client(api, bootstrapped):
    _org, _admin, key = bootstrapped
    with TestClient(api) as c:
        c.headers.update({"x-api-key": key})
        yield c


@pytest.fixture
def api_workspace(client) -> dict:
    response = client.post(
        "/api/v1/workspaces",
        json={"name": "W", "slug": f"w-{uuid.uuid4().hex[:8]}", "environment_version": "test"},
    )
    assert response.status_code == 201, response.text
    return response.json()


# --------------------------------------------------------------------------
# health and documentation
# --------------------------------------------------------------------------
def test_health_reports_the_sandbox_actually_in_force(api):
    """§46: a weaker guarantee must be visible, not implied."""
    with TestClient(api) as c:
        body = c.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["sandbox_backend"] in ("subprocess", "docker", "gvisor")
    assert isinstance(body["sandbox_unenforced_limits"], list)
    assert body["schema_version"]


def test_openapi_documents_every_required_route(api):
    """§50 lists the routes an implementation must provide."""
    with TestClient(api) as c:
        spec = c.get("/api/v1/openapi.json").json()
    paths = set(spec["paths"])
    for required in ("organizations", "workspaces", "projects", "tasks", "episodes", "agents",
                     "artifacts", "artifact-relations", "tools", "experiments", "evaluations",
                     "jobs", "metrics", "retrieval", "events"):
        assert any(required in p for p in paths), f"§50 requires a {required} route"


# --------------------------------------------------------------------------
# authentication (§51, §52)
# --------------------------------------------------------------------------
def test_an_unauthenticated_request_is_refused(api):
    with TestClient(api) as c:
        assert c.get("/api/v1/workspaces").status_code == 401


def test_an_invalid_key_gets_the_same_message_as_a_missing_one(api):
    """A distinguishable response would let a caller enumerate valid prefixes (§52)."""
    with TestClient(api) as c:
        bad = c.get("/api/v1/workspaces", headers={"x-api-key": "civ_notarealkey"})
    assert bad.status_code == 401
    assert "invalid API key" in bad.json()["detail"]


def test_a_revoked_key_stops_working(api, db, bootstrapped):
    from civitas.persistence.models import ApiKey
    from civitas.persistence.types import utcnow

    _org, _admin, key = bootstrapped
    with TestClient(api) as c:
        assert c.get("/api/v1/workspaces", headers={"x-api-key": key}).status_code == 200
        db.query(ApiKey).one().revoked_at = utcnow()
        db.commit()
        assert c.get("/api/v1/workspaces", headers={"x-api-key": key}).status_code == 401


def test_only_the_hash_of_a_key_is_stored(db, bootstrapped):
    """A database dump must not be a credential leak (§40)."""
    from civitas.persistence.models import ApiKey

    _org, _admin, plaintext = bootstrapped
    stored = db.query(ApiKey).one()
    assert stored.key_hash != plaintext
    assert plaintext not in str({c.name: getattr(stored, c.name) for c in ApiKey.__table__.columns})
    assert stored.prefix == plaintext[:12], "the prefix is stored so a key can be revoked"


def test_an_agent_identity_cannot_satisfy_a_human_role(api, db, bootstrapped):
    """§51 least privilege. An agent is a different kind of principal, not a weak operator."""
    org, _admin, _key = bootstrapped
    identity = ServiceIdentity(organization_id=org.id, name="agent-1",
                               kind=ActorKind.AGENT, role=Role.AGENT)
    db.add(identity)
    db.flush()
    _key_row, agent_key = create_api_key(
        db, organization_id=org.id, name="agent", service_identity_id=identity.id,
    )
    db.commit()

    with TestClient(api) as c:
        headers = {"x-api-key": agent_key}
        # An agent may not list organizations (a viewer-role endpoint).
        assert c.get("/api/v1/organizations", headers=headers).status_code == 403
        assert c.get("/api/v1/jobs", headers=headers).status_code == 403


def test_a_researcher_cannot_reach_an_operator_endpoint(api, db, bootstrapped):
    org, _admin, _key = bootstrapped
    user = User(organization_id=org.id, email="r@example.com", role=Role.RESEARCHER)
    db.add(user)
    db.flush()
    _row, key = create_api_key(db, organization_id=org.id, name="r", user_id=user.id)
    db.commit()

    with TestClient(api) as c:
        assert c.get("/api/v1/jobs", headers={"x-api-key": key}).status_code == 403
        assert c.get("/api/v1/workspaces", headers={"x-api-key": key}).status_code == 200


def test_a_viewer_cannot_create_a_workspace(api, db, bootstrapped):
    org, _admin, _key = bootstrapped
    user = User(organization_id=org.id, email="v@example.com", role=Role.VIEWER)
    db.add(user)
    db.flush()
    _row, key = create_api_key(db, organization_id=org.id, name="v", user_id=user.id)
    db.commit()

    with TestClient(api) as c:
        response = c.post("/api/v1/workspaces", json={"name": "X", "slug": "x"},
                          headers={"x-api-key": key})
    assert response.status_code == 403


# --------------------------------------------------------------------------
# the tenancy boundary (§52)
# --------------------------------------------------------------------------
def test_a_workspace_in_another_organization_is_not_found(api, db, bootstrapped, client):
    """404 rather than 403: telling a caller an id exists but is not theirs is a disclosure."""
    other_org = Organization(name="Other", slug=f"other-{uuid.uuid4().hex[:8]}")
    db.add(other_org)
    db.flush()
    foreign = Workspace(organization_id=other_org.id, name="Foreign", slug="foreign")
    db.add(foreign)
    db.commit()

    response = client.get(f"/api/v1/workspaces/{foreign.id}")
    assert response.status_code == 404
    assert "no such workspace" in response.json()["detail"]


def test_listing_never_crosses_organizations(api, db, bootstrapped, client):
    other_org = Organization(name="Other", slug=f"other-{uuid.uuid4().hex[:8]}")
    db.add(other_org)
    db.flush()
    db.add(Workspace(organization_id=other_org.id, name="Foreign", slug="foreign2"))
    db.commit()

    listed = client.get("/api/v1/workspaces").json()
    assert all(w["organization_id"] != str(other_org.id) for w in listed)


# --------------------------------------------------------------------------
# what the API must not disclose (§47)
# --------------------------------------------------------------------------
def test_a_task_never_discloses_its_evaluator_specification(client, api_workspace, db):
    """§47: an agent that can read its own success criteria can write to them — and so can
    anyone who can call this endpoint."""
    from civitas.persistence.models import Task

    task = Task(
        workspace_id=uuid.UUID(api_workspace["id"]), title="t", description="d",
        evaluator_spec={"kind": "exact_match", "expected": "SECRET-EXPECTED-ANSWER"},
    )
    db.add(task)
    db.commit()

    listed = client.get(f"/api/v1/workspaces/{api_workspace['id']}/tasks")
    assert listed.status_code == 200
    assert "SECRET-EXPECTED-ANSWER" not in listed.text
    assert "evaluator_spec" not in listed.text


# --------------------------------------------------------------------------
# the research surface
# --------------------------------------------------------------------------
def test_artifacts_can_be_created_read_and_related(client, api_workspace):
    ws = api_workspace["id"]
    first = client.post(f"/api/v1/workspaces/{ws}/artifacts", json={
        "type": "observation", "title": "the writer holds a lock", "body": "detail",
    })
    assert first.status_code == 201, first.text
    second = client.post(f"/api/v1/workspaces/{ws}/artifacts", json={
        "type": "hypothesis", "title": "the retry path is at fault", "body": "detail",
    })
    assert second.status_code == 201

    relation = client.post(f"/api/v1/workspaces/{ws}/artifact-relations", json={
        "source_id": second.json()["id"], "target_id": first.json()["id"],
        "type": "derived_from",
    })
    assert relation.status_code == 201

    provenance = client.get(
        f"/api/v1/workspaces/{ws}/artifacts/{second.json()['id']}/provenance"
    )
    assert provenance.status_code == 200
    body = provenance.json()
    assert body["depth"] == 1
    assert body["nodes"][0]["title"] == "the writer holds a lock"


def test_an_unknown_artifact_type_is_a_422_not_a_500(client, api_workspace):
    response = client.post(f"/api/v1/workspaces/{api_workspace['id']}/artifacts", json={
        "type": "not_a_real_type", "title": "x", "body": "y",
    })
    assert response.status_code == 422


def test_retrieval_enforces_the_arm_and_logs_the_decision(client, api_workspace):
    """§14, §21 over the wire."""
    ws = api_workspace["id"]
    client.post(f"/api/v1/workspaces/{ws}/artifacts", json={
        "type": "failure", "title": "the retry approach failed", "body": "detail",
    })

    collective = client.post(f"/api/v1/workspaces/{ws}/retrieval", json={
        "query": "retry approach", "arm": "collective", "limit": 5,
    }).json()
    assert collective["returned_count"] >= 1
    assert collective["decision_id"]
    assert "lexical" in collective["results"][0]["features"]

    solo = client.post(f"/api/v1/workspaces/{ws}/retrieval", json={
        "query": "retry approach", "arm": "solo", "limit": 5,
    }).json()
    assert solo["returned_count"] == 0
    assert solo["suppressed"][0]["reason"] == "arm_blind"


def test_an_unknown_arm_is_refused_over_the_wire(client, api_workspace):
    response = client.post(f"/api/v1/workspaces/{api_workspace['id']}/retrieval", json={
        "query": "x", "arm": "collective_with_extra_sauce",
    })
    assert response.status_code == 422


def test_collective_frozen_without_a_cut_is_refused_not_degraded(client, api_workspace):
    """Otherwise the API would silently run an unablated arm and label it a control."""
    response = client.post(f"/api/v1/workspaces/{api_workspace['id']}/retrieval", json={
        "query": "x", "arm": "collective_frozen",
    })
    assert response.status_code == 422
    assert "snapshot cut" in response.json()["detail"]


def test_the_event_stream_is_cursored_by_sequence(client, api_workspace):
    ws = api_workspace["id"]
    for i in range(5):
        client.post(f"/api/v1/workspaces/{ws}/artifacts", json={
            "type": "observation", "title": f"o{i}", "body": "b",
        })
    events = client.get(f"/api/v1/workspaces/{ws}/events").json()
    assert events == sorted(events, key=lambda e: e["sequence"])

    tail = client.get(f"/api/v1/workspaces/{ws}/events",
                      params={"after_sequence": events[0]["sequence"]}).json()
    assert all(e["sequence"] > events[0]["sequence"] for e in tail)


def test_metrics_report_absence_rather_than_zero(client, api_workspace):
    """§48, over the wire: a dashboard rendering 'not measurable' as 'measured zero'
    manufactures a finding."""
    body = client.get(f"/api/v1/workspaces/{api_workspace['id']}/metrics").json()
    assert body["unavailable"], "an empty workspace has unmeasurable metrics"
    for name in body["unavailable"]:
        assert body["metrics"][name]["value"] is None
        assert body["metrics"][name]["unavailable"]


def test_benchmark_metrics_are_withheld_when_a_gate_failed(client, api_workspace, db):
    """ARCHITECTURE §3.1 preserved across the wire: there is nothing for a client to render as a
    number the run did not earn."""
    from civitas.persistence.models import Experiment, ExperimentArm, ExperimentRun

    workspace_id = uuid.UUID(api_workspace["id"])
    experiment = Experiment(workspace_id=workspace_id, name="gated", kind="newcomer")
    db.add(experiment)
    db.flush()
    arm = ExperimentArm(experiment_id=experiment.id, name="a", arm="collective")
    db.add(arm)
    db.flush()
    db.add(ExperimentRun(
        experiment_id=experiment.id, experiment_arm_id=arm.id, seed=0, status="completed",
        config_hash="c", metrics={"newcomer_advantage": 0.9},
        gates={"baseline_has_headroom": {"passed": False, "detail": "at ceiling"}},
    ))
    db.commit()

    body = client.get(f"/api/v1/experiments/{experiment.id}/result").json()
    assert body["gates_passed"] is False
    assert body["metrics"] is None, "a withheld number must be absent, not zeroed"
    assert "baseline_has_headroom" in body["failed_gates"]
    assert "not read" in body["withheld_reason"]
    assert "0.9" not in str(body["metrics"])


# --------------------------------------------------------------------------- §5, §32 over HTTP


def test_a_high_level_request_returns_a_plan_over_the_wire(client, api_workspace):
    """§5: the API's answer to a request is the work it created, not a result."""
    response = client.post(
        f"/api/v1/workspaces/{api_workspace['id']}/requests",
        json={"request": "investigate the intermittent data corruption in the ledger writer"},
    )
    assert response.status_code == 201, response.text
    plan = response.json()
    assert [t["family"] for t in plan["tasks"]] == [
        "investigate", "hypothesise", "verify", "synthesise"
    ]
    assert plan["decomposer"] == "deterministic/1.0"


def test_the_tasks_a_request_created_disclose_no_evaluator_spec(client, api_workspace):
    """§47 over HTTP. `TaskOut` has no field for it, so this holds for any task the API returns —
    but a request-created chain is the path a user is most likely to read tasks through."""
    client.post(
        f"/api/v1/workspaces/{api_workspace['id']}/requests",
        json={"request": "investigate the intermittent data corruption in the ledger writer"},
    )
    tasks = client.get(f"/api/v1/workspaces/{api_workspace['id']}/tasks").json()
    assert tasks
    assert all("evaluator_spec" not in t for t in tasks)


def test_project_status_over_http_reports_blocked_stages(client, api_workspace):
    plan = client.post(
        f"/api/v1/workspaces/{api_workspace['id']}/requests",
        json={"request": "investigate the lock contention in the scheduler"},
    ).json()
    status = client.get(
        f"/api/v1/workspaces/{api_workspace['id']}/projects/{plan['project_id']}/status"
    ).json()
    assert status["tasks_total"] == 4
    assert status["stages"]["synthesise"] == "blocked"
    assert status["complete"] is False


def test_a_project_in_another_workspace_is_a_404_not_a_403(client, api_workspace):
    """Same discipline as everywhere else in the API: a 403 confirms the id exists."""
    plan = client.post(
        f"/api/v1/workspaces/{api_workspace['id']}/requests",
        json={"request": "investigate the lock contention in the scheduler"},
    ).json()
    other = client.post(
        "/api/v1/workspaces",
        json={"name": "Other", "slug": f"o-{uuid.uuid4().hex[:8]}", "environment_version": "test"},
    ).json()
    response = client.get(
        f"/api/v1/workspaces/{other['id']}/projects/{plan['project_id']}/status"
    )
    assert response.status_code == 404


def test_a_request_too_short_to_decompose_is_refused(client, api_workspace):
    response = client.post(
        f"/api/v1/workspaces/{api_workspace['id']}/requests", json={"request": "fix"}
    )
    assert response.status_code == 422


def test_the_domains_endpoint_lists_what_the_process_can_run(client):
    response = client.get("/api/v1/domains")
    assert response.status_code == 200
    assert {d["name"] for d in response.json()} == {"code_repair", "hidden_rule"}


def test_the_openapi_document_covers_the_new_routes(client):
    paths = client.get("/api/v1/openapi.json").json()["paths"]
    assert "/api/v1/domains" in paths
    assert "/api/v1/workspaces/{workspace_id}/requests" in paths
    assert "/api/v1/workspaces/{workspace_id}/projects/{project_id}/status" in paths
