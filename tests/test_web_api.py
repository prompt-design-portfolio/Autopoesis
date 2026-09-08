"""The endpoints the UI needs (Part B §39, §45, §49).

Driven through a real TestClient against a real database. The properties under test are the two
the UI exists to hold: a number the platform could not compute must reach the client as `null`
with a reason, and a benchmark whose gates failed must reach it with no metrics at all.
"""

from __future__ import annotations

import json
import uuid

import pytest

from civitas.domain.enums import EventType, ExperimentArm, TerminationReason
from civitas.persistence.events import emit
from civitas.persistence.models import AgentProfile, Artifact, Episode, Task


@pytest.fixture
def populated(db, api_workspace):
    """A workspace with one of everything the dashboards read."""
    workspace_id = uuid.UUID(api_workspace["id"])
    profile = AgentProfile(workspace_id=workspace_id, name="prober", role="explorer")
    db.add(profile)
    db.flush()
    task = Task(workspace_id=workspace_id, title="t", description="d", task_family="hidden_rule",
                status="ready", evaluator_spec={"kind": "exact_match", "expected": "ferrose"})
    db.add(task)
    db.flush()
    priced = Episode(
        workspace_id=workspace_id, agent_profile_id=profile.id, task_id=task.id,
        model_provider="anthropic", model_name="model-x",
        experiment_arm=ExperimentArm.COLLECTIVE,
        termination_reason=TerminationReason.EVALUATOR_SUCCESS,
        tokens_used=1200, tool_calls_used=4, cost_usd=0.25, environment_version="device-e1-s0",
    )
    free = Episode(
        workspace_id=workspace_id, agent_profile_id=profile.id, task_id=task.id,
        model_provider="deterministic", model_name="policy-v1",
        experiment_arm=ExperimentArm.SOLO,
        termination_reason=TerminationReason.EVALUATOR_FAILURE,
        tokens_used=800, tool_calls_used=3, cost_usd=0.0, environment_version="device-e1-s0",
    )
    db.add_all([priced, free])
    db.flush()
    db.add(Artifact(workspace_id=workspace_id, type="observation", title="an observation",
                    body="b", creator_episode_id=priced.id))
    archived = Artifact(workspace_id=workspace_id, type="failure", title="archived one", body="b")
    db.add(archived)
    db.flush()
    from civitas.persistence.types import utcnow

    archived.archived_at = utcnow()
    archived.archived_reason = "memory_reset arm"
    emit(db, workspace_id=workspace_id, type=EventType.EPISODE_STARTED,
         episode_id=priced.id, payload={"hello": "world"})
    db.commit()
    return {"workspace_id": workspace_id, "priced": priced, "free": free, "task": task}


# --------------------------------------------------------------------------- overview


def test_the_overview_keeps_archived_artifacts_out_of_the_live_count(client, api_workspace,
                                                                     populated):
    """Part A §A1.2: `memory_reset` archives rather than deletes. One total would make an arm that
    lost its memory indistinguishable from one that never had any."""
    data = client.get(f"/api/v1/workspaces/{api_workspace['id']}/overview").json()
    assert data["artifacts"] == 1
    assert data["artifacts_archived"] == 1
    assert data["episodes"] == 2
    assert data["episodes_succeeded"] == 1
    assert data["tokens_used"] == 2000
    assert data["latest_sequence"] >= 1


# --------------------------------------------------------------------------- cost


def test_an_unpriced_bucket_reports_no_mean_cost_rather_than_zero(client, api_workspace,
                                                                  populated):
    """§39 and ARCHITECTURE §3.9. A deterministic provider is genuinely free, so its cost per
    episode is *unknown*, not zero — and a dashboard that renders it as zero states something
    untrue about what the run cost to produce."""
    data = client.get(f"/api/v1/workspaces/{api_workspace['id']}/cost").json()
    by_model = {b["key"]: b for b in data["by_model"]}

    assert by_model["deterministic/policy-v1"]["mean_cost_usd"] is None
    assert by_model["anthropic/model-x"]["mean_cost_usd"] == pytest.approx(0.25)
    assert data["unpriced_episodes"] == 1
    assert data["total_cost_usd"] == pytest.approx(0.25)


def test_cost_is_bucketed_by_day_model_and_arm(client, api_workspace, populated):
    data = client.get(f"/api/v1/workspaces/{api_workspace['id']}/cost").json()
    assert {b["key"] for b in data["by_arm"]} == {"collective", "solo"}
    assert len(data["by_day"]) == 1
    assert sum(b["episodes"] for b in data["by_model"]) == data["episodes"]


def test_probe_episodes_can_be_excluded_from_the_cost_view(client, api_workspace, populated, db):
    """A benchmark probe costs real tokens but is not the collective's ordinary work; conflating
    them makes a benchmark run look like a spending anomaly."""
    populated["free"].is_benchmark_probe = True
    db.commit()
    both = client.get(f"/api/v1/workspaces/{api_workspace['id']}/cost").json()
    without = client.get(
        f"/api/v1/workspaces/{api_workspace['id']}/cost?include_probes=false"
    ).json()
    assert both["episodes"] == 2 and without["episodes"] == 1


# --------------------------------------------------------------------------- episode detail


def test_the_episode_inspector_discloses_no_reasoning_and_no_ground_truth(client, api_workspace,
                                                                          populated):
    """§4 forbids persisting reasoning at all, and §47 forbids the answer reaching a reader. The
    check is on the serialised response, not on a field list, because a leak would arrive nested
    inside `detail` or `args` rather than as a new top-level key."""
    episode_id = populated["priced"].id
    response = client.get(f"/api/v1/workspaces/{api_workspace['id']}/episodes/{episode_id}")
    assert response.status_code == 200
    body = response.json()
    rendered = json.dumps(body)

    assert "ferrose" not in rendered, "the task's expected answer reached the client"
    for forbidden in ("reasoning", "thought", "chain_of_thought", "prompt_text", "evaluator_spec"):
        assert forbidden not in rendered
    assert body["experiment_arm"] == "collective"
    assert body["config_hash"] is not None


def test_an_episode_in_another_workspace_is_a_404(client, api_workspace, populated):
    other = client.post("/api/v1/workspaces", json={
        "name": "Other", "slug": f"o-{uuid.uuid4().hex[:8]}", "environment_version": "test",
    }).json()
    response = client.get(
        f"/api/v1/workspaces/{other['id']}/episodes/{populated['priced'].id}"
    )
    assert response.status_code == 404


# --------------------------------------------------------------------------- dashboards


def test_the_specialization_index_is_absent_with_a_reason_not_zero(client, api_workspace):
    """An index of 0.0 means "every profile performs identically"; an absent index means "there is
    not enough evidence to say". A dashboard that showed the first for the second would report a
    finding nobody measured."""
    data = client.get(f"/api/v1/workspaces/{api_workspace['id']}/specialization").json()
    assert data["specialization_index"] is None
    assert data["index_unavailable_reason"]


def test_reading_the_specialization_screen_changes_nothing(client, api_workspace, populated, db):
    """A read that rewrote every profile's performance table would make opening a dashboard an
    experimental intervention."""
    profile = db.query(AgentProfile).filter_by(
        workspace_id=populated["workspace_id"]).first()
    before = dict(profile.performance_by_work_type or {})
    db.rollback()  # release SQLite's write lock before the API opens its own transaction

    client.get(f"/api/v1/workspaces/{api_workspace['id']}/specialization")
    db.expire_all()
    profile = db.query(AgentProfile).filter_by(
        workspace_id=populated["workspace_id"]).first()
    assert dict(profile.performance_by_work_type or {}) == before


def test_the_institutions_screen_reports_adoption_as_absent_over_an_empty_set(client,
                                                                             api_workspace):
    """§A2.2: an adoption rate over nothing proposed is not zero adoption."""
    data = client.get(f"/api/v1/workspaces/{api_workspace['id']}/institutions").json()
    assert set(data["gate"]) == {"policy", "procedure", "prompt"}
    for kind in data["gate"].values():
        assert kind["adoption_rate"] is None
        assert kind["proposed"] == 0
    assert data["policies"] == []


def test_a_policy_shows_whether_the_gate_approved_it(client, api_workspace, db):
    """The A2.2 gate in one field: an unapproved policy is one the loader will not return,
    however good it looks."""
    from civitas.institutions import gate

    workspace_id = uuid.UUID(api_workspace["id"])
    gate.propose_policy(db, workspace_id=workspace_id, kind="retrieval", name="hybrid",
                        body={"semantic": True}, description="d")
    db.commit()

    data = client.get(f"/api/v1/workspaces/{api_workspace['id']}/institutions").json()
    assert len(data["policies"]) == 1
    assert data["policies"][0]["approved"] is False
    assert data["gate"]["policy"]["proposed"] == 1
    assert data["gate"]["policy"]["adoption_rate"] == 0.0


# --------------------------------------------------------------------------- the stream


def _sse(text: str) -> list[dict]:
    out = []
    for block in text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                try:
                    out.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass
    return out


def test_the_stream_replays_the_event_log_in_sequence_order(client, api_workspace, populated):
    """§45, §49: the log is the transport. Ordered by `sequence`, never by timestamp — two events
    in the same millisecond are indistinguishable by clock, and a cursor built on timestamps
    silently skips or repeats rows."""
    response = client.get(
        f"/api/v1/workspaces/{api_workspace['id']}/stream?timeout_s=1&max_events=50"
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = [e for e in _sse(response.text) if "type" in e]
    assert events, "the stream replayed nothing"
    assert [e["sequence"] for e in events] == sorted(e["sequence"] for e in events)
    assert "id: " in response.text, "no SSE id, so a reconnecting browser cannot resume"


def test_a_reconnecting_client_resumes_from_the_id_it_last_saw(client, api_workspace, populated):
    first = client.get(
        f"/api/v1/workspaces/{api_workspace['id']}/stream?timeout_s=1&max_events=50"
    ).text
    seen = [e["sequence"] for e in _sse(first) if "type" in e]
    resumed = client.get(
        f"/api/v1/workspaces/{api_workspace['id']}/stream?timeout_s=1&max_events=50",
        headers={"last-event-id": str(max(seen))},
    ).text
    assert [e["sequence"] for e in _sse(resumed) if "type" in e] == []


def test_the_last_event_id_header_beats_the_after_parameter(client, api_workspace, populated):
    """`after` is what the browser believed when it opened the connection; the header is what it
    actually received. The header has to win or a reconnect replays events twice."""
    everything = _sse(client.get(
        f"/api/v1/workspaces/{api_workspace['id']}/stream?timeout_s=1"
    ).text)
    highest = max(e["sequence"] for e in everything if "type" in e)
    body = client.get(
        f"/api/v1/workspaces/{api_workspace['id']}/stream?after=0&timeout_s=1",
        headers={"last-event-id": str(highest)},
    ).text
    assert [e for e in _sse(body) if "type" in e] == []


def test_the_stream_ends_on_its_own(client, api_workspace, populated):
    """An unbounded generator holds a worker and a connection open for a client that may already
    be gone. `EventSource` reconnects by itself, so a bounded stream costs nothing."""
    body = client.get(
        f"/api/v1/workspaces/{api_workspace['id']}/stream?timeout_s=1&max_events=1"
    ).text
    assert "event: end" in body


def test_the_stream_of_another_organizations_workspace_is_a_404(client, api_workspace):
    response = client.get(f"/api/v1/workspaces/{uuid.uuid4()}/stream?timeout_s=1")
    assert response.status_code == 404


# --------------------------------------------------------------------------- the page itself


def test_the_ui_is_served_by_the_same_process_as_the_api(client):
    """§6, §33: one artifact, and a Colab runtime renders the same UI as production."""
    page = client.get("/ui/")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert "<title>Civitas</title>" in page.text

    for asset in ("/ui/app.js", "/ui/app.css"):
        assert client.get(asset).status_code == 200


def test_the_root_redirects_to_the_ui(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/ui/"


def test_the_page_loads_nothing_from_a_third_party(client):
    """No CDN and no build step. A UI that fetched a framework at load time would not render in an
    air-gapped deployment or a Colab runtime without outbound network (§33)."""
    for asset in ("/ui/", "/ui/app.js", "/ui/app.css"):
        body = client.get(asset).text
        for marker in ("https://", "http://", "cdn.", "unpkg", "jsdelivr"):
            assert marker not in body, f"{asset} references {marker}"
