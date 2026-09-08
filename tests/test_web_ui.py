"""The UI, driven in a real browser (Part B §49).

An assertion that a route returns JSON is not evidence that a screen renders it. These tests start
the real app on a real port, open the real page in Chromium, and read what a person would see.

Two of them are the reason this file exists rather than being a nicety. `absence is not zero`
(ARCHITECTURE §3.9) and `a failed gate withholds the result` (§3.1) are properties of the
*rendered page*: a client that received `null` and printed `0.0000` would satisfy every API test
in the suite and still lie to every reader.

Skipped, with a stated reason, when Chromium is not available — never silently passed.
"""

from __future__ import annotations

import json
import socket
import threading
import uuid

import pytest

from civitas.domain.enums import EventType, ExperimentArm, TerminationReason
from civitas.persistence.events import emit
from civitas.persistence.models import AgentProfile, Artifact, Episode, Task

playwright_api = pytest.importorskip(
    "playwright.sync_api", reason="playwright is not installed; the browser tests cannot run"
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


#: Where a preinstalled Chromium may live. Playwright pins a build number, and a runner that
#: shipped a different one is the common case rather than the exception — so the binary is looked
#: for rather than downloaded, and the test skips with a stated reason when there is none.
CHROMIUM_CANDIDATES = (
    "/opt/pw-browsers/chromium/chrome-linux/chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/usr/bin/google-chrome",
)


def _chromium_path() -> str | None:
    import glob
    import os

    for pattern in ("/opt/pw-browsers/chromium-*/chrome-linux/chrome", *CHROMIUM_CANDIDATES):
        for match in sorted(glob.glob(pattern), reverse=True):
            if os.access(match, os.X_OK):
                return match
    return None


@pytest.fixture(scope="function")
def browser():
    from playwright.sync_api import Error, sync_playwright

    with sync_playwright() as p:
        launch: dict = {"args": ["--no-sandbox"]}
        executable = _chromium_path()
        if executable:
            launch["executable_path"] = executable
        try:
            instance = p.chromium.launch(**launch)
        except Error as exc:  # pragma: no cover - environment without a browser binary
            pytest.skip(f"chromium could not launch: {exc}")
        try:
            yield instance
        finally:
            instance.close()


@pytest.fixture
def live_server(api):
    """The real app on a real port. `TestClient` cannot serve a browser."""
    import uvicorn

    port = _free_port()
    config = uvicorn.Config(api, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = 200
    while not server.started and deadline:
        threading.Event().wait(0.05)
        deadline -= 1
    if not server.started:  # pragma: no cover
        pytest.skip("the test server did not start")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.fixture
def seeded(db, api_workspace):
    """Enough rows for the dashboards to have something to render — and, deliberately, not enough
    for the specialization index to be computable. That absence is the thing under test."""
    workspace_id = uuid.UUID(api_workspace["id"])
    profile = AgentProfile(workspace_id=workspace_id, name="prober", role="explorer")
    db.add(profile)
    db.flush()
    task = Task(workspace_id=workspace_id, title="Determine the accepted operation",
                description="d", task_family="hidden_rule", status="ready",
                evaluator_spec={"kind": "exact_match", "expected": "ferrose"})
    db.add(task)
    db.flush()
    episode = Episode(
        workspace_id=workspace_id, agent_profile_id=profile.id, task_id=task.id,
        model_provider="deterministic", model_name="policy-v1",
        experiment_arm=ExperimentArm.COLLECTIVE,
        termination_reason=TerminationReason.EVALUATOR_SUCCESS,
        tokens_used=1200, tool_calls_used=4, cost_usd=0.0,
        environment_version="device-e1-s0",
    )
    db.add(episode)
    db.flush()
    db.add(Artifact(workspace_id=workspace_id, type="observation",
                    title="the device rejects fold for aurex", body="b",
                    creator_episode_id=episode.id))
    emit(db, workspace_id=workspace_id, type=EventType.EPISODE_STARTED,
         episode_id=episode.id, payload={"seeded": True})
    db.commit()
    return {"workspace_id": workspace_id, "episode": episode, "task": task}


@pytest.fixture
def page(browser, live_server, bootstrapped, seeded):
    _org, _admin, key = bootstrapped
    context = browser.new_context()
    page = context.new_page()
    page.goto(live_server + "/ui/", wait_until="domcontentloaded")
    # The key lives in sessionStorage exactly as the header dialog puts it there.
    page.evaluate("k => sessionStorage.setItem('civitas.apiKey', k)", key)
    page.reload(wait_until="domcontentloaded")
    page.wait_for_function("() => window.civitas && window.civitas.state.workspaceId")
    try:
        yield page
    finally:
        context.close()


def _open(page, screen: str):
    """Navigate and wait for *that* screen's render to land.

    Waiting on `main h1` alone is not enough: the page renders the default screen while the
    fixture is still setting up, and two renders in flight at once is exactly the race the UI's
    `renderToken` exists to settle. The wait is on the render having finished with nothing left
    loading."""
    page.evaluate("s => { location.hash = '#/' + s; }", screen)
    page.wait_for_function("s => window.civitas.state.screen === s", arg=screen.split("?")[0])
    page.wait_for_selector("main h1")
    page.wait_for_function(
        "() => { const n = document.querySelector('main .empty');"
        "        return !n || !n.textContent.includes('Loading'); }"
    )
    return page.inner_text("main")


# --------------------------------------------------------------------------- the page loads


def test_the_page_renders_with_no_console_error(page, live_server):
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    _open(page, "overview")
    assert errors == [], errors


def test_every_navigable_screen_renders(page):
    """Thirteen screens, each actually opened. A screen that throws on render is a screen that
    does not exist, however complete the route table looks."""
    screens = page.evaluate(
        "() => window.civitas.SCREENS.filter(s => s.id && !s.hidden).map(s => s.id)"
    )
    assert len(screens) >= 13, f"only {len(screens)} navigable screens"

    failures = {}
    for screen in screens:
        text = _open(page, screen)
        if "Error" in text or page.query_selector("main .err"):
            failures[screen] = page.inner_text("main .err")
    assert failures == {}, failures


# --------------------------------------------------------------------------- §3.9 on screen


def test_an_unavailable_metric_renders_as_unavailable_and_never_as_zero(page):
    """ARCHITECTURE §3.9, checked where it actually matters — on the screen.

    The specialization index is `null` here because there is not enough evidence to compute it.
    An index of 0.0000 would mean "every profile performs identically", which is a finding. A
    client that printed the second for the first would pass every API test in this suite.
    """
    text = _open(page, "specialization")
    assert "not available" in text
    assert "0.0000" not in text.split("Profile performance")[0]


def test_an_unpriced_bucket_renders_as_unknown_cost(page):
    """The seeded episodes run on a deterministic provider, which is genuinely free — so the mean
    cost per episode is unknown, not zero, and the cost screen has to say so (§39)."""
    text = _open(page, "cost")
    assert "not available" in text


def test_the_metrics_screen_states_how_many_metrics_are_not_computable(page):
    text = _open(page, "metrics")
    assert "not computable yet" in text
    assert "not available" in text


# --------------------------------------------------------------------------- §3.1 on screen


def test_a_gated_out_benchmark_shows_the_gates_and_withholds_the_numbers(page, db, seeded,
                                                                        api_workspace):
    """ARCHITECTURE §3.1 rendered: a failed gate does not annotate a result, it withholds it.

    The experiment below has a failed gate and metrics attached. If the screen showed the metrics
    anyway, a reader would have a number the run did not earn — which is the exact failure the
    gate discipline exists to prevent, arriving through the UI rather than through the API.
    """
    from civitas.persistence.models import Experiment, ExperimentRun
    from civitas.persistence.models import ExperimentArm as ExperimentArmRow

    experiment = Experiment(
        workspace_id=seeded["workspace_id"], name="newcomer_advantage",
        hypothesis="the collective helps", status="completed",
    )
    db.add(experiment)
    db.flush()
    arm = ExperimentArmRow(experiment_id=experiment.id, arm=ExperimentArm.COLLECTIVE,
                           name="collective")
    db.add(arm)
    db.flush()
    db.add(ExperimentRun(
        experiment_id=experiment.id, experiment_arm_id=arm.id, seed=0, status="completed",
        config_hash="deadbeef",
        gates={"accumulation_occurred": {"passed": False,
                                         "detail": "0 artifacts accumulated"},
               "frozen_model": {"passed": True, "detail": "one configuration hash"}},
        metrics={"newcomer_advantage": 0.42},
    ))
    db.commit()

    text = _open(page, "benchmarks")
    assert "Metrics withheld" in text
    assert "0 artifacts accumulated" in text
    assert "0.42" not in text, "a withheld metric was rendered anyway"


# --------------------------------------------------------------------------- §47 on screen


def test_no_screen_discloses_a_task_evaluator_specification(page):
    """§47 over the whole rendered surface, not one endpoint. The expected answer is seeded into
    the task's `evaluator_spec`; if any screen renders it, an agent's operator can read the
    benchmark."""
    for screen in ("overview", "tasks", "episodes", "artifacts", "events", "metrics"):
        text = _open(page, screen)
        assert "ferrose" not in text, f"the {screen} screen disclosed the expected answer"


def test_the_episode_inspector_shows_what_the_episode_did_and_not_what_it_thought(page, seeded):
    """§4: there is no reasoning to show, and the screen says so rather than leaving a reader to
    wonder whether it was hidden."""
    page.evaluate("id => { location.hash = '#/episode?id=' + id; }", str(seeded["episode"].id))
    page.wait_for_function("() => window.civitas.state.screen === 'episode'")
    page.wait_for_selector("main h1")
    text = page.inner_text("main")

    assert "private reasoning" in text
    assert "ferrose" not in text
    assert "collective" in text


# --------------------------------------------------------------------------- live updates


def test_the_page_receives_a_new_event_over_the_live_stream(page, db, seeded):
    """§45 as the transport for §49's live updates. Written to the database from outside the
    browser and expected to arrive on the open page without a reload."""
    _open(page, "events")
    page.wait_for_selector("#event-rows tbody tr")
    before = page.eval_on_selector_all("#event-rows tbody tr", "rows => rows.length")

    emit(db, workspace_id=seeded["workspace_id"], type=EventType.ARTIFACT_CREATED,
         payload={"marker": "live-update-probe"})
    db.commit()

    page.wait_for_function(
        "n => document.querySelectorAll('#event-rows tbody tr').length > n",
        arg=before, timeout=15000,
    )
    assert "live-update-probe" in page.inner_text("#event-rows")


def test_the_live_badge_reports_the_stream_state(page):
    _open(page, "overview")
    page.wait_for_function("() => document.getElementById('live').className.includes('on')",
                           timeout=15000)


# --------------------------------------------------------------------------- provenance


def test_the_provenance_inspector_reaches_an_artifact_from_a_link(page, db, seeded):
    """§13, §49: a reader must be able to go from a claim to what it rests on without leaving the
    page or constructing a URL by hand."""
    _open(page, "artifacts")
    page.wait_for_selector("main table tbody tr a")
    page.click("main table tbody tr a")
    page.wait_for_function("() => window.civitas.state.screen === 'provenance'")
    page.wait_for_selector("main h1")
    assert "Provenance" in page.inner_text("main h1")
    assert page.evaluate("() => window.civitas.state.params.artifact")


# --------------------------------------------------------------------------- auth


def test_without_a_key_the_page_says_so_rather_than_rendering_an_empty_collective(
    browser, live_server
):
    """An unauthenticated page that rendered zeros would be indistinguishable from a collective
    that had done nothing."""
    context = browser.new_context()
    page = context.new_page()
    try:
        page.goto(live_server + "/ui/", wait_until="domcontentloaded")
        # An <option> inside a <select> is never "visible" to Playwright, so the wait is on the
        # option existing rather than on it being rendered.
        page.wait_for_function(
            "() => document.querySelectorAll('#workspace-select option').length > 0"
        )
        assert "set an API key" in page.eval_on_selector(
            "#workspace-select", "el => el.textContent"
        )
        # And no screen may render a count: a page that showed zeros without a key would be
        # indistinguishable from a collective that had done nothing.
        assert page.query_selector("main .cards") is None
    finally:
        context.close()


def test_the_api_key_never_leaves_this_browser_except_as_a_header(page, live_server):
    """It is held in `sessionStorage`, not a cookie: a cookie would be attached to every request
    the browser makes to this origin, including ones the page did not initiate."""
    cookies = page.context.cookies()
    assert all("civitas" not in json.dumps(c).lower() or "apiKey" not in json.dumps(c)
               for c in cookies)
    stored = page.evaluate("() => sessionStorage.getItem('civitas.apiKey')")
    assert stored
    assert page.evaluate("() => localStorage.getItem('civitas.apiKey')") is None

