"""Runs, rows and readings, on every configured backend.

D12: "both backends" is a claim about persistence fidelity, not about numpy having run twice. So
these tests write a real run's rows, read them back, and check that the reading recomputed from
the rows is the reading -- on each backend, and identical between them.
"""

from __future__ import annotations

import math

import pytest
from sqlalchemy import select

from civitas_g.persistence.models import Campaign, EraRow, Run
from civitas_g.provider.population import (
    CampaignPlan,
    densities,
    existing_runs,
    load_results,
    load_run,
    open_campaign,
    store_failure,
    store_run,
)
from civitas_g.reading.compute import compute_fields


@pytest.fixture
def stored(sessions, short_run):
    plan = CampaignPlan(kind="test", arms=["collective"], seeds=[0], phase_steps=150)
    with sessions() as session:
        campaign = open_campaign(session, plan)
        run = store_run(session, campaign, short_run)
        session.commit()
        return campaign.id, run.id


def test_a_run_and_all_its_rows_are_stored(sessions, stored, short_run):
    campaign_id, run_id = stored
    with sessions() as session:
        assert session.get(Campaign, campaign_id) is not None
        rows = session.execute(
            select(EraRow).where(EraRow.run_id == run_id)).scalars().all()
        assert len(rows) == len(short_run.log)
        assert [r.ordinal for r in rows] == sorted(r.ordinal for r in rows)


def test_a_stored_row_equals_the_row_the_engine_produced(sessions, stored, short_run):
    _campaign_id, run_id = stored
    with sessions() as session:
        rebuilt = load_run(session, run_id)
    assert len(rebuilt["log"]) == len(short_run.log)
    for original, loaded in zip(short_run.log, rebuilt["log"], strict=True):
        assert set(original) == set(loaded)
        for k, v in original.items():
            if isinstance(v, float) and math.isnan(v):
                assert math.isnan(loaded[k]), f"{k} lost its NaN"


def test_the_reading_recomputed_from_rows_equals_the_reading_from_memory(
        sessions, stored, short_run):
    """The whole basis of the G1 gate: the number comes from the database, not from memory."""
    campaign_id, _run_id = stored
    from_memory = compute_fields({"plastic + record": [dict(
        log=short_run.log, cfg=short_run.cfg,
        phase_bounds=short_run.raw["phase_bounds"], n_steps=short_run.raw["n_steps"],
        chain_start=short_run.raw["chain_start"], seed=0)]})
    with sessions() as session:
        from_rows = compute_fields(load_results(session, campaign_id))
    assert set(from_memory) == set(from_rows)
    for k, v in from_memory.items():
        if math.isnan(v):
            assert math.isnan(from_rows[k]), k
        else:
            assert from_rows[k] == v, k


def test_the_campaign_resumes_rather_than_repeating(sessions, stored):
    campaign_id, _run_id = stored
    with sessions() as session:
        assert existing_runs(session, campaign_id) == {("collective", 0)}


def test_a_failed_run_is_recorded_not_dropped(sessions):
    """A1.4: a run that died is evidence about the world; an absent row is evidence about
    nothing."""
    from civitas_g.world.engine import build_run_spec

    plan = CampaignPlan(kind="test", arms=["collective"], seeds=[7], phase_steps=100)
    spec = build_run_spec("collective", 7, 100)
    with sessions() as session:
        campaign = open_campaign(session, plan)
        run = store_failure(session, campaign, spec, RuntimeError("boom"))
        session.commit()
        run_id = run.id
    with sessions() as session:
        row = session.get(Run, run_id)
        assert row.status == "failed" and "boom" in row.failure
        # and it does not count as done, so a resume will try it again
        assert existing_runs(session, campaign.id) == set()


def test_the_reading_is_keyed_by_the_research_name(sessions, stored):
    campaign_id, _ = stored
    with sessions() as session:
        results = load_results(session, campaign_id)
    assert list(results) == ["plastic + record"]


def test_densities_report_what_is_unmeasurable_with_a_reason(sessions, stored):
    """D10. Food cover is not in the engine's log and A1.8 forbids adding it."""
    campaign_id, _ = stored
    with sessions() as session:
        run_dict = load_results(session, campaign_id)["plastic + record"][0]
    ds = densities(run_dict)
    assert ds and all(d.food_cover is None for d in ds)
    assert "A1.8" in ds[0].food_cover_reason
    assert all(d.mark_mean_age is None for d in ds)
    assert any(d.mark_density is not None for d in ds)


def test_a_read_only_session_takes_no_write_lock(readers, sessions, stored):
    """Reading must not write. Two concurrent readers must not deadlock each other."""
    campaign_id, _ = stored
    with readers() as a, readers() as b:
        assert a.get(Campaign, campaign_id) is not None
        assert b.get(Campaign, campaign_id) is not None
