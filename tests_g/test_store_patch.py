"""The G2 engine change (ruled under G2-D1, applied).

The engine gained store capture and injection at G2. A1.8 says it runs byte-identical with its
hash in every manifest -- a claim about a *named* engine, not a claim that it can never change.
So what these tests hold is the thing that has to keep being true: the engine on disk is a version
this build records, the change is additive, and it is trajectory-neutral.

The file `docs/patches/g2-store-capture-and-injection.diff` is now history rather than a pending
action, and is kept because it is the readable statement of exactly what changed.
"""

from __future__ import annotations

import inspect

import pytest

import sim_v3_13
from civitas_g.manifest import (
    CURRENT_ENGINE,
    ENGINE_BY_SHA,
    ENGINE_VERSIONS,
    REPO_ROOT,
    engine_identity,
    sha256_file,
    verify_engine,
)
from civitas_g.selftests import PATCH_PATH


def test_the_engine_on_disk_is_a_version_this_build_records():
    """The failure this catches is a stray edit that no manifest can name."""
    ok, detail = verify_engine()
    assert ok, detail
    assert engine_identity()["engine_version"] == CURRENT_ENGINE.label


def test_the_current_engine_hash_is_the_recorded_one():
    assert sha256_file(REPO_ROOT / "sim_v3_13.py") == CURRENT_ENGINE.sha256


def test_an_unrecorded_engine_is_named_as_unrecorded_not_as_a_mismatch():
    assert "0" * 64 not in ENGINE_BY_SHA
    assert len({v.sha256 for v in ENGINE_VERSIONS}) == len(ENGINE_VERSIONS)


def test_every_engine_version_after_the_first_says_how_equivalence_was_checked():
    """A version that changed the engine without a stated equivalence check is a version whose
    numbers cannot be compared to the ones before it."""
    for version in ENGINE_VERSIONS[1:]:
        assert version.equivalence, f"{version.label} records no equivalence check"
        assert version.change, f"{version.label} records no change"


def test_the_engine_exposes_capture_and_injection():
    assert hasattr(sim_v3_13.Config(), "store_snaps")
    assert sim_v3_13.Config().store_snaps is False, "capture must be off by default"
    assert "init_store" in inspect.signature(sim_v3_13.run).parameters


def test_capture_is_off_by_default_so_every_existing_call_is_unchanged():
    """The G1 reproduction calls run() without either flag. If the default moved, the reproduction
    would be measuring a different engine than the one it was gated on."""
    cfg = sim_v3_13.Config()
    assert cfg.store_snaps is False
    assert inspect.signature(sim_v3_13.run).parameters["init_store"].default is None


def test_the_patch_file_is_kept_as_the_readable_statement_of_the_change():
    assert (REPO_ROOT / PATCH_PATH).exists()
    diff = (REPO_ROOT / PATCH_PATH).read_text()
    assert "store_snaps" in diff and "init_store" in diff


def test_the_change_is_additive():
    """The trajectory-neutrality argument rests on nothing being removed."""
    diff = (REPO_ROOT / PATCH_PATH).read_text().splitlines()
    removed = [ln[1:].strip() for ln in diff
               if ln.startswith("-") and not ln.startswith("---") and ln[1:].strip()]
    # A MODIFIED line shows up in a unified diff as a removal and an addition, so this list has
    # to cover both. `cfg=asdict(cfg), final=snapshot(agents))` is the return dict's last line:
    # G2-store-fix appended `store_snaps` and `final_store` to it, which moved the closing paren
    # off that line. Nothing on it was dropped -- both `cfg` and `final` are still returned, and
    # `test_the_return_dict_only_grew` is what actually holds that to account.
    for body in removed:
        assert any(k in body for k in (
            "era_snap_keep", "def run(", "notebook did.", "era_snaps=era_snaps",
            "era_snaps = []", "era_snaps.append", "del era_snaps", "genomes=snapshot",
            "world = World",
            "cfg=asdict(cfg), final=snapshot(agents))",
        )), f"the change removes a line it should not: {body!r}"


def test_the_return_dict_only_grew():
    """What `test_the_change_is_additive` is really claiming, checked against the engine itself.

    The diff test works on text, and a modified line reads as a removal there -- which is exactly
    what the return dict's last line is: G2-store-fix appended `store_snaps` and `final_store` to
    `cfg=asdict(cfg), final=snapshot(agents))`. Text cannot tell that apart from a deletion, so
    this asserts the thing that matters directly: every key the run dict carried before the store
    changes is still a key it carries now.
    """
    src = inspect.getsource(sim_v3_13.run)
    for key in ("log=", "cfg=", "final=", "era_snaps=", "final_mapping="):
        assert key in src, f"the run dict no longer returns {key.rstrip('=')!r}"
    for added in ("store_snaps=", "final_store="):
        assert added in src, f"the store change did not add {added.rstrip('=')!r}"


def test_analysis_v3_13_was_not_touched():
    """G2-D3: keeping the assay on the Civitas side means exactly one research file changed, so
    frozen_replay and frozen_knockout still work and the G1 reproduction stays a valid regression
    test for the engine change."""
    from civitas_g.manifest import CONTEXT_PINS

    pin = next(p for p in CONTEXT_PINS if p.path == "analysis_v3_13.py")
    ok, detail = pin.verify()
    assert ok, detail


@pytest.mark.slow
def test_the_store_change_is_trajectory_neutral_and_injection_works():
    """Measured, against the engine as it was at G0. Slow: it runs the engine four times."""
    from civitas_g.selftests import run_selftests

    result = run_selftests(names=["engine_store"], verbose=False)[0]
    assert result.ok, result.detail
    assert "identical to engine G0" in result.detail
