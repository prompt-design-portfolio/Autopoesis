"""The G0 pins, and the discipline around them (A3, B§5.1)."""

from __future__ import annotations

import subprocess

from civitas_g import manifest as M


def test_every_pinned_artifact_still_matches_the_working_tree():
    failures = [detail for ok, detail in M.verify_pins() if not ok]
    assert not failures, "\n".join(failures)


def test_the_gated_references_are_the_two_with_committed_producing_code():
    keys = {r.key for r in M.GATED_REFERENCES}
    assert keys == {"precheck_v3_13", "v3_11_grid"}


def test_the_v3_12_precheck_is_recorded_but_never_gated():
    """F4/D1: its producing code does not exist at the commit that added it."""
    ref = M.PRECHECK_V3_12
    assert ref.gated is False
    assert ref.sim is None and ref.analysis is None
    at_commit = subprocess.run(
        ("git", "show", f"{ref.summary.commit}:sim_v3_12.py"),
        cwd=M.REPO_ROOT, capture_output=True, timeout=30)
    assert at_commit.returncode != 0, (
        "sim_v3_12.py now resolves at cc8a0a6; if the history changed, D1 must be re-ruled")


def test_the_empty_sha_is_named_so_it_cannot_be_recorded_as_a_file_hash():
    """`git show <commit>:<absent path>` produces no bytes; hashing that yields a real-looking
    hash for a file that is not there."""
    assert M.sha256_bytes(b"") == M.EMPTY_SHA256
    for ref in M.REFERENCES:
        for pin in (ref.summary, ref.sim, ref.analysis):
            if pin is not None:
                assert pin.sha256 != M.EMPTY_SHA256


def test_the_reference_invocation_records_which_parts_are_inference():
    """F5. The file states neither its seeds nor its configuration."""
    inv = M.REFERENCE_INVOCATION
    assert inv["seed"] is None
    assert "NOT RECOVERABLE" in inv["seed_basis"]
    for k in ("arms", "phase_steps", "prep_every", "n_seeds"):
        assert inv[f"{k}_basis"], f"{k} has a value with no stated basis"
    assert inv["matches_a_committed_mode"] is False


def test_the_missing_artifacts_are_recorded_rather_than_reconstructed():
    names = {name for name, _ in M.MISSING_ARTIFACTS}
    assert "v3_12_finding.md" in names
    assert "assay_selftest" in names
    assert not (M.REPO_ROOT / "v3_12_finding.md").exists()


def test_a_manifest_carries_the_engine_hash():
    """A1.8: its hash is in every manifest."""
    m = M.build_manifest(kind="test", arms=["collective"], seeds=[0], phase_steps=100)
    assert len(m.engine["sim_v3_13.py"]) == 64
    assert m.engine["engine_tree_clean"] in ("yes", "no")
    assert m.world["prep_value"] == 1.0
    assert any(r["gated"] for r in m.references)


def test_a_pin_reports_a_mismatch_rather_than_passing_quietly(tmp_path):
    (tmp_path / "spec_v3_13.md").write_text("not the file")
    pin = next(p for p in M.CONTEXT_PINS if p.path == "spec_v3_13.md")
    ok, detail = pin.verify(tmp_path)
    assert not ok and "pinned" in detail


def test_an_absent_file_is_reported_as_absent_not_as_a_mismatch(tmp_path):
    pin = next(p for p in M.CONTEXT_PINS if p.path == "spec_v3_13.md")
    ok, detail = pin.verify(tmp_path)
    assert not ok and "absent" in detail
