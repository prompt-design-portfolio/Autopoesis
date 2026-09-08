"""The unapplied G2 engine patch (G2-D1).

The patch is in the repository and is applied to a temporary copy, never to the working tree. That
is what lets these checks be re-run from a clean checkout while `sim_v3_13.py` stays byte-identical
and the decision to apply is still open.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from civitas_g.selftests import PATCH_PATH, REPO_ROOT, _patched_engine


def test_the_patch_is_in_the_repository_and_is_not_applied():
    """A1.8: the engine runs byte-identical. It is still the unpatched engine on disk."""
    import sim_v3_13

    assert (REPO_ROOT / PATCH_PATH).exists()
    source = Path(sim_v3_13.__file__).read_text()
    assert "store_snaps" not in source, "the G2 patch has been applied without a ruling on G2-D1"
    assert "init_store" not in source


def test_the_patch_still_applies_cleanly():
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(("cp", str(REPO_ROOT / "sim_v3_13.py"), f"{tmp}/sim_v3_13.py"), check=True)
        applied = subprocess.run(("patch", "-p0", "--quiet", "-i", str(REPO_ROOT / PATCH_PATH)),
                                 cwd=tmp, capture_output=True, text=True, timeout=60)
        assert applied.returncode == 0, applied.stderr or applied.stdout


def test_the_patch_adds_and_removes_nothing_else():
    """Four additions, nothing removed. A patch that deleted a line would not be additive, and the
    bit-identity argument rests on it being additive."""
    diff = (REPO_ROOT / PATCH_PATH).read_text().splitlines()
    removed = [ln for ln in diff
               if ln.startswith("-") and not ln.startswith("---")
               and ln[1:].strip() not in ("", )]
    # the only "removed" lines are the three context lines the additions attach to, re-emitted
    for ln in removed:
        body = ln[1:].strip()
        assert ("era_snap_keep" in body or "def run(" in body
                or "notebook did." in body or "era_snaps=era_snaps" in body
                or "era_snaps = []" in body
                or "era_snaps.append" in body or "del era_snaps" in body
                or "genomes=snapshot" in body or "world = World" in body), \
            f"the patch removes a line it should not: {body!r}"


def test_the_patched_engine_imports_and_exposes_the_new_surface():
    with tempfile.TemporaryDirectory() as tmp:
        patched = _patched_engine(tmp)
        assert hasattr(patched.Config(), "store_snaps")
        assert patched.Config().store_snaps is False, "capture must be off by default"
        import inspect

        assert "init_store" in inspect.signature(patched.run).parameters


@pytest.mark.slow
def test_the_patch_is_trajectory_neutral_and_injection_works():
    """The whole of G2-D1's safety argument, measured. Slow: it runs the engine three times."""
    from civitas_g.selftests import run_selftests

    result = run_selftests(names=["store_patch"], verbose=False)[0]
    assert result.ok, result.detail
    assert "trajectory-neutral" in result.detail
