"""Manifests (§46, kept), and the G0 hash pins (A3, B§5.1).

A3's G1 row ends: "the reference summaries and the code they were produced by recorded by hash".
B§5.1 is sharper -- "record the reference files and the producing code by hash at G0 and reproduce
against those, **not against anything that lands later**". The G0 audit found why that sentence is
load-bearing: every candidate reference summary in this repository was produced by code that was
changed afterwards, so "the producing code" cannot mean HEAD.

    summary                    committed at   producing code touched after?
    v3_11_grid_summary.txt     1619ce4        yes -- d9f11fa, a9c3765
    precheck_v3_12.txt         cc8a0a6        the v3.12 code DOES NOT EXIST at that commit
    precheck_v3_13.txt         3d7b228        yes -- 41c100d

So each reference pins two code hashes: the blob **at the summary's own commit**, which is the
code that actually produced it, and the blob at HEAD, which is what a run today would use. Where
those differ, the difference is stated rather than assumed away -- `SIM_V3_13_DRIFT` below records
that the v3.13 engine drift is additive recording only and therefore trajectory-preserving, and
`civitas_g.selftests.engine_drift_selftest` turns that reading of a diff into a measurement.

`precheck_v3_12.txt` has no producing code at its own commit and is therefore recorded but never
gated on (D1). Recording it is not a formality: A1.4 says nothing is hard-deleted, and an artifact
whose provenance is broken is worth more as a documented gap than as an absence.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file on disk."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


#: SHA-256 of the empty string. `git show <commit>:<absent path>` produces no bytes, and hashing
#: that would record a real-looking hash for a file that is not there. Named so it can be refused.
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


@dataclass(frozen=True)
class Pin:
    """One file, recorded by hash at a named commit."""

    path: str
    sha256: str
    commit: str
    note: str = ""

    def verify(self, root: Path | None = None) -> tuple[bool, str]:
        """Does the file on disk still match this pin? Returns `(ok, detail)`."""
        p = (root or REPO_ROOT) / self.path
        if not p.exists():
            return False, f"{self.path}: absent from the working tree"
        actual = sha256_file(p)
        if actual == self.sha256:
            return True, f"{self.path}: matches {self.sha256[:12]}"
        return False, (f"{self.path}: on disk {actual[:12]}, pinned {self.sha256[:12]} "
                       f"({self.commit})")


@dataclass(frozen=True)
class Reference:
    """A reference summary, the code that produced it, and whether G1 gates on it."""

    key: str
    summary: Pin
    sim: Pin | None
    analysis: Pin | None
    gated: bool
    why: str


# ---------------------------------------------------------------------------------------------
# the G0 pins
# ---------------------------------------------------------------------------------------------

G0_COMMIT = "63d42f69d62c7490b5e6cb173a4f1b2e1258d3a2"

#: PRIMARY. B§5.1 asks for "whatever v3.13 output is committed at that moment"; this is it.
#: Its invocation is not recorded in the file -- see `REFERENCE_INVOCATION` and D4.
PRECHECK_V3_13 = Reference(
    key="precheck_v3_13",
    summary=Pin("precheck_v3_13.txt",
                "89b493e8de42dba7df5533816c35807c551636b115f9cdedfd3e8014065caeb9", "3d7b228"),
    sim=Pin("sim_v3_13.py",
            "fa577b9c171e27b967b4c6f21d11840b293ced5a46630c92cbc93e4c8f4afdcd", "3d7b228",
            "the blob that produced the summary; HEAD's differs, additively -- see\n"
            "SIM_V3_13_DRIFT"),
    analysis=Pin("analysis_v3_13.py",
                 "844b1487f302e63c277d0fb153e004c602ef2c10d45f1b7badd7b6cc54cf5058", "3d7b228",
                 "HEAD's prints an extra (ii-newborn) block and changes no number this file has"),
    gated=True,
    why="the v3.13 output committed at G0, which is exactly what B§5.1 names",
)

#: SECONDARY. Three seeds and per-arm, per-seed tables -- the only committed artifact that can show
#: a reproduction holding ACROSS seeds. A different world: K = 3, and no record at all.
V3_11_GRID = Reference(
    key="v3_11_grid",
    summary=Pin("v3_11_grid_summary.txt",
                "4a37d81fffe28f70a520f3d00cd92fcbf0019244e87fd0791624edae4591051b", "1619ce4"),
    sim=Pin("sim_v3_11.py",
            "469f3f31368d17c2328f4bafddc2afe127ed72583a67db0f7f3e7a78ea569c1e", "1619ce4"),
    analysis=Pin("analysis_v3_11.py",
                 "666827b1837a6ae7065122f96c1d7ee112a50b913606f110b766e400ed581134", "1619ce4"),
    gated=True,
    why="three seeds; the only committed multi-seed numeric artifact",
)

#: RECORDED, NEVER GATED (D1). `sim_v3_12.py` and `analysis_v3_12.py` do not exist at cc8a0a6, the
#: commit that added this summary. Reproducing it would mean reproducing against a guess that the
#: file on disk today is the file that ran.
PRECHECK_V3_12 = Reference(
    key="precheck_v3_12",
    summary=Pin("precheck_v3_12.txt",
                "0488d651a11acc893bb39b657d76af0f8fcd1ebb611fbb49c798d43b1026401f", "cc8a0a6"),
    sim=None,
    analysis=None,
    gated=False,
    why="no producing code is committed at its own commit; recorded as a documented gap",
)

REFERENCES: tuple[Reference, ...] = (PRECHECK_V3_13, V3_11_GRID, PRECHECK_V3_12)
GATED_REFERENCES: tuple[Reference, ...] = tuple(r for r in REFERENCES if r.gated)

#: Context, recorded at HEAD. Not reproduced against.
CONTEXT_PINS: tuple[Pin, ...] = (
    Pin("spec_v3_13.md",
        "efdd8a0951a20a6c7f514901010b3a60d256d67c385305a33737bd226bb806d0", G0_COMMIT),
    Pin("spec_v3_12.md",
        "fa7d095cbaed431a82476dbcbceabbc113b4f37d32fec3effa2ebf6204e4e23f", G0_COMMIT),
    Pin("spec_v3_11.md",
        "59d95ec623807a20b029b68dcea88a198d9f8c2c5ddbf0d67bc8d8bb3c941bb9", G0_COMMIT),
    Pin("v3_11_finding.md",
        "3a1ddf4fb9b3abbefffe365fdf4864c10f052a033702c3add9374a1d9d5da5eb", G0_COMMIT),
    Pin("make_notebook_v3_13.py",
        "b37e08877f8770c0d88d87a47acdf575ec46128600896f5159cbb385c89dc13c", G0_COMMIT),
    Pin("autopoiesis_v3_13_record.ipynb",
        "a3877ffd363a141f1b62ddcb08f48997ee391acc82a8960da1a784157834d829", G0_COMMIT),
    Pin("analysis_v3_13.py",
        "9e6911263f2ad3117a441fada46c25056989dcc84d22e456e48f71e3d4048aad", G0_COMMIT,
        "the reading. NOT touched by the G2 change -- frozen_replay and frozen_knockout still "
        "work exactly as they did, which is what keeps the G1 reproduction a valid regression "
        "test for the engine change."),
)


@dataclass(frozen=True)
class EngineVersion:
    """One version of the compute engine, and what made it a different one.

    A1.8 says the engine runs byte-identical and its hash is in every manifest. That is a claim
    about a *named* engine, not a claim that it can never change: the directive's own escape hatch
    is that an instrument change "arrives as a versioned change with its own gate". So the engine
    is versioned here, every version keeps its hash, and a manifest names which one produced each
    number. What is forbidden is an engine that moved without anyone recording that it did.
    """

    label: str
    sha256: str
    at_commit: str
    change: str
    #: How the claim that this version is trajectory-equivalent to its predecessor was checked.
    equivalence: str = ""


#: Oldest first. The current engine is the last entry.
ENGINE_VERSIONS: tuple[EngineVersion, ...] = (
    EngineVersion(
        "reference", "fa577b9c171e27b967b4c6f21d11840b293ced5a46630c92cbc93e4c8f4afdcd",
        "3d7b228",
        "the blob that produced precheck_v3_13.txt",
        ""),
    EngineVersion(
        "G0", "3b51b191d953a71691831095e1f631eb69ab07d1d17b180376547ea4fceb84f5",
        G0_COMMIT,
        "adds the four (ii-newborn) counters (41c100d)",
        "engine_drift_selftest: 129 shared log fields identical to `reference` across a mapping "
        "remap; the only difference is the four added counters"),
    EngineVersion(
        "G2-store", "d5bb11c8a15290e7b1a082e8afd1ac3f21f7c098c8099e1e45e93fd4b5a37920",
        "applied at G2, superseded within G2 by G2-store-fix",
        "adds Config.store_snaps (default False), run(..., init_store=None), an era-boundary copy "
        "of world.marks and pi beside the genome snapshot, and store_snaps in the returned dict. "
        "Nothing removed; analysis_v3_13.py untouched. Ruled under G2-D1 -- without it "
        "frozen_replay has no store to see, so A2.1's store arms and G3's population B are both "
        "unrunnable.",
        "engine_store_selftest: with both flags off, bit-identical to `G0`; with capture on, the "
        "trajectory still does not move. Plus the G1 reproduction re-run against it (182/182)."),
    EngineVersion(
        "G2-store-fix", "8e635fed1b0fbb19e7b04a67427b6f72cf143bfb18682a2fda4ab33e654a960e",
        "corrected within G2",
        "TWO CORRECTIONS TO G2-store, both found before any G3 number was produced. (1) The pi "
        "stored with a boundary capture was the pi of the era ABOUT TO START, not the one the "
        "marks were written under: new_recipe() redraws pi before the snapshot is taken. A store "
        "carrying that pi looks intact and decodes to the wrong preparation for every type -- "
        "measured, every type of every snapshot. The engine now tracks prev_pi exactly as it "
        "tracks prev_mapping. (2) Adds final_store: the record as it stood when the run ENDED, "
        "mid-era, marks live under the pi still in force. That is what a following population "
        "inherits; the era-boundary snapshots are what the frozen assay replays against, and they "
        "are not the same moment.",
        "engine_store_selftest (trajectory-neutrality, unchanged) plus "
        "store_decodes_selftest, which checks that the pi stored with a record actually decodes "
        "that record's marks -- the check that would have caught (1)."),
    EngineVersion(
        "G3-mapping", "383f24dd6abd5c223c22537262d5e6b6c7f7c163aaaed538c9ab36837f9b74fb",
        "added for G3",
        "adds run(..., init_mapping=...): start the world in a NAMED era and then let the clock "
        "run normally. Unlike force_mapping, which pins the mapping and stops every redraw. A "
        "population born into another's record needs it, because a record says which preparation "
        "succeeded on which type and that is only true of the era it was written in.",
        "the mapping is DRAWN first and then overwritten, so the RNG stream is untouched: "
        "measured, a run with init_mapping set to the mapping the world would have drawn anyway "
        "is identical across all 133 shared log fields. That matching is what makes B's arms "
        "differ in the store and in nothing else."),
)

CURRENT_ENGINE = ENGINE_VERSIONS[-1]
ENGINE_BY_SHA: dict[str, EngineVersion] = {v.sha256: v for v in ENGINE_VERSIONS}

#: Required reading that does not exist (F1). Recorded rather than reconstructed: writing a
#: substitute finding for runs nobody here observed would be worse than the gap.
MISSING_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("v3_12_finding.md",
     "named as required reading by the directive; not in the repository. The v3.12 conclusions "
     "survive as spec_v3_12.md, precheck_v3_12.txt, and the v3.12 paragraphs in "
     "analysis_v3_13.WORLD."),
    ("the v3.12 grid",
     "named by A3's G1 row; no such file exists. What exists is a 1-seed v3.12 pre-check."),
    ("assay_selftest",
     "named by A2.1 as an existing reference and by B§6 as a gate; not in the repository. "
     "Specified in D5 and built at G2 with the store."),
)

#: The invocation `precheck_v3_13.txt` was produced under, recovered by inference because the file
#: records none of it (F5). Every element has a stated basis; the seed does not, and says so.
REFERENCE_INVOCATION: dict[str, Any] = {
    "arms": ["plastic", "plastic + record", "plastic + noise", "fixed + record",
             "plastic + record (slow)"],
    "arms_basis": "all five of analysis_v3_13.VARIANTS appear in every table",
    "phase_steps": 3000,
    "phase_steps_basis": "every transition table is headed 'switch at t = 3000'",
    "prep_every": 700,
    "prep_every_basis": "Gate R reports 5 pi-epochs for the fast arms and 2 for the slow arm, "
                        "whose label_every is 3 * 700 = 2100; both counts follow from 700",
    "n_seeds": 1,
    "n_seeds_basis": "no per-seed table anywhere, and `plastic` and `plastic + record (slow)` "
                     "print identical phase-1 bins, which they can only do on one seed",
    "seed": None,
    "seed_basis": "NOT RECOVERABLE from the file. Almost certainly 0, since every committed stage "
                  "starts there, but that is inference and is not recorded as a value.",
    "matches_a_committed_mode": False,
    "modes": {"quick": "3 arms, 1500 steps", "acceptance": "3 arms, 8000 steps",
              "grid": "5 arms, 8000 steps"},
}

#: F3, as a claim this package is willing to have tested. The diff 3d7b228..HEAD on the engine adds
#: four counters and rewrites one `else:` as `elif True:`; it consumes no RNG and changes no
#: branch, so the trajectory is bit-identical and HEAD's engine reproduces 3d7b228's numbers.
SIM_V3_13_DRIFT: dict[str, Any] = {
    "from": "3d7b228", "to": G0_COMMIT,
    "note": "the FIRST hop. The engine moved again at G2 -- see ENGINE_VERSIONS and "
            "engine_store_selftest. This entry stays because it is what licenses reproducing "
            "precheck_v3_13.txt with anything other than the blob that produced it.",
    "change": "adds follgf_n/ok and follbf_n/ok counters, records two new log fields, and "
              "rewrites one `else:` as `elif True:`",
    "consumes_rng": False,
    "changes_a_branch": False,
    "claim": "trajectory bit-identical; HEAD's engine reproduces 3d7b228's numbers",
    "verified_by": "civitas_g.selftests.engine_drift_selftest",
}


# ---------------------------------------------------------------------------------------------
# a run's manifest
# ---------------------------------------------------------------------------------------------

@dataclass
class Manifest:
    """What a run was, recorded well enough that a reader needs nothing else (the `sr_w` lesson).

    A4.3 wants the spec, the DECISIONs, the pre-check numbers and the gate numbers per seed. This
    is the machine-readable half: everything a *re-run* would need. Prose belongs in the write-up.
    """

    kind: str
    civitas_version: str
    engine: dict[str, str] = field(default_factory=dict)
    world: dict[str, Any] = field(default_factory=dict)
    arms: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    phase_steps: int | None = None
    references: list[dict[str, Any]] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    created_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=True, default=str)


def _git(*args: str) -> str:
    try:
        return subprocess.run(("git", *args), cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=15).stdout.strip()
    except Exception:
        return ""


def engine_identity(root: Path | None = None) -> dict[str, str]:
    """A1.8: the engine's hash, in every manifest.

    Both v3.13 files, plus the commit the tree is on, plus whether either has uncommitted changes.
    A number produced from a dirty tree is not reproducible and the manifest says so rather than
    leaving a reader to find out.
    """
    root = root or REPO_ROOT
    dirty = _git("status", "--porcelain", "--", "sim_v3_13.py", "analysis_v3_13.py")
    engine_sha = sha256_file(root / "sim_v3_13.py")
    version = ENGINE_BY_SHA.get(engine_sha)
    return {
        "sim_v3_13.py": engine_sha,
        "engine_version": version.label if version else "UNRECORDED",
        "analysis_v3_13.py": sha256_file(root / "analysis_v3_13.py"),
        "commit": _git("rev-parse", "HEAD") or "unknown",
        "engine_tree_clean": "no" if dirty else "yes",
    }


def verify_pins(root: Path | None = None) -> list[tuple[bool, str]]:
    """Every G0 pin against the working tree. Order: gated references first, then context."""
    out: list[tuple[bool, str]] = []
    for ref in REFERENCES:
        for pin in (ref.summary, ref.sim, ref.analysis):
            if pin is None:
                continue
            # a pin at a non-HEAD commit is about a blob, not about the working tree
            if pin.commit != G0_COMMIT and pin.path.endswith(".py"):
                continue
            out.append(pin.verify(root))
    for pin in CONTEXT_PINS:
        out.append(pin.verify(root))
    out.append(verify_engine(root))
    return out


def verify_engine(root: Path | None = None) -> tuple[bool, str]:
    """The engine on disk must be a version this build knows about, and it says which.

    Not a plain hash pin: the engine is versioned (see `ENGINE_VERSIONS`), so an engine that moved
    deliberately reports the version it moved to, and an engine that moved by accident reports an
    unknown hash. The failure mode this exists to catch is the second one wearing the first one's
    clothes -- a stray edit to sim_v3_13.py that no manifest records.
    """
    actual = sha256_file((root or REPO_ROOT) / "sim_v3_13.py")
    version = ENGINE_BY_SHA.get(actual)
    if version is None:
        return False, (
            f"sim_v3_13.py is {actual[:12]}, which is no recorded engine version "
            f"({', '.join(f'{v.label}={v.sha256[:8]}' for v in ENGINE_VERSIONS)}). An engine that "
            f"moved without a version entry is an engine no manifest can name.")
    if version is not CURRENT_ENGINE:
        return False, (
            f"sim_v3_13.py is engine {version.label} ({actual[:12]}), but this build expects "
            f"{CURRENT_ENGINE.label} ({CURRENT_ENGINE.sha256[:12]}).")
    return True, f"sim_v3_13.py: engine {version.label} ({actual[:12]})"


def build_manifest(*, kind: str, arms: list[str], seeds: list[int],
                   phase_steps: int | None = None, world: dict[str, Any] | None = None,
                   notes: list[str] | None = None, root: Path | None = None) -> Manifest:
    """A manifest for one campaign."""
    from datetime import datetime, timezone

    from civitas_g import __version__
    from civitas_g.world.spec import WORLD

    return Manifest(
        kind=kind,
        civitas_version=__version__,
        engine=engine_identity(root),
        world=dict(world if world is not None else WORLD),
        arms=list(arms),
        seeds=list(seeds),
        phase_steps=phase_steps,
        references=[
            {"key": r.key, "summary": r.summary.path, "sha256": r.summary.sha256,
             "commit": r.summary.commit, "gated": r.gated, "why": r.why,
             "sim": None if r.sim is None else {"path": r.sim.path, "sha256": r.sim.sha256,
                                                "commit": r.sim.commit},
             "analysis": None if r.analysis is None else {
                 "path": r.analysis.path, "sha256": r.analysis.sha256,
                 "commit": r.analysis.commit}}
            for r in REFERENCES
        ],
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": __import__("numpy").__version__,
        },
        notes=list(notes or []),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
