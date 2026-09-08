"""The frozen assay (A2.1), through the store.

    Snapshot at an era boundary; births, deaths and injection disabled; energy pinned by identity;
    300 steps; `eta_scale` 0 and 1; matched and shuffled mapping; store visible / hidden /
    label-permuted. Nothing can change but `H`. Every claim is stated on this line.

Twelve cells: `mapping ∈ {matched, shuffled} × eta_scale ∈ {0, 1} × store ∈ {visible, hidden,
label_permuted}`. Before the G2 engine change the last axis did not exist — `frozen_replay` builds
a fresh `World` whose marks are zeroed and whose π is redrawn, so every replay ran against an empty
store whatever the population had left behind.

Three things about how the cells are matched, each of which could quietly turn a control into
something else:

**The store conditions differ only in what the population STARTS with.** All three keep
`record = "real"`, so agents go on writing marks during the window exactly as they would in a live
run. Setting `record = "none"` for the `hidden` arm would have been the obvious alternative and is
wrong: it removes the writing as well as the content, so the arm would differ from `visible` in two
ways at once and a gap between them could be either. `hidden` means *starts empty*, and the write
dynamics are identical across all three.

**`label_permuted` is a GLOBAL permutation, and that is deliberate.** The assay asks whether a
population that has already bound labels *inside its own life* was reading those particular labels.
A global permutation breaks that binding while preserving density, signs and cross-cell
consistency, so a hit rate that falls is evidence about the binding rather than about the store's
statistics. This is the opposite of what B§5.2's inherited-scrambled control needs — see
`ScrambleMode` — and using one for the other would break whichever it was used for.

**`sym_gain` cannot be zeroed with `cfg.sym_gain_lock` in a replay.** The lock is applied in
`Agent.__init__` and at reproduction, and `restore()` then writes every genome field back over the
top, `sym_gain` included. So a replay that set the lock would silently run at the snapshot's own
gain. `sym_gain_zero=True` zeroes it in the genome copy instead, which is the only place that
survives `restore`.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

import analysis_v3_13 as A
import sim_v3_13 as S
from civitas_g.store.record import Record, ScrambleMode

#: A2.1's window. ~5 preparations, which is what within-life learning takes (v3.11's finding).
FROZEN_STEPS = A.FROZEN_STEPS

STORE_CONDITIONS: tuple[str, ...] = ("visible", "hidden", "label_permuted")
MAPPING_CONDITIONS: tuple[str, ...] = ("matched", "shuffled")
ETA_SCALES: tuple[float, ...] = (0.0, 1.0)


@dataclass(frozen=True)
class AssayCell:
    """One of the twelve. `n` is the preparations the window actually produced -- a cell with a
    small `n` is not a small effect, it is a cell that could not be read."""

    store: str
    mapping: str
    eta_scale: float
    hit: float
    first_bin: float
    last_bin: float
    curve: list[float]
    pop: float
    n_preparations: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AssayResult:
    arm: str
    seed: int
    t: int
    era_index: int
    matched_mapping: tuple[int, ...]
    shuffled_mapping: tuple[int, ...]
    store_sha256: str
    sym_gain_zero: bool
    steps: int
    cells: list[AssayCell] = field(default_factory=list)

    def cell(self, store: str, mapping: str, eta: float) -> AssayCell:
        for c in self.cells:
            if (c.store, c.mapping, c.eta_scale) == (store, mapping, float(eta)):
                return c
        raise KeyError(f"no cell {(store, mapping, eta)!r}")

    def claim_line(self, store: str = "visible") -> float:
        """A2.1's line: `eta1 − eta0` on the SHUFFLED mapping.

        Shuffled because a genome sorted for the matched mapping is already right; a hit rate that
        rises on a mapping no genotype was selected under, with nothing able to change but `H`, is
        within-life learning and can be nothing else.
        """
        return (self.cell(store, "shuffled", 1.0).hit
                - self.cell(store, "shuffled", 0.0).hit)

    def store_effect(self, mapping: str = "shuffled", eta: float = 1.0) -> dict[str, float]:
        """Each store condition against `visible`, at one mapping and one learning rate."""
        base = self.cell("visible", mapping, eta).hit
        return {s: self.cell(s, mapping, eta).hit - base for s in STORE_CONDITIONS}

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cells"] = [c.as_dict() for c in self.cells]
        return d

    def table(self) -> str:
        rows = [f"  {'store':<16}{'mapping':<10}{'eta':>5}{'hit':>9}{'first':>9}{'last':>9}"
                f"{'pop':>8}{'n':>8}"]
        for store in STORE_CONDITIONS:
            for mapping in MAPPING_CONDITIONS:
                for eta in ETA_SCALES:
                    c = self.cell(store, mapping, eta)
                    rows.append(f"  {store:<16}{mapping:<10}{eta:>5.0f}{c.hit:>9.3f}"
                                f"{c.first_bin:>9.3f}{c.last_bin:>9.3f}{c.pop:>8.0f}"
                                f"{c.n_preparations:>8d}")
        return "\n".join(rows)


def _store_for(record: Record, condition: str, rng: np.random.Generator) -> dict[str, Any]:
    if condition == "visible":
        derived = record
    elif condition == "hidden":
        derived = record.hidden()
    elif condition == "label_permuted":
        derived = record.scrambled(rng, ScrambleMode.GLOBAL)
    else:
        raise ValueError(f"unknown store condition {condition!r}")
    return {"marks": derived.marks, "pi": derived.pi}


def _replay(run_cfg: dict[str, Any], genomes: list[dict[str, Any]], mapping: tuple[int, ...],
            eta: float, store: dict[str, Any] | None, *, steps: int, seed: int) -> dict[str, Any]:
    cfg = dict(run_cfg)
    for key in ("seed", "n_steps"):
        cfg.pop(key, None)
    cfg.update(eta_scale=float(eta), force_mapping=tuple(mapping), frozen=True,
               log_every=max(10, steps // 6),
               # a replay observes; it does not add to the record it was handed
               store_snaps=False)
    return S.run(S.Config(seed=seed, **cfg), verbose=False, init_genomes=genomes,
                 phases=[dict(n_steps=steps, chain=True)], init_store=store)


def _cell(raw: dict[str, Any], store: str, mapping: str, eta: float) -> AssayCell:
    log = raw["log"]
    curve = [A.prep_hit([w]) for w in log]
    n = int(sum(int(w.get("n_attempts_raw", w.get("attempts", 0)) or 0) for w in log))
    return AssayCell(
        store=store, mapping=mapping, eta_scale=float(eta),
        hit=float(A.prep_hit(log)),
        first_bin=float(curve[0]) if curve else float("nan"),
        last_bin=float(curve[-1]) if curve else float("nan"),
        curve=[float(c) for c in curve],
        pop=float(A.half(log, "pop")), n_preparations=n,
    )


def run_assay(run_dict: dict[str, Any], record: Record, *, snap: int = -1,
              steps: int = FROZEN_STEPS, seed_offset: int = 7000,
              sym_gain_zero: bool = False,
              permute_seed: int = 0) -> AssayResult:
    """A2.1's twelve cells on one era-boundary snapshot.

    `run_dict` is a stored run (from `load_run`); `record` is the store captured at the same
    boundary. Both are needed and they are different things -- the genomes are what the population
    IS, the record is what it LEFT.
    """
    snaps = run_dict.get("era_snaps") or []
    if not snaps:
        raise ValueError(
            "the run carries no era-boundary genome snapshots, so there is nothing to freeze. "
            "A run shorter than one era produces none.")
    sn = snaps[snap]
    genomes = copy.deepcopy(sn["genomes"])
    if sym_gain_zero:
        # cfg.sym_gain_lock cannot do this: restore() writes every genome field back over the top
        # of what Agent.__init__ set, sym_gain included.
        for g in genomes:
            g["sym_gain"] = 0.0

    matched = tuple(int(x) for x in sn["mapping"])
    shuffled = tuple(int(x) for x in A.shuffle_mapping(matched))
    seed = int(run_dict["cfg"]["seed"])

    result = AssayResult(
        arm=run_dict.get("arm", "?"), seed=seed, t=int(sn["t"]),
        era_index=len(snaps) + snap if snap < 0 else snap,
        matched_mapping=matched, shuffled_mapping=shuffled,
        store_sha256=record.sha256(), sym_gain_zero=sym_gain_zero, steps=steps,
    )
    for store in STORE_CONDITIONS:
        # one generator per condition, seeded from `permute_seed`, so the label permutation is a
        # recorded measurement condition rather than an unrecorded draw (the `sr_w` lesson)
        rng = np.random.default_rng(permute_seed)
        init_store = _store_for(record, store, rng)
        for mapping_name, mapping in (("matched", matched), ("shuffled", shuffled)):
            for eta in ETA_SCALES:
                raw = _replay(run_dict["cfg"], genomes, mapping, eta, init_store,
                              steps=steps, seed=seed_offset + seed)
                result.cells.append(_cell(raw, store, mapping_name, eta))
    return result
