"""The record, as an artifact rather than as a field of a running world (G2).

A2.1 and B§5.2 both need the store to be a *thing* — something that can be snapshotted at an era
boundary, saved, loaded byte-identically, scrambled on load, and handed to a population that never
wrote it. In `sim_v3_13` it is none of those: it is `world.marks`, a `(T, K, g, g)` float array
created zeroed inside `World.__init__` and never returned by `run()`. This module is what it
becomes on the way out.

**Provenance is at the artifact level, not per mark, and that is a limitation rather than a
choice.** §A1.2 (kept) asks for provenance on every mark. The engine's marks carry none: a write
is `marks[ftype, label, y, x] = ±1`, overwriting whatever was there, with no writer identity and
no timestamp. Nothing downstream can recover which agent wrote a mark or when, because that
information was never stored. So `RecordProvenance` records what *is* knowable — which run, which
era, which engine, what the world was — and the per-mark gap is stated here rather than papered
over.

**Two different permutations, and confusing them would break a control.**

* `ScrambleMode.PER_CELL` is B§5.2's `inherited scrambled`: an independent permutation of the
  label axis for every `(cell, food type)`. Density and the multiset of signs are preserved
  exactly, because a permutation is a bijection; what is destroyed is any *consistent* label →
  preparation association across cells.
* `ScrambleMode.GLOBAL` is A2.1's `label-permuted` assay arm: one permutation for the whole store.

They are not interchangeable, and using GLOBAL for the inherited-scrambled arm would be a broken
control. A globally permuted store is *isomorphic* to the real one — it says "label σ(j) marks
what label j marked" everywhere, consistently. A population born into it has to learn the binding
within life, which is exactly what it would have had to do with the real store, because π is
redrawn every era and no genome ever inherits the binding. So the arm would read as a null while
information had in fact passed. PER_CELL is the one that removes the information.

GLOBAL is right for the assay because the assay asks a different question: it takes a population
that has *already* bound labels inside its own life and permutes them, so a hit rate that falls is
evidence those particular labels were being read.
"""

from __future__ import annotations

import hashlib
import zlib
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from civitas_g.world.spec import N_TYPES

#: Serialisation format version. Bumped when the byte layout changes, so a stored artifact can
#: never be silently reinterpreted under a newer layout.
FORMAT_VERSION = 1

#: What the engine counts as a live mark (`mark_stats`: `np.abs(world.marks) > 1e-3`). Used here
#: for density so that a Civitas-side density and the engine's log agree by construction rather
#: than by coincidence.
LIVE_THRESHOLD = 1e-3


class ScrambleMode(StrEnum):
    """See the module docstring. These are different controls and are not interchangeable."""

    #: B§5.2's `inherited scrambled`. Independent permutation per (cell, food type).
    PER_CELL = "per_cell"
    #: A2.1's `label-permuted` assay arm. One permutation for the whole store.
    GLOBAL = "global"


@dataclass(frozen=True)
class RecordProvenance:
    """Where a record came from. A1.4: everything external is auditable.

    Deliberately says nothing about individual marks — see the module docstring.
    """

    run_seed: int
    arm: str
    t: int
    era_index: int
    engine_sha256: str
    cfg_digest: str
    #: The mapping in force when the snapshot was taken, and the one just lived under. A replay
    #: that re-seeds the world does NOT get either, so both travel with the artifact.
    mapping: tuple[int, ...] = ()
    prev_mapping: tuple[int, ...] = ()
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Record:
    """One era-boundary snapshot of the store.

    `marks[ftype, label, y, x]` is the age-decayed sign of the latest mark at that label for that
    food type at that cell. `pi` is the label permutation in force: label of preparation k is
    `pi[k]`, so a mark at label j endorses preparation `pi^-1(j)`.
    """

    marks: np.ndarray
    pi: tuple[int, ...]
    provenance: RecordProvenance
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.marks.ndim != 4:
            raise ValueError(f"marks must be (T, K, g, g); got shape {self.marks.shape}")
        t, k, gy, gx = self.marks.shape
        # T is pinned: the number of food types is not something any mechanic moves. K is read
        # off the marks, because G4's first mechanic raises it (spec.HARDENING_PARAMETERS) and a
        # record that could only ever be K = N_PREPS would make a hardened store unrepresentable.
        # This is the same K-blindness that `check_chance_ev_is_zero` had, in the store.
        if t != N_TYPES:
            raise ValueError(f"marks is ({t}, ...) but the world has {N_TYPES} food types")
        if k <= t:
            raise ValueError(f"marks is (..., {k}, ...) with {t} food types: K must exceed T, "
                             f"or a mapping is not an injection and there is nothing to record")
        if gy != gx:
            raise ValueError(f"the grid is not square: {gy} x {gx}")
        if sorted(self.pi) != list(range(k)):
            raise ValueError(f"pi is not a permutation of 0..{k - 1}: {self.pi}")

    # ---------------------------------------------------------------- shape and statistics

    @property
    def grid(self) -> int:
        return int(self.marks.shape[-1])

    @property
    def n_preps(self) -> int:
        """K, read off the marks rather than the module: a hardened store carries its own K."""
        return int(self.marks.shape[1])

    @property
    def live(self) -> np.ndarray:
        """The boolean mask the engine's own `mark_stats` uses."""
        return np.abs(self.marks) > LIVE_THRESHOLD

    def density(self) -> float:
        """Fraction of `(type, label, cell)` slots carrying a live mark. Matches
        `sim_v3_13.mark_stats`'s `mark_density` by construction."""
        return float(self.live.mean())

    def mean_abs(self) -> float:
        live = self.live
        return float(np.abs(self.marks)[live].mean()) if live.any() else 0.0

    def sign_counts(self) -> dict[str, int]:
        """Positive and negative live marks. Preserved by every scramble, and checked."""
        live = self.live
        return {"positive": int((self.marks[live] > 0).sum()),
                "negative": int((self.marks[live] < 0).sum())}

    def endorsed(self) -> tuple[int, ...]:
        """`endorsed[j]` is the preparation label j endorses, i.e. pi^-1."""
        inv = [0] * self.n_preps
        for k, label in enumerate(self.pi):
            inv[label] = k
        return tuple(inv)

    # ---------------------------------------------------------------- the arms

    def hidden(self) -> Record:
        """A2.1's `store hidden` arm: the array is there and every mark is gone.

        Not "no store". The read channels exist and are zero, exactly as in `memory_reset`, so the
        input layout is unchanged and a difference between arms cannot be a difference in what the
        network was given to work with.
        """
        return Record(marks=np.zeros_like(self.marks), pi=self.pi,
                      provenance=self.provenance,
                      meta=dict(self.meta, derived="hidden"))

    def scrambled(self, rng: np.random.Generator,
                  mode: ScrambleMode = ScrambleMode.PER_CELL) -> Record:
        """Labels randomised; density and the multiset of signs preserved exactly.

        A permutation is a bijection, so every live mark stays live and every sign survives — what
        moves is which label carries it. `PER_CELL` destroys any consistent label → preparation
        association; `GLOBAL` preserves it and is the assay's arm, not a control. See the module
        docstring for why that distinction is load-bearing.
        """
        out = np.empty_like(self.marks)
        if mode is ScrambleMode.GLOBAL:
            perm = rng.permutation(self.n_preps)
            out[:] = self.marks[:, perm, :, :]
        else:
            g = self.grid
            for ftype in range(N_TYPES):
                # one independent permutation per cell, drawn as a (g, g, K) argsort so the whole
                # type is permuted in one vectorised pass rather than g*g python-level draws
                order = np.argsort(rng.random((g, g, self.n_preps)), axis=-1)
                plane = np.moveaxis(self.marks[ftype], 0, -1)          # (g, g, K)
                out[ftype] = np.moveaxis(np.take_along_axis(plane, order, axis=-1), -1, 0)
        return Record(marks=out, pi=self.pi, provenance=self.provenance,
                      meta=dict(self.meta, derived=f"scrambled:{mode.value}"))

    # ---------------------------------------------------------------- bytes

    def to_bytes(self) -> bytes:
        """A byte-exact, self-describing serialisation.

        The marks are written as raw little-endian doubles rather than through a text or JSON
        form, because B§6 asks the round-trip to be *byte-identical* and a decimal round-trip
        through text is only identical if every writer and reader agrees on repr. `zlib` because a
        48x48 store is 276 KB of mostly zeros.
        """
        arr = np.ascontiguousarray(self.marks, dtype="<f8")
        header = (f"civitas-g-record\n{FORMAT_VERSION}\n"
                  f"{arr.shape[0]} {arr.shape[1]} {arr.shape[2]} {arr.shape[3]}\n"
                  f"{' '.join(str(p) for p in self.pi)}\n").encode()
        return zlib.compress(header + arr.tobytes(), level=6)

    @classmethod
    def from_bytes(cls, blob: bytes, provenance: RecordProvenance,
                   meta: dict[str, Any] | None = None) -> Record:
        raw = zlib.decompress(blob)
        head, rest = raw.split(b"\n", 1)
        if head != b"civitas-g-record":
            raise ValueError("not a civitas-g record blob")
        version_line, rest = rest.split(b"\n", 1)
        version = int(version_line)
        if version != FORMAT_VERSION:
            raise ValueError(
                f"record blob is format {version}, this build writes {FORMAT_VERSION}. A stored "
                f"artifact is never reinterpreted under a newer layout.")
        shape_line, rest = rest.split(b"\n", 1)
        pi_line, payload = rest.split(b"\n", 1)
        shape = tuple(int(x) for x in shape_line.split())
        pi = tuple(int(x) for x in pi_line.split())
        marks = np.frombuffer(payload, dtype="<f8").reshape(shape).copy()
        return cls(marks=marks, pi=pi, provenance=provenance, meta=dict(meta or {}))

    def sha256(self) -> str:
        """Over the marks and pi, not over the compressed blob.

        zlib output depends on the library version, so hashing the blob would make two identical
        records hash differently on two machines — which is exactly the kind of false alarm a
        byte-identity check must not raise.
        """
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.marks, dtype="<f8").tobytes())
        h.update(bytes(self.pi))
        return h.hexdigest()

    def equals(self, other: Record) -> tuple[bool, str]:
        """Byte-identical marks and pi. Returns `(ok, detail)` so a caller can report why not."""
        if self.pi != other.pi:
            return False, f"pi differs: {self.pi} vs {other.pi}"
        if self.marks.shape != other.marks.shape:
            return False, f"shape differs: {self.marks.shape} vs {other.marks.shape}"
        a = np.ascontiguousarray(self.marks, dtype="<f8").tobytes()
        b = np.ascontiguousarray(other.marks, dtype="<f8").tobytes()
        if a != b:
            n = int((self.marks != other.marks).sum())
            return False, f"{n} of {self.marks.size} marks differ"
        return True, f"byte-identical: {self.marks.size} marks, pi {self.pi}"


def record_from_world(world: Any, *, arm: str, t: int, era_index: int,
                      engine_sha256: str, cfg_digest: str,
                      prev_mapping: tuple[int, ...] = (), note: str = "") -> Record:
    """Lift a live `sim_v3_13.World`'s store out as an artifact.

    Takes a copy: the engine keeps mutating `world.marks` in place (decay runs every step), so a
    view would silently become a snapshot of a later moment.
    """
    return Record(
        marks=np.array(world.marks, dtype="<f8", copy=True),
        pi=tuple(int(x) for x in world.pi),
        provenance=RecordProvenance(
            run_seed=int(world.cfg.seed), arm=arm, t=int(t), era_index=int(era_index),
            engine_sha256=engine_sha256, cfg_digest=cfg_digest,
            mapping=tuple(int(x) for x in world.mapping),
            prev_mapping=tuple(int(x) for x in prev_mapping), note=note),
    )
