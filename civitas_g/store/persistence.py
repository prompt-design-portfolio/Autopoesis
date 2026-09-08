"""Saving and loading the record (G2).

B§6 asks for a store round-trip: "save, load, byte-identical marks and pi". That is the whole
contract of this module, and the two things that make it non-trivial are stated where they are
enforced rather than here:

* the byte layout is in `Record.to_bytes`, because a decimal round-trip through text is only
  identical if every writer and reader agrees on repr;
* `content_sha256` is over the marks and pi rather than over the compressed blob, because zlib
  output depends on the library version and two identical records would otherwise hash differently
  on two machines.

What this module adds is the database half: a derived record — hidden, scrambled, permuted — is
stored **beside** its parent with `derived_from` set, never in place of it (A1.4). A control that
overwrote the thing it is a control for would leave nothing to compare against.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from civitas_g.persistence.models import Run, StoreArtifact
from civitas_g.store.record import Record, RecordProvenance, ScrambleMode


def save_record(session: Session, run: Run | Any, record: Record, *,
                variant: str = "real",
                derived_from: StoreArtifact | None = None) -> StoreArtifact:
    """Persist one record. The statistics are measured on the way in, not on the way out."""
    run_id = run.id if isinstance(run, Run) else run
    signs = record.sign_counts()
    artifact = StoreArtifact(
        run_id=run_id, variant=variant,
        derived_from=None if derived_from is None else derived_from.id,
        era_index=int(record.provenance.era_index), t=int(record.provenance.t),
        pi=list(record.pi),
        mapping=list(record.provenance.mapping),
        prev_mapping=list(record.provenance.prev_mapping),
        density=record.density(), mean_abs=record.mean_abs(),
        n_positive=signs["positive"], n_negative=signs["negative"],
        blob=record.to_bytes(), content_sha256=record.sha256(),
        provenance=record.provenance.as_dict(),
    )
    session.add(artifact)
    session.flush()
    return artifact


def load_record(session: Session, artifact: StoreArtifact | Any) -> Record:
    """Rebuild a record from a stored artifact, and refuse one whose bytes have moved."""
    row = artifact if isinstance(artifact, StoreArtifact) else session.get(StoreArtifact, artifact)
    if row is None:
        raise LookupError(f"no store artifact {artifact!r}")
    prov = dict(row.provenance)
    prov["mapping"] = tuple(prov.get("mapping") or ())
    prov["prev_mapping"] = tuple(prov.get("prev_mapping") or ())
    record = Record.from_bytes(bytes(row.blob), RecordProvenance(**prov),
                              meta={"variant": row.variant})
    actual = record.sha256()
    if actual != row.content_sha256:
        raise ValueError(
            f"store artifact {row.id} does not match its recorded content hash "
            f"({actual[:12]} vs {row.content_sha256[:12]}). The bytes moved after they were "
            f"written; the artifact is not the artifact it claims to be.")
    return record


def records_for_run(session: Session, run: Run | Any,
                    variant: str = "real") -> list[Record]:
    """Every stored record for a run, in era order."""
    run_id = run.id if isinstance(run, Run) else run
    rows = session.execute(
        select(StoreArtifact)
        .where(StoreArtifact.run_id == run_id, StoreArtifact.variant == variant)
        .order_by(StoreArtifact.era_index)).scalars().all()
    return [load_record(session, row) for row in rows]


def save_control_variants(session: Session, run: Run | Any, artifact: StoreArtifact,
                          record: Record, *, seed: int = 0,
                          modes: Sequence[ScrambleMode] = (ScrambleMode.PER_CELL,),
                          include_hidden: bool = True) -> list[StoreArtifact]:
    """Store a record's controls beside it.

    The seed is explicit and recorded in `meta`, because a scramble is a measurement condition: an
    arm whose control was drawn from an unrecorded random state cannot be reproduced, which is the
    `sr_w` lesson applied to a control rather than to a parameter.
    """
    out: list[StoreArtifact] = []
    if include_hidden:
        out.append(save_record(session, run, record.hidden(), variant="hidden",
                               derived_from=artifact))
    for i, mode in enumerate(modes):
        rng = np.random.default_rng(seed + i)
        scrambled = record.scrambled(rng, mode)
        row = save_record(session, run, scrambled, variant=f"scrambled:{mode.value}",
                          derived_from=artifact)
        row.provenance = dict(row.provenance, scramble_seed=seed + i, scramble_mode=mode.value)
        out.append(row)
    session.flush()
    return out


def preserves_density_and_sign(original: Record, scrambled: Record) -> tuple[bool, str]:
    """B§6's scrambled-load clause: density and sign preserved, label destroyed.

    All three are checked, and the third is the one that is easy to get wrong: a scramble that
    happened to draw the identity permutation everywhere would pass the first two and be no
    control at all.
    """
    if scrambled.density() != original.density():
        return False, (f"density moved: {original.density():.6f} -> {scrambled.density():.6f}. "
                       f"A permutation is a bijection, so this cannot happen unless the scramble "
                       f"dropped or created marks.")
    if scrambled.sign_counts() != original.sign_counts():
        return False, (f"signs moved: {original.sign_counts()} -> {scrambled.sign_counts()}. "
                       f"The control must remove the label, not the valence.")
    identical = np.array_equal(original.marks, scrambled.marks)
    live = int(original.live.sum())
    if identical and live:
        return False, (f"the scramble left all {live} marks where they were, so it is not a "
                       f"control -- it is a copy.")
    moved = int((original.marks != scrambled.marks).sum())
    return True, (f"density {original.density():.6f} and signs {original.sign_counts()} "
                  f"preserved; {moved} of {original.marks.size} label slots moved")
