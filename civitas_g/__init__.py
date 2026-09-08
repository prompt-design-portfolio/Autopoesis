"""Civitas-G.

A rebuild under `CIVITAS_G_MASTER_BUILD_DIRECTIVE.md`, which supersedes the original combined
prompt wherever the two conflict. The inversion that defines it (A1.1):

    The learner is the only thing that thinks. Civitas provides persistence, measurement and the
    environment; it never provides cognition.

So there is no agent in this package. No LLM is a component of one, no hand-written policy stands
in for one, and nothing here reads the store on a learner's behalf. The learner is the grown
network of `sim_v3_13.py` and it is the only thing in the system that decides anything.

What Civitas-G is, exactly:

* `world/`      the world of record, the vector domain adapter, and the arms — a statement of
                what `sim_v3_13.py` already is, checked against it rather than reimplementing it.
* `provider/`   the population provider: runs the engine byte-identical, persists what it
                produced, and touches the world only at era boundaries.
* `persistence/` runs, era rows, snapshots, stores and manifests.
* `reading/`    recompute a reading from the persisted rows and diff it against a reference
                summary — the G1 gate.
* `manifest.py` §46: every artifact recorded by hash, including the engine's own.

The G0 audit that precedes this build, the reference hashes it reproduces against, and the twelve
DECISIONs it was built under are in `docs/G0_G1.md`.
"""

from __future__ import annotations

__all__ = ["__version__"]

#: G-lineage version. G0 delivered the audit; this is the G1 build.
__version__ = "0.1.0-g1"
