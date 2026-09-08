"""The experimental arms (B§4).

B§4 replaces §21's nine arms with seven, "mapped so the manifest vocabulary survives". This module
is that mapping, and it is the only place the two vocabularies are allowed to meet: the manifest
name is what Civitas persists, the research name is what `analysis_v3_13.VARIANTS` calls the same
thing, and everything downstream reads one or the other but never guesses between them.

Four things worth stating explicitly, because each is a place the mapping could be got wrong
silently:

1. **`collective_scrambled` is the NOISE RECORD, not v3.12's `scramble`.** They are different
   controls with confusingly similar names. `scramble=True` randomises the sign of the *modulator*
   -- it attacks learning. The noise record randomises the *label* on write, holding mark density,
   sign and decay identical -- it attacks the record's content while leaving its presence intact.
   B§4 maps `collective_scrambled` to "plastic + noise record", and A2.3 is why: a store changes
   the world by existing, so "a store existed" has to be separated from "information passed".

2. **`memory_reset` is not "no store".** The read channels exist and are zero. The input layout is
   identical across every arm, so no genome ever sees a different world shape, and a difference
   between arms cannot be a difference in what the network was given to work with.

3. **`collective_frozen` is not a run configuration.** It is A2.1's procedure over a snapshot --
   births, deaths and injection disabled, energy pinned, 300 steps, `eta_scale` 0 and 1, matched
   and shuffled mapping, store visible / hidden / label-permuted. It carries no kwargs, and asking
   for its kwargs raises rather than returning an empty dict that would silently run a live world.

4. **`plastic + record (slow)` is in the reference summary and not in B§4's table.** It is
   `collective` with `label_every = 3 * prep_every`, so a label's meaning outlives what it names by
   three eras -- v3.5's tempo condition. It is carried here because G1 reproduces
   `precheck_v3_13.txt`, which has it, and a reproduction that silently dropped an arm would be a
   reproduction of something else. Recorded as `collective_slow_labels`, flagged `in_directive =
   False`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from civitas_g.world.spec import WORLD


class ArmUnavailable(RuntimeError):
    """This arm cannot be run at this milestone, and will not be faked."""


@dataclass(frozen=True)
class Arm:
    """One arm: the manifest name, the research name, and exactly what it changes."""

    name: str                       # the manifest vocabulary Civitas persists
    research_name: str              # what analysis_v3_13.VARIANTS calls it
    holds_or_removes: str           # B§4's third column, verbatim where it exists
    kwargs: dict[str, Any] = field(default_factory=dict)
    #: A2.1's frozen replay is a procedure over a snapshot, not a world to run.
    is_replay: bool = False
    #: G5's reference arm. Reported beside, never combined, and not buildable before G5.
    milestone: str = "G1"
    #: False for arms carried for the reproduction that B§4's table does not list.
    in_directive: bool = True

    def config_kwargs(self, world: dict[str, Any] | None = None) -> dict[str, Any]:
        """The world of record with this arm's changes laid over it."""
        if self.is_replay:
            raise ArmUnavailable(
                f"{self.name!r} is A2.1's frozen replay, a procedure over an era-boundary "
                f"snapshot, not a world configuration. Run it with the frozen assay."
            )
        if self.milestone != "G1":
            raise ArmUnavailable(
                f"{self.name!r} is a {self.milestone} arm and is not available at G1."
            )
        return dict(world if world is not None else WORLD, **self.kwargs)


_SLOW_LABEL_EVERY = 3 * int(WORLD["prep_every"])

ARMS: tuple[Arm, ...] = (
    Arm("memory_reset", "plastic",
        "no record; the read channels exist and are zero",
        dict(mode="plastic", plastic_layers="W2", record="none")),
    Arm("collective", "plastic + record",
        "the record, written and read",
        dict(mode="plastic", plastic_layers="W2", record="real")),
    Arm("collective_scrambled", "plastic + noise record",
        "same writes, density and signs; labels randomised",
        dict(mode="plastic", plastic_layers="W2", record="noise")),
    Arm("no_plasticity", "fixed + record",
        "the record, a genome that cannot learn",
        dict(mode="fixed", record="real")),
    Arm("collective_frozen", "frozen assay",
        "snapshot, population frozen, store visible / hidden / permuted, eta_scale 0 / 1",
        is_replay=True),
    Arm("solo", "random policy",
        "the affordance null: uniform actions, same world",
        dict(mode="random")),
    Arm("reference", "frozen LLM",
        "one frozen LLM against the same store; reported beside, never combined",
        milestone="G5"),
    # carried for the reproduction; see the module docstring
    Arm("collective_slow_labels", "plastic + record (slow)",
        "the record, with a label's meaning outliving what it names by three eras",
        dict(mode="plastic", plastic_layers="W2", record="real",
             label_every=_SLOW_LABEL_EVERY),
        in_directive=False),
)

BY_NAME: dict[str, Arm] = {a.name: a for a in ARMS}
BY_RESEARCH_NAME: dict[str, Arm] = {a.research_name: a for a in ARMS}

#: §21's arms that B§4 removes. Kept as a list so a caller that meets one in old data can say so
#: precisely rather than failing on an unknown string.
RETIRED_ARMS: tuple[str, ...] = (
    "shared_memory", "independent", "collective_no_negative", "collective_no_provenance",
)

#: The five arms `precheck_v3_13.txt` was produced from, in the order it prints them. Reproduction
#: is per-arm, so the order matters only for reading the diff, but a fixed order makes two runs of
#: the reproduction comparable line by line.
REFERENCE_ARM_ORDER: tuple[str, ...] = (
    "memory_reset", "collective", "collective_scrambled", "no_plasticity",
    "collective_slow_labels",
)


def get_arm(name: str) -> Arm:
    """By manifest name or by research name. Raises with the alternatives, never silently."""
    if name in BY_NAME:
        return BY_NAME[name]
    if name in BY_RESEARCH_NAME:
        return BY_RESEARCH_NAME[name]
    if name in RETIRED_ARMS:
        raise ArmUnavailable(
            f"{name!r} is one of §21's arms that B§4 removes. It is not an unknown name -- it is "
            f"a retired one, and data carrying it belongs in results/legacy/."
        )
    raise KeyError(
        f"unknown arm {name!r}. Manifest names: {sorted(BY_NAME)}. "
        f"Research names: {sorted(BY_RESEARCH_NAME)}."
    )


def runnable_arms() -> tuple[Arm, ...]:
    """The arms that are a world to run at G1: not the replay, not G5's reference."""
    return tuple(a for a in ARMS if not a.is_replay and a.milestone == "G1")
