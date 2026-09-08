"""G5 step 1 — the presentation, and the leak rule applied to it.

`docs/G5_SPEC.md` §2. The whole difficulty of the reference arm is that an LLM cannot perceive a
vector world, and every rendering that makes it perceivable is a design choice that can hand the
model the answer. G5-D1 rules the rendering **mechanical and lossless**: each of the 31 inputs
printed by its layout position at a fixed precision, in a fixed order, with no interpretation, no
summary, and no natural-language gloss.

Two rules do the work, and both are checked rather than described:

**The read channels are never named by preparation.** A mark sits at label `pi(k)`, and decoding
`pi` is exactly the work the learner has to do inside its own life. So the rendering says
`read_channel_2 = +0.42` and never `preparation 2`. `check_presentation_no_leak` scans the
rendered text for that vocabulary, because a comment saying "we do not do this" is not a check.

**The modulator table is rendered without its notes.** G5-D2 gives the model the action set and
the seven events with their signs — the world's rules, which are not secret. `adapter`'s own table
carries a `note` column that says *"the slow fact: k == mapping[ftype]. Writes a mark at label
pi(k)"*. Rendering that would transmit the fact the population is supposed to acquire, so the
renderer takes the event and the sign and drops everything else.

Nothing here calls a model. Step 1 of §6 is deliberately the whole presentation with its leak
check, finished and tested, *before* any provider is revived.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from civitas_g.world.adapter import (
    ACTIONS,
    DEAD_INPUTS,
    MODULATOR_EVENTS,
    OBS_LAYOUT,
    READ,
    READ_IS_AN_ACTION,
    DomainLeak,
    check_no_leak,
    check_observation_layout,
)

#: Bumped when the rendering changes at all. The manifest records it beside the model identifier
#: and the prompt hash (G5-D5): a number produced under one presentation is not comparable to a
#: number produced under another, and an unreproducible number is worse than no number.
PRESENTATION_VERSION = 1

#: Decimals every float is printed to. Fixed, so two renderings of the same observation are the
#: same string and the prompt hash means something.
DECIMALS = 4


class PresentationLeak(DomainLeak):
    """The rendered text could name an answer."""


# ---------------------------------------------------------------------------------------------
# field names
# ---------------------------------------------------------------------------------------------

def _field_names() -> tuple[str, ...]:
    """One name per input index, from `OBS_LAYOUT`, in engine order.

    A field of width one keeps its layout name. A wider field is suffixed by its offset within
    the field -- `food_A_dirsum_0` -- and the read block becomes `read_channel_j`. `read_channel`
    rather than anything shorter because the name is the only thing the model has to go on, and
    it must not be a preparation index by another spelling.
    """
    names: list[str] = []
    for f in OBS_LAYOUT:
        if f.name == "read_channels":
            names.extend(f"read_channel_{j}" for j in range(f.width))
        elif f.width == 1:
            names.append(f.name)
        else:
            names.extend(f"{f.name}_{j}" for j in range(f.width))
    return tuple(names)


FIELD_NAMES: tuple[str, ...] = _field_names()


# ---------------------------------------------------------------------------------------------
# the renderings
# ---------------------------------------------------------------------------------------------

def render_observation(obs: Any, *, field_names: tuple[str, ...] | None = None) -> str:
    """The 31 floats, one per line, `index name = value`. Lossless and uninterpreted.

    Dead inputs are rendered like everything else. `OBS_LAYOUT` names them DEAD -- the appetite
    channel is fed zeros because `scaffold_food` is False -- and G5-D1 is explicit that they go in
    as zeros rather than being helpfully removed. Removing them would be the experimenter telling
    the model which inputs not to bother with, which is a small piece of the search the learner
    has to do for itself.

    The names themselves are more than the learner gets -- its network receives thirty-one
    unlabelled floats. G5-D6 rules that they stay, and records the consequence: the reference arm
    is generously provisioned on purpose, so a poor number from it is more informative than a good
    one. The names never cross the line that matters, because `read_channel_j` is a position and
    not a preparation.
    """
    a = np.asarray(obs, dtype=float).ravel()
    names = field_names if field_names is not None else FIELD_NAMES
    if a.size != len(names):
        raise ValueError(f"observation has {a.size} inputs, the layout names {len(names)}")
    return "\n".join(f"{i:>3} {n:<20} = {a[i]:+.{DECIMALS}f}" for i, n in enumerate(names))


def render_action_set() -> str:
    """The ten actions by engine index. The action index *is* the preparation index for the
    preparations -- that is what the network's output units are -- so naming them `prepare_k` is
    not a leak. The leak would be connecting a read channel to one of them, which is why the
    channels are named `read_channel_j` and no line here mentions a channel.
    """
    return "\n".join(f"{a.index:>3} {a.name}" for a in ACTIONS)


def render_modulator_table() -> str:
    """The seven events and their signs. No notes, no energy column.

    See the module docstring: `adapter.MODULATOR_EVENTS` carries a `note` that states the mapping
    rule outright, and an `energy` column that names `prep_value`. G5-D2 gives the model the
    events and their signs and nothing else.
    """
    return "\n".join(f"{e.event:<16} {e.m:+.1f}" for e in MODULATOR_EVENTS)


# ---------------------------------------------------------------------------------------------
# the leak rule, applied to the text
# ---------------------------------------------------------------------------------------------

#: Vocabulary that would mean the rendering had decoded pi on the model's behalf. Matched
#: case-insensitively against the whole rendered prompt. `mapping`, `era` and `label` are here
#: because naming any of them tells the model there is a permutation to invert and a clock to
#: track -- G5-D2 rules that it is told none of it.
FORBIDDEN = (
    r"\bpi\b", r"\bpermut\w*", r"\bmapping\w*", r"\bera\b", r"\beras\b",
    r"\blabel\w*", r"\bendorse\w*", r"\bcorrect\s+preparation\b", r"\brecipe\w*",
    r"\bprevious\s+agent\w*", r"\bsucceed\w*\s+with\b",
)

#: `read_channel_j` is the only place a channel index may appear. Anything that puts a channel
#: index next to the word `preparation` has decoded the store.
_CHANNEL_MEANS_PREP = re.compile(
    r"(channel|read)\D{0,24}\bprepar\w*|prepar\w*\D{0,24}\b(channel|read)\b", re.IGNORECASE)


def check_presentation_no_leak(text: str, *, marks_are_labels: bool = True,
                               pi_redrawn_every_era: bool = True) -> None:
    """§47 for the presentation. Raises `PresentationLeak` on the first thing found.

    Runs `adapter.check_no_leak` first -- the four structural conditions, unchanged, because a
    rendering of a leaky world is leaky whatever the text says -- and then the two conditions that
    exist only once the world is text. `read_is_observation` and `read_is_here_only` are read off
    the adapter rather than passed, because the presentation cannot change either.
    """
    check_no_leak(marks_are_labels=marks_are_labels,
                  pi_redrawn_every_era=pi_redrawn_every_era,
                  read_is_observation=not READ_IS_AN_ACTION,
                  read_is_here_only=True)
    for pattern in FORBIDDEN:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            raise PresentationLeak(
                f"the presentation contains {m.group(0)!r}, which names machinery the learner has "
                f"to discover within its own life. G5-D2: the model is told the action set and "
                f"the modulator, and nothing about the mapping, pi, the era length, or what the "
                f"read channels mean.")
    m = _CHANNEL_MEANS_PREP.search(text)
    if m:
        raise PresentationLeak(
            f"the presentation connects a read channel to a preparation ({m.group(0)!r}). "
            f"Decoding pi is the work the learner does; a rendering that does it has handed over "
            f"the answer and the arm measures nothing.")


# ---------------------------------------------------------------------------------------------
# the whole prompt
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Presentation:
    """One frozen rendering: the fixed preamble, and how a step is rendered.

    Frozen in the manifest sense of G5-D5 -- `digest()` goes in beside the model identifier, and
    a run refuses to start if it is unset. It is not a prompt template with knobs; if it needs a
    knob, that is a second presentation with a second version.
    """

    version: int = PRESENTATION_VERSION
    variant: str = "blind"

    def preamble(self) -> str:
        """Everything the model is told about the task, once. G5-D2, verbatim in effect.

        Note what is absent: any statement of what the read channels are, how many food types
        there are, that a permutation exists, or that anything changes over time. The model is
        given what an agent's own network is given -- the actions it can take, and a modulator.
        """
        return (
            "You are choosing one action per step in an environment you cannot see directly.\n"
            "Each step you receive a fixed-length vector of numbers. The vector's fields are "
            "named by position; the names are not explanations and nothing tells you what any "
            "field means.\n\n"
            "Actions, by index:\n" + render_action_set() + "\n\n"
            "After each action a number arrives. Larger is better. The number depends only on "
            "which of these outcomes occurred:\n" + render_modulator_table() + "\n\n"
            "Reply with a single action index and nothing else."
        )

    def step(self, obs: Any, *, history: str = "") -> str:
        """One step's user message: the observation, and whatever context the variant carries.

        `history` is supplied by the caller, not built here, because the three variants of G5-D3
        differ in exactly that and in nothing else -- `blind` passes none, `life` passes this
        life's rendered steps within a bounded window, `store` passes the same. The read channels
        are in the observation either way; `store` differs from `life` in whether the replayed
        world has an inherited record, which is a property of the run, not of the text.
        """
        parts = []
        if history:
            parts.append("Earlier steps in this life:\n" + history)
        parts.append("Observation:\n" + render_observation(obs))
        return "\n\n".join(parts)

    def digest(self) -> str:
        """Hash over the preamble and the layout the observations are rendered under.

        Over the *layout* rather than a sample observation, so the digest is a property of the
        presentation and not of whichever step happened to be rendered when it was taken.
        """
        h = hashlib.sha256()
        h.update(f"civitas-g5-presentation:{self.version}:{self.variant}\n".encode())
        h.update(self.preamble().encode())
        h.update("\n".join(FIELD_NAMES).encode())
        h.update(f"decimals={DECIMALS}".encode())
        return h.hexdigest()

    def check(self) -> str:
        """Run the leak rule over this presentation's own text. Returns a one-line report."""
        check_observation_layout()
        zeros = render_observation(np.zeros(len(FIELD_NAMES)))
        check_presentation_no_leak(self.preamble() + "\n" + zeros)
        return (f"presentation v{self.version} ({self.variant}): {len(FIELD_NAMES)} fields, "
                f"{len(DEAD_INPUTS)} dead, read block at {READ}, digest {self.digest()[:12]}")
