"""G5 §7 — what substitutes for the network, and the manifest that makes it quotable.

The hook itself is an engine change (G5-D7, engine version `G5-policy`). This module is
everything on Civitas's side of it: the policy protocol, the two policies that need no provider,
and the frozen manifest a reference run refuses to start without.

**Nothing here is a component of an agent.** A1.1 is absolute and G5 §7.2 says how this stays
true: the hook replaces the policy of a *population in a frozen replay*, in an arm with no claim
line, whose results are stored in their own table, and no genome is written back.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from civitas_g.g5.presentation import Presentation

# `PREP0` and `N_ACTIONS` are the reference world's. `PREP0` is fixed -- 4 moves plus eat, before
# the preparations begin -- and does not move with K; `N_ACTIONS` does, which is why the check
# below takes it as a parameter rather than reading it here.
from civitas_g.world.adapter import N_ACTIONS, PREP0


class IllegalAction(ValueError):
    """A policy proposed an action the world does not have, or one the engine has masked.

    Refused rather than clipped (G5-D7). A model that proposes a preparation in phase 1 has told
    us something; recording a clipped legal action instead would record a choice it did not make.
    """


class Policy(Protocol):
    """The engine-side contract. `None` means *this step is the network's*.

    Returning `None` is how G5-D4's sampling works: a policy applied to a subset says `None` for
    every agent outside it, and there is no second mechanism for the sampling to disagree with.
    """

    def __call__(self, lineage: int, obs: np.ndarray, chain_on: bool,
                 logits: np.ndarray) -> int | None: ...


def check_action(action: int, chain_on: bool, n_actions: int = N_ACTIONS) -> int:
    """G5-D7's masking rule. The engine's mask, not the policy's.

    `n_actions` is a parameter and not the module constant, for the reason `Record`, the matched
    null and `check_chance_ev_is_zero` all had to learn: at K = 5 the constant is the right
    number, so K-blindness here would never fail a test and would silently accept action 10 on a
    K = 7 world as out of range.
    """
    if not isinstance(action, (int, np.integer)) or isinstance(action, bool):
        raise IllegalAction(f"a policy returned {action!r}, which is not an action index")
    a = int(action)
    if not 0 <= a < n_actions:
        raise IllegalAction(f"action {a} is outside 0..{n_actions - 1}")
    if not chain_on and a >= PREP0:
        raise IllegalAction(
            f"action {a} is a preparation and the chain is off. Phase 1 is exactly the five "
            f"actions v3.1 had; a preparation there is not a choice the world offers.")
    return a


# ---------------------------------------------------------------------------------------------
# the policies that need no provider
# ---------------------------------------------------------------------------------------------

@dataclass
class NetworkPolicy:
    """Returns `None` always: every step is the network's.

    The first half of G5 §7.1's equivalence check. A run under this policy must be bit-identical
    to a run with no policy at all, which is what proves the hook costs nothing when unused.
    """

    calls: int = 0

    def __call__(self, lineage: int, obs: np.ndarray, chain_on: bool,
                 logits: np.ndarray) -> int | None:
        self.calls += 1
        return None


@dataclass
class MirrorPolicy:
    """Returns the network's own choice, computed outside the engine.

    **The second and interesting half of §7.1.** A run under this policy must be bit-identical to
    a run with no policy, even though every action now arrives through the hook. That exercises
    the whole loop -- hook, observation, action, modulator, statistics -- against a stand-in whose
    answers are known, and it costs nothing. A model in the loop could never prove this, because
    with a model there is no trajectory to compare against.

    It is handed the *same* logits the engine computed, rather than recomputing them: recomputing
    would need the action-noise draw, and drawing it again would move the RNG stream and make the
    check fail for a reason that has nothing to do with the hook. That is also why `logits` is the
    fourth argument of the policy contract -- a policy that ignores it (every model policy does)
    costs nothing, and one that needs it cannot get it any other way.
    """

    calls: int = 0

    def __call__(self, lineage: int, obs: np.ndarray, chain_on: bool,
                 logits: np.ndarray) -> int | None:
        if logits is None:
            raise RuntimeError(
                "MirrorPolicy was called without logits. It mirrors the engine's own choice and "
                "cannot recompute it: the action-noise draw is the engine's, and drawing it again "
                "would move the RNG stream.")
        self.calls += 1
        return check_action(int(np.argmax(logits)), chain_on, len(logits))


# ---------------------------------------------------------------------------------------------
# G5-D5: the manifest, and the refusal
# ---------------------------------------------------------------------------------------------

class ReferenceArmNotFrozen(RuntimeError):
    """A reference run was started without something the manifest needs.

    G5-D5: *"an unreproducible number is worse than no number, and this arm's whole purpose is to
    be quotable."* So this is a refusal at the start of the run, not a warning in the write-up.
    """


@dataclass
class ReferenceManifest:
    """Everything that has to be pinned before a token is spent.

    `sample_n` and `steps` are in here rather than in the write-up because G5-D4 requires the
    table to report N: *"so nobody reads a number from twelve agents as though it came from three
    hundred."* A manifest that recorded the model but not the sample size would let exactly that
    happen.
    """

    model: str = ""
    variant: str = ""
    presentation: Presentation | None = None
    temperature: float = 0.0
    max_tokens: int = 8
    sample_n: int = 0
    steps: int = 0
    seed: int = 0
    store_sha256: str = ""
    era_index: int = -1
    #: Filled in as the run proceeds. Reported, because cost is part of what this arm is for.
    tokens_in: int = 0
    tokens_out: int = 0
    calls: int = 0
    refusals: int = 0
    illegal: int = 0
    notes: list[str] = field(default_factory=list)

    REQUIRED = ("model", "variant", "presentation", "sample_n", "steps", "store_sha256")

    def check(self) -> None:
        missing = [k for k in self.REQUIRED if not getattr(self, k)]
        if missing:
            raise ReferenceArmNotFrozen(
                f"the reference arm is missing {', '.join(missing)}. G5-D5: the model identifier, "
                f"the prompt, the sampling parameters and the sample size are recorded before the "
                f"run, and a run without them produces a number nobody can reproduce.")
        if self.temperature != 0.0:
            raise ReferenceArmNotFrozen(
                f"temperature is {self.temperature}, not 0. A provider that could answer "
                f"differently on a re-run makes the table unreproducible, which is the one thing "
                f"this project has been most careful about.")
        if self.variant not in ("blind", "life", "store"):
            raise ReferenceArmNotFrozen(
                f"unknown variant {self.variant!r}; G5-D3 names exactly three")

    def digest(self) -> str:
        """One hash over everything that could change a number. Goes in the results table."""
        self.check()
        assert self.presentation is not None
        h = hashlib.sha256()
        for part in (f"model={self.model}", f"variant={self.variant}",
                     f"presentation={self.presentation.digest()}",
                     f"temperature={self.temperature}", f"max_tokens={self.max_tokens}",
                     f"sample_n={self.sample_n}", f"steps={self.steps}",
                     f"store={self.store_sha256}"):
            h.update(part.encode())
        return h.hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model, "variant": self.variant,
            "presentation_version": None if self.presentation is None
            else self.presentation.version,
            "presentation_digest": None if self.presentation is None
            else self.presentation.digest(),
            "temperature": self.temperature, "max_tokens": self.max_tokens,
            "sample_n": self.sample_n, "steps": self.steps, "seed": self.seed,
            "store_sha256": self.store_sha256, "era_index": self.era_index,
            "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
            "calls": self.calls, "refusals": self.refusals, "illegal": self.illegal,
            "digest": self.digest(), "notes": list(self.notes),
        }
