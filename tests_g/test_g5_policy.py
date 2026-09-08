"""G5 §7 — the policy layer, and the refusals that keep the arm quotable.

The engine hook itself (engine version `G5-policy`) has its own equivalence check; these are the
Civitas-side pieces, which need no engine and no provider.
"""

from __future__ import annotations

import numpy as np
import pytest

from civitas_g.g5.policy import (
    IllegalAction,
    MirrorPolicy,
    NetworkPolicy,
    ReferenceArmNotFrozen,
    ReferenceManifest,
    check_action,
)
from civitas_g.g5.presentation import Presentation
from civitas_g.world.adapter import N_ACTIONS, PREP0


def _manifest(**kw):
    base = dict(model="pinned-model-id", variant="blind",
                presentation=Presentation(variant="blind"),
                sample_n=12, steps=300, store_sha256="ab" * 32)
    base.update(kw)
    return ReferenceManifest(**base)


# --------------------------------------------------------------------------- masking

def test_an_action_outside_the_set_is_refused_not_clipped():
    """G5-D7. A model that proposes an illegal action has told us something; clipping it would
    record a choice it did not make."""
    for bad in (-1, N_ACTIONS, 99):
        with pytest.raises(IllegalAction, match="outside"):
            check_action(bad, chain_on=True)


def test_a_preparation_is_refused_while_the_chain_is_off():
    with pytest.raises(IllegalAction, match="chain is off"):
        check_action(PREP0, chain_on=False)
    check_action(PREP0 - 1, chain_on=False)          # the five phase-1 actions are fine


def test_a_non_index_is_refused():
    for bad in (None, "3", 3.5, True):
        with pytest.raises(IllegalAction, match="not an action index"):
            check_action(bad, chain_on=True)


def test_the_action_range_is_read_from_the_world_not_the_module():
    """The K-blindness that `Record`, the matched null and `check_chance_ev_is_zero` each had.
    At K = 5 the module constant is right, so a K-blind check never fails a test -- it just
    accepts nothing wrong and rejects action 10 on a K = 7 world, where it is legal."""
    check_action(N_ACTIONS + 1, chain_on=True, n_actions=N_ACTIONS + 2)
    with pytest.raises(IllegalAction, match="outside"):
        check_action(N_ACTIONS + 1, chain_on=True)


# --------------------------------------------------------------------------- the two policies

def test_the_network_policy_defers_every_step():
    """The first half of §7.1: a run under this must be bit-identical to a run with no policy."""
    p = NetworkPolicy()
    assert all(p(i, np.zeros(31), True, np.zeros(10)) is None for i in range(5))
    assert p.calls == 5


def test_the_mirror_policy_returns_the_engines_own_choice():
    """The second half of §7.1 -- the whole loop against a stand-in whose answers are known."""
    p = MirrorPolicy()
    logits = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0])
    assert p(0, np.zeros(31), True, logits) == 5
    assert p.calls == 1


def test_the_mirror_policy_refuses_to_recompute_the_logits():
    """Recomputing them needs the action-noise draw, and drawing it again moves the RNG stream --
    which would make the equivalence check fail for a reason that is not the hook."""
    with pytest.raises(RuntimeError, match="cannot recompute"):
        MirrorPolicy()(0, np.zeros(31), True, None)


# --------------------------------------------------------------------------- G5-D5

def test_a_reference_run_refuses_to_start_unfrozen():
    """*An unreproducible number is worse than no number.* So this is a refusal at the start of
    the run, not a caveat in the write-up."""
    for missing in ("model", "sample_n", "steps", "store_sha256"):
        with pytest.raises(ReferenceArmNotFrozen, match=missing):
            _manifest(**{missing: type(getattr(_manifest(), missing))()}).check()


def test_a_reference_run_refuses_a_nonzero_temperature():
    with pytest.raises(ReferenceArmNotFrozen, match="temperature"):
        _manifest(temperature=0.7).check()


def test_only_the_three_named_variants_exist():
    for v in ("blind", "life", "store"):
        _manifest(variant=v).check()
    with pytest.raises(ReferenceArmNotFrozen, match="exactly three"):
        _manifest(variant="store_and_life").check()


def test_the_digest_moves_with_everything_that_could_change_a_number():
    base = _manifest().digest()
    assert _manifest().digest() == base
    for kw in ({"model": "other"}, {"variant": "life"}, {"sample_n": 13}, {"steps": 301},
               {"store_sha256": "cd" * 32}, {"max_tokens": 16},
               {"presentation": Presentation(variant="life")}):
        assert _manifest(**kw).digest() != base, kw


def test_the_manifest_reports_n_so_twelve_agents_cannot_read_as_three_hundred():
    """G5-D4's stated reason for putting the sample size in the manifest rather than the prose."""
    d = _manifest(sample_n=12).as_dict()
    assert d["sample_n"] == 12
    assert d["steps"] == 300
    assert "digest" in d and "presentation_digest" in d
