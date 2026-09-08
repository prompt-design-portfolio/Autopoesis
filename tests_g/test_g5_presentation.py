"""G5 step 1: the presentation, and the leak rule over rendered text.

`docs/G5_SPEC.md` §6 puts this before any model is called, and §2.1 says plainly where the rule
will be broken: in the rendering of the read channels. So the tests here are mostly attempts to
break it -- a check that has never rejected anything is not evidence that anything is safe.
"""

from __future__ import annotations

import numpy as np
import pytest

from civitas_g.g5 import presentation as P
from civitas_g.world.adapter import DEAD_INPUTS, MODULATOR_EVENTS, N_IN, READ

# --------------------------------------------------------------------------- the rendering

def test_every_input_is_rendered_and_named_by_its_layout_position():
    text = P.render_observation(np.arange(N_IN, dtype=float))
    lines = text.splitlines()
    assert len(lines) == N_IN
    assert len(P.FIELD_NAMES) == N_IN
    assert len(set(P.FIELD_NAMES)) == N_IN, "two inputs share a name; the rendering is not lossless"
    for i, line in enumerate(lines):
        assert line.startswith(f"{i:>3} "), line


def test_the_rendering_is_lossless_to_its_stated_precision():
    rng = np.random.default_rng(0)
    obs = rng.normal(size=N_IN)
    back = np.array([float(line.split("=")[1]) for line in
                     P.render_observation(obs).splitlines()])
    assert np.allclose(back, obs, atol=10.0 ** -P.DECIMALS)


def test_dead_inputs_are_rendered_as_zeros_rather_than_removed():
    """G5-D1. Removing them would tell the model which inputs not to bother with, which is a
    piece of the search the learner does for itself."""
    obs = np.zeros(N_IN)
    lines = P.render_observation(obs).splitlines()
    assert DEAD_INPUTS, "the layout claims no dead inputs; this test is checking nothing"
    for i in DEAD_INPUTS:
        assert lines[i].endswith("+0.0000"), lines[i]


def test_the_rendering_refuses_an_observation_of_the_wrong_width():
    with pytest.raises(ValueError, match="the layout names"):
        P.render_observation(np.zeros(N_IN - 1))


def test_the_read_channels_are_named_by_channel_not_by_preparation():
    """§2.1. A mark sits at label pi(k); naming the channel by k would decode pi in the prompt."""
    read_names = P.FIELD_NAMES[READ:]
    assert read_names, "no read channels in the layout"
    for j, name in enumerate(read_names):
        assert name == f"read_channel_{j}"
        assert "prep" not in name


# --------------------------------------------------------------------------- the preamble

def test_the_modulator_table_is_rendered_without_its_notes():
    """`adapter`'s own note for `prep_ok` says `k == mapping[ftype]. Writes a mark at label
    pi(k)`. Rendering the note would transmit the fact the population has to acquire."""
    text = P.render_modulator_table()
    assert len(text.splitlines()) == len(MODULATOR_EVENTS)
    for e in MODULATOR_EVENTS:
        assert e.event in text
        if e.note:
            assert e.note not in text, f"the note for {e.event} was rendered"
    assert "mapping" not in text and "pi(" not in text


def test_the_preamble_says_nothing_about_the_mapping_the_clock_or_the_channels():
    P.Presentation().check()          # raises on any leak


def test_the_presentation_digest_is_stable_and_variant_specific():
    a, b = P.Presentation(variant="blind"), P.Presentation(variant="blind")
    assert a.digest() == b.digest()
    assert a.digest() != P.Presentation(variant="store").digest()


def test_a_step_carries_history_only_when_the_variant_supplies_it():
    """G5-D3's three variants differ in the context and in nothing else."""
    obs = np.zeros(N_IN)
    blind = P.Presentation(variant="blind").step(obs)
    life = P.Presentation(variant="life").step(obs, history="  0 move_+y  +0.0000")
    assert "Earlier steps" not in blind
    assert "Earlier steps" in life
    assert blind in life or P.render_observation(obs) in life


# --------------------------------------------------------------------------- breaking it

@pytest.mark.parametrize("leak", [
    "a previous agent succeeded with preparation 3 on this food here",
    "read_channel_3 is preparation 3",
    "the label permutation pi is redrawn every era",
    "channel 3 endorses preparation 3",
    "the correct preparation for this food type",
    "the current mapping from food type to preparation",
    "this is the recipe for the food underfoot",
])
def test_the_check_catches_a_rendering_that_decoded_pi(leak):
    """§2.1 names the exact sentence that breaks the rule; these are it and its neighbours.
    A check that has never rejected anything is not evidence."""
    with pytest.raises(P.PresentationLeak):
        P.check_presentation_no_leak(leak)


def test_the_check_passes_the_honest_rendering():
    """§2.1's honest form: `channel 3 here reads +0.42`, with no statement of what it means."""
    P.check_presentation_no_leak("read_channel_3 = +0.4200")


def test_the_structural_conditions_are_checked_before_the_text():
    """A rendering of a leaky world is leaky whatever the text says."""
    from civitas_g.world.adapter import DomainLeak
    with pytest.raises(DomainLeak, match="preparation indices"):
        P.check_presentation_no_leak("read_channel_0 = +0.0000", marks_are_labels=False)
    with pytest.raises(DomainLeak, match="pi is not redrawn"):
        P.check_presentation_no_leak("read_channel_0 = +0.0000", pi_redrawn_every_era=False)


# --------------------------------------------------------------------------- what it is not

def test_no_acceptance_function_reads_the_reference_arm():
    """§3.1, structurally rather than editorially: G5 has no claim line, and the way that is
    kept true is that nothing in the acceptance path imports it."""
    import inspect

    import civitas_g.g3 as g3
    import civitas_g.g4 as g4
    for mod in (g3, g4):
        src = inspect.getsource(mod)
        assert "g5" not in src.lower().replace("g5_spec", ""), f"{mod.__name__} reaches into G5"
