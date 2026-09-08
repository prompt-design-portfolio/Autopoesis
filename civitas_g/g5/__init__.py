"""G5 — the reference arm. Reported, never claimed.

`docs/G5_SPEC.md` is the specification. The one thing worth repeating here, because this is the
package most likely to be imported by someone who has not read it: **nothing in this package is
eligible for a claim.** No acceptance function reads it, no gate reads it, and its numbers are
not comparable to the learner's without the reduction recorded beside them (G5-D4).
"""

from civitas_g.g5.presentation import (
    PRESENTATION_VERSION,
    Presentation,
    PresentationLeak,
    check_presentation_no_leak,
    render_action_set,
    render_modulator_table,
    render_observation,
)

__all__ = [
    "PRESENTATION_VERSION",
    "Presentation",
    "PresentationLeak",
    "check_presentation_no_leak",
    "render_action_set",
    "render_modulator_table",
    "render_observation",
]
