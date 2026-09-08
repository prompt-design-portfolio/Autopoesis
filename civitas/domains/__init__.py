"""Domain adapters (Part B §5, §47 -- the leak rule, which B§1 KEEPS).

Importing this package registers the built-in domains. A domain that is not imported is not
registered, which is deliberate: the registry is a record of what this process can actually run,
not a catalogue of what exists somewhere. **After G0's removal it registers none**, and that is
the correct state rather than a broken one: B§1 removes `hidden_rule` and `code_repair` as
agent-facing domains, and A1.1 removed the reason for a text domain at all -- the learner is the
grown network of `sim_v3_13.py`, which reads channels, not words.

What survives here is `base.py`: §47's leak rule and the four ways a task can give itself away.
B§1 keeps §47, and this is its implementation. `civitas_g.world.adapter.check_no_leak` states the
same rule for the vector domain, which is the one the live lineage actually uses.
"""

from __future__ import annotations

from civitas.domains.base import (
    LEAK_MARKERS,
    Domain,
    DomainLeak,
    DomainTask,
    Evaluator,
    Stage,
    all_domains,
    available_domains,
    check_no_leak,
    get_domain,
    register_domain,
)

__all__ = [
    "LEAK_MARKERS",
    "Domain",
    "DomainLeak",
    "DomainTask",
    "Evaluator",
    "Stage",
    "all_domains",
    "available_domains",
    "check_no_leak",
    "get_domain",
    "register_domain",
]
