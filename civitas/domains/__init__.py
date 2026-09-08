"""Domain adapters (Part B §5).

Importing this package registers the built-in domains. A domain that is not imported is not
registered, which is deliberate: the registry is a record of what this process can actually run,
not a catalogue of what exists somewhere.
"""

from __future__ import annotations

from civitas.domains import code_repair, hidden_rule  # noqa: F401  (registration side effect)
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
