"""Domain adapters (Part B §5, §47).

Everything the core does — bounded episodes, hybrid retrieval, arm enforcement, credit
assignment, the scheduler, the experiment gate — is domain-agnostic. What is *not* agnostic is
four things: what a task says, which tools can act on it, who judges the answer, and what the
stages of work are called. A domain adapter is exactly those four things and nothing else.

The reason this layer exists at all is §65: a result measured on one task family is a result about
that task family. The newcomer advantage of +0.300 measured on the hidden-rule device is not a
claim about collective intelligence until the same machinery, unchanged, produces a comparable
number on a task of a different shape. An adapter is what makes that comparison possible without
forking the runtime.

**The leak rule.** §47 says an agent must not be able to read its own success criteria. A domain
is where that is easiest to violate, because the domain author writes both the description and the
evaluator spec, often in the same function. So the boundary enforces it rather than trusting it:
every domain declares `hidden_terms` (strings that must not appear in agent-visible text at all)
and `candidate_terms` (the alternatives among which the answer must appear indistinguishably), and
`check_no_leak` is run over every generated task by a test that iterates the registry. A new
domain cannot be added without stating its own indistinguishability argument and passing it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from civitas.domain.enums import AgentRole, ArtifactType

#: Words that would single out the answer even if the answer itself is one of many candidates.
LEAK_MARKERS: tuple[str, ...] = (
    "expected", "the correct", "the answer is", "ground truth", "solution:", "evaluator",
)


class DomainLeak(AssertionError):
    """Raised when a generated task's agent-visible text reveals its own success criteria."""


@dataclass(frozen=True)
class Stage:
    """One step of §5's decomposition, as a domain declares it."""

    family: str
    title: str
    description: str
    role: AgentRole


@dataclass
class DomainTask:
    """One unit of work, with the agent-visible half and the held-out half kept apart.

    `title` and `description` are what an episode reads. `evaluator_spec` is what an evaluator
    reads, and never reaches an `EpisodeContext` — it lives on the `Task` row and the API's
    `TaskOut` has no field for it.
    """

    title: str
    description: str
    evaluator_spec: dict[str, Any]
    task_family: str = "investigate"
    #: Position within the domain's generated set. Used to seed a per-agent policy order that is
    #: identical across arms, so any difference between arms comes from what they showed the
    #: agent rather than from which agent they got (§20).
    index: int = 0
    #: The era this task belongs to (§15). An artifact recorded under one is visibly inapplicable
    #: under the next, which is what makes recall distinguishable from knowledge.
    environment_version: str = "dev"
    #: Chance-level success on this task, for a reader to compare a rate against. A success rate
    #: with no chance level beside it is not interpretable.
    chance_level: float = 0.0
    difficulty: float = 0.0
    #: Strings that must not appear in `title` or `description` under any circumstances.
    hidden_terms: tuple[str, ...] = ()
    #: The alternatives the answer must be indistinguishable from. Empty means the answer is not
    #: drawn from a presented set — then `hidden_terms` carries the whole burden.
    candidate_terms: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def agent_visible_text(self) -> str:
        return f"{self.title}\n{self.description}"


def check_no_leak(task: DomainTask) -> None:
    """§47, enforced at the boundary where domains are written.

    Three separate ways a task can give itself away, and all three are checked:

    1. a hidden term appears verbatim in what the agent reads;
    2. a marker word singles the answer out ("the correct operation is ...");
    3. the answer is presented differently from the alternatives it is supposed to hide among —
       mentioned more often, or mentioned when a candidate is omitted. Presence-by-omission is the
       subtle one, and it is the one an author is most likely to introduce by accident.
    """
    text = task.agent_visible_text
    lowered = text.lower()

    for term in task.hidden_terms:
        if term and term.lower() in lowered:
            raise DomainLeak(f"hidden term {term!r} appears in agent-visible text")

    for marker in LEAK_MARKERS:
        if marker in lowered:
            raise DomainLeak(f"marker {marker!r} singles out the answer")

    if task.candidate_terms:
        counts = {c: len(re.findall(re.escape(c.lower()), lowered)) for c in task.candidate_terms}
        missing = [c for c, n in counts.items() if n == 0]
        if missing:
            raise DomainLeak(
                f"candidates {missing!r} are absent while others are present — the answer is "
                f"distinguishable by omission"
            )
        if len(set(counts.values())) != 1:
            raise DomainLeak(
                f"candidates are mentioned unequally ({counts!r}) — the answer is distinguishable "
                f"by frequency"
            )


@runtime_checkable
class Evaluator(Protocol):
    """Judges a submission against a spec no episode can see (§47)."""

    kind: str
    version: str

    def judge(
        self, spec: dict[str, Any], submitted: str | None
    ) -> tuple[bool, float, dict[str, Any]]:
        """Return (succeeded, score, detail). `detail` is API-readable, so it must not contain
        the ground truth — an evaluation row that leaks the answer leaks the benchmark."""


@runtime_checkable
class Domain(Protocol):
    """A problem domain, as a bundle of the four things the core cannot supply itself."""

    name: str
    version: str

    def stages(self) -> tuple[Stage, ...]: ...

    def system_prompt(self) -> str: ...

    def tool_names(self) -> tuple[str, ...]:
        """Domain tools *in addition to* the builtins. Names only — instances are built per
        episode, because a tool holding cross-episode state would breach §4."""

    def build_tools(self, task: DomainTask) -> list[Any]:
        """Instances of the tools this task needs, built fresh for this episode.

        Separate from `tool_names` because some domain tools cannot exist without a task: the
        device prober needs the device, and a globally registered one would either hold a device
        across episodes (breaching §4) or have none to probe. A domain whose tools are already in
        a shared registry returns nothing here.
        """
        return []

    def evaluators(self) -> tuple[Evaluator, ...]: ...

    def generate(self, *, seed: int, count: int, **kwargs: Any) -> list[DomainTask]: ...

    def artifact_vocabulary(self) -> tuple[ArtifactType, ...]:
        """The artifact types this domain expects work to produce. Used by the UI and by
        consolidation to know what a normal corpus looks like here."""

    # -- benchmark hooks (Part B §22) --------------------------------------------------------
    #
    # A domain that supplies these can be run through the newcomer procedure. They are on the
    # same object as `generate` deliberately: the agent, the tools and the tasks have to agree
    # about the environment, and splitting them across two registries is how they stop agreeing.

    def system_prompt_version(self) -> str:
        """Part of the frozen configuration hash (§20), so it must change when the prompt does."""

    def build_agent(
        self, task: DomainTask, *, record_findings: bool, probe_order_seed: int
    ) -> Any:
        """A deterministic policy agent for this task (§20).

        Deterministic rather than model-backed because the benchmark measures the *platform's*
        contribution: with the agent a pure function of what it is shown, a difference between
        arms is caused by what the arms showed it. A domain that cannot supply one can still be
        run with a real provider, but not under frozen-model mode.
        """

    def naive_probe_ceiling(self, task: DomainTask, tool_calls: int) -> float:
        """The success rate an agent that learns nothing would reach at this budget.

        The reference a measured rate has to beat before "the collective helped" means anything.
        """


_REGISTRY: dict[str, Domain] = {}


def register_domain(domain: Domain) -> Domain:
    if domain.name in _REGISTRY:
        raise ValueError(f"domain {domain.name!r} is already registered")
    _REGISTRY[domain.name] = domain
    return domain


def get_domain(name: str) -> Domain:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"no domain {name!r}; registered: {sorted(_REGISTRY)}"
        ) from None


def available_domains() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def all_domains() -> tuple[Domain, ...]:
    return tuple(_REGISTRY[n] for n in sorted(_REGISTRY))
