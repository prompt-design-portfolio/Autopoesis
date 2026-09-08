"""The tool interface (Part B §17, §18).

Two kinds of tool, one interface:

* **Built-in tools** are how an episode touches the collective — create an artifact, search
  knowledge, record a failure. They run in-process because they *are* the process: they are
  Civitas's own code operating on its own database, not agent-generated code.
* **Agent-created tools** are untrusted source and always run in the sandbox (§18).

`ToolContext` is what a tool is allowed to see. It carries the episode, its session and its budget
tracker — and deliberately not the settings object, so no tool can read a credential (§40).
"""

from __future__ import annotations

import abc
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from civitas.persistence.types import canonical_json

if TYPE_CHECKING:  # pragma: no cover
    from sqlalchemy.orm import Session

    from civitas.runtime.budgets import BudgetTracker


@dataclass
class ToolContext:
    """What a tool may see. Anything not here is out of a tool's reach by construction."""

    session: Session
    workspace_id: uuid.UUID
    episode_id: uuid.UUID
    budgets: BudgetTracker
    experiment_arm: str
    environment_version: str
    config_hash: str = ""
    project_id: uuid.UUID | None = None
    task_id: uuid.UUID | None = None
    #: Artifacts the episode has actually read, accumulated for `ArtifactUsage` (Part A §A2.1).
    #: An id here means the episode consumed the artifact, not merely that retrieval returned it.
    read_artifacts: dict[uuid.UUID, int] = field(default_factory=dict)
    #: Set when the episode produced something the collective keeps. Drives the no-progress
    #: termination reason, which is diagnostic where `budget_exhausted` is not.
    made_progress: bool = False
    #: Parameters of the active retrieval policy, from the episode spec. Threaded through rather
    #: than read from a module global so that a matched A/B (Part A §A2.2) can vary the policy
    #: between arms while everything else stays identical — and so the parameters a run actually
    #: used are recoverable from the episode.
    retrieval_options: dict[str, Any] = field(default_factory=dict)
    scratch: dict[str, Any] = field(default_factory=dict)

    def note_read(self, artifact_id: uuid.UUID) -> None:
        self.read_artifacts[artifact_id] = self.read_artifacts.get(artifact_id, 0) + 1


@dataclass
class ToolResult:
    ok: bool
    content: str = ""
    data: Any = None
    error: str = ""
    #: Set when the runtime blocked the call before executing it — a duplicate failure
    #: (Part A §A2.3) or a policy violation. Distinct from `ok=False`, which means the tool ran
    #: and failed.
    blocked_reason: str | None = None

    def to_message(self) -> str:
        if self.blocked_reason:
            return f"BLOCKED: {self.blocked_reason}\n{self.content}"
        return self.content if self.ok else f"ERROR: {self.error}"


class Tool(abc.ABC):
    """A capability an episode can invoke."""

    name: str = ""
    description: str = ""
    #: JSON Schema for the arguments. Validated before the tool runs, so a malformed call is a
    #: validation error the agent can read rather than an exception in tool code.
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    #: Tools that only read cost a budget slot but cannot make the collective worse. Marked so
    #: policy can allow a read-only tool where it would refuse a writing one.
    read_only: bool = False

    @abc.abstractmethod
    def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        ...

    def spec(self):
        from civitas.runtime.providers.base import ToolSpec

        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)

    def args_hash(self, args: dict[str, Any]) -> str:
        """Canonical identity of (this tool, these arguments) — Part A §A2.3.

        Canonical because "the same call" must not depend on key order or float formatting;
        otherwise duplicate-failure detection misses the repeat it exists to catch.
        """
        import hashlib

        return hashlib.sha256(f"{self.name}|{canonical_json(args)}".encode()).hexdigest()


class ToolRegistry:
    """The tools available to an episode.

    Built per episode from the profile's tool policy, so which tools exist is an *experimental
    variable* recorded on the episode (§8) rather than a global the runtime happens to have.
    """

    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {t.name: t for t in (tools or [])}

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> tuple:
        return tuple(t.spec() for t in self._tools.values())

    def filtered(self, allowed: list[str] | None) -> ToolRegistry:
        if allowed is None:
            return self
        return ToolRegistry([t for name, t in self._tools.items() if name in allowed])

    def __len__(self) -> int:
        return len(self._tools)


def validate_arguments(schema: dict[str, Any], args: dict[str, Any]) -> str | None:
    """Check arguments against a JSON Schema subset. Returns an error message, or None.

    A deliberately small validator rather than a `jsonschema` dependency: it covers the shapes
    tool parameters actually take (object with typed properties and a required list), and an
    unsupported construct is *accepted* rather than rejected, so a tool is never blocked by the
    validator failing to understand its own schema.
    """
    if schema.get("type") not in (None, "object"):
        return None
    props: dict[str, Any] = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in args:
            return f"missing required argument {name!r}"
    for key, value in args.items():
        if key not in props:
            if schema.get("additionalProperties") is False:
                return f"unexpected argument {key!r}; expected one of {sorted(props)}"
            continue
        expected = props[key].get("type")
        if expected and not _type_ok(expected, value):
            return f"argument {key!r} must be {expected}, got {type(value).__name__}"
        if "enum" in props[key] and value not in props[key]["enum"]:
            return f"argument {key!r} must be one of {props[key]['enum']}"
    return None


_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
}


def _type_ok(expected: str, value: Any) -> bool:
    if expected == "integer" and isinstance(value, bool):
        return False  # bool is an int in Python; a schema asking for an integer does not mean bool
    types = _TYPES.get(expected)
    return True if types is None else isinstance(value, types)
