"""The hidden-rule task family — the benchmark's substrate (Part B §22, §47).

The design question ARCHITECTURE §6.1 left open: a task family must be objectively evaluable
(§47), must not be solvable by memorising one answer, and must have enough surface variation that
a mature collective's advantage is *knowledge* rather than recall.

The answer is taken directly from the research lineage (ARCHITECTURE §2). A **device** has a hidden
bijection from input classes to operations. An episode discovers it by probing, one probe per tool
call, against a hard budget. Three properties make it the right substrate:

1. **Objectively evaluable.** The true mapping is known to the evaluator and to nothing an agent
   can read. There is no judgement call and no self-certification (§47).
2. **The mapping rotates between eras**, and so do the surface labels. This is `π`: a memorised
   answer is worthless one era later, so a collective that appears to help *by recall* is caught
   by the era rotation, and stale-artifact usage becomes measurable (§15, §48).
3. **Structure persists across eras** even though facts do not. The mapping is a bijection, so
   ruling out an operation for one class constrains every other class. That is the difference
   between a world where nothing transfers — which would make the collective pointless — and one
   where what transfers has to be *understood* rather than copied.

The budget is set so a probing agent with no prior information usually fails and one that can read
a confirmed prior finding usually succeeds. That gap is the newcomer advantage, and its size is a
property of the task, reported in the manifest rather than left implicit.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

#: Opaque labels. Deliberately meaningless: a label that hinted at its operation would let a model
#: solve the task from prior knowledge rather than from the collective, and the benchmark would
#: measure the model instead of the platform.
CLASS_POOL = [
    "aurex", "belnor", "cirith", "dovak", "ethrin", "fandel", "gyrix", "halvor",
    "ithara", "jorlen", "kaspir", "lumeth", "morvan", "nydris", "obrek", "pelvar",
]
OP_POOL = [
    "fold", "quench", "sift", "braid", "temper", "anneal", "lattice", "shear",
    "distil", "graft", "burnish", "cleave", "ferment", "rivet", "smelt", "warp",
]


@dataclass(frozen=True)
class DeviceSpec:
    """One era's device. Fully determined by `(seed, era)`, so any run is reconstructable."""

    era: int
    seed: int
    classes: tuple[str, ...]
    operations: tuple[str, ...]
    #: input class -> the one operation the device accepts. A bijection, which is the structural
    #: fact that survives a rotation and is therefore the only thing worth learning *about* the
    #: device rather than *from* an era of it.
    mapping: dict[str, str]

    @property
    def environment_version(self) -> str:
        """What an artifact records as its applicability (Part B §15).

        Era-scoped, so an artifact from era 3 is visibly inapplicable in era 4 rather than
        silently wrong.
        """
        return f"device-e{self.era}-s{self.seed}"

    def accepts(self, input_class: str, operation: str) -> bool:
        return self.mapping.get(input_class) == operation

    def answer_for(self, input_class: str) -> str | None:
        return self.mapping.get(input_class)

    def digest(self) -> str:
        blob = f"{self.era}|{self.seed}|" + "|".join(
            f"{k}={v}" for k, v in sorted(self.mapping.items())
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def build_device(*, seed: int, era: int, n_classes: int = 6, n_ops: int = 10) -> DeviceSpec:
    """Construct an era's device.

    Both the mapping **and the labels** are redrawn per era. Rotating the mapping alone would
    leave the labels as a stable index a collective could key on, which is the leak `π` exists to
    close: the association must be unavailable to anything but within-era learning.
    """
    if n_ops < n_classes:
        raise ValueError("a bijection needs at least as many operations as classes")
    if n_ops > len(OP_POOL) or n_classes > len(CLASS_POOL):
        raise ValueError("not enough labels in the pool for the requested size")

    rng = random.Random(f"device|{seed}|{era}")
    classes = tuple(rng.sample(CLASS_POOL, n_classes))
    operations = tuple(rng.sample(OP_POOL, n_ops))
    chosen = rng.sample(operations, n_classes)
    return DeviceSpec(
        era=era,
        seed=seed,
        classes=classes,
        operations=operations,
        mapping=dict(zip(classes, chosen, strict=True)),
    )


@dataclass
class TaskInstance:
    """One question about the device: which operation does it accept for this class?"""

    device: DeviceSpec
    input_class: str
    index: int = 0

    @property
    def title(self) -> str:
        return f"Determine the accepted operation for input class '{self.input_class}'"

    @property
    def description(self) -> str:
        ops = ", ".join(self.device.operations)
        return (
            f"A device accepts exactly one operation for each input class.\n\n"
            f"Input class under investigation: {self.input_class}\n"
            f"Available operations: {ops}\n\n"
            f"Use probe_device(input_class, operation) to test one combination per call — each "
            f"probe costs a tool call and your budget is small. The device's mapping is a "
            f"bijection: no two input classes accept the same operation.\n\n"
            f"Search the collective knowledge base first. Earlier agents may have established "
            f"or ruled out operations for this class. Record what you establish, then call "
            f"submit_result with the operation name."
        )

    @property
    def evaluator_spec(self) -> dict[str, Any]:
        """Held on the `Task` row, never shown to the agent (Part B §47)."""
        return {
            "kind": "exact_match",
            "expected": self.device.answer_for(self.input_class),
            "device_digest": self.device.digest(),
            "environment_version": self.device.environment_version,
        }

    @property
    def chance_success_rate(self) -> float:
        """The level a guess achieves. Reported alongside every result, so a success rate is
        always read against the baseline it must beat rather than against zero."""
        return 1.0 / len(self.device.operations)


def era_instances(device: DeviceSpec, *, count: int | None = None) -> list[TaskInstance]:
    """One instance per input class, in a stable order."""
    classes = device.classes if count is None else device.classes[:count]
    return [TaskInstance(device=device, input_class=c, index=i) for i, c in enumerate(classes)]


def probes_needed_by_chance(n_ops: int, n_probes: int) -> float:
    """Probability that `n_probes` distinct random probes find the accepted operation.

    The honest null for this task. Sampling *without* replacement, because an agent that probes
    the same pair twice has wasted a call rather than taken a second draw — modelling it with
    replacement would understate the naive agent and inflate every advantage measured against it.
    """
    return min(1.0, n_probes / n_ops)
