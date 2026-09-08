"""The distributed-knowledge task family (Part B §23).

> Build tasks where no single episode can access all information required for the solution. [...]
> The solution must emerge by combining information across the collective environment.

The design has one hard requirement: it must be **provably** unsolvable by a single episode, not
merely hard for one. A task that a lucky agent could brute-force would make a positive result
uninterpretable — you could not tell distributed cognition from a good guess.

So the device is a **two-factor rule**:

* every input class belongs to a hidden *family*, discoverable only by `probe_family`;
* every family maps to a hidden *operation*, discoverable only by `probe_table`.

The accepted operation for a class is `table[family[class]]`. Neither probe alone determines it,
and the two probes are given to **different populations**: an episode in partition A has
`probe_family` and not `probe_table`, and vice versa. There is no third route — the composed answer
is never exposed by any tool.

`assert_unsolvable_alone` proves the property rather than assuming it, by enumerating what each
partition can learn with unlimited probing and checking that neither determines the answer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from civitas.experiments.tasks.hidden_rule import CLASS_POOL, OP_POOL

FAMILY_POOL = ["ferrous", "saline", "vitric", "amber", "basalt", "nitre"]


@dataclass(frozen=True)
class SplitDevice:
    """A device whose rule is split across two populations (§23)."""

    era: int
    seed: int
    classes: tuple[str, ...]
    families: tuple[str, ...]
    operations: tuple[str, ...]
    #: class -> family. Learnable only in partition A.
    family_of: dict[str, str]
    #: family -> operation. Learnable only in partition B.
    operation_of: dict[str, str]

    @property
    def environment_version(self) -> str:
        return f"split-e{self.era}-s{self.seed}"

    def answer_for(self, input_class: str) -> str | None:
        family = self.family_of.get(input_class)
        return self.operation_of.get(family) if family else None

    def family_accepts(self, input_class: str, family: str) -> bool:
        return self.family_of.get(input_class) == family

    def table_accepts(self, family: str, operation: str) -> bool:
        return self.operation_of.get(family) == operation


def build_split_device(
    *, seed: int, era: int = 1, n_classes: int = 4, n_families: int = 3, n_ops: int = 8
) -> SplitDevice:
    if n_ops < n_families:
        raise ValueError("each family needs a distinct operation")
    rng = random.Random(f"split|{seed}|{era}")
    classes = tuple(rng.sample(CLASS_POOL, n_classes))
    families = tuple(rng.sample(FAMILY_POOL, n_families))
    operations = tuple(rng.sample(OP_POOL, n_ops))

    # Every family must be used by at least one class, or a partition-B finding about an unused
    # family is dead information and the benchmark quietly gets easier.
    assigned = list(families) + [rng.choice(families) for _ in range(n_classes - n_families)]
    rng.shuffle(assigned)
    family_of = dict(zip(classes, assigned[:n_classes], strict=True))
    operation_of = dict(zip(families, rng.sample(operations, n_families), strict=True))
    return SplitDevice(
        era=era, seed=seed, classes=classes, families=families, operations=operations,
        family_of=family_of, operation_of=operation_of,
    )


def assert_unsolvable_alone(device: SplitDevice) -> dict[str, Any]:
    """Prove no single partition can determine the answer, however much it probes (§23).

    Enumerates the *complete* knowledge each partition could obtain — partition A learns the whole
    class→family map, partition B the whole family→operation map — and checks that neither, alone,
    narrows any class to one operation.

    Called before the benchmark runs. A distributed-cognition result on a task that turned out to
    be individually solvable would be evidence of nothing, and this is cheaper than discovering
    that afterwards.
    """
    report: dict[str, Any] = {"classes": len(device.classes), "violations": []}

    # Partition A knows every class's family, and nothing about operations. Its candidate set for
    # any class is therefore every operation.
    for input_class in device.classes:
        if len(device.operations) <= 1:
            report["violations"].append(
                f"partition A determines {input_class}: only one operation exists"
            )
    # Partition B knows every family's operation, and nothing about which family a class is in.
    # Its candidate set for a class is one operation per family.
    distinct_family_ops = {device.operation_of[f] for f in device.families}
    if len(distinct_family_ops) <= 1:
        report["violations"].append(
            "partition B determines every class: all families share one operation"
        )
    for input_class in device.classes:
        if len({device.operation_of[f] for f in device.families}) == 1:
            report["violations"].append(f"partition B determines {input_class}")

    report["partition_a_candidates_per_class"] = len(device.operations)
    report["partition_b_candidates_per_class"] = len(distinct_family_ops)
    report["unsolvable_alone"] = not report["violations"]
    if not report["unsolvable_alone"]:
        raise AssertionError(
            "the split device is individually solvable, so a positive result would be "
            f"uninterpretable: {report['violations']}"
        )
    return report


@dataclass
class SplitTaskInstance:
    device: SplitDevice
    input_class: str
    #: "a" learns families, "b" learns the table, "integrator" combines what both left behind.
    partition: str
    index: int = 0

    @property
    def title(self) -> str:
        if self.partition == "a":
            return f"Determine the family of input class '{self.input_class}'"
        if self.partition == "b":
            return "Determine which operation each material family accepts"
        return f"Determine the accepted operation for input class '{self.input_class}'"

    @property
    def description(self) -> str:
        common = (
            "\nRecord what you establish in the collective knowledge base. Another population is "
            "working on the other half of this problem and cannot see your probes, only what you "
            "write down.\n"
        )
        if self.partition == "a":
            return (
                f"Every input class belongs to one material family. Determine the family of "
                f"'{self.input_class}'.\n\n"
                f"Families: {', '.join(self.device.families)}\n"
                f"Use probe_family(input_class, family).{common}"
            )
        if self.partition == "b":
            return (
                "Every material family accepts exactly one operation. Determine the operation "
                "for each family.\n\n"
                f"Families: {', '.join(self.device.families)}\n"
                f"Operations: {', '.join(self.device.operations)}\n"
                f"Use probe_table(family, operation).{common}"
            )
        return (
            f"Determine the operation the device accepts for input class "
            f"'{self.input_class}'.\n\n"
            f"The rule has two parts: each class belongs to a family, and each family accepts one "
            f"operation. You have **neither probe**. Other agents have established both halves "
            f"and recorded them — search the collective knowledge base and combine what you "
            f"find.\n\n"
            f"Operations: {', '.join(self.device.operations)}\n"
        )

    @property
    def evaluator_spec(self) -> dict[str, Any]:
        if self.partition == "a":
            expected = self.device.family_of[self.input_class]
        elif self.partition == "b":
            expected = ",".join(
                f"{f}={self.device.operation_of[f]}" for f in sorted(self.device.families)
            )
        else:
            expected = self.device.answer_for(self.input_class)
        return {
            "kind": "exact_match",
            "expected": expected,
            "partition": self.partition,
            "environment_version": self.device.environment_version,
        }


def partition_instances(device: SplitDevice) -> dict[str, list[SplitTaskInstance]]:
    """One instance per class for partition A, one table task for B, and the integrator tasks."""
    return {
        "a": [
            SplitTaskInstance(device=device, input_class=c, partition="a", index=i)
            for i, c in enumerate(device.classes)
        ],
        "b": [SplitTaskInstance(device=device, input_class=device.classes[0], partition="b")],
        "integrator": [
            SplitTaskInstance(device=device, input_class=c, partition="integrator", index=i)
            for i, c in enumerate(device.classes)
        ],
    }
