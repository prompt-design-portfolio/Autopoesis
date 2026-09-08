"""The hidden-rule device, as a domain adapter (Part B §5, §21).

This adapter wraps machinery that already existed and is already measured — the device from
`civitas.experiments.tasks.hidden_rule`, on which M4 measured a newcomer advantage of +0.300. It
is here to establish that the adapter interface is expressive enough to hold a domain that
*works*, rather than being an abstraction fitted to a domain invented to fit it.

Nothing about the device changed. The adapter is a re-description of what the benchmark already
does, which is the point: if wrapping it had required altering the device, the interface would be
the wrong shape.
"""

from __future__ import annotations

from typing import Any

from civitas.domain.enums import AgentRole, ArtifactType
from civitas.domains.base import (
    Domain,
    DomainTask,
    Evaluator,
    Stage,
    register_domain,
)


class ExactMatchEvaluator:
    """The evaluator §47 already uses: compare a submitted string to a held-out one.

    `detail` deliberately omits the expected value. An evaluation row is readable through the API,
    and ground truth in it would leak the benchmark to anything with read access.
    """

    kind = "exact_match"
    version = "exact_match/1.0"

    def judge(
        self, spec: dict[str, Any], submitted: str | None
    ) -> tuple[bool, float, dict[str, Any]]:
        expected = str(spec.get("expected", "")).strip().lower()
        given = (submitted or "").strip().lower()
        succeeded = bool(expected) and given == expected
        return succeeded, 1.0 if succeeded else 0.0, {
            "submitted": submitted, "matched": succeeded, "evaluator": self.kind,
        }


class HiddenRuleDomain:
    """Establishing a bijection between input classes and operations, one probe at a time."""

    name = "hidden_rule"
    version = "1.0"

    def stages(self) -> tuple[Stage, ...]:
        return (
            Stage("investigate", "Probe the device for: {focus}",
                  "Establish which operations the device rejects for {focus}. A rejection is a "
                  "result, not a failure — record it.", AgentRole.EXPLORER),
            Stage("verify", "Confirm the accepted operation for: {focus}",
                  "Re-probe the candidate that survived and confirm it directly.",
                  AgentRole.VERIFIER),
            Stage("synthesise", "State the mapping for: {focus}",
                  "Record the established pair and what it rules out for other classes, since "
                  "the mapping is a bijection.", AgentRole.SYNTHESIZER),
        )

    def system_prompt(self) -> str:
        return (
            "You are investigating a device that accepts exactly one operation for each input "
            "class. Probes cost budget. Read what earlier agents established before probing, and "
            "record what you establish — including what you ruled out."
        )

    def tool_names(self) -> tuple[str, ...]:
        return ("probe_device",)

    def build_tools(self, task: DomainTask) -> list[Any]:
        """A prober bound to *this* task's device, built per episode.

        The device is rebuilt from the task's recorded seed and era rather than carried on the
        domain object. A domain instance holding a device would be exactly the hidden cross-episode
        state §4 forbids — and worse, a single mutable one shared between arms.
        """
        from civitas.legacy.tasks.hidden_rule import build_device
        from civitas.legacy.tasks.probe_tool import ProbeDeviceTool

        device = build_device(
            seed=task.meta["device_seed"], era=task.meta["era"],
            n_classes=task.meta.get("n_classes", 6), n_ops=task.meta.get("n_ops", 10),
        )
        return [ProbeDeviceTool(device)]

    def evaluators(self) -> tuple[Evaluator, ...]:
        return (ExactMatchEvaluator(),)

    def artifact_vocabulary(self) -> tuple[ArtifactType, ...]:
        return (
            ArtifactType.OBSERVATION, ArtifactType.FAILURE, ArtifactType.CONCLUSION,
            ArtifactType.HYPOTHESIS,
        )

    def generate(
        self, *, seed: int, count: int, era: int = 1, n_classes: int = 6, n_ops: int = 10,
        **kwargs: Any,
    ) -> list[DomainTask]:
        from civitas.legacy.tasks.hidden_rule import build_device, era_instances

        device = build_device(seed=seed, era=era, n_classes=n_classes, n_ops=n_ops)
        tasks = []
        for instance in era_instances(device)[:count]:
            tasks.append(DomainTask(
                title=instance.title,
                description=instance.description,
                evaluator_spec=instance.evaluator_spec,
                task_family="hidden_rule",
                index=instance.index,
                environment_version=device.environment_version,
                chance_level=1.0 / max(1, len(device.operations)),
                difficulty=1.0 / max(1, len(device.operations)),
                # The answer is one of the offered operations, so it cannot be hidden by absence.
                # It is hidden by being presented identically to every alternative, and
                # `check_no_leak` is what holds that claim to account.
                candidate_terms=tuple(device.operations),
                meta={"device_seed": seed, "era": era, "input_class": instance.input_class,
                      "n_classes": n_classes, "n_ops": n_ops,
                      "environment_version": device.environment_version},
            ))
        return tasks


    # -- benchmark hooks -------------------------------------------------------------------

    def system_prompt_version(self) -> str:
        return "hidden-rule/1.0"

    def build_agent(
        self, task: DomainTask, *, record_findings: bool, probe_order_seed: int
    ) -> Any:
        from civitas.legacy.policy_agent import PolicyAgentProvider
        from civitas.legacy.tasks.hidden_rule import build_device

        device = build_device(
            seed=task.meta["device_seed"], era=task.meta["era"],
            n_classes=task.meta.get("n_classes", 6), n_ops=task.meta.get("n_ops", 10),
        )
        return PolicyAgentProvider(
            input_class=task.meta["input_class"],
            operations=list(device.operations),
            environment_version=device.environment_version,
            record_findings=record_findings,
            probe_order_seed=probe_order_seed,
        )

    def naive_probe_ceiling(self, task: DomainTask, tool_calls: int) -> float:
        from civitas.legacy.tasks.hidden_rule import probes_needed_by_chance

        return probes_needed_by_chance(task.meta.get("n_ops", 10), max(0, tool_calls - 2))


DOMAIN = register_domain(HiddenRuleDomain())
assert isinstance(DOMAIN, Domain)
