"""Repairing a defective function, as a domain adapter (Part B §5, §18, §47).

The hidden-rule domain and this one share nothing but the core. There the answer is one of ten
offered strings and the evaluator is a string comparison; here the answer is a program the agent
writes and the evaluator *executes it in the hardened sandbox* against checks the agent never
sees. If the same episode runtime, the same arms, the same retrieval and the same credit
assignment carry both without modification, then the adapter boundary is real and the M4 result
is a result about the machinery rather than about one device.

**Why the checks are data, not code.** `evaluator_spec` lives in a JSON column on the `Task` row.
Holding the checks as `[[args], result]` pairs rather than as a Python test module means the spec
round-trips through the database and through a manifest unchanged, so a replay (§46) runs the
identical evaluator — and means the harness that runs them is one audited piece of code rather
than arbitrary source that arrived with the task.

**Why the failing check is reported by index.** An `Evaluation` row is readable through the API.
Reporting *which* input failed would hand a second attempt the test suite one row at a time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from civitas.domain.enums import AgentRole, ArtifactType
from civitas.domains.base import Domain, DomainTask, Evaluator, Stage, register_domain
from civitas.runtime.sandbox import SandboxLimits

#: Deliberately tight. A repair that needs more than this is not being measured by these checks,
#: and a runaway submission must cost a bounded amount of wall clock rather than a worker.
EVAL_LIMITS = SandboxLimits(cpu_seconds=5, wall_seconds=10, memory_mb=256, max_processes=1)

HARNESS = '''
import json as _json

_CHECKS = _json.loads({checks!r})


def main():
    passed = 0
    first_failure = None
    errors = 0
    for i, (args, want) in enumerate(_CHECKS):
        try:
            got = {entry}(*args)
        except Exception:
            errors += 1
            if first_failure is None:
                first_failure = i
            continue
        if got == want:
            passed += 1
        elif first_failure is None:
            first_failure = i
    return {{"passed": passed, "total": len(_CHECKS), "first_failure": first_failure,
             "errors": errors}}
'''


class SandboxedTestsEvaluator:
    """Runs a submitted implementation against held-out checks inside the sandbox (§18, §47).

    The sandbox is not an optimisation here, it is the evaluator's correctness condition: the
    submission is agent-written code, and running it in this process would let a task's answer
    reach the evaluator's own memory.
    """

    kind = "sandboxed_tests"
    version = "sandboxed_tests/1.0"

    def __init__(self, sandbox: Any = None):
        self._sandbox = sandbox

    @property
    def sandbox(self):
        if self._sandbox is None:
            from civitas.runtime.sandbox import get_sandbox

            self._sandbox = get_sandbox()
        return self._sandbox

    def judge(
        self, spec: dict[str, Any], submitted: str | None
    ) -> tuple[bool, float, dict[str, Any]]:
        checks = spec.get("checks") or []
        entry = spec.get("entry", "solve")
        if not submitted or not submitted.strip():
            return False, 0.0, {"evaluator": self.kind, "reason": "no submission",
                                "passed": 0, "total": len(checks)}

        source = submitted + "\n\n" + HARNESS.format(checks=json.dumps(checks), entry=entry)
        result = self.sandbox.run_python(source, limits=EVAL_LIMITS, entrypoint="main")

        detail: dict[str, Any] = {
            "evaluator": self.kind,
            "backend": result.backend,
            "isolation_level": result.isolation_level.value,
            "total": len(checks),
        }
        if result.unenforced_limits:
            # §46: a bound the caller believes is in force but is not must reach the record.
            detail["unenforced_limits"] = list(result.unenforced_limits)

        if not result.succeeded:
            detail.update({
                "passed": 0,
                "reason": "timeout" if result.timed_out else "submission did not run",
                "limit_hit": result.limit_hit,
                # stderr is the agent's own traceback, not ground truth, so it is safe to keep.
                "stderr": result.stderr[-2000:],
            })
            return False, 0.0, detail

        from civitas.runtime.sandbox import parse_result

        _, payload = parse_result(result.stdout)
        if not isinstance(payload, dict):
            detail.update({"passed": 0, "reason": "harness produced no result"})
            return False, 0.0, detail

        passed = int(payload.get("passed", 0))
        total = len(checks) or 1
        detail.update({
            "passed": passed,
            "errors": payload.get("errors", 0),
            # By index only: naming the failing input would hand out the check suite one row per
            # attempt (§47).
            "first_failing_check": payload.get("first_failure"),
        })
        return passed == len(checks) and bool(checks), passed / total, detail


@dataclass(frozen=True)
class Program:
    """A function with a real defect, a real specification and a catalogue of candidate edits.

    `patches` is what makes this domain runnable under §22's newcomer procedure. Each entry is a
    real textual edit to `broken`; exactly one of them repairs it. An agent tries them one at a
    time against `example`, which costs a tool call — structurally the same search the hidden-rule
    device presents, over a different kind of object.

    `example` is a *visible* worked case, deliberately disjoint from `checks`. Without one an
    agent has no local oracle and the task degenerates into guessing; with the hidden checks
    instead, §47 would be violated. Passing the example is necessary and not sufficient, which is
    the honest shape of a repair task.
    """

    entry: str
    signature: str
    spec: str
    broken: str
    checks: tuple[tuple[list[Any], Any], ...]
    defect: str
    #: (name, old, new) — applied to `broken` by substring replacement. Exactly one repairs it.
    patches: tuple[tuple[str, str, str], ...] = ()
    #: (args, result) shown in the task description, and the oracle `try_patch` checks against.
    example: tuple[list[Any], Any] = ((), None)

    def apply(self, patch_name: str) -> str | None:
        """Apply one named edit. The anchor must match exactly once.

        Not a style rule. `"    return out"` also occurs inside `"        return out"`, so an
        ambiguous anchor silently edits a different line and the patch quietly stops repairing
        anything — which is exactly what happened to `flush_final_run`, and was caught only
        because the catalogue is verified against its own worked examples. Failing loudly is the
        difference between a broken catalogue and a benchmark that reads 0.000 for a reason
        nobody can see.
        """
        for name, old, new in self.patches:
            if name == patch_name:
                occurrences = self.broken.count(old)
                if occurrences != 1:
                    raise ValueError(
                        f"patch {name!r} for {self.entry} anchors on a string occurring "
                        f"{occurrences} times; an anchor must be unambiguous"
                    )
                return self.broken.replace(old, new, 1)
        return None

    @property
    def patch_names(self) -> tuple[str, ...]:
        return tuple(name for name, _, _ in self.patches)


#: Four defects of different kinds: a wrong operator, a lost final iteration, a boundary
#: comparison, and a wrong return value. Different kinds matter because a domain whose defects are
#: all one shape measures one skill and reports it as "repair".
PROGRAMS: tuple[Program, ...] = (
    Program(
        entry="running_max",
        signature="running_max(xs: list[int]) -> list[int]",
        spec="Return a list whose i-th element is the largest value in xs[:i+1].",
        broken=(
            "def running_max(xs):\n"
            "    out = []\n"
            "    best = None\n"
            "    for x in xs:\n"
            "        best = x if best is None else min(best, x)\n"
            "        out.append(best)\n"
            "    return out\n"
        ),
        checks=(
            ([[1, 3, 2, 7, 5]], [1, 3, 3, 7, 7]),
            ([[]], []),
            ([[-4, -9, -1]], [-4, -4, -1]),
            ([[5, 5, 5]], [5, 5, 5]),
        ),
        defect="wrong operator",
        patches=(
            ("use_max", "min(best, x)", "max(best, x)"),
            ("sort_input", "for x in xs:", "for x in sorted(xs):"),
            ("reverse_output", "\n    return out\n", "\n    return out[::-1]\n"),
            ("swap_arguments", "min(best, x)", "min(x, best)"),
            ("absolute_values", "min(best, x)", "min(best, abs(x))"),
            ("drop_first", "\n    return out\n", "\n    return out[1:]\n"),
        ),
        example=([[2, 1, 4]], [2, 2, 4]),
    ),
    Program(
        entry="rle_encode",
        signature="rle_encode(s: str) -> list[list]",
        spec=(
            "Return the run-length encoding of s as a list of [character, run length] pairs, in "
            "order of first appearance of each run."
        ),
        broken=(
            "def rle_encode(s):\n"
            "    out = []\n"
            "    if not s:\n"
            "        return out\n"
            "    current = s[0]\n"
            "    count = 1\n"
            "    for ch in s[1:]:\n"
            "        if ch == current:\n"
            "            count += 1\n"
            "        else:\n"
            "            out.append([current, count])\n"
            "            current = ch\n"
            "            count = 1\n"
            "    return out\n"
        ),
        checks=(
            (["aaabbc"], [["a", 3], ["b", 2], ["c", 1]]),
            ([""], []),
            (["z"], [["z", 1]]),
            (["abab"], [["a", 1], ["b", 1], ["a", 1], ["b", 1]]),
        ),
        defect="lost final iteration",
        patches=(
            ("flush_final_run", "\n    return out\n",
             "\n    out.append([current, count])\n    return out\n"),
            ("count_from_zero", "\n    count = 1\n", "\n    count = 0\n"),
            ("emit_on_match", "        if ch == current:\n            count += 1\n",
                              "        if ch == current:\n"
                              "            out.append([current, count])\n"),
            ("iterate_whole_string", "for ch in s[1:]:", "for ch in s:"),
            ("swap_pair_order", "out.append([current, count])",
                               "out.append([count, current])"),
            ("reset_count_to_zero", "            current = ch\n            count = 1\n",
                                    "            current = ch\n            count = 0\n"),
        ),
        example=(["ppq"], [["p", 2], ["q", 1]]),
    ),
    Program(
        entry="merge_intervals",
        signature="merge_intervals(intervals: list[list[int]]) -> list[list[int]]",
        spec=(
            "Merge overlapping or touching closed intervals and return them sorted by start. "
            "[1, 3] and [3, 5] touch and merge into [1, 5]."
        ),
        broken=(
            "def merge_intervals(intervals):\n"
            "    if not intervals:\n"
            "        return []\n"
            "    ordered = sorted(intervals)\n"
            "    out = [list(ordered[0])]\n"
            "    for start, end in ordered[1:]:\n"
            "        if start < out[-1][1]:\n"
            "            out[-1][1] = max(out[-1][1], end)\n"
            "        else:\n"
            "            out.append([start, end])\n"
            "    return out\n"
        ),
        checks=(
            ([[[1, 3], [3, 5], [8, 9]]], [[1, 5], [8, 9]]),
            ([[]], []),
            ([[[4, 6], [1, 2]]], [[1, 2], [4, 6]]),
            ([[[1, 10], [2, 3]]], [[1, 10]]),
        ),
        defect="boundary comparison",
        patches=(
            ("include_touching", "if start < out[-1][1]:", "if start <= out[-1][1]:"),
            ("compare_starts", "if start < out[-1][1]:", "if start < out[-1][0]:"),
            ("drop_sort", "ordered = sorted(intervals)", "ordered = list(intervals)"),
            ("take_min_end", "out[-1][1] = max(out[-1][1], end)",
                             "out[-1][1] = min(out[-1][1], end)"),
            ("strictly_greater", "if start < out[-1][1]:", "if start > out[-1][1]:"),
            ("reverse_result", "\n    return out\n", "\n    return out[::-1]\n"),
        ),
        example=([[[2, 4], [4, 6]]], [[2, 6]]),
    ),
    Program(
        entry="first_index",
        signature="first_index(xs: list[int], target: int) -> int",
        spec=(
            "Return the index of the first occurrence of target in the sorted list xs, or -1 if "
            "it is not present."
        ),
        broken=(
            "def first_index(xs, target):\n"
            "    lo, hi, found = 0, len(xs) - 1, -1\n"
            "    while lo <= hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if xs[mid] == target:\n"
            "            found = mid\n"
            "            hi = mid - 1\n"
            "        elif xs[mid] < target:\n"
            "            lo = mid + 1\n"
            "        else:\n"
            "            hi = mid - 1\n"
            "    return found + 1\n"
        ),
        checks=(
            ([[1, 2, 2, 2, 5], 2], 1),
            ([[1, 2, 3], 4], -1),
            ([[], 1], -1),
            ([[7, 7, 7], 7], 0),
        ),
        defect="wrong return value",
        patches=(
            ("return_found", "return found + 1", "return found"),
            ("subtract_one", "return found + 1", "return found - 1"),
            ("return_high", "return found + 1", "return hi"),
            ("return_low", "return found + 1", "return lo"),
            ("absolute_result", "return found + 1", "return abs(found)"),
            ("search_right", "            found = mid\n            hi = mid - 1\n",
                             "            found = mid\n            lo = mid + 1\n"),
        ),
        example=([[5, 6, 7], 9], -1),
    ),
)


#: Opaque surface labels for candidate edits. Deliberately meaningless, for two reasons at once.
#:
#: **§47.** A descriptive name is a giveaway: an edit called `use_max` on a function whose defect
#: is a `min` announces itself, and the candidate set would no longer be indistinguishable.
#:
#: **The `π` discipline (ARCHITECTURE §2).** Both the mapping *and* the labels are redrawn per
#: era. Rotating which edit repairs the function while leaving the labels stable would leave the
#: labels as an index a collective could key on; rotating both makes a memorised label worthless
#: and forces knowledge rather than recall.
PATCH_LABEL_POOL = [
    "alcor", "brenth", "caldis", "dorune", "esker", "faltha", "girn", "halcyn",
    "ivren", "jasque", "kelvor", "lumis", "myrrh", "nocturne", "ossian", "peltar",
]


@dataclass(frozen=True)
class EraProgram:
    """A program as one era presents it: the same defect under a rotated label surface.

    Wraps rather than mutates `Program`, so the catalogue stays a single readable declaration and
    the era's presentation is derived from `(seed, era)` — reconstructable from the manifest
    alone (§46).
    """

    program: Program
    seed: int
    era: int
    #: surface label -> internal patch name
    labels: dict[str, str]

    @property
    def entry(self) -> str:
        return self.program.entry

    @property
    def broken(self) -> str:
        return self.program.broken

    @property
    def checks(self) -> tuple[tuple[list[Any], Any], ...]:
        return self.program.checks

    @property
    def example(self) -> tuple[list[Any], Any]:
        return self.program.example

    @property
    def patch_names(self) -> tuple[str, ...]:
        return tuple(self.labels)

    @property
    def environment_version(self) -> str:
        return f"repair-e{self.era}-s{self.seed}"

    def apply(self, label: str) -> str | None:
        internal = self.labels.get(label)
        return None if internal is None else self.program.apply(internal)

    def repairing_label(self) -> str | None:
        """Which surface label repairs it. Used by tests and by nothing on the agent path."""
        for label, internal in self.labels.items():
            source = self.program.apply(internal)
            namespace: dict[str, Any] = {}
            try:
                exec(source, namespace)  # noqa: S102 - domain-authored, not agent-authored
                args, want = self.program.example
                if namespace[self.program.entry](*args) == want:
                    return label
            except Exception:
                continue
        return None


def era_program(program: Program, *, seed: int, era: int) -> EraProgram:
    """Present one program under an era's label surface."""
    import random as _random

    rng = _random.Random(f"repair|{seed}|{era}|{program.entry}")
    labels = rng.sample(PATCH_LABEL_POOL, len(program.patches))
    internal = list(program.patch_names)
    rng.shuffle(internal)
    return EraProgram(program=program, seed=seed, era=era,
                      labels=dict(zip(labels, internal, strict=True)))


class CodeRepairDomain:
    """Find and fix a defect in a small function, judged by execution rather than by argument."""

    name = "code_repair"
    version = "1.0"

    def stages(self) -> tuple[Stage, ...]:
        return (
            Stage("investigate", "Characterise the failure in: {focus}",
                  "Find inputs on which {focus} behaves differently from its specification. "
                  "Record the inputs, not a diagnosis.", AgentRole.EXPLORER),
            Stage("hypothesise", "Locate the defect in: {focus}",
                  "Propose which line is wrong and why, and what would falsify that.",
                  AgentRole.RESEARCHER),
            Stage("verify", "Test the repair for: {focus}",
                  "Run the repaired function on the recorded inputs before submitting it.",
                  AgentRole.VERIFIER),
            Stage("synthesise", "Submit the repaired: {focus}",
                  "Submit the full repaired function source and record what the defect was.",
                  AgentRole.SYNTHESIZER),
        )

    def system_prompt(self) -> str:
        return (
            "You repair defective functions. You are given a specification and an implementation "
            "that does not meet it. Submit the complete repaired function source with "
            "submit_result. Your submission is executed against checks you cannot see, so test "
            "your repair before submitting rather than reasoning about it."
        )

    def tool_names(self) -> tuple[str, ...]:
        # Not a bespoke tool: §16's ecology already gives an agent the means to write code and
        # run it in the sandbox, and a domain that added a second way to execute code would have
        # a second sandbox policy to keep in step with the first.
        # `try_patch` is task-bound and built per episode; the other two are already in §16's
        # ecology registry, so a domain that added its own way to run code would leave two
        # sandbox policies to keep in step.
        return ("try_patch", "create_tool", "run_tool")

    def evaluators(self) -> tuple[Evaluator, ...]:
        return (SandboxedTestsEvaluator(),)

    def artifact_vocabulary(self) -> tuple[ArtifactType, ...]:
        return (
            ArtifactType.OBSERVATION, ArtifactType.HYPOTHESIS, ArtifactType.FAILURE,
            ArtifactType.CODE_PATCH, ArtifactType.CONCLUSION,
        )

    def generate(
        self, *, seed: int, count: int, era: int = 1, **kwargs: Any
    ) -> list[DomainTask]:
        tasks: list[DomainTask] = []
        for i in range(count):
            index = (seed + i) % len(PROGRAMS)
            presented = era_program(PROGRAMS[index], seed=seed, era=era)
            example_args, example_want = presented.example
            labels = ", ".join(presented.patch_names)
            tasks.append(DomainTask(
                title=f"Repair {presented.entry}",
                description=(
                    f"The function below does not meet its specification.\n\n"
                    f"Specification of {presented.program.signature}:\n"
                    f"  {presented.program.spec}\n\n"
                    f"Worked example: {presented.entry}"
                    f"({', '.join(json.dumps(a) for a in example_args)}) "
                    f"is {json.dumps(example_want)}\n\n"
                    f"Current implementation:\n\n{presented.broken}\n"
                    f"Candidate edits: {labels}\n\n"
                    f"Use try_patch(patch) to apply one candidate edit and run it on the worked "
                    f"example — each attempt costs a tool call and your budget is small. Exactly "
                    f"one candidate repairs the function.\n\n"
                    f"Search the collective knowledge base first. Earlier agents may have "
                    f"established or ruled out edits for this function. Record what you "
                    f"establish, then call submit_result with the repaired function source. "
                    f"Your submission is executed against checks that are not shown to you, so "
                    f"passing the worked example is necessary and not sufficient."
                ),
                evaluator_spec={
                    "kind": "sandboxed_tests",
                    "entry": presented.entry,
                    "checks": [[args, want] for args, want in presented.checks],
                },
                task_family="code_repair",
                index=index,
                environment_version=presented.environment_version,
                chance_level=1.0 / max(1, len(presented.patch_names)),
                difficulty=1.0 / max(1, len(presented.patch_names)),
                # The whole check suite is held out, so every result literal is a hidden term...
                hidden_terms=tuple(
                    json.dumps(want) for _, want in presented.checks
                    if len(json.dumps(want)) > 3 and json.dumps(want) != json.dumps(example_want)
                ),
                # ...and the repairing edit is hidden by being presented exactly like the five
                # that do not repair it.
                candidate_terms=presented.patch_names,
                meta={"defect_kind": presented.program.defect, "entry": presented.entry,
                      "program_index": index, "seed": seed, "era": era},
            ))
        return tasks

    # -- benchmark hooks -------------------------------------------------------------------

    def _presented(self, task: DomainTask) -> EraProgram:
        """Rebuild this task's era presentation from its recorded seed and era.

        Not carried on the domain object: a domain instance holding a program would be exactly the
        hidden cross-episode state §4 forbids, and would be shared between arms.
        """
        return era_program(
            PROGRAMS[task.meta["program_index"]], seed=task.meta["seed"], era=task.meta["era"]
        )

    def build_tools(self, task: DomainTask) -> list[Any]:
        from civitas.experiments.tasks.patch_tool import TryPatchTool

        return [TryPatchTool(self._presented(task))]

    def system_prompt_version(self) -> str:
        return "code-repair/1.0"

    def build_agent(
        self, task: DomainTask, *, record_findings: bool, probe_order_seed: int
    ) -> Any:
        from civitas.experiments.repair_policy_agent import RepairPolicyAgentProvider

        presented = self._presented(task)
        return RepairPolicyAgentProvider(
            program=presented,
            environment_version=presented.environment_version,
            record_findings=record_findings,
            probe_order_seed=probe_order_seed,
        )

    def naive_probe_ceiling(self, task: DomainTask, tool_calls: int) -> float:
        """The rate an agent that learns nothing reaches: it tries candidates in a fixed random
        order and wins if the repairing one falls within the attempts its budget allows.

        Two calls of the budget go to searching and submitting, exactly as on the device domain,
        so the comparison between the two ceilings is like for like.
        """
        n = len(self._presented(task).patch_names)
        attempts = max(0, min(n, tool_calls - 2))
        return attempts / n if n else 0.0


DOMAIN = register_domain(CodeRepairDomain())
assert isinstance(DOMAIN, Domain)
