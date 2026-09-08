"""The bounded agent episode (Part B §8) — and the enforcement of Part B §4.

An episode is a fixed model, a fixed prompt, fixed budgets and no hidden state carried in or out.
`EpisodeContext` holds everything episode-local, it is constructed inside `run()`, and it is
dropped when `run()` returns. There is no field on it that is persisted as text and no path by
which a later episode could read it.

What survives is what the episode wrote through a tool. That is the entire distinction between
temporary individual state and persistent collective state, and it is structural here rather than
a matter of care: `EpisodeRunner` has no parameter through which a prior episode's context could
be supplied, and `tests/test_state_isolation.py` drives two consecutive episodes to prove a canary
in the first reaches no input of the second under any arm.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from civitas.domain.enums import (
    EpisodeStatus,
    EventType,
    ExperimentArm,
    TerminationReason,
)
from civitas.knowledge.arms import arm_policy
from civitas.observability import REGISTRY
from civitas.persistence.events import emit
from civitas.persistence.models import (
    Artifact,
    ArtifactUsage,
    Episode,
    ModelCall,
    Task,
)
from civitas.persistence.types import utcnow
from civitas.runtime.budgets import BudgetExceeded, Budgets, BudgetTracker
from civitas.runtime.providers.base import (
    CompletionRequest,
    Message,
    Provider,
    ProviderError,
    Role,
)
from civitas.runtime.tools.base import ToolContext, ToolRegistry, ToolResult, validate_arguments

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpisodeSpec:
    """Everything an episode is configured with — and nothing carried from a previous one.

    Frozen, because a spec that could be mutated mid-episode would make the recorded
    configuration a description of the start rather than of the run (Part B §8, §46).
    """

    workspace_id: uuid.UUID
    agent_profile_id: uuid.UUID
    provider_name: str
    model_name: str
    system_prompt: str
    system_prompt_version: str = "v1"
    model_version: str = ""
    model_parameters: dict[str, Any] = field(default_factory=dict)
    budgets: Budgets = field(default_factory=Budgets)
    experiment_arm: ExperimentArm = ExperimentArm.SOLO
    retrieval_policy_version: str = ""
    #: Parameters of the retrieval policy — which hybrid components are active, and their
    #: weights. A versioned `Policy` body under the A2.2 gate; here so a benchmark can run a
    #: matched A/B over it.
    retrieval_options: dict[str, Any] = field(default_factory=dict)
    tool_policy: dict[str, Any] = field(default_factory=dict)
    project_id: uuid.UUID | None = None
    task_id: uuid.UUID | None = None
    assignment_id: uuid.UUID | None = None
    agent_instance_id: uuid.UUID | None = None
    experiment_run_id: uuid.UUID | None = None
    environment_version: str = "dev"
    code_version: str = ""
    config_hash: str = ""
    seed: int | None = None
    is_benchmark_probe: bool = False
    #: `collective_frozen` needs the snapshot cut. Validated at construction of the arm policy, so
    #: a frozen arm without a cut fails before the episode starts rather than degrading silently
    #: into `collective`.
    frozen_as_of: Any = None


@dataclass
class EpisodeOutcome:
    episode_id: uuid.UUID
    termination_reason: TerminationReason
    detail: str = ""
    artifacts_created: int = 0
    artifacts_read: int = 0
    tokens_used: int = 0
    tool_calls: int = 0
    cost_usd: float = 0.0
    duplicate_failures: int = 0
    submitted_answer: str | None = None
    submitted_artifact_id: uuid.UUID | None = None
    turns: int = 0

    @property
    def solved(self) -> bool:
        """Whether the episode *claims* success. Not whether it succeeded — §47 reserves that
        for an evaluator, and this flag never sets a validation state."""
        return self.termination_reason in (
            TerminationReason.SOLVED, TerminationReason.EVALUATOR_SUCCESS
        )


class EpisodeRunner:
    """Runs one bounded episode to termination.

    Deliberately has no constructor parameter through which prior-episode state could arrive: it
    takes a session, a provider, a tool registry and a spec. The registry holds *tool objects*,
    which are stateless, and the provider is shared only because it holds an HTTP client.
    """

    def __init__(
        self,
        session: Session,
        *,
        provider: Provider,
        tools: ToolRegistry,
        max_tool_result_chars: int = 8000,
    ):
        self._session = session
        self._provider = provider
        self._tools = tools
        self._max_tool_result_chars = max_tool_result_chars

    # ------------------------------------------------------------------
    def run(self, spec: EpisodeSpec, *, initial_instruction: str | None = None) -> EpisodeOutcome:
        # Resolved once, before anything is written: a mis-specified arm must not produce a
        # half-run episode whose row claims a condition it never ran under. Passing it down also
        # means every arm decision in this episode comes from one object, so a policy cannot be
        # re-derived differently halfway through.
        policy = arm_policy(spec.experiment_arm, as_of=spec.frozen_as_of)

        # §58, checked before the episode row exists. An organization over its limit must not
        # produce an episode at all: a row created and then refused would be counted by the very
        # usage query the quota is measured with, so exceeding a quota would raise the usage that
        # proves it was exceeded.
        from civitas.quotas import check_workspace

        check_workspace(self._session, spec.workspace_id)

        episode = self._create_episode(spec)
        tracker = BudgetTracker(spec.budgets)
        ctx = ToolContext(
            session=self._session,
            workspace_id=spec.workspace_id,
            episode_id=episode.id,
            budgets=tracker,
            experiment_arm=spec.experiment_arm.value,
            environment_version=spec.environment_version,
            config_hash=spec.config_hash,
            project_id=spec.project_id,
            task_id=spec.task_id,
            retrieval_options=dict(spec.retrieval_options),
        )
        tools = self._tools.filtered(spec.tool_policy.get("allowed"))

        # The conversation is a local variable. It is never stored, never returned, and is
        # unreachable once `run` exits — Part B §4.
        messages: list[Message] = [
            Message(Role.SYSTEM, spec.system_prompt),
            Message(Role.USER, initial_instruction or self._task_brief(spec)),
        ]

        reason, detail = TerminationReason.FAILED, ""
        try:
            reason, detail = self._loop(spec, policy, episode, ctx, tracker, tools, messages)
        except BudgetExceeded as exc:
            reason, detail = exc.termination_reason, str(exc)
        except ProviderError as exc:
            # A circuit-breaker skip arrives here too, and that is the point: it is recorded as
            # `provider_failure`, which `evaluation.is_readable` excludes from success rates. An
            # outage must not be counted as an agent's failure (§47).
            reason, detail = TerminationReason.PROVIDER_FAILURE, str(exc)[:2000]
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("episode %s failed", episode.id)
            reason, detail = TerminationReason.FAILED, f"{type(exc).__name__}: {exc}"[:2000]

        return self._finish(spec, episode, ctx, tracker, reason, detail)

    # ------------------------------------------------------------------
    def _loop(self, spec, policy, episode, ctx, tracker, tools, messages):
        while True:
            try:
                tracker.check_wall_clock()
                tracker.check_turn()
            except BudgetExceeded as exc:
                return exc.termination_reason, str(exc)

            info = self._provider.model_info(spec.model_name)
            # Includes tool-call arguments and tool declarations. A content-only count
            # under-estimates most for exactly the agents a budget most needs to bound.
            prompt_tokens = self._provider.count_message_tokens(messages)
            if info.supports_tools:
                for spec_ in tools.specs():
                    prompt_tokens += self._provider.count_tokens(spec_.description)
            max_out = min(
                spec.model_parameters.get("max_tokens", 2048), info.max_output_tokens
            )
            tracker.check_model_call(
                estimated_prompt_tokens=prompt_tokens, max_output_tokens=max_out
            )

            request = CompletionRequest(
                messages=tuple(messages),
                model=spec.model_name,
                max_tokens=max_out,
                temperature=spec.model_parameters.get("temperature", 0.0),
                top_p=spec.model_parameters.get("top_p", 1.0),
                tools=tools.specs() if info.supports_tools else (),
                seed=spec.seed,
            )
            completion = self._call_model(spec, episode, request, tracker)

            ctx.made_progress = False
            messages.append(
                Message(Role.ASSISTANT, completion.text, tool_calls=completion.tool_calls)
            )

            if not completion.tool_calls:
                # No tool call means nothing reached the collective this turn. Repeated, that is
                # `no_progress` — a diagnostic reason, unlike `budget_exhausted`.
                tracker.record_progress(False)
                if tracker.stalled:
                    return TerminationReason.NO_PROGRESS, (
                        f"{tracker.state.no_progress_turns} consecutive turns with no tool call"
                    )
                messages.append(Message(
                    Role.USER,
                    "You have not used a tool. Use one, or call submit_result to finish.",
                ))
                continue

            terminated = None
            for call in completion.tool_calls:
                result = self._invoke(spec, policy, ctx, tracker, tools, call)
                messages.append(
                    Message(
                        Role.TOOL,
                        result.to_message()[: self._max_tool_result_chars],
                        tool_call_id=call.id,
                        name=call.name,
                    )
                )
                if call.name == "submit_result" and result.ok:
                    terminated = TerminationReason.SOLVED

            tracker.record_progress(ctx.made_progress)
            if terminated is not None:
                return terminated, "submitted a result"
            if tracker.stalled:
                return TerminationReason.NO_PROGRESS, (
                    f"{tracker.state.no_progress_turns} consecutive turns without progress"
                )

    # ------------------------------------------------------------------
    def _invoke(self, spec, policy, ctx, tracker, tools, call) -> ToolResult:
        tool = tools.get(call.name)
        if tool is None:
            return ToolResult(
                ok=False, error=f"no tool named {call.name!r}; available: {tools.names()}"
            )
        # A tool-call budget breach ends the episode, so it propagates rather than becoming a
        # ToolResult the agent could ignore and retry.
        tracker.check_tool_call()

        error = validate_arguments(tool.parameters, call.arguments)
        if error:
            # A malformed call still costs a budget slot: otherwise an agent emitting invalid
            # arguments loops without bound, which is precisely what §8's budgets exist to stop.
            tracker.record_tool_call()
            return ToolResult(ok=False, error=error)

        blocked = self._duplicate_check(spec, policy, ctx, tool, call)
        if blocked is not None:
            tracker.record_tool_call()
            return blocked

        tracker.record_tool_call()
        try:
            result = tool.run(ctx, **call.arguments)
        except Exception as exc:  # a tool must not be able to kill the episode
            log.warning("tool %s failed in episode %s: %s", call.name, ctx.episode_id, exc)
            return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")

        emit(
            self._session, workspace_id=spec.workspace_id, type=EventType.TOOL_CALLED,
            episode_id=ctx.episode_id, actor_kind="agent",
            payload={"tool": call.name, "ok": result.ok, "blocked": bool(result.blocked_reason)},
            config_hash=spec.config_hash,
        )
        return result

    def _duplicate_check(self, spec, policy, ctx, tool, call) -> ToolResult | None:
        """Detect a repeat *before* the action runs (Part A §A2.3).

        Detection happens in every arm — `duplicate_failure_rate` is a benchmark metric and has to
        be comparable across arms (§48). Only *surfacing* is arm-gated, because surfacing is the
        intervention.
        """
        from civitas.knowledge import duplicate

        if tool.read_only:
            return None

        hypothesis = None
        if call.name in ("create_artifact", "record_failure"):
            hypothesis = f"{call.arguments.get('title', '')} {call.arguments.get('approach', '')}"

        hit = duplicate.check(
            self._session,
            workspace_id=spec.workspace_id,
            environment_version=spec.environment_version,
            tool_name=call.name if not tool.read_only else None,
            tool_args=call.arguments,
            hypothesis=hypothesis.strip() if hypothesis and hypothesis.strip() else None,
            exclude_episode_id=ctx.episode_id,
        )
        if hit is None:
            return None

        surfaced = policy.duplicate_warnings
        duplicate.report(
            self._session, hit, workspace_id=spec.workspace_id, episode_id=ctx.episode_id,
            surfaced=surfaced, config_hash=spec.config_hash,
        )
        ctx.scratch["duplicate_failures"] = ctx.scratch.get("duplicate_failures", 0) + 1

        if not surfaced:
            return None

        # Being told "this was already tried and it failed" is information the episode did not
        # have a moment ago, so it is progress.
        #
        # Without this the mechanism designed to stop wasted work instead kills the episode: a run
        # of blocked duplicates counts as consecutive no-progress turns and the agent terminates
        # for `no_progress` with budget unspent. Measured before the fix — 4 of 5 collective-arm
        # failures terminated that way at 4.6 tool calls of a 7 budget — and it cost a third of
        # the newcomer advantage.
        ctx.made_progress = True

        if hit.artifact is not None:
            ctx.note_read(hit.artifact.id)
            # One usage row per (episode, artifact) — the table is uniquely keyed on the pair.
            # An episode can be warned about the same prior failure several times, and inserting a
            # row per warning violates that key; the repeat count belongs on `read_count`.
            existing = (
                self._session.query(ArtifactUsage)
                .filter(
                    ArtifactUsage.episode_id == ctx.episode_id,
                    ArtifactUsage.artifact_id == hit.artifact.id,
                )
                .one_or_none()
            )
            if existing is None:
                self._session.add(
                    ArtifactUsage(
                        episode_id=ctx.episode_id, artifact_id=hit.artifact.id,
                        surfaced_as_duplicate_warning=True,
                    )
                )
            else:
                existing.read_count += 1
            self._session.flush()
        return ToolResult(
            ok=False, blocked_reason="this has already been tried and failed",
            content=hit.warning(),
        )

    # ------------------------------------------------------------------
    def _call_model(self, spec, episode, request, tracker: BudgetTracker):
        """One provider call, accounted for on both the `ModelCall` row and the budget tracker.

        Prompts are not persisted. `request_hash` identifies the call so it is repeatable, while
        the content stays episode-local — storing it would create exactly the hidden channel
        Part B §4 forbids.
        """
        record = ModelCall(
            episode_id=episode.id,
            provider=spec.provider_name,
            model_name=spec.model_name,
            model_version=spec.model_version,
            request_hash=request.hash(),
        )
        try:
            completion = self._provider.complete(request)
        except ProviderError as exc:
            record.error = str(exc)[:4000]
            self._session.add(record)
            self._session.flush()
            raise

        cost = completion.cost_usd or self._provider.estimate_cost(
            spec.model_name, completion.usage
        )
        record.prompt_tokens = completion.usage.prompt_tokens
        record.completion_tokens = completion.usage.completion_tokens
        record.total_tokens = completion.usage.total_tokens
        record.latency_ms = completion.latency_ms
        record.cost_usd = cost
        record.stop_reason = completion.stop_reason
        record.tool_calls = len(completion.tool_calls)
        record.model_version = completion.model_version or spec.model_version
        self._session.add(record)
        self._session.flush()

        emit(
            self._session, workspace_id=spec.workspace_id, type=EventType.MODEL_CALLED,
            episode_id=episode.id,
            payload={"provider": spec.provider_name, "model": spec.model_name,
                     "tokens": completion.usage.total_tokens, "cost_usd": round(cost, 6)},
            config_hash=spec.config_hash,
        )
        # Recorded after the call, against the *actual* usage: the pre-check used an estimate, and
        # a budget held against an estimate is not held.
        tracker.record_model_call(
            prompt_tokens=completion.usage.prompt_tokens,
            completion_tokens=completion.usage.completion_tokens,
            cost_usd=cost,
        )
        return completion

    # ------------------------------------------------------------------
    def _create_episode(self, spec: EpisodeSpec) -> Episode:
        episode = Episode(
            workspace_id=spec.workspace_id,
            project_id=spec.project_id,
            task_id=spec.task_id,
            assignment_id=spec.assignment_id,
            agent_profile_id=spec.agent_profile_id,
            agent_instance_id=spec.agent_instance_id,
            model_provider=spec.provider_name,
            model_name=spec.model_name,
            model_version=spec.model_version,
            model_parameters=dict(spec.model_parameters),
            system_prompt_version=spec.system_prompt_version,
            tool_policy=dict(spec.tool_policy),
            retrieval_policy_version=spec.retrieval_policy_version,
            experiment_arm=spec.experiment_arm,
            experiment_run_id=spec.experiment_run_id,
            token_budget=spec.budgets.tokens,
            context_budget=spec.budgets.context,
            tool_call_budget=spec.budgets.tool_calls,
            cost_budget_usd=spec.budgets.cost_usd,
            wall_clock_budget_s=spec.budgets.wall_clock_s,
            status=EpisodeStatus.RUNNING,
            started_at=utcnow(),
            environment_version=spec.environment_version,
            code_version=spec.code_version,
            config_hash=spec.config_hash,
            seed=spec.seed,
            is_benchmark_probe=spec.is_benchmark_probe,
        )
        self._session.add(episode)
        self._session.flush()
        emit(
            self._session, workspace_id=spec.workspace_id, type=EventType.EPISODE_STARTED,
            episode_id=episode.id, task_id=spec.task_id, actor_kind="agent",
            payload={"arm": spec.experiment_arm.value, "profile": str(spec.agent_profile_id),
                     "model": spec.model_name},
            config_hash=spec.config_hash,
        )
        return episode

    def _task_brief(self, spec: EpisodeSpec) -> str:
        """The task as the agent sees it.

        The evaluator specification is deliberately **not** included: an agent that can read its
        own success criteria can write to them, and §47 forbids self-certification.
        """
        if spec.task_id is None:
            return "No task assigned. Use submit_result to report that."
        task = self._session.get(Task, spec.task_id)
        if task is None:
            return "The assigned task no longer exists."
        return (
            f"Task: {task.title}\n\n{task.description}\n\n"
            "Search the collective knowledge base before you begin: earlier agents may have "
            "already established, or ruled out, part of this. Record what you find, and call "
            "submit_result when you have an answer."
        )

    def _finish(self, spec, episode, ctx, tracker, reason, detail) -> EpisodeOutcome:
        created = (
            self._session.query(Artifact)
            .filter(Artifact.creator_episode_id == episode.id)
            .count()
        )
        # `ArtifactUsage` rows are the credit-assignment links (Part A §A2.1): they say what the
        # episode *read*, which is not what retrieval returned.
        for artifact_id, count in ctx.read_artifacts.items():
            existing = (
                self._session.query(ArtifactUsage)
                .filter(
                    ArtifactUsage.episode_id == episode.id,
                    ArtifactUsage.artifact_id == artifact_id,
                )
                .one_or_none()
            )
            if existing is None:
                self._session.add(
                    ArtifactUsage(
                        episode_id=episode.id, artifact_id=artifact_id, read_count=count
                    )
                )
            else:
                existing.read_count = max(existing.read_count, count)

        episode.status = EpisodeStatus.TERMINATED
        episode.ended_at = utcnow()
        episode.termination_reason = reason
        episode.termination_detail = detail[:4000]
        episode.tokens_used = tracker.state.tokens_used
        episode.prompt_tokens = tracker.state.prompt_tokens
        episode.completion_tokens = tracker.state.completion_tokens
        episode.tool_calls_used = tracker.state.tool_calls_used
        episode.cost_usd = tracker.state.cost_usd
        episode.model_calls = tracker.state.turns
        episode.artifacts_created = created
        episode.artifacts_read = len(ctx.read_artifacts)
        episode.duplicate_failures = ctx.scratch.get("duplicate_failures", 0)
        self._session.flush()

        # §53. Labelled by arm and reason, never by workspace or task: those are per-tenant and
        # would put a tenant list on an endpoint people expose to a scraper.
        REGISTRY.inc(
            "civitas_episodes_total", arm=spec.experiment_arm.value, reason=reason.value,
        )
        REGISTRY.inc(
            "civitas_episode_tokens_total", float(episode.tokens_used),
            arm=spec.experiment_arm.value,
        )

        emit(
            self._session, workspace_id=spec.workspace_id, type=EventType.EPISODE_TERMINATED,
            episode_id=episode.id, task_id=spec.task_id, actor_kind="agent",
            payload={
                "reason": reason.value, "tokens": episode.tokens_used,
                "tool_calls": episode.tool_calls_used, "artifacts_created": created,
                "artifacts_read": episode.artifacts_read,
                "duplicate_failures": episode.duplicate_failures,
                "cost_usd": round(episode.cost_usd, 6),
            },
            config_hash=spec.config_hash,
        )
        return EpisodeOutcome(
            episode_id=episode.id,
            termination_reason=reason,
            detail=detail,
            artifacts_created=created,
            artifacts_read=episode.artifacts_read,
            tokens_used=episode.tokens_used,
            tool_calls=episode.tool_calls_used,
            cost_usd=episode.cost_usd,
            duplicate_failures=episode.duplicate_failures,
            submitted_answer=ctx.scratch.get("submitted_answer"),
            submitted_artifact_id=ctx.scratch.get("submitted_artifact_id"),
            turns=tracker.state.turns,
        )
