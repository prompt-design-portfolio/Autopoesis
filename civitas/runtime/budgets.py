"""Budget enforcement (Part B §8, §58).

"No agent may run indefinitely" (§8) is a property of the runtime, not a hope about prompts. Every
bound is checked *before* the action that would exceed it, so an episode terminates with
`budget_exhausted` rather than overrunning and being noticed afterwards.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from civitas.domain.enums import TerminationReason


class BudgetExceeded(Exception):
    """Raised when an action would exceed a bound. Carries which one, so the episode's
    termination reason is a fact rather than an inference."""

    def __init__(self, kind: str, limit: float, would_be: float):
        super().__init__(f"{kind} budget exceeded: {would_be:.4g} > {limit:.4g}")
        self.kind = kind
        self.limit = limit
        self.would_be = would_be

    @property
    def termination_reason(self) -> TerminationReason:
        return TerminationReason.BUDGET_EXHAUSTED


@dataclass
class Budgets:
    tokens: int = 100_000
    context: int = 32_000
    tool_calls: int = 50
    cost_usd: float = 1.0
    wall_clock_s: float = 600.0
    #: Consecutive turns producing no artifact, no tool call and no progress signal. Distinct from
    #: a token budget: an agent looping on model calls that achieve nothing should stop for
    #: `no_progress`, which is diagnostic, rather than burn to `budget_exhausted`, which is not.
    max_no_progress_turns: int = 3
    max_turns: int = 40


@dataclass
class BudgetState:
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls_used: int = 0
    cost_usd: float = 0.0
    turns: int = 0
    no_progress_turns: int = 0
    started_at: float = field(default_factory=time.monotonic)

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at


class BudgetTracker:
    """Holds the bounds and the consumption, and refuses the action that would breach them."""

    def __init__(self, budgets: Budgets, *, clock=time.monotonic):
        self.budgets = budgets
        self._clock = clock
        self.state = BudgetState(started_at=clock())

    # --- pre-action checks ----------------------------------------------
    def check_wall_clock(self) -> None:
        elapsed = self._clock() - self.state.started_at
        if elapsed >= self.budgets.wall_clock_s:
            raise BudgetExceeded("wall_clock_s", self.budgets.wall_clock_s, elapsed)

    def check_turn(self) -> None:
        if self.state.turns >= self.budgets.max_turns:
            raise BudgetExceeded("max_turns", self.budgets.max_turns, self.state.turns + 1)

    def check_model_call(self, *, estimated_prompt_tokens: int, max_output_tokens: int) -> None:
        """Refuse a call that could not fit inside the remaining budgets.

        The check uses the *maximum* the call could consume, not an expectation. A check against
        the average would let the last call overshoot, and the whole point of a bound is that it
        is not exceeded.
        """
        self.check_wall_clock()
        self.check_turn()
        if estimated_prompt_tokens > self.budgets.context:
            raise BudgetExceeded("context", self.budgets.context, estimated_prompt_tokens)
        would_be = self.state.tokens_used + estimated_prompt_tokens + max_output_tokens
        if would_be > self.budgets.tokens:
            raise BudgetExceeded("tokens", self.budgets.tokens, would_be)

    def check_tool_call(self) -> None:
        self.check_wall_clock()
        if self.state.tool_calls_used >= self.budgets.tool_calls:
            raise BudgetExceeded(
                "tool_calls", self.budgets.tool_calls, self.state.tool_calls_used + 1
            )

    def check_cost(self, additional_usd: float = 0.0) -> None:
        would_be = self.state.cost_usd + additional_usd
        if would_be > self.budgets.cost_usd:
            raise BudgetExceeded("cost_usd", self.budgets.cost_usd, would_be)

    # --- recording ------------------------------------------------------
    def record_model_call(
        self, *, prompt_tokens: int, completion_tokens: int, cost_usd: float
    ) -> None:
        self.state.prompt_tokens += prompt_tokens
        self.state.completion_tokens += completion_tokens
        self.state.tokens_used += prompt_tokens + completion_tokens
        self.state.cost_usd += cost_usd
        self.state.turns += 1
        # Cost is checked *after* recording as well as before: an actual response can exceed the
        # estimate, and an episode that has already overspent must stop on the next check rather
        # than continue because the pre-check passed.
        self.check_cost()

    def record_tool_call(self) -> None:
        self.state.tool_calls_used += 1

    def record_progress(self, made_progress: bool) -> None:
        self.state.no_progress_turns = 0 if made_progress else self.state.no_progress_turns + 1

    @property
    def stalled(self) -> bool:
        return self.state.no_progress_turns >= self.budgets.max_no_progress_turns

    def remaining(self) -> dict[str, float]:
        b, s = self.budgets, self.state
        return {
            "tokens": max(0, b.tokens - s.tokens_used),
            "tool_calls": max(0, b.tool_calls - s.tool_calls_used),
            "cost_usd": max(0.0, b.cost_usd - s.cost_usd),
            "wall_clock_s": max(0.0, b.wall_clock_s - (self._clock() - s.started_at)),
            "turns": max(0, b.max_turns - s.turns),
        }
