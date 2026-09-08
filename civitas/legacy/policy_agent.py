"""The frozen policy agent (Part B §20).

Frozen-model mode asks that every individual-agent variable be held constant so that a change in
outcome is attributable to the collective environment. A deterministic *policy* is that
requirement taken to its limit: the agent is a pure function of what it is shown, so any
difference between arms is caused by what the arms showed it — not by sampling, not by
temperature, not by a model revision.

That is a stronger instrument than a frozen LLM, which can only support the claim statistically.
It is also honest about what it measures: this benchmark measures **the platform's contribution**,
not a model's intelligence. The same benchmark runs against any provider in §19, and the manifest
records which one produced a given result, so the two readings are never confused.

The policy is deliberately simple and deliberately *not* clairvoyant:

* search the collective knowledge base once;
* believe a confirmed finding only when it applies to the current environment version;
* use ruled-out operations to shrink the search space (the bijection is the structure that
  survives an era rotation);
* probe the cheapest untested operation;
* record what it establishes, so the next agent inherits it;
* submit when it knows, or when the budget is nearly gone.

It has no memory between episodes and no access to the mapping. Everything it knows at the start
of an episode came through retrieval.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from civitas.runtime.providers.base import (
    Completion,
    CompletionRequest,
    ModelInfo,
    Provider,
    Role,
    ToolCall,
    Usage,
)

#: Marker written into an artifact body so a later episode can parse a finding back out. Real
#: agents would read prose; a deterministic policy needs an unambiguous format, and using one
#: keeps the measured advantage attributable to *retrieval* rather than to parsing luck.
FINDING_RE = re.compile(
    r"DEVICE-FINDING\s+env=(?P<env>\S+)\s+class=(?P<cls>\S+)\s+op=(?P<op>\S+)\s+"
    r"verdict=(?P<verdict>accepted|rejected)"
)

#: Artifact ids as `search_knowledge` prints them. Search returns identities, not contents, so an
#: id is all the agent gets until it spends a call reading one.
ID_RE = re.compile(r"id=([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})")


def finding_line(env: str, input_class: str, operation: str, accepted: bool) -> str:
    return (
        f"DEVICE-FINDING env={env} class={input_class} op={operation} "
        f"verdict={'accepted' if accepted else 'rejected'}"
    )


@dataclass
class _Belief:
    """What the agent has worked out this episode. Dies with the episode (Part B §4)."""

    accepted: str | None = None
    ruled_out: set[str] = field(default_factory=set)
    #: Operations known to be taken by *another* class. Under a bijection they cannot be the
    #: answer for this one — the structural inference that makes prior work worth reading even
    #: when it was about a different class.
    taken_by_others: set[str] = field(default_factory=set)
    searched: bool = False
    probed: set[str] = field(default_factory=set)
    recorded: bool = False
    #: Tracked separately from `recorded`: a positive finding and a set of rejections are
    #: different contributions, and an agent that recorded one has not thereby recorded the other.
    recorded_negative: bool = False
    submitted: bool = False
    #: Artifact ids the last search returned, in rank order, and those already read. Reading is
    #: rationed by the tool budget, so the agent takes them in the order retrieval gave them —
    #: which is exactly what makes rank consequential and `collective_scrambled` a real control.
    ranked_ids: list[str] = field(default_factory=list)
    read_ids: set[str] = field(default_factory=set)


class PolicyAgentProvider(Provider):
    """A provider whose `complete` is a fixed policy over the message history.

    Stateless between episodes by construction: the belief state is rebuilt from the messages on
    every call, so two episodes with identical inputs behave identically and no belief can survive
    an episode boundary.
    """

    name = "policy"

    def __init__(self, *, input_class: str, operations: list[str], environment_version: str,
                 record_findings: bool = True, verify_prior: bool = False, max_reads: int = 2,
                 probe_order_seed: int = 0):
        self._input_class = input_class.lower()
        # The order this agent will try operations in.
        #
        # Varied per episode, and this is not a detail. With every agent probing in one fixed
        # order, the operations a failed predecessor ruled out are exactly the prefix its
        # successor would have tried first — so negative knowledge saves tool calls while
        # carrying no information about *where* the answer is, and the collective cannot raise a
        # success rate however much it accumulates. Measured directly: at five different budgets
        # the collective arm matched the baseline to three decimal places while halving probe
        # count. Real agents differ; modelling them as identical makes §12's negative knowledge
        # worthless by construction.
        #
        # The seed is a function of the task instance and the pass, never of the arm, so every
        # arm faces the same sequence of agents and the comparison stays matched (§20).
        ordered = [o.lower() for o in operations]
        random.Random(f"probe-order|{probe_order_seed}|{input_class}").shuffle(ordered)
        self._operations = ordered
        self._env = environment_version
        self._record = record_findings
        #: When true the agent spends one probe confirming a retrieved finding before submitting.
        #: Off by default: trusting the record is what the newcomer benchmark is measuring, and a
        #: mandatory verification probe would mask the advantage it exists to detect.
        self._verify = verify_prior
        #: How many artifacts the agent will read before falling back to probing. Rationed on
        #: purpose: an agent that reads every result would make retrieval rank irrelevant, and
        #: `collective_scrambled` would stop being a control.
        self._max_reads = max_reads

    def model_info(self, model: str) -> ModelInfo:
        return ModelInfo(
            name=model or "policy-v1",
            provider=self.name,
            context_window=32_000,
            max_output_tokens=1024,
            supports_tools=True,
            is_deterministic=True,
            version="policy-v1",
        )

    # ------------------------------------------------------------------
    def _read_history(self, request: CompletionRequest) -> _Belief:
        belief = _Belief()
        pending: list[ToolCall] = []

        for message in request.messages:
            if message.role is Role.ASSISTANT:
                pending = list(message.tool_calls)
                continue
            if message.role is not Role.TOOL:
                continue

            call = next((c for c in pending if c.id == message.tool_call_id), None)
            name = message.name or (call.name if call else "")

            if name == "search_knowledge":
                belief.searched = True
                belief.ranked_ids = ID_RE.findall(message.content or "")
            elif name == "read_artifact":
                if call is not None:
                    belief.read_ids.add(str(call.arguments.get("artifact_id", "")))
                self._absorb_findings(belief, message.content)
            elif name == "probe_device" and call is not None:
                operation = str(call.arguments.get("operation", "")).lower()
                belief.probed.add(operation)
                if "ACCEPTS" in message.content:
                    belief.accepted = operation
                elif "REJECTS" in message.content:
                    belief.ruled_out.add(operation)
                elif "already been tried and failed" in message.content:
                    # The runtime blocked the call because a previous episode documented this
                    # exact probe failing (Part A §A2.3). That carries the same information a
                    # rejection does, and an agent that ignored it would re-derive it.
                    belief.ruled_out.add(operation)
            elif name == "create_artifact":
                belief.recorded = True
            elif name == "record_failure":
                belief.recorded_negative = True
            elif name == "submit_result":
                belief.submitted = True
        return belief

    def _absorb_findings(self, belief: _Belief, text: str) -> None:
        """Parse findings out of retrieved text.

        The environment-version check is the `π` discipline (ARCHITECTURE §2): a finding from a
        previous era describes a device that no longer exists. Believing it would not merely waste
        a probe — it would produce a *confident wrong answer*, which is why stale-artifact usage
        is a metric (§48) rather than a curiosity.
        """
        for match in FINDING_RE.finditer(text or ""):
            if match.group("env") != self._env:
                continue  # stale: a different era's device
            operation = match.group("op").lower()
            same_class = match.group("cls").lower() == self._input_class
            accepted = match.group("verdict") == "accepted"
            if same_class and accepted:
                belief.accepted = operation
            elif same_class:
                belief.ruled_out.add(operation)
            elif accepted:
                # A different class accepts it, so under the bijection this class cannot.
                belief.taken_by_others.add(operation)

    # ------------------------------------------------------------------
    def complete(self, request: CompletionRequest) -> Completion:
        belief = self._read_history(request)
        available = {t.name for t in request.tools}
        prompt_tokens = self.count_request_tokens(request)

        call, text = self._decide(belief, available)
        return Completion(
            text=text,
            tool_calls=(call,) if call else (),
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=self.count_tokens(text)),
            stop_reason="tool_use" if call else "end_turn",
            model=request.model,
            model_version="policy-v1",
        )

    def _decide(self, belief: _Belief, available: set[str]) -> tuple[ToolCall | None, str]:
        step = (
            len(belief.probed) + len(belief.read_ids) + int(belief.searched) + int(belief.recorded)
        )

        if not belief.searched and "search_knowledge" in available:
            return (
                ToolCall(f"c{step}", "search_knowledge", {
                    "query": f"device input class {self._input_class} accepted operation",
                }),
                "Checking what the collective already knows about this class.",
            )

        unread = [i for i in belief.ranked_ids if i not in belief.read_ids]
        if (
            belief.accepted is None
            and unread
            and len(belief.read_ids) < self._max_reads
            and "read_artifact" in available
        ):
            return (
                ToolCall(f"c{step}", "read_artifact", {"artifact_id": unread[0]}),
                f"Reading the highest-ranked result not yet read ({len(belief.read_ids) + 1}"
                f" of at most {self._max_reads}).",
            )

        if belief.accepted is not None:
            if self._record and not belief.recorded and "create_artifact" in available:
                return (
                    ToolCall(f"c{step}", "create_artifact", {
                        "type": "evidence",
                        "title": f"Device accepts {belief.accepted} for {self._input_class}",
                        "body": finding_line(self._env, self._input_class, belief.accepted, True),
                        "confidence": 0.9,
                    }),
                    "Recording the confirmed mapping for later agents.",
                )
            return (
                ToolCall(f"c{step}", "submit_result", {
                    "answer": belief.accepted,
                    "reasoning": "established by probe or by a confirmed prior finding",
                    "confidence": 0.9,
                }),
                "Submitting.",
            )

        candidates = [
            op for op in self._operations
            if op not in belief.probed
            and op not in belief.ruled_out
            and op not in belief.taken_by_others
        ]
        if candidates and "probe_device" in available:
            return (
                ToolCall(f"c{step}", "probe_device", {
                    "input_class": self._input_class, "operation": candidates[0],
                }),
                f"Probing {candidates[0]}.",
            )

        # Out of candidates, or out of budget to probe them. Before giving up, write down what was
        # ruled out.
        #
        # This is Part B §12 in the one place it actually bites. An agent that probes four
        # operations and finds none of them learns four facts, and throwing them away is what
        # keeps a collective from ever maturing: every later agent starts the same blind search.
        # Recording them is also what makes `collective_no_negative` a control with something to
        # remove rather than a relabelling of `collective`.
        if (
            self._record
            and not belief.recorded_negative
            and belief.ruled_out
            and "record_failure" in available
        ):
            ruled = sorted(belief.ruled_out)
            lines = [
                finding_line(self._env, self._input_class, op, False) for op in ruled
            ]
            return (
                ToolCall(f"c{step}", "record_failure", {
                    "title": f"Ruled out {len(ruled)} operation(s) for {self._input_class}",
                    "approach": f"probed operations for input class {self._input_class}",
                    "what_happened": "\n".join(lines),
                    "ruled_out": ruled,
                    "hypothesis": f"the device accepts one of {ruled} for {self._input_class}",
                    "reproducible": True,
                }),
                f"Recording {len(ruled)} ruled-out operation(s) so the next agent need not "
                f"repeat them.",
            )

        # Nothing left to try. Submit the best remaining guess rather than stalling: an episode
        # that gives up silently is indistinguishable from one that crashed, and §8 wants the
        # termination reason to be informative.
        fallback = candidates[0] if candidates else (
            sorted(set(self._operations) - belief.ruled_out - belief.taken_by_others)
            or self._operations
        )[0]
        return (
            ToolCall(f"c{step}", "submit_result", {
                "answer": fallback,
                "reasoning": "budget nearly exhausted; unverified best remaining candidate",
                "confidence": 0.1,
            }),
            "Submitting an unverified candidate.",
        )
