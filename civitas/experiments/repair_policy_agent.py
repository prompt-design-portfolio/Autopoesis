"""The frozen policy agent for the code-repair domain (Part B §20, §22).

The same instrument as `policy_agent.PolicyAgentProvider`, pointed at a different kind of object.
That is the claim the domain layer exists to test: if a newcomer advantage appears here too, under
the same arms and the same gates, then M4's +0.300 is a property of the platform rather than of
one device.

The policy is the same five moves — search, read what ranked highest, act on a believed finding,
otherwise test the next untried candidate, record what was established — and the same refusals:
no memory between episodes, no access to the held-out checks, and no belief in a finding recorded
under a different environment version.
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

#: Same role as the device domain's marker: an unambiguous format so the measured advantage is
#: attributable to retrieval rather than to how well a policy parses prose.
FINDING_RE = re.compile(
    r"REPAIR-FINDING\s+env=(?P<env>\S+)\s+entry=(?P<entry>\S+)\s+patch=(?P<patch>\S+)\s+"
    r"verdict=(?P<verdict>accepted|rejected)"
)

ID_RE = re.compile(r"id=([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})")


def finding_line(env: str, entry: str, patch: str, accepted: bool) -> str:
    return (
        f"REPAIR-FINDING env={env} entry={entry} patch={patch} "
        f"verdict={'accepted' if accepted else 'rejected'}"
    )


@dataclass
class _Belief:
    """What this episode has worked out. Dies with the episode (Part B §4)."""

    repaired_by: str | None = None
    ruled_out: set[str] = field(default_factory=set)
    searched: bool = False
    tried: set[str] = field(default_factory=set)
    recorded: bool = False
    recorded_negative: bool = False
    submitted: bool = False
    ranked_ids: list[str] = field(default_factory=list)
    read_ids: set[str] = field(default_factory=set)


class RepairPolicyAgentProvider(Provider):
    """A deterministic repair policy: a pure function of the message history."""

    name = "policy"

    def __init__(self, *, program, environment_version: str, record_findings: bool = True,
                 max_reads: int = 2, probe_order_seed: int = 0):
        self._program = program
        self._entry = program.entry
        # Varied per agent and seeded from (seed, task, pass) — never from the arm. With one fixed
        # order every agent would try exactly the prefix its predecessor ruled out, so negative
        # knowledge would save calls while carrying no information about where the answer is. That
        # defect was measured on the device domain in M4 and is not going to be reintroduced here.
        ordered = list(program.patch_names)
        random.Random(f"patch-order|{probe_order_seed}|{program.entry}").shuffle(ordered)
        self._patches = ordered
        self._env = environment_version
        self._record = record_findings
        self._max_reads = max_reads

    def model_info(self, model: str) -> ModelInfo:
        return ModelInfo(
            name=model or "policy-v1", provider=self.name, context_window=32_000,
            max_output_tokens=1024, supports_tools=True, is_deterministic=True,
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
            elif name == "try_patch" and call is not None:
                patch = str(call.arguments.get("patch", ""))
                belief.tried.add(patch)
                if "verdict=accepted" in (message.content or ""):
                    belief.repaired_by = patch
                elif "verdict=rejected" in (message.content or ""):
                    belief.ruled_out.add(patch)
                elif "already been tried and failed" in (message.content or ""):
                    # §A2.3 blocked it: a previous episode documented this exact edit failing.
                    # That is the same information a rejection carries.
                    belief.ruled_out.add(patch)
            elif name == "create_artifact":
                belief.recorded = True
            elif name == "record_failure":
                belief.recorded_negative = True
            elif name == "submit_result":
                belief.submitted = True
        return belief

    def _absorb_findings(self, belief: _Belief, text: str) -> None:
        """The `π` discipline (ARCHITECTURE §2): a finding recorded under a different environment
        version describes a different catalogue of edits, and believing it produces a *confident
        wrong answer* rather than merely a wasted call."""
        for match in FINDING_RE.finditer(text or ""):
            if match.group("env") != self._env or match.group("entry") != self._entry:
                continue
            patch = match.group("patch")
            if patch not in self._patches:
                continue
            if match.group("verdict") == "accepted":
                belief.repaired_by = patch
            else:
                belief.ruled_out.add(patch)

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
            len(belief.tried) + len(belief.read_ids) + int(belief.searched) + int(belief.recorded)
        )

        if not belief.searched and "search_knowledge" in available:
            return (
                ToolCall(f"c{step}", "search_knowledge", {
                    "query": f"repair {self._entry} defective function candidate edit",
                }),
                "Checking what the collective already knows about this function.",
            )

        unread = [i for i in belief.ranked_ids if i not in belief.read_ids]
        if (
            belief.repaired_by is None
            and unread
            and len(belief.read_ids) < self._max_reads
            and "read_artifact" in available
        ):
            return (
                ToolCall(f"c{step}", "read_artifact", {"artifact_id": unread[0]}),
                f"Reading the highest-ranked result not yet read ({len(belief.read_ids) + 1} of "
                f"at most {self._max_reads}).",
            )

        if belief.repaired_by is not None:
            if self._record and not belief.recorded and "create_artifact" in available:
                return (
                    ToolCall(f"c{step}", "create_artifact", {
                        "type": "evidence",
                        "title": f"{self._entry} is repaired by {belief.repaired_by}",
                        "body": finding_line(self._env, self._entry, belief.repaired_by, True),
                        "confidence": 0.9,
                    }),
                    "Recording the repair for later agents.",
                )
            source = self._program.apply(belief.repaired_by)
            return (
                ToolCall(f"c{step}", "submit_result", {
                    "answer": source,
                    "reasoning": "the candidate edit passes the worked example",
                    "confidence": 0.9,
                }),
                "Submitting the repaired source.",
            )

        candidates = [
            p for p in self._patches if p not in belief.tried and p not in belief.ruled_out
        ]
        if candidates and "try_patch" in available:
            return (
                ToolCall(f"c{step}", "try_patch",
                         {"patch": candidates[0], "entry": self._entry}),
                f"Trying {candidates[0]}.",
            )

        # §12: four rejected edits are four facts. Discarding them is what keeps a collective from
        # ever maturing, and recording them is what gives `collective_no_negative` something to
        # remove.
        if (
            self._record
            and not belief.recorded_negative
            and belief.ruled_out
            and "record_failure" in available
        ):
            ruled = sorted(belief.ruled_out)
            return (
                ToolCall(f"c{step}", "record_failure", {
                    "title": f"Ruled out {len(ruled)} candidate edit(s) for {self._entry}",
                    "body": "\n".join(
                        finding_line(self._env, self._entry, p, False) for p in ruled
                    ),
                    "tool_name": "try_patch",
                    "ruled_out": ruled,
                }),
                "Recording what did not repair it.",
            )

        return (None, "No candidate edits remain within budget.")
