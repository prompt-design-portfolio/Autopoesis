"""What B§1 removes from the live surface, kept as history (A1.4).

The directive's G0 removal list and the reasons for each entry are in `docs/G0_G1.md` §1. Two of
its instructions meet awkwardly here and this package is the resolution.

* "**Removed, with tests:** ... `ChainPolicy`, `PartitionPolicy`, `IntegratorPolicy`,
  `ToolmakerPolicy`, every deterministic policy provider."
* "**Dormant:** ... §17-§18 tool ecology; §26-§32 specialization, allocation, disagreement,
  reputation, institutions, meta-learning, long-horizon projects."

Those four benchmarks *are* their policies — the policy is the agent and the module exists to run
it. Deleting the policy guts the module; keeping the module without its policy leaves something
that cannot import. So the whole surface moves here: it is on no acceptance path, nothing in
`civitas_g` imports it, no test exercises it, and A1.4's "nothing is hard-deleted" is honoured
rather than argued with.

**Why it is not simply deleted.** These modules produced the measurements in `results/legacy/`,
and B§1 keeps those "as history". A result whose producing code has been deleted is exactly the
gap the G0 audit found for `precheck_v3_12.txt`, and it would be perverse to create another one
while documenting that as a defect.

**Why it is not revived.** A1.1: the learner is the only thing that thinks. Every agent in here is
a hand-written policy, which is what the rebuild exists to stop standing in for a learner. This is
a record of what was measured before that inversion, not a component of anything.
"""
