# v3.9 — sparse chains are not discoverable under autopoietic economics

*4 September 2026. The recipe world is closed for individual-learner tests. Three economics passes;
the stop condition agreed in the spec was invoked before the acceptance grid.*

## The finding

**A multi-step chain whose payoff arrives only at the end is not discoverable by an individual
learner under autopoietic economics, without an instinct.** Not because the learner cannot hold the
fact, and not because the chain does not pay — but because the chain is *rare*, and a rare
opportunity cannot generate the selection differential that would build the approach behaviour
needed to make it less rare.

## The mechanism

At the densities the world supports, an agent gets roughly **half a tool attempt per lifetime**. An
attempt at chance is worth about **+0.37 energy** (`1/6 × 2.5 − 5/6 × 0.05`), against a lifetime
income of several meals at 0.7 each — **about 5% of what an agent earns in a life**. The behaviour
that would raise that number is a four-step sequence: notice an item, pick it up, carry it to a
station, act there. **A four-step approach behaviour cannot be selected out of a 5% differential in
twenty generations.**

The three passes each removed one candidate explanation and left the differential intact:

| pass | change | `fixed` att/life | `random policy` att/life | P(interact \| at station) vs null | plastic P2 pop |
|---|---|---|---|---|---|
| 1 — payoff size | nuts, oracle arms, `tool_bonus`/`fail_cost` 0.05 | 0.21 | 0.16 | — (P(int\|on item) 0.124 vs 0.165) | 41–61 |
| 2 — journey length | chain shortened to pickup→station→attempt, `tool_value` 1.5, item cover 21%, station cover 7% | 0.56 | 0.96 | — | **153** |
| 3 — co-location | items and stations inside the food patches, stations travel with them, `carry_cost` 0, `tool_value` 2.5 | **0.38** | **0.58** | 0.135 vs **0.156** | 45 (window transient; the full run held 107–209) |

Neither lever moved it. Station density three times the agreed band took the null from 0.96 to only
2.16. Raising `tool_value` from 1.5 to 8.0 *lowered* attempts per life (0.96 → 0.85). **Co-location
halved item supply without creating a gradient** — confining items to patches cut global item cover
from ~21% to ~10%, and shortening the journey does nothing if nothing selects for taking it.

**A correction to an earlier reading.** In the amendment-3 pre-check I read plastic's phase-2
population of 45 as co-location turning the chain into an "unavoidable tax on foraging". That is
wrong. A full run at seed 0 held `plastic` at **107–209 through phase 2** with pickups ~1/1k and
attempts ~0.3/1k: **the chain costs almost nothing because almost nobody touches it.** The 45 was a
window transient. The chain is not a tax; it is ignored.

## Pre-check note — the abstention under pass 2

*An observation from a one-seed pre-check, not a claim.* With `tool_value` 1.5 and a carry tax,
`plastic (W2)` **declined 99.7% of station opportunities** and made **0.006 attempts per life** —
160× below the random-walk null — while being the **only arm not at the population floor** (153
against 41–52, zero injections) and holding the best safe rate (0.622). Given a genuine choice, the
best learner rejected the chain and specialised in eating. That reading is what pass 3 was built to
answer.

## What the rig fixes did establish

The audit fixes are not wasted, and one of them produced a real positive:

- **Fix A (splitting `eat` from `interact`) restored food learning across the chain.** v3.8 saw the
  safe rate decay from 0.65 to 0.50 in *every* arm once the chain switched on. That did not recur:
  `plastic (W2)` holds **0.622** (pass 2) and **0.579** (pass 3) in phase 2, with `probe_adv` (food)
  above 1.0 and zero injections. The v3.8 decay was the shared action, exactly as the audit said.
- **Fix B, D, F** hold: densities are set to a measured target, the modulator docstring matches the
  event table, and there is a random-walk null.
- **The semantics self-test** (16 rows, 0.02 s) and the learning-rule test now run in the setup cell
  and would have caught the shared action.
- Two defects in the *stopping rule itself* were found by running it: per-1k contact rates are
  confounded by the action budget (an arm that learns to eat necessarily interacts less than a
  uniform walker), and attempts-per-life measured on an arm at the injection floor measures churn
  rather than the world.

## Consequence

**The recipe world is closed for individual-learner tests.** Any future test of a conjunctive or
delayed-credit fact needs the opportunity to be *dense* — where the agent already is, on every
meal — rather than sparse and sought. That is the preparation world (v3.10).

What is **not** claimed: that the learner cannot hold a conjunction; that the rule cannot bridge
delayed credit. Neither was ever tested on a rig that could show it. The one agreed rule-form change
is **not** spent.
