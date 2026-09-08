"""The compute engine, untouched (A1.8).

    The compute engine is untouched. `sim_v3_13.py` runs byte-identical; its hash is in every
    manifest. Civitas touches the world only at era boundaries -- snapshots, the store, the
    manifest -- never per agent-step.

This module is the whole of Civitas's contact with the engine, and it is deliberately thin. It
builds a `Config`, calls `sim_v3_13.run`, and hands back what came out. It does not wrap
`resolve_action`, does not subclass `World` or `Agent`, does not install a per-step callback, and
does not observe on any agent's behalf. There is nothing here for a learner to lean on, which is
the point: A1.1 says the learner is the only thing that thinks, and a per-step hook is the shape
cognition would arrive in if it ever did.

The engine's own logging is already era-scoped -- `run()` appends one row every `log_every` steps
and clears its accumulators -- so "at era boundaries" is not a discipline this module has to
impose. It is a property of the engine that this module must not break, and
`touches_world_only_at_era_boundaries` states it as a check rather than a promise.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import sim_v3_13 as _sim
from civitas_g.world.spec import STAGED, WORLD, check_chance_ev_is_zero, check_world


class EngineRefusal(RuntimeError):
    """The engine will not be run, and the reason is not a gate failure."""


@dataclass(frozen=True)
class RunSpec:
    """One run: an arm's world, a seed, and how long each phase is.

    `phase_steps` scales BOTH phases, exactly as `analysis_v3_13.run_experiment` does -- each
    phase's `n_steps` is `n_steps / PHASE_STEPS * phase_steps`. Scaling only one would change the
    ratio of the food-only phase to the chain phase, which is a different staging and therefore a
    different experiment.
    """

    arm: str
    seed: int
    phase_steps: int
    kwargs: dict[str, Any]

    def phases(self) -> list[dict[str, Any]]:
        base = int(STAGED[0]["n_steps"])
        return [dict(p, n_steps=int(p["n_steps"] / base * self.phase_steps)) for p in STAGED]

    def config(self) -> _sim.Config:
        return _sim.Config(seed=self.seed, **self.kwargs)


def build_run_spec(arm: str, seed: int, phase_steps: int,
                   world: dict[str, Any] | None = None,
                   overrides: dict[str, Any] | None = None) -> RunSpec:
    """An arm's `RunSpec`, with the world checked before anything is built.

    The checks run here rather than at read time on purpose. A number produced against an
    unexpected world is not a number that failed a gate; it is a number about a different
    experiment, and the only safe thing to do with it is not to produce it.
    """
    from civitas_g.world.arms import get_arm

    world = WORLD if world is None else world
    check_world(world)
    check_chance_ev_is_zero(world)
    kwargs = get_arm(arm).config_kwargs(world)
    if overrides:
        kwargs.update(overrides)          # overrides win over WORLD, as run_experiment does
    return RunSpec(arm=get_arm(arm).name, seed=seed, phase_steps=phase_steps, kwargs=kwargs)


@dataclass
class RunResult:
    """What one run produced, plus how long it took and what produced it."""

    arm: str
    seed: int
    phase_steps: int
    raw: dict[str, Any]
    wall_seconds: float
    engine_sha256: str

    @property
    def log(self) -> list[dict[str, Any]]:
        return self.raw["log"]

    @property
    def cfg(self) -> dict[str, Any]:
        return self.raw["cfg"]

    @property
    def final_mapping(self) -> tuple[int, ...]:
        return tuple(self.raw["final_mapping"])

    @property
    def n_era_snapshots(self) -> int:
        return len(self.raw.get("era_snaps") or [])


def run(spec: RunSpec, *, verbose: bool = False,
        init_genomes: Any = None) -> RunResult:
    """Run the engine. Byte-identical, hash recorded.

    `init_genomes` exists for G3 -- population B born into population A's record -- and is passed
    straight through. At G1 it is always None, and a caller that passes one is told so rather than
    quietly getting a founder-seeded run under a G1 manifest.
    """
    from civitas_g.manifest import engine_identity

    if init_genomes is not None:
        raise EngineRefusal(
            "init_genomes is G3's mechanism (population B born into population A's record). "
            "At G1 every run starts from fresh founders, and a run that did not would not be a "
            "reproduction of anything committed."
        )
    identity = engine_identity()
    if identity["engine_tree_clean"] == "no":
        raise EngineRefusal(
            "sim_v3_13.py or analysis_v3_13.py has uncommitted changes. A1.8 requires the engine "
            "to run byte-identical with its hash in the manifest; a dirty tree has no hash a "
            "reader could resolve. Commit or stash before running."
        )
    t0 = time.time()
    raw = _sim.run(spec.config(), verbose=verbose, phases=spec.phases())
    return RunResult(
        arm=spec.arm, seed=spec.seed, phase_steps=spec.phase_steps, raw=raw,
        wall_seconds=time.time() - t0, engine_sha256=identity["sim_v3_13.py"],
    )


def touches_world_only_at_era_boundaries() -> tuple[bool, str]:
    """A1.8's second clause, as a check on this module rather than a promise about it.

    Civitas's contact with the engine is the `run()` above. The property that makes it era-scoped
    is the engine's: `run()` appends to `log` only every `log_every` steps and clears its
    accumulators there. So the thing to verify is that this module holds no per-step surface at
    all -- no callback parameter, no monkeypatch of `resolve_action`, no subclass of `World` or
    `Agent`.
    """
    import inspect

    import civitas_g.world.engine as _self

    problems: list[str] = []

    sig = inspect.signature(run)
    for suspicious in ("on_step", "callback", "hook", "per_step", "observer"):
        if suspicious in sig.parameters:
            problems.append(f"run() takes a per-step parameter {suspicious!r}")

    if _sim.resolve_action.__module__ != "sim_v3_13":
        problems.append("sim_v3_13.resolve_action has been replaced")
    for name in ("World", "Agent", "run"):
        obj = getattr(_sim, name)
        if getattr(obj, "__module__", "sim_v3_13") != "sim_v3_13":
            problems.append(f"sim_v3_13.{name} has been replaced")

    for name, obj in vars(_self).items():
        if isinstance(obj, type) and issubclass(obj, (_sim.World, _sim.Agent)) \
                and obj not in (_sim.World, _sim.Agent):
            problems.append(f"{name} subclasses the engine's {obj.__mro__[1].__name__}")

    if problems:
        return False, "; ".join(problems)
    return True, ("no per-step surface: run() takes no callback, the engine's resolve_action, "
                  "World, Agent and run are unreplaced, and nothing here subclasses them")
