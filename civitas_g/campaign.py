"""Reading a G3 campaign across seeds, under criteria fixed BEFORE the seeds are read.

`docs/G3_WRITEUP.md` §5.6 found that the acceptance's coherence criterion scales its tolerance by
the effect, so the bar shrinks as the effect shrinks and a seed with no effect cannot pass however
well its controls agree. Fixing that after seeing which seeds it rejects would be fitting the gate
to the result, so the criterion in `g3.acceptance` is untouched and the alternatives live here,
as a **report** with no authority over any gate.

**When this was written.** Seeds 0-2 were on disk and had been read; seeds 3-7 were running and
had not. So with respect to seeds 3-7 every threshold below is pre-registered, and that is the
only sense in which any of them is out of sample. The distinction is recorded in the output
itself rather than left to a reader's memory, because a criterion's provenance is the whole of
its value.
"""

from __future__ import annotations

import json
import math
import pathlib
import statistics as st
from dataclasses import dataclass
from typing import Any

from civitas_g.g3 import G3Result

#: Seeds that had been read when the criteria below were fixed. A verdict over only these is
#: in-sample and says nothing; a verdict over the rest is a test.
SEEDS_SEEN_WHEN_WRITTEN = (0, 1, 2)


@dataclass(frozen=True)
class Criterion:
    """One answer to "do the two information-removing controls agree?"."""

    name: str
    note: str

    def tolerance(self, content: float, reading: float, noise: float | None) -> float | None:
        raise NotImplementedError

    def agrees(self, gap: float, content: float, reading: float,
               noise: float | None) -> bool | None:
        tol = self.tolerance(content, reading, noise)
        if tol is None or not math.isfinite(gap):
            return None
        return abs(gap) <= tol


@dataclass(frozen=True)
class RelativeToEffect(Criterion):
    fraction: float = 0.35

    def tolerance(self, content, reading, noise):
        scale = max(abs(content), abs(reading))
        return self.fraction * scale if scale > 0 else None


@dataclass(frozen=True)
class AbsolutePreparations(Criterion):
    limit: float = 0.02

    def tolerance(self, content, reading, noise):
        return self.limit


@dataclass(frozen=True)
class SamplingNoise(Criterion):
    multiple: float = 2.0

    def tolerance(self, content, reading, noise):
        #: `noise` is the standard error of the difference between two arms' `nfc_mean`. It is
        #: None on every run made so far: the engine logs `nfc_sum` and `n_nfc` and not
        #: `nfc_sumsq`, so the second moment does not exist in any stored run. This criterion is
        #: therefore NOT COMPUTABLE rather than passing or failing, and says so.
        return None if noise is None else self.multiple * noise


CRITERIA: tuple[Criterion, ...] = (
    RelativeToEffect("relative-to-effect (the gate)", fraction=0.35,
                     note="what `g3.acceptance` applies. Counts effect size twice: the bar goes "
                          "to zero with the effect, so a seed with no effect cannot pass."),
    AbsolutePreparations("absolute 0.02 preparations", limit=0.02,
                         note="arbitrary but visible, and independent of the effect. 0.02 is "
                              "about 1% of a typical `nfc_mean` of ~1.8."),
    AbsolutePreparations("absolute 0.05 preparations", limit=0.05,
                         note="a looser absolute bar, included so the choice of 0.02 is not "
                              "load-bearing on its own."),
    SamplingNoise("2 x sampling noise", multiple=2.0,
                  note="the principled one, and NOT COMPUTABLE from any run made so far. Needs "
                       "the engine change staged at docs/patches/apply-g3-nfc-variance.py."),
)


def _cells(r: G3Result) -> dict[str, Any] | None:
    by = {a.arm: a for a in r.arms}
    need = ("fresh store", "inherited store", "inherited scrambled", "inherited gain-zero")
    if any(n not in by for n in need):
        return None
    treat, scram, gain0, fresh = (by["inherited store"], by["inherited scrambled"],
                                  by["inherited gain-zero"], by["fresh store"])
    return {
        "seed": r.succession.seed,
        "alignment": r.succession.alignment,
        "content": treat.nfc_mean - scram.nfc_mean,
        "reading": treat.nfc_mean - gain0.nfc_mean,
        "total": treat.nfc_mean - fresh.nfc_mean,
        "gap": scram.nfc_mean - gain0.nfc_mean,
        "stale_content": treat.ratio - scram.ratio,
        "stale_reading": treat.ratio - gain0.ratio,
        "fresh_hit": fresh.prep_hit,
        # None until the engine logs the second moment. Carried explicitly so the difference
        # between "agrees" and "cannot be evaluated" is never silently collapsed to "agrees".
        "noise": None,
    }


def load(directory: str | pathlib.Path = "var/g3") -> list[dict[str, Any]]:
    out = []
    for f in sorted(pathlib.Path(directory).glob("*.json")):
        c = _cells(G3Result.from_dict(json.loads(f.read_text())))
        if c is not None:
            out.append(c)
    return out


def summarise(cells: list[dict[str, Any]], alignment: str) -> dict[str, Any]:
    xs = [c for c in cells if c["alignment"] == alignment]
    content = [c["content"] for c in xs]
    if not content:
        return {"alignment": alignment, "n": 0}
    n = len(content)
    sd = st.stdev(content) if n > 1 else float("nan")
    return {
        "alignment": alignment, "n": n, "seeds": sorted(c["seed"] for c in xs),
        "mean": st.mean(content), "sd": sd,
        "se": sd / math.sqrt(n) if n > 1 else float("nan"),
        "min": min(content), "max": max(content),
        "n_negative": sum(1 for x in content if x < 0),
        # The smallest p a sign test can produce at this n, reported so a run of consistent
        # signs is never read as significance it cannot reach.
        "sign_test_p_if_unanimous": 0.5 ** n,
    }


#: Measured, not assumed: `inherited store` replicated 8 times over B's RNG seed with A, the
#: record and the arm all held fixed. The gain-zero arm gives 0.088 the same way.
SINGLE_ARM_SD = 0.144


def _aligned_sd(cells: list[dict[str, Any]], key: str = "content") -> float:
    """The paired contrast's own spread, COMPUTED. It was once written into the prose as a
    literal (0.061, from five seeds) and was wrong by the sixth. A number quoted in a sentence
    beside numbers that update is a number that will go stale."""
    xs = [c[key] for c in cells if c["alignment"] == "aligned"]
    return st.stdev(xs) if len(xs) > 1 else float("nan")


def paired_test(cells: list[dict[str, Any]], alignment: str,
                key: str = "content") -> dict[str, Any]:
    """The contrast across seeds, with the error term the design actually supports.

    **The G3 contrast is paired and this is easy to get wrong.** All four arms of a succession run
    under one `succession.seed`, so they consume the same random draws in the same order and
    differ only in the store they are handed — common random numbers, a variance-reduction design.
    The consequence is that the spread of a SINGLE arm's `nfc_mean` over RNG seeds is *not* the
    yardstick for the contrast: measured directly, one arm replicated over B's seed has sd 0.144,
    while the paired contrast across seeds has sd 0.061. Reading the first as the noise floor for
    the second understates the design by a factor of about two and would reject a real effect.

    So the error term is the between-seed spread of the paired difference, and the statistic is a
    one-sample t on the per-seed contrasts. Reported with its df, because at these n the t is
    fragile and the number of seeds is the whole story.
    """
    xs = [c[key] for c in cells if c["alignment"] == alignment]
    n = len(xs)
    if n < 2:
        return {"alignment": alignment, "key": key, "n": n, "t": None,
                "note": "fewer than two seeds: no spread, so no test"}
    mean, sd = st.mean(xs), st.stdev(xs)
    se = sd / math.sqrt(n)
    return {"alignment": alignment, "key": key, "n": n, "df": n - 1,
            "mean": mean, "sd": sd, "se": se,
            "t": mean / se if se else float("nan"),
            "all_same_sign": all(x < 0 for x in xs) or all(x > 0 for x in xs)}


def alignment_contrast(cells: list[dict[str, Any]], key: str = "content") -> dict[str, Any]:
    """Aligned minus misaligned, paired **within** seed.

    The sharpest check the design contains, and one no control arm can supply. Both sides share
    the seed, the A population, and the record A left. The only difference is whether B's era
    clock matches A's — that is, whether the record is TRUE of the world B lives in. A store
    effect that appeared in both would be the store changing behaviour by existing; an effect
    that appears only when aligned is the record's *content* being used.

    Pairing within seed removes the between-seed variation that both sides share, which is why
    this is computed as a difference per seed rather than as two independent means.
    """
    by_seed: dict[int, dict[str, float]] = {}
    for c in cells:
        by_seed.setdefault(c["seed"], {})[c["alignment"]] = c[key]
    diffs = [(s, d["aligned"] - d["misaligned"]) for s, d in sorted(by_seed.items())
             if "aligned" in d and "misaligned" in d]
    n = len(diffs)
    if n < 2:
        return {"n": n, "t": None, "note": "fewer than two complete seed pairs"}
    xs = [d for _, d in diffs]
    mean, sd = st.mean(xs), st.stdev(xs)
    se = sd / math.sqrt(n)
    return {"n": n, "df": n - 1, "per_seed": diffs, "mean": mean, "sd": sd, "se": se,
            "t": mean / se if se else float("nan"),
            "n_negative": sum(1 for x in xs if x < 0)}


def report(directory: str | pathlib.Path = "var/g3") -> str:
    cells = load(directory)
    seeds = sorted({c["seed"] for c in cells})
    out_of_sample = [s for s in seeds if s not in SEEDS_SEEN_WHEN_WRITTEN]
    rows = [
        "G3 CAMPAIGN",
        "",
        f"  seeds on disk: {seeds}",
        f"  in-sample when the criteria were fixed: {sorted(SEEDS_SEEN_WHEN_WRITTEN)}",
        f"  OUT OF SAMPLE (the only ones that test anything): {out_of_sample or 'none yet'}",
        "",
    ]
    for alignment in ("aligned", "misaligned"):
        s = summarise(cells, alignment)
        if not s["n"]:
            continue
        rows += [f"  {alignment}: n={s['n']} seeds {s['seeds']}",
                 f"    content mean {s['mean']:+.4f}  sd {s['sd']:.4f}  se {s['se']:.4f}"
                 f"  range [{s['min']:+.4f}, {s['max']:+.4f}]",
                 f"    negative in {s['n_negative']} of {s['n']}"
                 f"  (unanimous would be p={s['sign_test_p_if_unanimous']:.3f} by sign test)", ""]

    rows += ["  THE CONTRAST, with the error term the paired design supports", ""]
    for alignment in ("aligned", "misaligned"):
        t = paired_test(cells, alignment)
        if t.get("t") is None:
            continue
        rows.append(f"    {alignment:<12} content {t['mean']:+.4f} +/- {t['se']:.4f} (se)"
                    f"   t = {t['t']:+.2f} on {t['df']} df"
                    f"   {'all one sign' if t['all_same_sign'] else 'signs mixed'}")
    # The unbalanced-seeds warning. This is not hypothetical: the alignment contrast was once
    # reported here at n=5 while the aligned column showed n=6, and the missing seed was the one
    # that had just moved every other number. A reader comparing the two columns would have been
    # comparing different seed sets. The tool now says so rather than relying on anyone noticing.
    by_seed: dict[int, set[str]] = {}
    for c in cells:
        by_seed.setdefault(c["seed"], set()).add(c["alignment"])
    half = sorted(s for s, al in by_seed.items() if len(al) < 2)
    if half:
        rows += [f"    !! seeds {half} have only one alignment on disk. The per-alignment means",
                 "       above and the paired contrast below are over DIFFERENT seed sets, and a",
                 "       half-finished seed can move either one. Do not compare them until this",
                 "       line is gone.", ""]

    ac = alignment_contrast(cells)
    if ac.get("t") is not None:
        rows += ["",
                 f"    alignment    aligned - misaligned {ac['mean']:+.4f} +/- {ac['se']:.4f} (se)"
                 f"   t = {ac['t']:+.2f} on {ac['df']} df"
                 f"   negative in {ac['n_negative']} of {ac['n']}",
                 "    (paired within seed: same A, same record, same draws -- only whether the",
                 "     record is TRUE of B's world differs. No control arm can supply this.)"]
    rows += ["",
             "    The four arms of a succession share one seed, so they consume the same draws",
             "    in the same order and differ only in the store. The spread of ONE arm over RNG",
             f"    seeds (measured, 8 replicates: sd {SINGLE_ARM_SD:.3f}) is therefore the wrong "
             f"yardstick for the",
             f"    contrast (this campaign: sd {_aligned_sd(cells):.3f}) -- the pairing is what "
             f"makes the difference.", "",
             "  CONTROLS AGREE? by criterion, aligned seeds only", ""]
    aligned = sorted((c for c in cells if c["alignment"] == "aligned"), key=lambda c: c["seed"])
    head = f"    {'seed':>5}{'content':>10}{'gap':>10}" + "".join(
        f"{c.name.split(' (')[0][:22]:>24}" for c in CRITERIA)
    rows.append(head)
    for c in aligned:
        line = f"    {c['seed']:>5}{c['content']:>+10.4f}{c['gap']:>+10.4f}"
        for crit in CRITERIA:
            v = crit.agrees(c["gap"], c["content"], c["reading"], c["noise"])
            line += f"{('NOT COMPUTABLE' if v is None else ('agree' if v else 'DISAGREE')):>24}"
        rows.append(line)
    rows.append("")
    for crit in CRITERIA:
        verdicts = [crit.agrees(c["gap"], c["content"], c["reading"], c["noise"])
                    for c in aligned]
        if all(v is None for v in verdicts):
            rows.append(f"    {crit.name}: NOT COMPUTABLE -- {crit.note}")
        else:
            rows.append(f"    {crit.name}: {sum(1 for v in verdicts if v)}/{len(verdicts)} "
                        f"agree -- {crit.note}")
    rows += ["", "  No criterion here decides a gate. `g3.acceptance` is the gate and is "
                 "untouched;", "  these exist so §5.6's defect is visible against alternatives "
                 "rather than argued."]
    return "\n".join(rows)
