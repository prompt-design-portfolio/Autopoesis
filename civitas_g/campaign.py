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

import itertools
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


def sign_test(xs: list[float]) -> dict[str, Any]:
    """Exact binomial on the signs. Distribution-free, which is why it is here.

    At these n the t-test's normality assumption cannot be checked and the estimate is visibly
    heavy-tailed — one seed in six has flipped the verdict twice. The sign test throws away the
    magnitudes and keeps only the direction, so it is much less powerful and much harder to
    mislead. When the two disagree, the sign test is the one to believe at n < 10.

    Both tails are reported. B§5.2 states the direction in advance, so one-tailed is defensible;
    two-tailed is the conservative reading and is what `p_two` gives.
    """
    n = len(xs)
    neg = sum(1 for x in xs if x < 0)
    if n == 0:
        return {"n": 0}
    c = math.comb
    # P(X >= neg) under p=0.5, the one-tailed probability of a run this negative or more so
    p_one = sum(c(n, k) for k in range(neg, n + 1)) / 2 ** n
    return {"n": n, "negative": neg, "p_one_tailed": p_one,
            "p_two_tailed": min(1.0, 2 * min(p_one, sum(c(n, k) for k in range(0, neg + 1))
                                             / 2 ** n))}


def exact_sign_permutation(xs: list[float], max_n: int = 20) -> dict[str, Any]:
    """Exact paired permutation test: enumerate all 2^n sign assignments.

    **The right test for this design.** Under the null the sign of each seed's contrast is
    exchangeable, so enumerating every assignment gives an exact p with no distribution
    assumption at all. At n = 8 that is 256 arrangements — cheap, and it does not ask the
    normality question that a t-test at this n cannot answer.

    It is also the test least able to be talked into a result, which is why it is here: this
    campaign's t crossed ±2 three times and came back, and each crossing was reported as a
    finding before the next seed withdrew it.

    Uses the magnitudes, unlike `sign_test`, so it is more powerful than that and less
    assumption-laden than the t. Returns None above `max_n`, where 2^n stops being free.
    """
    n = len(xs)
    if n == 0:
        return {"n": 0, "p": None}
    if n > max_n:
        return {"n": n, "p": None, "note": f"2^{n} is too many; exact enumeration stops at n = "
                                           f"{max_n}"}
    obs = st.mean(xs)
    hits = 0
    for signs in itertools.product((1.0, -1.0), repeat=n):
        if st.mean([s * x for s, x in zip(signs, xs, strict=True)]) <= obs:
            hits += 1
    return {"n": n, "p": hits / 2 ** n, "arrangements": 2 ** n, "observed_mean": obs}


def seeds_for_significance(mean: float, sd: float, target_t: float = 2.0) -> int | None:
    """Roughly how many seeds this effect would need to reach `target_t`: `n >= (t*sd/mean)^2`.

    Not a power calculation in the proper sense -- it treats the current mean and sd as the truth,
    so it understates what is needed about half the time and is worthless when the mean is near
    zero. It is here because "how many more seeds?" is the question every one of these tables
    raises, and an approximate answer with its limits stated beats leaving a reader to guess.
    """
    if not mean or not math.isfinite(mean) or not math.isfinite(sd) or sd <= 0:
        return None
    return max(2, math.ceil((target_t * sd / abs(mean)) ** 2))


def running_estimate(cells: list[dict[str, Any]], alignment: str = "aligned",
                     key: str = "content") -> list[tuple[int, float, float]]:
    """The estimate after each seed, in seed order: `(n, mean, t)`.

    Shows whether an effect is settling or drifting. Under a true null the t wanders and crosses
    +/-2 from time to time, so any single reading can look decisive on its own -- which is the
    point of printing the trajectory. This campaign has twice now reported a result at n-1 seeds
    that weakened at n, and a reader who sees only the latest row cannot tell that happened.
    """
    xs = [c[key] for c in sorted((c for c in cells if c["alignment"] == alignment),
                                 key=lambda c: c["seed"])]
    out = []
    for i in range(2, len(xs) + 1):
        head = xs[:i]
        se = st.stdev(head) / math.sqrt(i)
        out.append((i, st.mean(head), st.mean(head) / se if se else float("nan")))
    return out


def paired_test(cells: list[dict[str, Any]], alignment: str,
                key: str = "content") -> dict[str, Any]:
    """The contrast across seeds, with the error term the design actually supports.

    **The G3 contrast is paired and this is easy to get wrong.** All four arms of a succession run
    under one `succession.seed`, so they consume the same random draws in the same order and
    differ only in the store they are handed — common random numbers, a variance-reduction design.
    The consequence is that the spread of a SINGLE arm's `nfc_mean` over RNG seeds is *not* the
    yardstick for the contrast: measured directly, one arm replicated over B's seed has sd 0.144,
    while the paired contrast across seeds is smaller (`_aligned_sd` computes it; it was 0.061 at
    five seeds and 0.100 at six, which is why it is computed and not written down here). Reading
    the single-arm figure as the noise floor for the contrast understates the design and could
    reject a real effect.

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
        sg = sign_test([c["content"] for c in cells if c["alignment"] == alignment])
        rows += [f"  {alignment}: n={s['n']} seeds {s['seeds']}",
                 f"    content mean {s['mean']:+.4f}  sd {s['sd']:.4f}  se {s['se']:.4f}"
                 f"  range [{s['min']:+.4f}, {s['max']:+.4f}]",
                 f"    negative in {sg['negative']} of {sg['n']}"
                 f"   sign test p = {sg['p_one_tailed']:.3f} one-tailed,"
                 f" {sg['p_two_tailed']:.3f} two-tailed"
                 f"   (unanimous would reach {s['sign_test_p_if_unanimous']:.3f})", ""]

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

    for label, xs in (("aligned content", [c["content"] for c in cells
                                           if c["alignment"] == "aligned"]),):
        ex = exact_sign_permutation(xs)
        if ex.get("p") is not None:
            rows += ["", f"    {label}: exact paired permutation p = {ex['p']:.4f} one-tailed "
                         f"over all {ex['arrangements']} sign assignments",
                     "    (no normality assumption -- the t at this n cannot check one)"]

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
             f"makes the difference.", ""]

    rows += ["  HOW THE ESTIMATE MOVED AS SEEDS ARRIVED (aligned content)", ""]
    for n, mean, t in running_estimate(cells):
        rows.append(f"    n={n:<3} mean {mean:+.4f}   t {t:+.2f}")
    t_al = paired_test(cells, "aligned")
    if t_al.get("t") is not None:
        need = seeds_for_significance(t_al["mean"], t_al["sd"])
        rows += ["", f"    at the current estimate, t = 2 would need about n = {need} seeds "
                     f"(have {t_al['n']}).",
                 "    That treats the current mean and sd as true, so it is a rough guide, and it",
                 "    is worthless if the effect is actually zero.", ""]

    rows += ["  CONTROLS AGREE? by criterion, aligned seeds only", ""]
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
