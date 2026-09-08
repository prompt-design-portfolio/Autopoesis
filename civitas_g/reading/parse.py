"""Parse a reference summary into the same field set the reading computes.

The summaries are fixed-width console output, not a data format, which makes parsing them a place
where a reproduction can quietly become a smaller reproduction. Two rules keep that from
happening:

1. **Tables are declared, not discovered.** A header is recognised by a signature of its column
   names, and each table's columns are stated. A header that stops matching produces zero rows for
   that table, and the gate reports the missing fields rather than passing on the ones it still
   found.
2. **A cell that is not a number is absent, never zero.** `plastic  --   (no record)` in the Gate R
   permutation table means the arm has no record to permute. It parses to no fields at all. The
   alternative -- writing 0.0 -- would compare a real 0.000 against an absence and call it a match.

`--`, `nan`, `NOT AVAILABLE` and a bare verdict word all fall under rule 2.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from civitas_g.reading.compute import TABLES, key

#: Research names as the summaries print them, longest first so that `plastic + record (slow)` is
#: never parsed as `plastic + record` with a stray `(slow)` token.
ARM_NAMES: tuple[str, ...] = tuple(sorted(
    ("plastic", "plastic + record", "plastic + noise", "fixed + record",
     "plastic + record (slow)"),
    key=len, reverse=True))

#: Each table's header signature: every fragment must appear in the header line.
HEADER_SIGNATURES: dict[str, tuple[str, ...]] = {
    "gate_r": ("arm", "pooled", "per-era mean", "n writes"),
    "gate_r_perm": ("arm", "observed", "null", "sd", "z", "epochs"),
    "sym_gain": ("arm", "phase 1", "phase 2", "change", "frac > 0"),
    "sym_gain_delta": ("arm", "|gain|", "baseline", "delta"),
    "store_gain": ("arm", "learned", "innate", "learned - innate"),
    "first_prep_by_mark": ("arm", "P(ok|mark)", "P(ok|none)", "gap", "n mark", "n none"),
    "follow": ("arm", "all", "endorse=ok", "STALE", "null", "ratio", "n stale"),
    "nfc": ("arm", "mean preps", "censored"),
    "population": ("arm", "pop p1", "pop p2", "inj p2", "prep hit", "mark dens"),
}

_NUMBER = re.compile(r"[+-]?(?:\d+\.\d+|\d+|nan|inf)", re.IGNORECASE)


@dataclass
class ParsedSummary:
    path: str
    fields: dict[str, float]
    tables_found: list[str]
    tables_missing: list[str]
    #: Rows that matched an arm but yielded no numbers -- an absence, recorded as one.
    empty_rows: list[tuple[str, str]]

    def __len__(self) -> int:
        return len(self.fields)


def _match_arm(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    for arm in ARM_NAMES:
        if stripped.startswith(arm):
            rest = stripped[len(arm):]
            # guard against `plastic` matching the start of a word it is not
            if rest[:1] in ("", " "):
                return arm, rest
    return None


def _numbers(text: str) -> list[float]:
    out: list[float] = []
    for tok in _NUMBER.findall(text):
        low = tok.lower().lstrip("+-")
        if low == "nan":
            out.append(math.nan)
        elif low == "inf":
            out.append(math.inf if not tok.startswith("-") else -math.inf)
        else:
            out.append(float(tok))
    return out


def _header_table(line: str) -> str | None:
    low = line.lower()
    for name, sig in HEADER_SIGNATURES.items():
        if all(frag.lower() in low for frag in sig):
            return name
    return None


def parse_summary(path: str | Path) -> ParsedSummary:
    """Every declared table in a reference summary, as `{field key: value}`."""
    text = Path(path).read_text(errors="replace")
    fields: dict[str, float] = {}
    found: list[str] = []
    empty: list[tuple[str, str]] = []

    table: str | None = None
    for line in text.splitlines():
        header = _header_table(line)
        if header is not None:
            table = header
            if header not in found:
                found.append(header)
            continue
        if table is None:
            continue
        if not line.strip():
            table = None
            continue
        matched = _match_arm(line)
        if matched is None:
            # a prose line inside a table block ends it; a continuation line does not restart it
            if line.startswith("  ") and not line.startswith("    "):
                table = None
            continue
        arm, rest = matched
        values = _numbers(rest)
        if not values:
            empty.append((table, arm))
            continue
        columns = TABLES[table]
        # a row may print fewer numbers than the table declares (a trailing verdict word, an
        # arm with nothing to report); zip stops at the shorter, and the missing columns are
        # then absent rather than filled
        for column, value in zip(columns, values, strict=False):
            fields[key(table, arm, column)] = value

    return ParsedSummary(
        path=str(path), fields=fields, tables_found=found,
        tables_missing=[t for t in TABLES if t not in found], empty_rows=empty,
    )
