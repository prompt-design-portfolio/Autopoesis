"""Making an engine log row survive a database round-trip without losing what it means.

The engine's log rows are numpy-flavoured Python: `np.float64` scalars, tuples, small ndarrays
(`mi_counts` is (K, 2, K)), and -- crucially -- `np.nan` wherever a metric could not be computed.
Three of those need care, and one of them is a correctness issue rather than a convenience.

**NaN is the correctness issue.** PostgreSQL's `JSONB` rejects NaN outright, and Python's `json`
writes a bare `NaN` token that is not valid JSON. The tempting fixes are both wrong: dropping the
key makes an absent measurement look like a field the sim never recorded, and coercing it to 0.0
makes it look like a measured zero. Both are the failure A5 names -- "unrecomputable fields print
NOT AVAILABLE, never a guess" -- committed in storage rather than in a report. So NaN, +inf and
-inf are encoded as tagged strings and decoded back to exactly the float they were.

**Tuples become lists.** JSON has no tuple. The reading never distinguishes them (`np.asarray` and
indexing treat them alike), so this is safe, and it is stated here rather than discovered later.

**Everything else is exact.** Python's float repr round-trips, so a `float64` written and read
back is the same double, which is what makes a three-decimal reproduction from stored rows mean
anything at all.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

#: Tagged forms for the three non-finite doubles. Strings so that JSONB accepts them, prefixed so
#: that a genuine data string could not collide by accident.
NAN_TAG = "__nonfinite__:nan"
POS_INF_TAG = "__nonfinite__:+inf"
NEG_INF_TAG = "__nonfinite__:-inf"

_TAGS = {NAN_TAG: math.nan, POS_INF_TAG: math.inf, NEG_INF_TAG: -math.inf}


def encode(value: Any) -> Any:
    """A log row, or any part of one, as something both JSONB and SQLite text will hold."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if math.isnan(f):
            return NAN_TAG
        if math.isinf(f):
            return POS_INF_TAG if f > 0 else NEG_INF_TAG
        return f
    if isinstance(value, np.ndarray):
        return encode(value.tolist())
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    return str(value)


def decode(value: Any) -> Any:
    """The inverse. A tagged non-finite comes back as the float it was, not as its tag."""
    if isinstance(value, str):
        return _TAGS.get(value, value)
    if isinstance(value, list):
        return [decode(v) for v in value]
    if isinstance(value, dict):
        return {k: decode(v) for k, v in value.items()}
    return value


def round_trips(value: Any) -> tuple[bool, str]:
    """Does `decode(encode(x))` equal `x`, treating NaN as equal to NaN?

    Used by the store round-trip self-test (B§6). Equality with NaN on both sides is the whole
    point: `nan != nan`, so a naive comparison would report every absent measurement as a
    round-trip failure and hide the real ones.
    """
    def same(a: Any, b: Any) -> bool:
        if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
            fa, fb = float(a), float(b)
            return (math.isnan(fa) and math.isnan(fb)) or fa == fb
        if isinstance(a, np.ndarray):
            return same(a.tolist(), b)
        if isinstance(a, (list, tuple)):
            return isinstance(b, list) and len(a) == len(b) and all(
                same(x, y) for x, y in zip(a, b, strict=True))
        if isinstance(a, dict):
            return isinstance(b, dict) and set(map(str, a)) == set(b) and all(
                same(v, b[str(k)]) for k, v in a.items())
        if isinstance(a, (bool, np.bool_)):
            return bool(a) == b
        if isinstance(a, (int, np.integer)):
            return int(a) == b
        return a == b or str(a) == b

    out = decode(encode(value))
    if same(value, out):
        return True, "round-trips"
    return False, f"changed: {value!r} -> {out!r}"
