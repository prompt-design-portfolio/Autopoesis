"""A log row has to survive the database unchanged, and the NaNs are the hard part."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from civitas_g.persistence import encoding as E


@pytest.mark.parametrize("value", [
    np.nan, np.inf, -np.inf, 0.0, -0.0, 0.1 + 0.2, np.float64(1.5), np.int64(3), np.bool_(True),
    (1, 2, 3), [1.0, np.nan], np.zeros((5, 2, 5)), {"a": np.nan, "b": [np.inf, -np.inf]},
    None, True, False, "text",
])
def test_everything_a_log_row_contains_round_trips(value):
    ok, why = E.round_trips(value)
    assert ok, why


def test_nan_survives_as_nan_rather_than_as_zero_or_as_an_absent_key():
    """The failure this guards is A5's, committed in storage: an unrecomputable field must not
    become a measured zero, and must not vanish into 'the sim never recorded it'."""
    encoded = E.encode({"trace_recency": np.nan})
    assert encoded == {"trace_recency": E.NAN_TAG}
    assert "trace_recency" in encoded
    assert math.isnan(E.decode(encoded)["trace_recency"])


def test_the_encoded_form_is_valid_json_which_jsonb_accepts():
    """PostgreSQL's JSONB rejects NaN outright and Python's json writes a bare NaN token."""
    payload = E.encode({"a": np.nan, "b": np.inf, "c": np.zeros((2, 2))})
    text = json.dumps(payload, allow_nan=False)          # raises if a bare NaN got through
    assert "NaN" not in text
    assert math.isnan(E.decode(json.loads(text))["a"])


def test_a_double_survives_to_the_last_bit():
    """A three-decimal reproduction from stored rows means nothing if storage rounds."""
    x = 0.1 + 0.2
    assert E.decode(E.encode(x)) == x
    assert E.decode(E.encode(x)).hex() == x.hex()


def test_round_trips_reports_a_change_rather_than_raising():
    ok, why = E.round_trips(object())
    assert isinstance(ok, bool) and isinstance(why, str)


def test_a_real_log_row_round_trips(short_run):
    for row in short_run.log:
        ok, why = E.round_trips(row)
        assert ok, why
    assert any(isinstance(v, np.ndarray) for v in short_run.log[0].values()), \
        "the row carries no ndarray, so this test is not exercising the hard case"
