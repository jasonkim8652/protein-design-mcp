import json
import math

import numpy as np
import pytest

from protein_design_mcp.dispatch.serialize import to_jsonable


@pytest.mark.parametrize(
    "value,expected",
    [
        (np.float32(0.5), 0.5),
        (np.float64(1.5), 1.5),
        (np.int64(7), 7),
        (np.int32(-3), -3),
        (np.bool_(True), True),
    ],
)
def test_numpy_scalars_become_python_scalars(value, expected):
    result = to_jsonable(value)
    assert result == expected
    assert type(result) in (int, float, bool)


def test_numpy_array_becomes_nested_lists():
    assert to_jsonable(np.array([[1, 2], [3, 4]])) == [[1, 2], [3, 4]]


def test_nested_structures_are_converted():
    payload = {"scores": [np.float32(0.5)], "meta": {"n": np.int64(2)}}
    assert to_jsonable(payload) == {"scores": [0.5], "meta": {"n": 2}}


def test_result_is_json_serializable():
    payload = {"iptm": np.float32(0.83), "pae": np.zeros((2, 2))}
    json.dumps(to_jsonable(payload))


def test_nan_and_inf_become_none_because_json_has_no_literal():
    assert to_jsonable(np.float32("nan")) is None
    assert to_jsonable(float("inf")) is None
    assert to_jsonable(-math.inf) is None


def test_plain_values_pass_through_unchanged():
    assert to_jsonable({"a": 1, "b": "x", "c": None, "d": True}) == {
        "a": 1,
        "b": "x",
        "c": None,
        "d": True,
    }


def test_tuples_and_sets_become_lists():
    assert to_jsonable((1, 2)) == [1, 2]
    assert sorted(to_jsonable({1, 2})) == [1, 2]


def test_unknown_object_becomes_its_string_form():
    class Opaque:
        def __repr__(self) -> str:
            return "<opaque>"

    assert to_jsonable(Opaque()) == "<opaque>"
