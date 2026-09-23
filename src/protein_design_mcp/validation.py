"""Server-side enforcement of manifest schema constraints.

JSON Schema support differs across models: Gemini's function calling accepts a
restricted OpenAPI subset and does not enforce ``pattern``, and no provider
reliably applies ``default``. Client-side validation is therefore advisory.
This module is the actual boundary.

Error messages name the offending parameter, state the constraint, and show a
correct example, because the caller is a language model that will read the
error and retry.
"""

from __future__ import annotations

import re
from typing import Any

from protein_design_mcp.manifest.schema import Manifest

_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


class ToolInputError(ValueError):
    """Input failed a manifest constraint."""


def _example_clause(spec: dict[str, Any]) -> str:
    if "example" in spec:
        return f" Example: {spec['example']!r}."
    return ""


def _check_type(label: str, value: Any, spec: dict[str, Any]) -> None:
    expected = spec.get("type")
    if expected is None:
        return
    check = _TYPE_CHECKS.get(expected)
    if check is not None and not check(value):
        raise ToolInputError(
            f"{label} must be of type {expected}, got "
            f"{type(value).__name__}.{_example_clause(spec)}"
        )


def _check_scalar(label: str, value: Any, spec: dict[str, Any]) -> None:
    _check_type(label, value, spec)

    pattern = spec.get("pattern")
    if pattern is not None and isinstance(value, str):
        # PART A: reject leading/trailing whitespace
        if value != value.strip():
            raise ToolInputError(
                f"{label} = {value!r} has leading or trailing whitespace. "
                f"Stripped value: {value.strip()!r}."
            )
        # PART B: use fullmatch for anchored patterns, search for others
        match_func = (
            re.fullmatch if pattern.startswith("^") and pattern.endswith("$")
            else re.search
        )
        if match_func(pattern, value) is None:
            raise ToolInputError(
                f"{label} = {value!r} does not match the required format "
                f"{pattern}.{_example_clause(spec)}"
            )

    enum = spec.get("enum")
    if enum is not None and value not in enum:
        raise ToolInputError(
            f"{label} = {value!r} is not allowed; expected one of "
            f"{list(enum)}.{_example_clause(spec)}"
        )

    minimum = spec.get("minimum")
    if minimum is not None and isinstance(value, (int, float)) and value < minimum:
        raise ToolInputError(f"{label} = {value!r} is below the minimum of {minimum}.")

    maximum = spec.get("maximum")
    if maximum is not None and isinstance(value, (int, float)) and value > maximum:
        raise ToolInputError(f"{label} = {value!r} is above the maximum of {maximum}.")


def _check_value(label: str, value: Any, spec: dict[str, Any]) -> None:
    _check_scalar(label, value, spec)

    if spec.get("type") == "array" and isinstance(value, list):
        min_items = spec.get("minItems")
        if min_items is not None and len(value) < min_items:
            raise ToolInputError(
                f"{label} has {len(value)} entries but minItems is {min_items}."
                f"{_example_clause(spec)}"
            )
        max_items = spec.get("maxItems")
        if max_items is not None and len(value) > max_items:
            raise ToolInputError(
                f"{label} has {len(value)} entries but maxItems is {max_items}."
            )
        item_spec = spec.get("items")
        if isinstance(item_spec, dict):
            merged = dict(item_spec)
            merged.setdefault("example", spec.get("example", [None])[0]
                              if isinstance(spec.get("example"), list) and spec["example"]
                              else None)
            if merged.get("example") is None:
                merged.pop("example", None)
            for index, item in enumerate(value):
                _check_scalar(f"{label}[{index}]", item, merged)


def validate_and_fill(manifest: Manifest, arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate ``arguments`` against ``manifest`` and apply defaults.

    Returns a new dict. Raises ToolInputError with a message written for a
    model that will retry.
    """
    arguments = dict(arguments or {})

    unexpected = sorted(set(arguments) - set(manifest.schema))
    if unexpected:
        raise ToolInputError(
            f"{manifest.name} received unexpected parameter(s): "
            f"{', '.join(unexpected)}. Accepted parameters: "
            f"{', '.join(sorted(manifest.schema))}."
        )

    result: dict[str, Any] = {}
    for key, spec in manifest.schema.items():
        label = f"{manifest.name}.{key}"
        if key in arguments:
            value = arguments[key]
            _check_value(label, value, spec)
            result[key] = value
        elif "default" in spec:
            result[key] = spec["default"]
        elif spec.get("required"):
            description = spec.get("description", "")
            raise ToolInputError(
                f"{label} is required but was not provided. "
                f"{description}{_example_clause(spec)}".strip()
            )
    return result
