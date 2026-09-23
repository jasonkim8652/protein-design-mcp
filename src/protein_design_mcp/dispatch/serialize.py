"""Coerce scientific Python results into JSON-serializable values.

Engine runners return numpy scalars and arrays, which ``json.dumps`` rejects.
NaN and infinity are mapped to ``None`` because JSON has no literal for them
and emitting bare ``NaN`` produces output that strict parsers reject.
"""

from __future__ import annotations

import math
from typing import Any


def _is_nonfinite(value: Any) -> bool:
    return isinstance(value, float) and not math.isfinite(value)


def to_jsonable(obj: Any) -> Any:
    """Recursively convert ``obj`` into JSON-serializable values."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj

    if isinstance(obj, int):
        return obj

    # numpy scalars and arrays, without importing numpy at module scope.
    if hasattr(obj, "item") and hasattr(obj, "dtype") and getattr(obj, "shape", None) == ():
        return to_jsonable(obj.item())

    if isinstance(obj, float):
        return None if _is_nonfinite(obj) else obj

    if hasattr(obj, "tolist") and hasattr(obj, "dtype"):
        return to_jsonable(obj.tolist())

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]

    return str(obj)
