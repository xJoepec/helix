from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy types and arrays to JSON-serializable Python types.

    - numpy arrays -> lists
    - numpy scalars -> native Python scalars via .item()
    - mappings -> dict with stringified keys and converted values
    - sequences -> list with converted items (tuples become lists)
    - sets -> list with converted items
    Other types are returned as-is.
    """
    # numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # numpy scalars
    if isinstance(obj, np.generic):
        return obj.item()
    # mappings
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    # sequences (but not strings/bytes)
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    # sets -> list
    if isinstance(obj, set):
        return [to_jsonable(v) for v in obj]
    return obj


def json_dumps(obj: Any, **kwargs: Any) -> str:
    """json.dumps wrapper that first converts numpy types via to_jsonable."""
    import json

    return json.dumps(to_jsonable(obj), **kwargs)

