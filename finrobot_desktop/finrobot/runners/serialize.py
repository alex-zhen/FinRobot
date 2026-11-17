"""JSON round-trip for pipeline objects that are models, not dicts.

``ThesisResult`` is a Pydantic model. ``json.dumps(thesis)`` raises
``TypeError``. Callers must use :func:`dump_jsonable` and
:func:`load_jsonable` instead of treating the object as a dict.
"""

from __future__ import annotations

import importlib
import json
from typing import Any

from pydantic import BaseModel

_TAG = "__pydantic__"


def dump_jsonable(value: Any) -> Any:
    """Convert models, dates, and containers into JSON-compatible data."""
    if isinstance(value, BaseModel):
        return {
            _TAG: f"{type(value).__module__}.{type(value).__qualname__}",
            "data": dump_jsonable(value.model_dump(mode="json")),
        }
    if isinstance(value, dict):
        return {str(key): dump_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [dump_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def load_jsonable(value: Any) -> Any:
    """Restore a payload written by :func:`dump_jsonable`."""
    if isinstance(value, dict) and _TAG in value and "data" in value:
        module_name, _, class_name = str(value[_TAG]).rpartition(".")
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        if not isinstance(cls, type) or not issubclass(cls, BaseModel):
            raise TypeError(f"{value[_TAG]} is not a Pydantic model")
        return cls.model_validate(load_jsonable(value["data"]))
    if isinstance(value, dict):
        return {key: load_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [load_jsonable(item) for item in value]
    return value


def dumps_model(value: Any) -> str:
    return json.dumps(dump_jsonable(value), ensure_ascii=False, indent=2)


def loads_model(payload: str) -> Any:
    return load_jsonable(json.loads(payload))
