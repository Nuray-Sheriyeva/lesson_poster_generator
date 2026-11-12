"""Convert Pydantic models into the strict JSON Schema subset Luna accepts.

OpenAI structured outputs require every object to set ``additionalProperties:
false`` and list every property in ``required``. They also reject several JSON
Schema keywords that Pydantic normally emits (titles, defaults, lengths, and
numeric bounds). We validate those constraints after parsing instead.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

_MAX_DEPTH = 40


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})
    resolved = _resolve(raw, defs, depth=0)
    cleaned = _clean(resolved)
    if not isinstance(cleaned, dict):
        raise ValueError("model schema did not produce an object")
    return cleaned


def _resolve(node: Any, defs: dict[str, Any], *, depth: int) -> Any:
    if depth > _MAX_DEPTH:
        raise ValueError("JSON schema nesting is too deep")
    if isinstance(node, list):
        return [_resolve(item, defs, depth=depth + 1) for item in node]
    if not isinstance(node, dict):
        return node
    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        name = ref.rsplit("/", 1)[-1]
        if name not in defs:
            raise ValueError(f"unknown schema reference: {ref}")
        return _resolve(defs[name], defs, depth=depth + 1)
    return {key: _resolve(value, defs, depth=depth + 1) for key, value in node.items()}


def _clean(node: Any) -> Any:
    if isinstance(node, list):
        return [_clean(item) for item in node]
    if not isinstance(node, dict):
        return node

    if "const" in node:
        return {"enum": [node["const"]]}
    if "anyOf" in node:
        return {"anyOf": [_clean(item) for item in node["anyOf"]]}
    if "oneOf" in node:
        return {"oneOf": [_clean(item) for item in node["oneOf"]]}
    if "allOf" in node:
        return {"allOf": [_clean(item) for item in node["allOf"]]}

    node_type = node.get("type")
    if node_type == "object" or "properties" in node:
        properties = {
            key: _clean(value) for key, value in (node.get("properties") or {}).items()
        }
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }
    if node_type == "array":
        return {"type": "array", "items": _clean(node.get("items", {}))}
    if "enum" in node:
        return {"type": node_type or "string", "enum": node["enum"]}
    if node_type == "null":
        return {"type": "null"}
    if node_type:
        return {"type": node_type}
    return {}
