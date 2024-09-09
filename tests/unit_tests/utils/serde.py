from typing import Any


def recursive_dump(obj: Any) -> Any:
    """Recursively dump the object if encountering any pydantic models."""
    if isinstance(obj, dict):
        return {
            k: recursive_dump(v)
            for k, v in obj.items()
            if k != "id"  # Remove the id field for testing purposes
        }
    if isinstance(obj, list):
        return [recursive_dump(v) for v in obj]
    if hasattr(obj, "model_dump"):
        # if the object contains an ID field, we'll remove it for testing purposes
        d = obj.model_dump()
        if "id" in d:
            d.pop("id")
        return recursive_dump(d)
    if hasattr(obj, "dict"):
        # if the object contains an ID field, we'll remove it for testing purposes
        if hasattr(obj, "id"):
            d = obj.dict()
            d.pop("id")
            return recursive_dump(d)
        return recursive_dump(obj.dict())
    return obj
