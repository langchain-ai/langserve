from typing import Any

import pytest
from langchain.schema.messages import (
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
)

try:
    from pydantic.v1 import BaseModel
except ImportError:
    from pydantic import BaseModel

from langserve.serialization import WellKnownLCSerializer


@pytest.mark.parametrize(
    "data",
    [
        # Test with python primitives
        1,
        [],
        {},
        {"a": 1},
        {"output": [HumanMessage(content="hello")]},
        # Test with a single message (HumanMessage)
        HumanMessage(content="Hello"),
        # Test with a list containing mixed elements
        [HumanMessage(content="Hello"), SystemMessage(content="Hi"), 42, "world"],
        # Uncomment when langchain 0.0.306 is released
        [HumanMessage(content="Hello"), HumanMessageChunk(content="Hi")],
        # Attention: This test is not correct right now
        # Test with full and chunk messages
        [HumanMessageChunk(content="Hello"), HumanMessage(content="Hi")],
        # Test with a dictionary containing mixed elements
        {
            "message": HumanMessage(content="Greetings"),
            "numbers": [1, 2, 3],
            "boom": "Hello, world!",
        },
    ],
)
def test_serialization(data: Any) -> None:
    """There and back again! :)"""
    # Test encoding
    lc_serializer = WellKnownLCSerializer()

    assert isinstance(lc_serializer.dumps(data), str)
    # Translate to python primitives and load back into object
    assert lc_serializer.loadd(lc_serializer.dumpd(data)) == data
    # Test simple equality (does not include pydantic class names)
    assert lc_serializer.loads(lc_serializer.dumps(data)) == data
    # Test full representation equality (includes pydantic class names)
    assert _get_full_representation(
        lc_serializer.loads(lc_serializer.dumps(data))
    ) == _get_full_representation(data)


def _get_full_representation(data: Any) -> Any:
    """Get the full representation of the data, replacing pydantic models with schema.

    Pydantic tests two different models for equality based on equality
    of their schema; instead we will rely on the equality of their full
    schema representation. This will make sure that both models have the
    same name (e.g., HumanMessage vs. HumanMessageChunk).

    Args:
        data: python primitives + pydantic models

    Returns:
        data represented entirely with python primitives
    """
    if isinstance(data, dict):
        return {key: _get_full_representation(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [_get_full_representation(value) for value in data]
    elif isinstance(data, BaseModel):
        return data.schema()
    else:
        return data
