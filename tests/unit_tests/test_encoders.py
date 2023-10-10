import json
from typing import Any

import pytest
from langchain.schema.messages import (
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
)
from langchain.schema.document import Document


try:
    from pydantic.v1 import BaseModel
except ImportError:
    from pydantic import BaseModel

from langserve.serialization import simple_dumps, simple_loads


@pytest.mark.parametrize(
    "data",
    [
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
            "meow": "Hello, world!",
        },
    ],
)
def test_serialization(data: Any) -> None:
    """Test that the LangChainEncoder encodes the data as expected."""
    serialized = simple_dumps(data)
    assert isinstance(serialized, str)
    deserialized = simple_loads(serialized)
    # Verify equality (pydantic only checks schema equality, not class names.)
    assert deserialized == data
    # Verify equality of class names as well
    assert _get_full_representation(data) == _get_full_representation(deserialized)


def test_serialization_of_well_known_types() -> None:
    """Test that the LangChainEncoder encodes the well known types as expected."""
    well_known_types = [
        HumanMessage(content="hello"),
        HumanMessageChunk(content="hello"),
        SystemMessage(content="goodbye"),
        Document(page_content="hello", metadata={"page_number": 1}),
    ]

    for well_known_type in well_known_types:
        assert _get_full_representation(well_known_type) == _get_full_representation(
            simple_loads(simple_dumps(well_known_type))
        )


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
