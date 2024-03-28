import datetime
import uuid
from enum import Enum
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, HumanMessageChunk, SystemMessage
from langchain_core.outputs import ChatGeneration

try:
    from pydantic.v1 import BaseModel
except ImportError:
    from pydantic import BaseModel

from langserve.serialization import WellKnownLCSerializer, load_events


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
        [ChatGeneration(message=HumanMessage(content="Hello"))],
    ],
)
def test_serialization(data: Any) -> None:
    """There and back again! :)"""
    # Test encoding
    lc_serializer = WellKnownLCSerializer()

    assert isinstance(lc_serializer.dumps(data), bytes)
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


@pytest.mark.parametrize(
    "data,expected",
    [
        ([], []),
        (
            [
                {
                    "type": "on_llm_start",
                    "serialized": {},
                    "prompts": [],
                    "run_id": str(uuid.UUID(int=2)),
                    "parent_run_id": str(uuid.UUID(int=1)),
                    "tags": ["h"],
                    "metadata": {},
                    "kwargs": {},
                }
            ],
            [
                {
                    "type": "on_llm_start",
                    "serialized": {},
                    "prompts": [],
                    "run_id": uuid.UUID(int=2),
                    "parent_run_id": uuid.UUID(int=1),
                    "tags": ["h"],
                    "metadata": {},
                    "kwargs": {},
                }
            ],
        ),
    ],
)
def test_decode_events(data: Any, expected: Any) -> None:
    """Test decoding events."""
    assert load_events(data) == expected


class SimpleEnum(Enum):
    a = "a"
    b = "b"


class SimpleModel(BaseModel):
    x: int
    y: SimpleEnum
    z: uuid.UUID
    dt: datetime.datetime
    d: datetime.date
    t: datetime.time


@pytest.mark.parametrize(
    "obj, expected",
    [
        ({"key": "value"}, {"key": "value"}),
        ([1, 2, 3], [1, 2, 3]),
        (123, 123),
        (uuid.UUID(int=1), "00000000-0000-0000-0000-000000000001"),
        (datetime.datetime(2020, 1, 1), "2020-01-01T00:00:00"),
        (datetime.date(2020, 1, 1), "2020-01-01"),
        (datetime.time(0, 0, 0), "00:00:00"),
        (datetime.time(0, 0, 0, 1), "00:00:00.000001"),
        (SimpleEnum.a, "a"),
        (
            SimpleModel(
                x=1,
                y=SimpleEnum.a,
                z=uuid.UUID(int=1),
                dt=datetime.datetime(2020, 1, 1),
                d=datetime.date(2020, 1, 1),
                t=datetime.time(0, 0, 0),
            ),
            {
                "x": 1,
                "y": "a",
                "z": "00000000-0000-0000-0000-000000000001",
                "dt": "2020-01-01T00:00:00",
                "d": "2020-01-01",
                "t": "00:00:00",
            },
        ),
        ("string", "string"),
        (True, True),
        (None, None),
    ],
)
def test_encoding_of_well_known_types(obj: Any, expected: str) -> None:
    """Test encoding of well known types.

    This tests verifies that our custom serializer is able to encode some well
    known types; e.g., uuid, datetime, date, time

    It doesn't handle types like Decimal or frozenset etc just yet while we determine
    how to roll out a more complete solution.
    """
    lc_serializer = WellKnownLCSerializer()
    assert lc_serializer.dumpd(obj) == expected
