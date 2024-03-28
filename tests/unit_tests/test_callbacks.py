import uuid

from langchain_core.prompts import ChatPromptTemplate

from langserve.callbacks import AsyncEventAggregatorCallback, replace_uuids
from tests.unit_tests.utils.llms import FakeListLLM


async def test_event_aggregator() -> None:
    """Test that the event aggregator is aggregating events."""
    prompt = ChatPromptTemplate.from_template("{question}")
    llm = FakeListLLM(responses=["hello", "world"])

    chain = prompt | llm
    callback = AsyncEventAggregatorCallback()
    assert callback.callback_events == []
    assert chain.invoke({"question": "hello"}, {"callbacks": [callback]}) == "hello"
    callback_events = callback.callback_events
    assert isinstance(callback_events, list)
    assert len(callback_events) == 6
    assert [event["type"] for event in callback_events] == [
        "on_chain_start",
        "on_chain_start",
        "on_chain_end",
        "on_llm_start",
        "on_llm_end",
        "on_chain_end",
    ]


def test_replace_uuids() -> None:
    """Test replace uuids in place."""
    uuid1 = uuid.UUID(int=1)
    uuid2 = uuid.UUID(int=2)

    events = [
        {
            "type": "on_llm_start",
            "run_id": uuid1,
            "parent_run_id": None,
        },
        {
            "type": "on_llm_start",
            "run_id": uuid1,
            "parent_run_id": uuid2,
        },
    ]
    new_events = replace_uuids(events)
    # Assert original event is unchanged
    assert events[0]["run_id"] == uuid1
    assert isinstance(new_events, list)
    assert len(new_events) == 2

    assert new_events[0]["run_id"] != uuid1
    assert new_events[1]["run_id"] != uuid2
    assert new_events[0]["run_id"] == new_events[1]["run_id"]
    assert new_events[0]["run_id"] != new_events[1]["parent_run_id"]
    assert new_events[0]["parent_run_id"] is None
