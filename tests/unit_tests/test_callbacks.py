import uuid

from langserve.callbacks import replace_uuids_in_place

# @pytest.mark.asyncio
# async def test_chain() -> None:
#     """Test that we can run a chain."""
#
#     from langchain.llms import FakeListLLM
#     from langchain.prompts import ChatPromptTemplate
#
#     prompt = ChatPromptTemplate.from_template("{question}")
#     llm = FakeListLLM(responses=["hello", "world"])
#
#     chain = prompt | llm
#     callback = EventAggregatorHandler()
#     assert chain.invoke({"question": "hello"}, {"callbacks": [callback]}) == "hello"
#     serializer = CallbackEventSerializer()
#     assert serializer.dumpd(callback.callback_events) == []


# Unit tests using pytest


def test_replace_uuids() -> None:
    """Test replace uuids in place."""
    uuid1 = uuid.UUID(int=1)
    uuid2 = uuid.UUID(int=2)

    events = [
        {
            "type": "on_llm_start",
            "data": {
                "run_id": uuid1,
                "parent_run_id": None,
            },
        },
        {
            "type": "on_llm_start",
            "data": {
                "run_id": uuid1,
                "parent_run_id": uuid2,
            },
        },
    ]
    replace_uuids_in_place(events)
    assert isinstance(events, list)
    assert len(events) == 2
    assert events[0]["data"]["run_id"] != uuid1
    assert events[0]["data"]["run_id"] == events[1]["data"]["run_id"]
    assert events[0]["data"]["run_id"] != events[1]["data"]["parent_run_id"]
    assert events[0]["data"]["parent_run_id"] is None
