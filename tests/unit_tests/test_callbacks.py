import pytest

from langserve.callbacks import EventAggregatorHandler
from langserve.serialization import CallbackEventSerializer


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
