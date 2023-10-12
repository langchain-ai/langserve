from typing import Dict, Any, Optional, List
from uuid import UUID

from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema import AgentAction, AgentFinish
from langchain.schema.runnable import RunnableLambda

from langserve.schema import CallbackEvent
from langserve.serialization import CallbackEventSerializer


class EventAggregatorHandler(AsyncCallbackHandler):
    """Callback Handler that prints to std out."""

    def __init__(self) -> None:
        """Get a list of all the callback events that have been called."""
        self.callback_events: List[CallbackEvent] = []

    async def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Attempt to serialize the callback event."""
        self.callback_events.append(
            {
                "type": "on_chain_start",
                "data": {
                    "serialized": serialized,
                    "inputs": inputs,
                    "run_id": run_id,
                    "parent_run_id": parent_run_id,
                    "tags": tags,
                    "metadata": metadata,
                    "kwargs": kwargs,
                },
            }
        )

    async def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        self.callback_events.append(
            {
                "type": "on_chain_end",
                "data": {
                    "outputs": outputs,
                    "run_id": run_id,
                    "parent_run_id": parent_run_id,
                    "tags": tags,
                    "kwargs": kwargs,
                },
            }
        )

    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        self.callback_events.append(
            {
                "type": "on_chain_error",
                "data": {
                    "error": error,
                    "run_id": run_id,
                    "parent_run_id": parent_run_id,
                    "tags": tags,
                    "kwargs": kwargs,
                },
            }
        )

    async def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.callback_events.append(
            {
                "type": "on_retriever_start",
                "data": {
                    "serialized": serialized,
                    "query": query,
                    "run_id": run_id,
                    "parent_run_id": parent_run_id,
                    "tags": tags,
                    "metadata": metadata,
                    "kwargs": kwargs,
                },
            }
        )

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.callback_events.append(
            {
                "type": "on_tool_start",
                "data": {
                    "serialized": serialized,
                    "input_str": input_str,
                    "run_id": run_id,
                    "parent_run_id": parent_run_id,
                    "tags": tags,
                    "metadata": metadata,
                    "kwargs": kwargs,
                },
            }
        )

    async def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        self.callback_events.append(
            {
                "type": "on_tool_end",
                "data": {
                    "output": output,
                    "run_id": run_id,
                    "parent_run_id": parent_run_id,
                    "tags": tags,
                    "kwargs": kwargs,
                },
            }
        )

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        self.callback_events.append(
            {
                "type": "on_tool_error",
                "data": {
                    "error": error,
                    "run_id": run_id,
                    "parent_run_id": parent_run_id,
                    "tags": tags,
                    "kwargs": kwargs,
                },
            }
        )

    async def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        self.callback_events.append(
            {
                "type": "on_agent_action",
                "data": {
                    "action": action,
                    "run_id": run_id,
                    "parent_run_id": parent_run_id,
                    "tags": tags,
                    "kwargs": kwargs,
                },
            }
        )

    async def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        self.callback_events.append(
            {
                "type": "on_agent_finish",
                "data": {
                    "finish": finish,
                    "run_id": run_id,
                    "parent_run_id": parent_run_id,
                    "tags": tags,
                    "kwargs": kwargs,
                },
            }
        )

    async def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.callback_events.append(
            {
                "type": "on_llm_start",
                "data": {
                    "serialized": serialized,
                    "prompts": prompts,
                    "run_id": run_id,
                    "parent_run_id": parent_run_id,
                    "tags": tags,
                    "metadata": metadata,
                    "kwargs": kwargs,
                },
            }
        )

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        self.callback_events.append(
            {
                "type": "on_llm_error",
                "data": {
                    "error": error,
                    "run_id": run_id,
                    "parent_run_id": parent_run_id,
                    "tags": tags,
                    "kwargs": kwargs,
                },
            }
        )


def test_chain() -> None:
    """Test that we can run a chain."""

    from langchain.llms import FakeListLLM
    from langchain.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template("{question}")
    llm = FakeListLLM(responses=["hello", "world"])

    chain = prompt | llm
    callback = EventAggregatorHandler()
    assert chain.invoke({"question": "hello"}, {"callbacks": [callback]}) == "hello"
    serializer = CallbackEventSerializer()
    assert serializer.dumpd(callback.callback_events) == []
