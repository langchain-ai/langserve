from __future__ import annotations

from typing import List, Dict, Any, Optional, Sequence
from uuid import UUID

from langchain.callbacks.base import AsyncCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager, CallbackManager
from langchain.schema import AgentAction, AgentFinish

from langserve.schema import CallbackEvent


class EventAggregatorHandler(AsyncCallbackHandler):
    """A callback handler that aggregates all the events that have been called."""

    def __init__(self) -> None:
        """Get a list of all the callback events that have been called."""
        super().__init__()
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


async def ahandle_callbacks(
    callback_manager: AsyncCallbackManager,
    callback_events: Sequence[CallbackEvent],
) -> None:
    """Invoke all the callbacks."""
    # 1. Do I need inheritable handlers
    for callback_event in callback_events:
        event_type = callback_event["type"]
        data = callback_event["data"]
        if data["parent_run_id"] is None:  # How do we make sure it's None!?
            data["parent_run_id"] = callback_manager.parent_run_id

        for handler in callback_manager.handlers:
            if event_type == "on_tool_start":
                await handler.on_tool_start(**data)
            elif event_type == "on_tool_end":
                await handler.on_tool_end(**data)
            elif event_type == "on_chain_start":
                await handler.on_chain_start(**data)
            elif event_type == "on_chain_end":
                await handler.on_chain_end(**data)
            elif event_type == "on_chain_error":
                await handler.on_chain_error(**data)
            elif event_type == "on_llm_start":
                await handler.on_llm_start(**data)
            else:
                # Not handled yet
                pass


def handle_callbacks(
    callback_manager: CallbackManager,
    run_id: UUID,
    callback_events: Sequence[CallbackEvent],
) -> None:
    """Invoke all the callbacks."""
    new_callback_events = []
    for callback_event in callback_events:
        event_type = callback_event["type"]
        data = callback_event["data"]
        if data["parent_run_id"] is None:  # How do we make sure it's None!?
            data["parent_run_id"] = run_id
        event = callback_event

        # print("Event type:", event_type)
        # print("parent: ", event["data"]["parent_run_id"])
        # print("current:", event["data"]["run_id"])
        #
        new_callback_events.append(callback_event)

        for handler in (
            callback_manager.handlers
        ):
            if event_type == "on_tool_start":
                handler.on_tool_start(**data)
            elif event_type == "on_tool_end":
                handler.on_tool_end(**data)
            elif event_type == "on_chain_start":
                handler.on_chain_start(**data)
            elif event_type == "on_chain_end":
                handler.on_chain_end(**data)
            elif event_type == "on_chain_error":
                handler.on_chain_error(**data)
            elif event_type == "on_llm_start":
                handler.on_llm_start(**data)
            else:
                # Not handled yet
                pass

    return new_callback_events
