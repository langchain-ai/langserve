from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.callbacks.manager import (
    BaseRunManager,
    ahandle_event,
    handle_event,
)
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from typing_extensions import TypedDict


class CallbackEventDict(TypedDict, total=False):
    """A dictionary representation of a callback event."""

    type: str
    parent_run_id: Optional[UUID]
    run_id: UUID


class AsyncEventAggregatorCallback(AsyncCallbackHandler):
    """A callback handler that aggregates all the events that have been called.

    This callback handler aggregates all the events that have been called placing
    them in a single mutable list.

    This callback handler is not threading safe, and is meant to be used in an async
    context only.
    """

    def __init__(self) -> None:
        """Get a list of all the callback events that have been called."""
        super().__init__()
        # Callback events is a mutable state that is used only in an async context,
        # so it should be safe to mutate without the usage of a lock.
        self.callback_events: List[CallbackEventDict] = []

    def log_callback(self, event: CallbackEventDict) -> None:
        """Log the callback event."""
        self.callback_events.append(event)

    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Attempt to serialize the callback event."""
        self.log_callback(
            {
                "type": "on_chat_model_start",
                "serialized": serialized,
                "messages": messages,
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "tags": tags,
                "metadata": metadata,
                "kwargs": kwargs,
            }
        )

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
        self.log_callback(
            {
                "type": "on_chain_start",
                "serialized": serialized,
                "inputs": inputs,
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "tags": tags,
                "metadata": metadata,
                "kwargs": kwargs,
            }
        )

    async def on_chain_end(
        self,
        outputs: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        self.log_callback(
            {
                "type": "on_chain_end",
                "outputs": outputs,
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "tags": tags,
                "kwargs": kwargs,
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
        self.log_callback(
            {
                "type": "on_chain_error",
                "error": error,
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "tags": tags,
                "kwargs": kwargs,
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
        self.log_callback(
            {
                "type": "on_retriever_start",
                "serialized": serialized,
                "query": query,
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "tags": tags,
                "metadata": metadata,
                "kwargs": kwargs,
            }
        )

    async def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        self.log_callback(
            {
                "type": "on_retriever_end",
                "documents": documents,
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "tags": tags,
                "kwargs": kwargs,
            }
        )

    async def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        self.log_callback(
            {
                "type": "on_retriever_error",
                "error": error,
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "tags": tags,
                "kwargs": kwargs,
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
        self.log_callback(
            {
                "type": "on_tool_start",
                "serialized": serialized,
                "input_str": input_str,
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "tags": tags,
                "metadata": metadata,
                "kwargs": kwargs,
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
        self.log_callback(
            {
                "type": "on_tool_end",
                "output": output,
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "tags": tags,
                "kwargs": kwargs,
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
        self.log_callback(
            {
                "type": "on_tool_error",
                "error": error,
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "tags": tags,
                "kwargs": kwargs,
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
        self.log_callback(
            {
                "type": "on_agent_action",
                "action": action,
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "tags": tags,
                "kwargs": kwargs,
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
        self.log_callback(
            {
                "type": "on_agent_finish",
                "finish": finish,
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "tags": tags,
                "kwargs": kwargs,
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
        self.log_callback(
            {
                "type": "on_llm_start",
                "serialized": serialized,
                "prompts": prompts,
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "tags": tags,
                "metadata": metadata,
                "kwargs": kwargs,
            }
        )

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        self.log_callback(
            {
                "type": "on_llm_end",
                "response": response,
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "tags": tags,
                "kwargs": kwargs,
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
        self.log_callback(
            {
                "type": "on_llm_error",
                "error": error,
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "tags": tags,
                "kwargs": kwargs,
            }
        )


def replace_uuids(
    callback_events: Sequence[CallbackEventDict],
) -> List[CallbackEventDict]:
    """Replace uids in the event callbacks with new uids.

    This function mutates the event callback events in place.

    Args:
        callback_events: A list of event callbacks.
    """
    # Create a dictionary to store mappings from old UID to new UID
    uid_mapping: dict = {}

    updated_events = []

    # Iterate through the list of event callbacks
    for event in callback_events:
        updated_event = event.copy()
        # Replace UIDs in the 'run_id' field
        if "run_id" in updated_event and updated_event["run_id"] is not None:
            if updated_event["run_id"] not in uid_mapping:
                # Generate a new UUID
                new_uid = uuid.uuid4()
                uid_mapping[updated_event["run_id"]] = new_uid
            # Replace the old UID with the new one
            updated_event["run_id"] = uid_mapping[updated_event["run_id"]]

        # Replace UIDs in the 'parent_run_id' field if it's not None
        if (
            "parent_run_id" in updated_event
            and updated_event["parent_run_id"] is not None
        ):
            if updated_event["parent_run_id"] not in uid_mapping:
                # Generate a new UUID
                new_uid = uuid.uuid4()
                uid_mapping[updated_event["parent_run_id"]] = new_uid
            # Replace the old UID with the new one
            updated_event["parent_run_id"] = uid_mapping[updated_event["parent_run_id"]]
        updated_events.append(updated_event)
    return updated_events


# Mapping from event name to ignore condition name
NAME_TO_IGNORE_CONDITION = {
    "on_retry": "ignore_retry",
    "on_text": None,
    "on_agent_action": "ignore_agent",
    "on_agent_finish": "ignore_agent",
    "on_llm_start": "ignore_llm",
    "on_llm_end": "ignore_llm",
    "on_llm_error": "ignore_llm",
    "on_chain_start": "ignore_chain",
    "on_chain_end": "ignore_chain",
    "on_chain_error": "ignore_chain",
    "on_chat_model_start": "ignore_chat_model",
    "on_tool_start": "ignore_agent",
    "on_tool_end": "ignore_agent",
    "on_tool_error": "ignore_agent",
    "on_retriever_start": "ignore_retriever",
    "on_retriever_end": "ignore_retriever",
    "on_retriever_error": "ignore_retriever",
}


async def ahandle_callbacks(
    callback_manager: BaseRunManager,
    callback_events: Sequence[CallbackEventDict],
) -> None:
    """Invoke all the callback handlers with the given callback events."""
    callback_events = replace_uuids(callback_events)

    # 1. Do I need inheritable handlers
    for event in callback_events:
        if event["parent_run_id"] is None:  # How do we make sure it's None!?
            event["parent_run_id"] = callback_manager.run_id

        event_data = {key: value for key, value in event.items() if key != "type"}

        await ahandle_event(
            # Unpacking like this may not work
            callback_manager.handlers,
            event["type"],
            ignore_condition_name=NAME_TO_IGNORE_CONDITION.get(event["type"], None),
            **event_data,
        )


def handle_callbacks(
    callback_manager: BaseRunManager,
    callback_events: Sequence[CallbackEventDict],
) -> None:
    """Invoke all the callback handlers with the given callback events."""
    callback_events = replace_uuids(callback_events)

    for event in callback_events:
        if event["parent_run_id"] is None:  # How do we make sure it's None!?
            event["parent_run_id"] = callback_manager.run_id

        event_data = {key: value for key, value in event.items() if key != "type"}

        handle_event(
            # Unpacking like this may not work
            callback_manager.handlers,
            event["type"],
            ignore_condition_name=NAME_TO_IGNORE_CONDITION.get(event["type"], None),
            **event_data,
        )
