from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

from langchain.callbacks.base import AsyncCallbackHandler
from langchain.callbacks.manager import (
    BaseRunManager,
    _ahandle_event,
    _handle_event,
)
from langchain.schema import AgentAction, AgentFinish, LLMResult, Document

from langserve.schema import CallbackEvent


class AsyncEventAggregatorCallback(AsyncCallbackHandler):
    """A callback handler that aggregates all the events that have been called.

    This callback handler aggregates all the events that have been called, placing
    them in a single mutable list.

    This callback handler is not thread-safe and is meant to be used in an async
    context only.
    """

    def __init__(self, events_to_intercept: Optional[Sequence[str]] = None) -> None:
        """Initialize the callback handler with an empty list."""
        super().__init__()
        self.callback_events: List[Dict[str, Any]] = []

    async def log_event(self, event_name: str, **kwargs: Any) -> None:
        """Log a generic event with its name and data."""
        event_data = {"name": event_name, "data": kwargs}
        self.callback_events.append(event_data)

    def __getattr__(self, name: str) -> Any:
        """Override __getattr__ to handle events dynamically."""
        if not name.startswith("on_"):
            raise AttributeError(f"Event {name} not found.")

        async def generic_event_handler(**kwargs: Any) -> None:
            await self.log_event(name, **kwargs)

        return generic_event_handler


#
# class AsyncEventAggregatorCallback(AsyncCallbackHandler):
#     """A callback handler that aggregates all the events that have been called.
#
#     This callback handler aggregates all the events that have been called placing
#     them in a single mutable list.
#
#     This callback handler is not threading safe, and is meant to be used in an async
#     context only.
#     """
#
#     def __init__(self) -> None:
#         """Get a list of all the callback events that have been called."""
#         super().__init__()
#         # Callback events is a mutable state that is used only in an async context,
#         # so it should be safe to mutate without the usage of a lock.
#         self.callback_events: List[CallbackEvent] = []
#
#     async def on_chain_start(
#         self,
#         serialized: Dict[str, Any],
#         inputs: Dict[str, Any],
#         *,
#         run_id: UUID,
#         parent_run_id: Optional[UUID] = None,
#         tags: Optional[List[str]] = None,
#         metadata: Optional[Dict[str, Any]] = None,
#         **kwargs: Any,
#     ) -> None:
#         """Attempt to serialize the callback event."""
#         self.callback_events.append(
#             {
#                 "type": "on_chain_start",
#                 "data": {
#                     "serialized": serialized,
#                     "inputs": inputs,
#                     "run_id": run_id,
#                     "parent_run_id": parent_run_id,
#                     "tags": tags,
#                     "metadata": metadata,
#                     "kwargs": kwargs,
#                 },
#             }
#         )
#
#     async def on_chain_end(
#         self,
#         outputs: Dict[str, Any],
#         *,
#         run_id: UUID,
#         parent_run_id: Optional[UUID] = None,
#         tags: Optional[List[str]] = None,
#         **kwargs: Any,
#     ) -> None:
#         self.callback_events.append(
#             {
#                 "type": "on_chain_end",
#                 "data": {
#                     "outputs": outputs,
#                     "run_id": run_id,
#                     "parent_run_id": parent_run_id,
#                     "tags": tags,
#                     "kwargs": kwargs,
#                 },
#             }
#         )
#
#     async def on_chain_error(
#         self,
#         error: BaseException,
#         *,
#         run_id: UUID,
#         parent_run_id: Optional[UUID] = None,
#         tags: Optional[List[str]] = None,
#         **kwargs: Any,
#     ) -> None:
#         self.callback_events.append(
#             {
#                 "type": "on_chain_error",
#                 "data": {
#                     "error": error,
#                     "run_id": run_id,
#                     "parent_run_id": parent_run_id,
#                     "tags": tags,
#                     "kwargs": kwargs,
#                 },
#             }
#         )
#
#     async def on_retriever_start(
#         self,
#         serialized: Dict[str, Any],
#         query: str,
#         *,
#         run_id: UUID,
#         parent_run_id: Optional[UUID] = None,
#         tags: Optional[List[str]] = None,
#         metadata: Optional[Dict[str, Any]] = None,
#         **kwargs: Any,
#     ) -> None:
#         self.callback_events.append(
#             {
#                 "type": "on_retriever_start",
#                 "data": {
#                     "serialized": serialized,
#                     "query": query,
#                     "run_id": run_id,
#                     "parent_run_id": parent_run_id,
#                     "tags": tags,
#                     "metadata": metadata,
#                     "kwargs": kwargs,
#                 },
#             }
#         )
#
#     async def on_retriever_end(
#         self,
#         documents: Sequence[Document],
#         *,
#         run_id: UUID,
#         parent_run_id: Optional[UUID] = None,
#         tags: Optional[List[str]] = None,
#         **kwargs: Any,
#     ) -> None:
#         self.callback_events.append(
#             {
#                 "type": "on_retriever_end",
#                 "data": {
#                     "documents": documents,
#                     "run_id": run_id,
#                     "parent_run_id": parent_run_id,
#                     "tags": tags,
#                     "kwargs": kwargs,
#                 },
#             }
#         )
#
#     async def on_retriever_error(
#         self,
#         error: BaseException,
#         *,
#         run_id: UUID,
#         parent_run_id: Optional[UUID] = None,
#         tags: Optional[List[str]] = None,
#         **kwargs: Any,
#     ) -> None:
#         self.callback_events.append(
#             {
#                 "type": "on_retriever_error",
#                 "data": {
#                     "error": error,
#                     "run_id": run_id,
#                     "parent_run_id": parent_run_id,
#                     "tags": tags,
#                     "kwargs": kwargs,
#                 },
#             }
#         )
#
#     async def on_tool_start(
#         self,
#         serialized: Dict[str, Any],
#         input_str: str,
#         *,
#         run_id: UUID,
#         parent_run_id: Optional[UUID] = None,
#         tags: Optional[List[str]] = None,
#         metadata: Optional[Dict[str, Any]] = None,
#         **kwargs: Any,
#     ) -> None:
#         self.callback_events.append(
#             {
#                 "type": "on_tool_start",
#                 "data": {
#                     "serialized": serialized,
#                     "input_str": input_str,
#                     "run_id": run_id,
#                     "parent_run_id": parent_run_id,
#                     "tags": tags,
#                     "metadata": metadata,
#                     "kwargs": kwargs,
#                 },
#             }
#         )
#
#     async def on_tool_end(
#         self,
#         output: str,
#         *,
#         run_id: UUID,
#         parent_run_id: Optional[UUID] = None,
#         tags: Optional[List[str]] = None,
#         **kwargs: Any,
#     ) -> None:
#         self.callback_events.append(
#             {
#                 "type": "on_tool_end",
#                 "data": {
#                     "output": output,
#                     "run_id": run_id,
#                     "parent_run_id": parent_run_id,
#                     "tags": tags,
#                     "kwargs": kwargs,
#                 },
#             }
#         )
#
#     async def on_tool_error(
#         self,
#         error: BaseException,
#         *,
#         run_id: UUID,
#         parent_run_id: Optional[UUID] = None,
#         tags: Optional[List[str]] = None,
#         **kwargs: Any,
#     ) -> None:
#         self.callback_events.append(
#             {
#                 "type": "on_tool_error",
#                 "data": {
#                     "error": error,
#                     "run_id": run_id,
#                     "parent_run_id": parent_run_id,
#                     "tags": tags,
#                     "kwargs": kwargs,
#                 },
#             }
#         )
#
#     async def on_agent_action(
#         self,
#         action: AgentAction,
#         *,
#         run_id: UUID,
#         parent_run_id: Optional[UUID] = None,
#         tags: Optional[List[str]] = None,
#         **kwargs: Any,
#     ) -> None:
#         self.callback_events.append(
#             {
#                 "type": "on_agent_action",
#                 "data": {
#                     "action": action,
#                     "run_id": run_id,
#                     "parent_run_id": parent_run_id,
#                     "tags": tags,
#                     "kwargs": kwargs,
#                 },
#             }
#         )
#
#     async def on_agent_finish(
#         self,
#         finish: AgentFinish,
#         *,
#         run_id: UUID,
#         parent_run_id: Optional[UUID] = None,
#         tags: Optional[List[str]] = None,
#         **kwargs: Any,
#     ) -> None:
#         self.callback_events.append(
#             {
#                 "type": "on_agent_finish",
#                 "data": {
#                     "finish": finish,
#                     "run_id": run_id,
#                     "parent_run_id": parent_run_id,
#                     "tags": tags,
#                     "kwargs": kwargs,
#                 },
#             }
#         )
#
#     async def on_llm_start(
#         self,
#         serialized: Dict[str, Any],
#         prompts: List[str],
#         *,
#         run_id: UUID,
#         parent_run_id: Optional[UUID] = None,
#         tags: Optional[List[str]] = None,
#         metadata: Optional[Dict[str, Any]] = None,
#         **kwargs: Any,
#     ) -> None:
#         self.callback_events.append(
#             {
#                 "type": "on_llm_start",
#                 "data": {
#                     "serialized": serialized,
#                     "prompts": prompts,
#                     "run_id": run_id,
#                     "parent_run_id": parent_run_id,
#                     "tags": tags,
#                     "metadata": metadata,
#                     "kwargs": kwargs,
#                 },
#             }
#         )
#
#     async def on_llm_end(
#         self,
#         response: LLMResult,
#         *,
#         run_id: UUID,
#         parent_run_id: Optional[UUID] = None,
#         tags: Optional[List[str]] = None,
#         **kwargs: Any,
#     ) -> None:
#         self.callback_events.append(
#             {
#                 "type": "on_llm_end",
#                 "data": {
#                     "response": response,
#                     "run_id": run_id,
#                     "parent_run_id": parent_run_id,
#                     "tags": tags,
#                     "kwargs": kwargs,
#                 },
#             }
#         )
#
#     async def on_llm_error(
#         self,
#         error: BaseException,
#         *,
#         run_id: UUID,
#         parent_run_id: Optional[UUID] = None,
#         tags: Optional[List[str]] = None,
#         **kwargs: Any,
#     ) -> None:
#         self.callback_events.append(
#             {
#                 "type": "on_llm_error",
#                 "data": {
#                     "error": error,
#                     "run_id": run_id,
#                     "parent_run_id": parent_run_id,
#                     "tags": tags,
#                     "kwargs": kwargs,
#                 },
#             }
#         )


def replace_uuids_in_place(callback_events: Sequence[CallbackEvent]) -> None:
    """Replace uids in the event callbacks with new uids.

    This function mutates the event callback events in place.

    Args:
        callback_events: A list of event callbacks.
    """
    # Create a dictionary to store mappings from old UID to new UID
    uid_mapping: dict = {}

    # Iterate through the list of event callbacks
    for event in callback_events:
        data = event["data"]

        # Replace UIDs in the 'run_id' field
        if "run_id" in data and data["run_id"] is not None:
            if data["run_id"] not in uid_mapping:
                # Generate a new UUID
                new_uid = uuid.uuid4()
                uid_mapping[data["run_id"]] = new_uid
            # Replace the old UID with the new one
            data["run_id"] = uid_mapping[data["run_id"]]

        # Replace UIDs in the 'parent_run_id' field if it's not None
        if "parent_run_id" in data and data["parent_run_id"] is not None:
            if data["parent_run_id"] not in uid_mapping:
                # Generate a new UUID
                new_uid = uuid.uuid4()
                uid_mapping[data["parent_run_id"]] = new_uid
            # Replace the old UID with the new one
            data["parent_run_id"] = uid_mapping[data["parent_run_id"]]


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
    callback_events: Sequence[CallbackEvent],
) -> None:
    """Invoke all the callback handlers with the given callback events."""
    replace_uuids_in_place(callback_events)

    # 1. Do I need inheritable handlers
    for callback_event in callback_events:
        event_name = callback_event["name"]
        data = callback_event["data"]
        if data["parent_run_id"] is None:  # How do we make sure it's None!?
            data["parent_run_id"] = callback_manager.run_id

        await _ahandle_event(
            # Unpacking like this may not work
            callback_manager.handlers,
            event_name,
            ignore_condition_name=NAME_TO_IGNORE_CONDITION.get(event_name, None),
            **data,
        )


def handle_callbacks(
    callback_manager: BaseRunManager,
    callback_events: Sequence[CallbackEvent],
) -> None:
    """Invoke all the callback handlers with the given callback events."""
    replace_uuids_in_place(callback_events)

    for callback_event in callback_events:
        name = callback_event["name"]
        data = callback_event["data"]
        if data["parent_run_id"] is None:  # How do we make sure it's None!?
            data["parent_run_id"] = callback_manager.run_id

        _handle_event(
            # Unpacking like this may not work
            callback_manager.handlers,
            name,
            ignore_condition_name=NAME_TO_IGNORE_CONDITION.get(name, None),
            **data,
        )
