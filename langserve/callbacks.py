from __future__ import annotations

import uuid
from typing import List, Dict, Any, Optional, Sequence
from uuid import UUID

from langchain.callbacks.base import AsyncCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager, CallbackManager
from langchain.callbacks.manager import _ahandle_event, _handle_event
from langchain.schema import AgentAction, AgentFinish
from typing_extensions import TypedDict

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
        data: EventData = event["data"]

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


async def ahandle_callbacks(
    callback_manager: AsyncCallbackManager,
    parent_id: UUID,
    callback_events: Sequence[CallbackEvent],
) -> None:
    """Invoke all the callbacks."""
    replace_uuids_in_place(callback_events)

    # 1. Do I need inheritable handlers
    for callback_event in callback_events:
        data = callback_event["data"]
        if data["parent_run_id"] is None:  # How do we make sure it's None!?
            data["parent_run_id"] = parent_id
        event = callback_event
        event_name_to_ignore_condition = {
            "on_llm_start": "ignore_llm",
            "on_chat_model_start": "ignore_chat_model",
            "on_chain_start": "ignore_chain",
            "on_tool_start": "ignore_agent",
            "on_retriever_start": "ignore_retriever",
        }

        ignore_condition = event_name_to_ignore_condition.get(event["type"], None)

        await _ahandle_event(
            # Unpacking like this may not work
            callback_manager.handlers,
            event["type"],
            ignore_condition_name=ignore_condition,
            **event["data"],
        )


def handle_callbacks(
    callback_manager: CallbackManager,
    parent_id: UUID,
    callback_events: Sequence[CallbackEvent],
) -> None:
    """Invoke all the callbacks."""
    replace_uuids_in_place(callback_events)

    for callback_event in callback_events:
        data = callback_event["data"]
        if data["parent_run_id"] is None:  # How do we make sure it's None!?
            data["parent_run_id"] = parent_id

        event_name_to_ignore_condition = {
            "on_llm_start": "ignore_llm",
            "on_chat_model_start": "ignore_chat_model",
            "on_chain_start": "ignore_chain",
            "on_tool_start": "ignore_agent",
            "on_retriever_start": "ignore_retriever",
        }
        ignore_condition = event_name_to_ignore_condition.get(
            callback_event["type"], None
        )
        _handle_event(
            # Unpacking like this may not work
            callback_manager.handlers,
            callback_event["type"],
            ignore_condition_name=ignore_condition,
            **callback_event["data"],
        )


#
#
# def _sort_callback_events(events: List[CallbackEvent]) -> List[CallbackEvent]:
#     """Sort callback events by parent_uid and uid."""
#     child_to_parent = {
#         event["data"]["run_id"]: event["data"]["parent_run_id"] for event in events
#     }
#
#     parent_to_children = defaultdict(list)
#     for child, parent in child_to_parent.items():
#         parent_to_children[parent].append(child)
#
#     sorted_events = []
#
#     nodes_to_add = parent_to_children[None]
#
#     while nodes_to_add:
#         sorted_events.extend(nodes_to_add)
#         new_nodes_to_add = []
#
#     return sorted_events
