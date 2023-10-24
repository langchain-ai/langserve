from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema import AgentAction, AgentFinish, BaseMessage, Document, LLMResult
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
