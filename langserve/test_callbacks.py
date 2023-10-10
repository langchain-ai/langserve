from langchain.callbacks.base import AsyncCallbackHandler
from typing import Dict, Any, Optional, List
from uuid import UUID
from langchain.schema.runnable import RunnableLambda
from langserve.server import CallbackEvent


class ServerEventSerializer(AsyncCallbackHandler):
    """Callback Handler that prints to std out."""
    def __int__(self) -> None:
        """Get a list of all the callback events that have been called."""
        super().__init__(self)
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
            CallbackEvent(
                "on_chain_start",
                serialized,
                inputs,
                run_id,
                parent_run_id,
                tags,
                metadata,
                **kwargs,
            )
        )


from langsmith.client import Client

def test_chain() -> None:
    """Test that we can run a chain."""

    def func(x):
        return x + 1

    r = RunnableLambda(func)

    callback = ServerEventSerializer()
    assert r.invoke(1, {"callbacks": [callback()]}) == 2
    callback.
