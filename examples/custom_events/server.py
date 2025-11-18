"""
Allows the `/stream` endpoint to return `sse_starlette.ServerSentEvent` from runnable,
allowing you to return custom events such as `event: error`.
"""

from typing import Any, AsyncIterator, Dict

from fastapi import FastAPI
from langchain_core.runnables import RunnableConfig, RunnableLambda
from sse_starlette import ServerSentEvent

from langserve import add_routes
from langserve.pydantic_v1 import BaseModel

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)


class InputType(BaseModel): ...


class OutputType(BaseModel):
    message: str


async def error_event(
    _: InputType,
    config: RunnableConfig,
) -> AsyncIterator[Dict[str, Any] | ServerSentEvent]:
    for i in range(4):
        yield {
            "message": f"Message {i}",
        }

    is_streaming = False
    if "metadata" in config:
        metadata = config["metadata"]
        if "__langserve_endpoint" in metadata:
            is_streaming = metadata["__langserve_endpoint"] == "stream"

    if is_streaming:
        yield ServerSentEvent(
            data={
                "message": "An error occurred",
            },
            event="error",
        )
    else:
        yield {
            "message": "An error occurred",
        }


add_routes(
    app,
    RunnableLambda(error_event),
    input_type=InputType,
    output_type=OutputType,
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
