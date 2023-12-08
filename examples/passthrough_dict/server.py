#!/usr/bin/env python
"""Example LangChain server exposes multiple runnables (LLMs in this case)."""

from typing import Any, Callable, Dict, List, Optional, TypedDict

from fastapi import FastAPI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap

from langserve import add_routes

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)


def _create_projection(
    *, include_keys: Optional[List] = None, exclude_keys: Optional[List[str]] = None
) -> Callable[[dict], dict]:
    """Create a projection function."""

    def _project_dict(
        d: dict,
    ) -> dict:
        """Project dictionary."""
        keys = d.keys()
        if include_keys is not None:
            keys = set(keys) & set(include_keys)
        if exclude_keys is not None:
            keys = set(keys) - set(exclude_keys)
        return {k: d[k] for k in keys}

    return _project_dict


prompt = ChatPromptTemplate.from_messages(
    [("human", "translate `{thing}` to {language}")]
)
model = ChatOpenAI()

chain = prompt | model

wrapped_chain = RunnableMap(
    {
        "output": _create_projection(exclude_keys=["info"]) | chain,
        "info": _create_projection(include_keys=["info"]),
    }
)


class Input(TypedDict):
    thing: str
    language: str
    info: Dict[str, Any]


class Output(TypedDict):
    output: chain.output_schema
    info: Any


add_routes(
    app,
    wrapped_chain.with_types(input_type=Input, output_type=Output),
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
