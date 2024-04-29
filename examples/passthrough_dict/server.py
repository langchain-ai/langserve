#!/usr/bin/env python
"""Example LangChain server passes through some of the inputs in the response."""

from typing import Any, Callable, Dict, List, Optional, TypedDict

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

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

underlying_chain = prompt | model

wrapped_chain = RunnableParallel(
    {
        "output": _create_projection(exclude_keys=["info"]) | underlying_chain,
        "info": _create_projection(include_keys=["info"]),
    }
)


class Input(TypedDict):
    thing: str
    language: str
    info: Dict[str, Any]


class Output(TypedDict):
    output: underlying_chain.output_schema
    info: Dict[str, Any]


add_routes(
    app, wrapped_chain.with_types(input_type=Input, output_type=Output), path="/v1"
)


# Version 2
# Uses RunnablePassthrough.assign
wrapped_chain_2 = RunnablePassthrough.assign(output=underlying_chain) | {
    "output": lambda x: x["output"],
    "info": lambda x: x["info"],
}

add_routes(
    app,
    wrapped_chain_2.with_types(input_type=Input, output_type=Output),
    path="/v2",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
