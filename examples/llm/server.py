#!/usr/bin/env python
"""Example LangChain server exposes multiple runnables (LLMs in this case)."""

from fastapi import FastAPI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from langserve import add_routes

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    ChatOpenAI(model="gpt-3.5-turbo-0125"),
    path="/openai",
)
add_routes(
    app,
    ChatAnthropic(model="claude-3-haiku-20240307"),
    path="/anthropic",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
