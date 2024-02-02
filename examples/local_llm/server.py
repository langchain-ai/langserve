#!/usr/bin/env python
"""Example LangChain Server that runs a local llm.

**Attention** This is OK for prototyping / dev usage, but should not be used
for production cases when there might be concurrent requests from different
users. As of the time of writing, Ollama is designed for single user and cannot
handle concurrent requests see this issue: 
https://github.com/ollama/ollama/issues/358

When deploying local models, make sure you understand whether the model is able
to handle concurrent requests or not. If concurrent requests are not handled
properly, the server will either crash or will just not be able to handle more
than one user at a time.
"""
from fastapi import FastAPI
from langchain_community.chat_models import ChatOllama

from langserve import add_routes

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)


llm = ChatOllama(model="llama2")

add_routes(
    app,
    llm,
    path="/ollama",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
