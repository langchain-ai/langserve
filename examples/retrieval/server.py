#!/usr/bin/env python
"""Example LangChain server exposes a retriever."""
from fastapi import FastAPI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from langserve import add_routes

vectorstore = FAISS.from_texts(
    ["cats like fish", "dogs like sticks"], embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)
# Adds routes to the app for using the retriever under:
# /invoke
# /batch
# /stream
add_routes(app, retriever)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
