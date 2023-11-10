#!/usr/bin/env python
"""Example LangChain server exposes a conversational retrieval chain."""
from typing import List, Tuple

from fastapi import FastAPI
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.pydantic_v1 import BaseModel, Field
from langchain.vectorstores import FAISS

from langserve import add_routes

vectorstore = FAISS.from_texts(
    ["cats like fish", "dogs like sticks"], embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

model = ChatOpenAI()

chain = ConversationalRetrievalChain.from_llm(model, retriever)


# User input
class ChatHistory(BaseModel):
    """Chat history with the bot."""

    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question", "output": "answer"}},
    )
    question: str


chain = ConversationalRetrievalChain.from_llm(model, retriever).with_types(
    input_type=ChatHistory
)

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)
# Adds routes to the app for using the chain under:
# /invoke
# /batch
# /stream
add_routes(app, chain, enable_feedback_endpoint=True)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
