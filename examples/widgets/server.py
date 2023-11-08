#!/usr/bin/env python
"""Endpoint shows off available playground widgets."""
import base64
from json import dumps
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.document_loaders.blob_loaders import Blob
from langchain.document_loaders.parsers.pdf import PDFMinerParser
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
)
from langchain.schema.runnable import RunnableLambda

from langserve.server import add_routes

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)


# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


class ChatHistory(BaseModel):
    chat_history: List[Tuple[str, str]] = Field(
        ...,
        examples=[[("a", "aa")]],
        extra={"widget": {"type": "chat", "input": "question", "output": "answer"}},
    )
    question: str


class ChatHistoryMessage(BaseModel):
    chat_history: List[BaseMessage] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "location", "output": "output"}},
    )
    location: str


class FileProcessingRequest(BaseModel):
    file: bytes = Field(..., extra={"widget": {"type": "base64file"}})
    num_chars: int = 100


def chat_with_bot(input: Dict[str, Any]) -> Dict[str, Any]:
    """Bot that repeats the question twice."""
    return {
        "answer": input["question"] * 2,
        "woof": "its so bad to woof, meow is better",
    }


def chat_message_bot(input: Dict[str, Any]) -> List[BaseMessage]:
    """Bot that repeats the question twice."""
    return [
        AIMessage(
            content="",
            additional_kwargs={
                "function_call": {
                    "name": "get_weather",
                    "arguments": dumps({"location": input["location"]}),
                }
            },
        ),
        FunctionMessage(name="get_weather", content='{"value": 32}'),
        AIMessage(content=f"Weather in {input['location']}: 32"),
    ]


def process_file(input: Dict[str, Any]) -> str:
    """Extract the text from the first page of the PDF."""
    content = base64.decodebytes(input["file"])
    blob = Blob(data=content)
    documents = list(PDFMinerParser().lazy_parse(blob))
    content = documents[0].page_content
    return content[: input["num_chars"]]


add_routes(
    app,
    RunnableLambda(chat_with_bot).with_types(input_type=ChatHistory),
    config_keys=["configurable"],
    path="/chat",
)

add_routes(
    app,
    RunnableLambda(process_file).with_types(input_type=FileProcessingRequest),
    config_keys=["configurable"],
    path="/pdf",
)

add_routes(
    app,
    RunnableLambda(chat_message_bot).with_types(input_type=ChatHistoryMessage),
    config_keys=["configurable"],
    path="/chat_message",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
