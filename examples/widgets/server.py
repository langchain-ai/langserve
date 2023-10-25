import base64
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.document_loaders.blob_loaders import Blob
from langchain.document_loaders.parsers.pdf import PDFMinerParser
from langchain.schema.runnable import RunnableLambda
from pydantic import BaseModel, Field

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


class FileProcessingRequest(BaseModel):
    file: bytes = Field(..., extra={"widget": {"type": "base64file"}})
    num_chars: int = 100


def chat_with_bot(input: Dict[str, Any]) -> Dict[str, Any]:
    """Bot that repeats the question twice."""
    return {
        "answer": input["question"] * 2,
        "woof": "its so bad to woof, meow is better",
    }


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
