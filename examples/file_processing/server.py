"""Example that shows how to upload files and process files in the server.

This example uses a very simple architecture for dealing with file uploads
and processing.

The main issue with this approach is that processing is done in
the same process rather than offloaded to a process pool. A smaller
issue is that the base64 encoding incurs an additional encoding/decoding
overhead.

This example also specifies a "base64file" widget, which will create a widget
allowing one to upload a binary file using the langserve playground UI.
"""
import base64
from typing import Any, Dict

from fastapi import FastAPI
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


class FileProcessingRequest(BaseModel):
    """Request including a base64 encoded file."""

    file: bytes = Field(..., extra={"widget": {"type": "base64file"}})
    first_num_chars: int = Field(
        default=100,
        description="Will extract up to this number of characters from the first page.",
    )


def _process_file(thingy: Dict[str, Any]) -> str:
    """Extract the text from the first page of the PDF."""
    content = base64.decodebytes(thingy["file"])
    blob = Blob(data=content)
    documents = list(PDFMinerParser().lazy_parse(blob))
    content = documents[0].page_content
    return content[: input["num_chars"]]


add_routes(
    app,
    RunnableLambda(_process_file).with_types(input_type=FileProcessingRequest),
    config_keys=["configurable"],
    path="/pdf",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
