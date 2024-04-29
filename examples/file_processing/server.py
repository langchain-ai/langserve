#!/usr/bin/env python
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

from fastapi import FastAPI
from langchain.pydantic_v1 import Field
from langchain_community.document_loaders.parsers.pdf import PDFMinerParser
from langchain_core.document_loaders import Blob
from langchain_core.runnables import RunnableLambda

from langserve import CustomUserType, add_routes

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)


# ATTENTION: Inherit from CustomUserType instead of BaseModel otherwise
#            the server will decode it into a dict instead of a pydantic model.
class FileProcessingRequest(CustomUserType):
    """Request including a base64 encoded file."""

    # The extra field is used to specify a widget for the playground UI.
    file: str = Field(..., extra={"widget": {"type": "base64file"}})
    num_chars: int = 100


def _process_file(request: FileProcessingRequest) -> str:
    """Extract the text from the first page of the PDF."""
    content = base64.b64decode(request.file.encode("utf-8"))
    blob = Blob(data=content)
    documents = list(PDFMinerParser().lazy_parse(blob))
    content = documents[0].page_content
    return content[: request.num_chars]


add_routes(
    app,
    RunnableLambda(_process_file).with_types(input_type=FileProcessingRequest),
    config_keys=["configurable"],
    path="/pdf",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
