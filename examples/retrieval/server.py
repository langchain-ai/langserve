#!/usr/bin/env python
"""A more complex example that shows"""
from typing import Optional

from fastapi import FastAPI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import Runnable, RunnableConfig
from langchain.schema.runnable.utils import Input, Output
from langchain.vectorstores import FAISS, VectorStore

vectorstore1 = FAISS.from_texts(
    ["cats like fish", "dogs like sticks"], embedding=OpenAIEmbeddings()
)

vectorstore2 = FAISS.from_texts(["x_n+1=a * xn * (1-xn)"], embedding=OpenAIEmbeddings())


class FakeVectorstore(VectorStore):
    """This is a fake vectorstore for demo purposes."""

    def __init__(self, collection_name: str) -> None:
        """Fake vectorstore that has a collection name."""
        self.collection_name = collection_name

    def as_retriever(self) -> BaseRetriever:
        if self.collection_name == "index1":
            return vectorstore1.as_retriever()
        elif self.collection_name == "index2":
            return vectorstore2.as_retriever()
        else:
            raise NotImplementedError(
                f"No retriever for collection {self.collection_name}"
            )


## Retriever with configuration
vectorstore2 = FAISS.from_texts(
    ["cats like fish", "dogs like sticks"], embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()


class ConfigurableVectorstore(Runnable):
    vectorstore: FAISS

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        """Invoke the retriever."""
        retriever = self.vectorstore.as_retriever()
        return self.vectorstore.as_retriever()


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
