#!/usr/bin/env python
"""A more complex example that shows how to configure index name at run time."""
from typing import Any, Iterable, List, Optional, Type

from fastapi import FastAPI
from langchain.schema.vectorstore import VST
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    ConfigurableFieldSingleOption,
    RunnableConfig,
    RunnableSerializable,
)
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings

from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field

vectorstore1 = FAISS.from_texts(
    ["cats like fish", "dogs like sticks"], embedding=OpenAIEmbeddings()
)

vectorstore2 = FAISS.from_texts(["x_n+1=a * xn * (1-xn)"], embedding=OpenAIEmbeddings())


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)


class UnderlyingVectorStore(VectorStore):
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

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        raise NotImplementedError()

    @classmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> VST:
        raise NotImplementedError()

    def similarity_search(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        raise NotImplementedError()


class ConfigurableRetriever(RunnableSerializable[str, List[Document]]):
    """Create a custom retriever that can be configured by the user.

    This is an example of how to create a custom runnable that can be configured
    to use a different collection name at run time.

    Configuration involves instantiating a VectorStore with a collection name.
    at run time, so the underlying vectorstore should be *cheap* to instantiate.

    For example, it should not be making any network requests at instantiation time.

    Make sure that the vectorstore you use meets this criteria.
    """

    collection_name: str

    def invoke(
        self, input: str, config: Optional[RunnableConfig] = None
    ) -> List[Document]:
        """Invoke the retriever."""
        vectorstore = UnderlyingVectorStore(self.collection_name)
        retriever = vectorstore.as_retriever()
        return retriever.invoke(input, config=config)


configurable_collection_name = ConfigurableRetriever(
    collection_name="index1"
).configurable_fields(
    collection_name=ConfigurableFieldSingleOption(
        id="collection_name",
        name="Collection Name",
        description="The name of the collection to use for the retriever.",
        options={
            "Index 1": "index1",
            "Index 2": "index2",
        },
        default="Index 1",
    )
)


class Request(BaseModel):
    __root__: str = Field(default="cat", description="Search query")


add_routes(app, configurable_collection_name.with_types(input_type=Request))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
