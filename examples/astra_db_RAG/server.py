import sys
import os

import uvicorn
import cassio
from fastapi import FastAPI

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.cache import CassandraCache
from langchain.schema import BaseMessage
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores import Cassandra
from langchain.embeddings import OpenAIEmbeddings

import langchain
from langserve import add_routes

# DB init
cassio.init(
    token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
    database_id=os.environ["ASTRA_DB_ID"],
    keyspace=os.environ.get("ASTRA_DB_KEYSPACE"),
)

# AI general init
langchain.llm_cache = CassandraCache(session=None, keyspace=None)
if "clear_cache" in sys.argv[1:]:
    print("Clearing LLM cache!")
    langchain.llm_cache.clear()
llm = ChatOpenAI()

# synonym-route preparation
synonym_prompt = ChatPromptTemplate.from_template(
    "List up to five comma-separated synonyms of this word: {word}"
)

# entomology QA-route preparation
embeddings = OpenAIEmbeddings()
vector_store = Cassandra(
    session=None,
    keyspace=None,
    embedding=embeddings,
    table_name="langserve_demo_store",
)
retriever = vector_store.as_retriever(search_kwargs={'k': 3})
ento_template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
ento_prompt = ChatPromptTemplate.from_template(ento_template)
ento_chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | ento_prompt 
    | llm 
    | StrOutputParser()
)

# custom runnables
def msg_splitter(msg: BaseMessage):
    return [
        w.strip()
        for w in msg.content.split(",")
        if w.strip()
    ]


app = FastAPI(
  title="Astra LangServe",
  description="A demo for LangServe powered by Astra DB",
)


add_routes(
    app,
    synonym_prompt | llm | RunnableLambda(msg_splitter),
    path="/synonyms",
)

add_routes(
    app,
    ento_chain,
    path="/entomology",
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
