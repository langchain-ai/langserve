#!/usr/bin/env python
"""Example LangChain server exposes a chain composed of a prompt and an LLM."""
from fastapi import FastAPI
from typing_extensions import TypedDict

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema.runnable.utils import ConfigurableField
from langserve import add_routes

model = ChatOpenAI().configurable_alternatives(
    ConfigurableField(id="llm", name="LLM"),
    high_temp=ChatOpenAI(temperature=0.9),
    low_temp=ChatOpenAI(temperature=0.1, max_tokens=1),
    mid_temp=ChatOpenAI(temperature=0.5),
)

prompt = PromptTemplate.from_template(
    "tell me a joke about {topic}"
).configurable_fields(
    template=ConfigurableField(
        id="topic", name="Topic", description="The topic of the joke"
    )
)

chain = prompt | model

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)


add_routes(app, chain, config_keys=["configurable"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
