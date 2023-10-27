#!/usr/bin/env python
"""Example LangChain server exposes a chain composed of a prompt and an LLM."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import ConfigurableField

from langserve import add_routes

model = ChatOpenAI(temperature=0.5).configurable_alternatives(
    ConfigurableField(id="llm", name="LLM"),
    high_temp=ChatOpenAI(temperature=0.9),
    low_temp=ChatOpenAI(temperature=0.1, max_tokens=1),
    default_key="medium_temp",
)
prompt = PromptTemplate.from_template(
    "tell me a joke about {topic}."
).configurable_fields(
    template=ConfigurableField(
        id="prompt",
        name="Prompt",
        description="The prompt to use. Must contain {topic}",
    )
)
chain = prompt | model | StrOutputParser()

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


# The input type is automatically inferred from the runnable
# interface; however, if you want to override it, you can do so
# by passing in the input_type argument to add_routes.
class ChainInput(BaseModel):
    """The input to the chain."""

    topic: str


add_routes(app, chain, input_type=ChainInput, config_keys=["configurable"])

# Alternatively, you can rely on langchain's type inference
# to infer the input type from the runnable interface.
# add_routes(app, chain)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
