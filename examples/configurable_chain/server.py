#!/usr/bin/env python
"""Example of configurable runnables.

This example shows how to use two options for configuration of runnables:

1) Configurable Fields: Use this to specify values for a given initialization parameter
2) Configurable Alternatives: Use this to specify complete alternative runnables
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import ConfigurableField

from langserve import add_routes

model = ChatOpenAI(temperature=0.5).configurable_alternatives(
    ConfigurableField(
        id="llm",
        name="LLM",
        description=(
            "Decide whether to use a high or a low temperature parameter for the LLM."
        ),
    ),
    high_temp=ChatOpenAI(temperature=0.9),
    low_temp=ChatOpenAI(temperature=0.1),
    default_key="medium_temp",
)
prompt = PromptTemplate.from_template(
    "tell me a joke about {topic}."
).configurable_fields(  # Example of a configurable field
    template=ConfigurableField(
        id="prompt",
        name="Prompt",
        description="The prompt to use. Must contain {topic}.",
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


# Add routes requires you to specify which config keys are accepted
# specifically, you must accept `configurable` as a config key.
add_routes(app, chain, config_keys=["configurable"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
