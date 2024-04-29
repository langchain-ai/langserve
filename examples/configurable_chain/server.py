#!/usr/bin/env python
"""Example of configurable runnables.

This example shows how to use two options for configuration of runnables:

1) Configurable Fields: Use this to specify values for a given initialization parameter
2) Configurable Alternatives: Use this to specify complete alternative runnables
"""
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

from langserve import add_routes

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

###############################################################################
#                EXAMPLE 1: Configure fields based on RunnableConfig          #
###############################################################################
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

add_routes(app, chain, path="/configurable_temp")


###############################################################################
#                EXAMPLE 2: Configure prompt based on RunnableConfig          #
###############################################################################
configurable_prompt = PromptTemplate.from_template(
    "tell me a joke about {topic}."
).configurable_alternatives(
    ConfigurableField(
        id="prompt",
        name="Prompt",
        description="The prompt to use. Must contain {topic}.",
    ),
    default_key="joke",
    fact=PromptTemplate.from_template(
        "tell me a fact about {topic} in {language} language."
    ),
)
prompt_chain = configurable_prompt | model | StrOutputParser()

add_routes(app, prompt_chain, path="/configurable_prompt")


###############################################################################
#             EXAMPLE 3: Configure fields based on Request metadata           #
###############################################################################


# Add another example route where you can configure the model based
# on properties of the request. This is useful for passing in API
# keys from request headers (WITH CAUTION) or using other properties
# of the request to configure the model.
def fetch_api_key_from_header(config: Dict[str, Any], req: Request) -> Dict[str, Any]:
    if "x-api-key" in req.headers:
        config["configurable"]["openai_api_key"] = req.headers["x-api-key"]
    else:
        raise HTTPException(401, "No API key provided")

    return config


dynamic_auth_model = ChatOpenAI(openai_api_key="placeholder").configurable_fields(
    openai_api_key=ConfigurableField(
        id="openai_api_key",
        name="OpenAI API Key",
        description=("API Key for OpenAI interactions"),
    ),
)

dynamic_auth_chain = dynamic_auth_model | StrOutputParser()

add_routes(
    app,
    dynamic_auth_chain,
    path="/auth_from_header",
    per_req_config_modifier=fetch_api_key_from_header,
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
