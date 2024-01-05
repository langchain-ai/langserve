#!/usr/bin/env python
"""Example LangChain server exposes a conversational retrieval chain."""
from fastapi import FastAPI
from langchain.agents import AgentExecutor, tool
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.pydantic_v1 import BaseModel
from langchain.tools.render import format_tool_to_openai_function
from langchain.vectorstores import FAISS
from langchain_core.runnables import ConfigurableField

from langserve import add_routes

vectorstore = FAISS.from_texts(
    ["cats like fish", "dogs like sticks"], embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()


@tool
def get_eugene_thoughts(query: str) -> list:
    """Returns Eugene's thoughts on a topic."""
    return retriever.get_relevant_documents(query)


tools = [get_eugene_thoughts]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


llm = ChatOpenAI(temperature=0.5).configurable_alternatives(
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

llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_functions(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools)

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)


# We need to add these input/output schemas because the current AgentExecutor
# is lacking in schemas.
class Input(BaseModel):
    input: str


class Output(BaseModel):
    output: str


# Adds routes to the app for using the chain under:
# /invoke
# /batch
# /stream
add_routes(app, agent_executor.with_types(input_type=Input, output_type=Output))
add_routes(app, llm, path='/llms')

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
