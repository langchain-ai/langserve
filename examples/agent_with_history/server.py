#!/usr/bin/env python
"""Example LangChain server exposes and agent that has history.

Please see documentation for custom agent streaming here:

https://python.langchain.com/docs/modules/agents/how_to/streaming#stream-tokens

**ATTENTION**
To support streaming individual tokens you will need to manually set the streaming=True
on the LLM and use the stream_log endpoint rather than stream endpoint.
"""
from typing import Any, List, Union

from fastapi import FastAPI
from langchain.agents import AgentExecutor, tool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.prompts import MessagesPlaceholder
from langchain_community.tools.convert_to_openai import format_tool_to_openai_tool
from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from langserve import add_routes
from langserve.pydantic_v1 import BaseModel

MEMORY_KEY = "chat_history"


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, but bad at calculating lengths of words. "
            "Talk with the user as normal. "
            "If they ask you to calculate the length of a word, use a tool",
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


@tool
def word_length(word: str) -> int:
    """Returns a counter word"""
    return len(word)


# We need to set streaming=True on the LLM to support streaming individual tokens.
# when using the stream_log endpoint.
# .stream for agents streams action observation pairs not individual tokens.
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True)

tools = [word_length]


llm_with_tools = llm.bind(tools=[format_tool_to_openai_tool(tool) for tool in tools])


agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)


# We need to add these input/output schemas because the current AgentExecutor
# is lacking in schemas.
class Input(BaseModel):
    input: str
    chat_history: List[Union[HumanMessage, AIMessage, FunctionMessage]]


class Output(BaseModel):
    output: Any


# Adds routes to the app for using the chain under:
# /invoke
# /batch
# /stream
add_routes(app, agent_executor.with_types(input_type=Input, output_type=Output))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
