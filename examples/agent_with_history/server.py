#!/usr/bin/env python
"""Example LangChain server exposes and agent that has conversation history.

In this example, the history is stored entirely on the client's side.

Please see other examples in LangServe on how to use RunnableWithHistory to
store history on the server side.

In addition, see agent documentation in LangChain:

https://python.langchain.com/docs/modules/agents/how_to/custom_agent

**ATTENTION** This exampl does not truncate message history, so it will crash
if you send too many messages (exceed token length).
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

# ATTENTION: For production use case, it's a good idea to trim the prompt to avoid
#            exceeding the context window length used by the model.
#
# To fix that simply adjust the chain to trim the prompt in whatever way
# is appropriate for your use case.
# For example, you may want to keep the system message and the last 10 messages.
# Or you may want to trim based on the number of tokens.
# Or you may want to also summarize the messages to keep information about things
# that were learned about the user.
#
# def prompt_trimmer(messages: List[Union[HumanMessage, AIMessage, FunctionMessage]]):
#     '''Trims the prompt to a reasonable length.'''
#     # Keep in mind that when trimming you may want to keep the system message!
#     return messages[-10:] # Keep last 10 messages.

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    # | prompt_trimmer # See comment above.
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
