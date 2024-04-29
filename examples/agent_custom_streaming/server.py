#!/usr/bin/env python
"""Example LangChain server that shows how to customize streaming for an agent.

Example uses a RunnableLambda that:

1) Uses the agent's astream events method to create a custom streaming API endpoint.
2) Instantiates an agent with custom tools (based on the user request).

In this example, we kept things simple and are outputting strings to the client
with all the intermediate steps of the agent. This is just for demonstration
purposes, and usually you would want to return more structured output in the form
of a dictionary.

To add history to the agent you can use RunnableWithHistory. Please see the
other examples in LangServe for how to use RunnableWithHistory to store history
on the server side.

Alternatively, you can keep track of history on the client side and send it to the
server with each request. For that to work, you will definitely want to modify the
streaming output to yield dictionaries with structured output, so it's easy
to determine what the final agent output was on the client side.

Customize the streaming output to your use case!

Note that we configure the agent using the `tools` field in the input rather
than using configurable fields. Using custom runnables and configurable fields
is another option to customize the agent. 

Please see configurable_agent_executor: https://github.com/langchain-ai/langserve/blob/main/examples/configurable_agent_executor/server.py
for an example that uses a custom runnable with configurable fields.

Relevant LangChain documentation:

* Creating a custom agent: https://python.langchain.com/docs/modules/agents/how_to/custom_agent
* Streaming with agents: https://python.langchain.com/docs/modules/agents/how_to/streaming#custom-streaming-with-events
* General streaming documentation: https://python.langchain.com/docs/expression_language/streaming
* Message History: https://python.langchain.com/docs/expression_language/how_to/message_history

**ATTENTION**
1. This example does not truncate message history, so it will crash if you
   send too many messages (exceed token length).
2. The playground at the moment does not render agent output well! If you want to
   use the playground you need to customize it's output server side using astream
   events by wrapping it within another runnable.
3. See the client notebook to see how .stream() behaves!
"""  # noqa: E501
from typing import Any, AsyncIterator, List, Literal

from fastapi import FastAPI
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_core.utils.function_calling import format_tool_to_openai_tool
from langchain_openai import ChatOpenAI

from langserve import add_routes
from langserve.pydantic_v1 import BaseModel

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, but bad at calculating lengths of words. "
            "Talk with the user as normal. "
            "If they ask you to calculate the length of a word, use a tool",
        ),
        # Please note the ordering of the fields in the prompt!
        # The correct ordering is:
        # 1. user - the user's current input
        # 2. agent_scratchpad - the agent's working space for thinking and
        #    invoking tools to respond to the user's input.
        # If you change the ordering, the agent will not work correctly since
        # the messages will be shown to the underlying LLM in the wrong order.
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


@tool
def word_length(word: str) -> int:
    """Returns a counter word"""
    return len(word)


@tool
def favorite_animal(name: str) -> str:
    """Get the favorite animal of the person with the given name"""
    if name.lower().strip() == "eugene":
        return "cat"
    return "dog"


# We need to set streaming=True on the LLM to support streaming individual tokens.
# Tokens will be available when using the stream_log / stream events endpoints,
# but not when using the stream endpoint since the stream implementation for agent
# streams action observation pairs not individual tokens.
# See the client notebook that shows how to use the stream events endpoint.
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True)

TOOL_MAPPING = {
    "word_length": word_length,
    "favorite_animal": favorite_animal,
}
KnownTool = Literal["word_length", "favorite_animal"]


def _create_agent_with_tools(requested_tools: List[KnownTool]) -> AgentExecutor:
    """Create an agent with custom tools."""
    tools = []

    for requested_tool in requested_tools:
        if requested_tool not in TOOL_MAPPING:
            raise ValueError(f"Unknown tool: {requested_tool}")
        tools.append(TOOL_MAPPING[requested_tool])

    if tools:
        llm_with_tools = llm.bind(
            tools=[format_tool_to_openai_tool(tool) for tool in tools]
        )
    else:
        llm_with_tools = llm

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True).with_config(
        {"run_name": "agent"}
    )
    return agent_executor


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using LangChain's Runnable interfaces",
)


# We need to add these input/output schemas because the current AgentExecutor
# is lacking in schemas.
class Input(BaseModel):
    input: str
    tools: List[KnownTool]


async def custom_stream(input: Input) -> AsyncIterator[str]:
    """A custom runnable that can stream content.

    Args:
        input: The input to the agent. See the Input model for more details.

    Yields:
        strings that are streamed to the client.


    Strings were chosen for simplicity, feel free to adapt to your use case.

    You will almost certainly want to return more structured output in the form
    of a dictionary, so it's easy to determine what the agent is doing without
    parsing the output.

    Before creating a custom streaming API, you should consider if you can use
    the existing astream events API and customize the output on the client side
    (potentially less overall work both server and client side).
    """
    agent_executor = _create_agent_with_tools(input["tools"])
    async for event in agent_executor.astream_events(
        {
            "input": input["input"],
        },
        version="v1",
    ):
        kind = event["event"]
        if kind == "on_chain_start":
            if (
                event["name"] == "agent"
            ):  # matches `.with_config({"run_name": "Agent"})` in agent_executor
                yield "\n"
                yield (
                    f"Starting agent: {event['name']} "
                    f"with input: {event['data'].get('input')}"
                )
                yield "\n"
        elif kind == "on_chain_end":
            if (
                event["name"] == "agent"
            ):  # matches `.with_config({"run_name": "Agent"})` in agent_executor
                yield "\n"
                yield (
                    f"Done agent: {event['name']} "
                    f"with output: {event['data'].get('output')['output']}"
                )
                yield "\n"
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                # Empty content in the context of OpenAI means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content
                yield content
        elif kind == "on_tool_start":
            yield "\n"
            yield (
                f"Starting tool: {event['name']} "
                f"with inputs: {event['data'].get('input')}"
            )
            yield "\n"
        elif kind == "on_tool_end":
            yield "\n"
            yield (
                f"Done tool: {event['name']} "
                f"with output: {event['data'].get('output')}"
            )
            yield "\n"


class Output(BaseModel):
    output: Any


# Adds routes to the app for using the chain under:
# /invoke
# /batch
# /stream
# /stream_events
add_routes(
    app,
    RunnableLambda(custom_stream),
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
