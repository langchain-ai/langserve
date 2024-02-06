#!/usr/bin/env python
"""Example LangChain server that shows how to customize streaming for an agent.

Example uses a RunnableLambda that:

1) Instantiates an agent with custom tools (based on the user request).
2) Uses an agent with a chat history with history stored on the client side.
3) Uses the agent's astream events method to create a custom streaming API endpoint.

In this example, we kept things simple and simply output strings from the streaming
endpoint. 

Please customize the streaming output to your use case!

Please note that we configure the agent using the `tools` field in the input rather
than using configurable fields. Using custom runnables and configurable fields
is another option to customize the agent. 

Please see configurable_agent_executor: https://github.com/langchain-ai/langserve/blob/main/examples/configurable_agent_executor/server.py
for an example that uses a custom runnable with configurable fields.

Please see other examples in LangServe on how to use RunnableWithHistory to
store history on the server side.

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
from typing import Any, AsyncIterator, List, Union, Literal

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
from langchain_core.runnables import RunnableLambda
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
        # 1. history - the past messages between the user and the agent
        # 2. user - the user's current input
        # 3. agent_scratchpad - the agent's working space for thinking and
        #    invoking tools to respond to the user's input.
        # If you change the ordering, the agent will not work correctly since
        # the messages will be shown to the underlying LLM in the wrong order.
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


@tool
def word_length(word: str) -> int:
    """Returns a counter word"""
    return len(word)

@tool
def favorite_animal(name: str) -> int:
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

    for tool in requested_tools:
        if tool not in TOOL_MAPPING:
            raise ValueError(f"Unknown tool: {tool}")
        tools.append(TOOL_MAPPING[tool])

    if tools:
        llm_with_tools = llm.bind(tools=[format_tool_to_openai_tool(tool) for tool in tools])
    else:
        llm_with_tools = llm

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
    # The field extra defines a chat widget.
    # Please see documentation about widgets in the main README.
    # The widget is used in the playground.
    # Keep in mind that playground support for agents is not great at the moment.
    # To get a better experience, you'll need to customize the streaming output
    # for now.
    chat_history: List[Union[HumanMessage, AIMessage, FunctionMessage]]
    tools: List[KnownTool]



async def custom_stream(input: Input) -> AsyncIterator[str]:
    """A custom runnable that can stream content.

    Args:
        input: The input to the agent. See the Input model for more details.

    Yields:
        strings that are streamed to the client.

        The strings were chosen for simplicity, feel free to adapt to your use case.
    """
    agent_executor = _create_agent_with_tools(input['tools'])
    async for event in agent_executor.astream_events(
        {
            "input": input["input"],
            "chat_history": input["chat_history"],
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
