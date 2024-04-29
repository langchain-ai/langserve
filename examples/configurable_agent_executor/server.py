#!/usr/bin/env python
"""An example that shows how to create a custom agent executor like Runnable.

At the time of writing, there is a bug in the current AgentExecutor that
prevents it from correctly propagating configuration of the underlying
runnable. While that bug should be fixed, this is an example shows
how to create a more complex custom runnable.

Please see documentation for custom agent streaming here:

https://python.langchain.com/docs/modules/agents/how_to/streaming#stream-tokens

**ATTENTION**
To support streaming individual tokens you will need to manually set the streaming=True
on the LLM and use the stream_log endpoint rather than stream endpoint.
"""
from typing import Any, AsyncIterator, Dict, List, Optional, cast

from fastapi import FastAPI
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.pydantic_v1 import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    ConfigurableField,
    ConfigurableFieldSpec,
    Runnable,
    RunnableConfig,
)
from langchain_core.runnables.utils import Input, Output
from langchain_core.tools import tool
from langchain_core.utils.function_calling import format_tool_to_openai_function
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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

# We need to set streaming=True on the LLM to support streaming individual tokens.
# when using the stream_log endpoint.
# .stream for agents streams action observation pairs not individual tokens.
llm = ChatOpenAI(temperature=0, streaming=True).configurable_fields(
    temperature=ConfigurableField(
        id="llm_temperature",
        name="LLM Temperature",
        description="The temperature of the LLM",
    )
)

llm_with_tools = llm.bind(
    functions=[format_tool_to_openai_function(t) for t in tools]
).with_config({"run_name": "LLM"})


class CustomAgentExecutor(Runnable):
    """A custom runnable that will be used by the agent executor."""

    def __init__(self, **kwargs):
        """Initialize the runnable."""
        super().__init__(**kwargs)
        self.agent = (
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

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        """Will not be used."""
        raise NotImplementedError()

    @property
    def config_specs(self) -> List[ConfigurableFieldSpec]:
        return self.agent.config_specs

    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        """Stream the agent's output."""
        configurable = cast(Dict[str, Any], config.pop("configurable", {}))

        if configurable:
            configured_agent = self.agent.with_config(
                {
                    "configurable": configurable,
                }
            )
        else:
            configured_agent = self.agent

        executor = AgentExecutor(
            agent=configured_agent,
            tools=tools,
        ).with_config({"run_name": "executor"})

        async for output in executor.astream(input, config=config, **kwargs):
            yield output


app = FastAPI()


# We need to add these input/output schemas because the current AgentExecutor
# is lacking in schemas.
class Input(BaseModel):
    input: str


class Output(BaseModel):
    output: Any


runnable = CustomAgentExecutor()

# Add routes to the app for using the custom agent executor.
add_routes(
    app,
    runnable.with_types(input_type=Input, output_type=Output),
    disabled_endpoints=["invoke", "batch"],  # not implemented
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
