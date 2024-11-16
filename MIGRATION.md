# LangGraph Platform Migration Guide

We have [recently announced](https://blog.langchain.dev/langgraph-platform-announce/) LangGraph Platform, a ***significantly*** enhanced solution for deploying agentic applications at scale.

LangGraph Platform incorporates [key design patterns and capabilities](https://langchain-ai.github.io/langgraph/concepts/langgraph_platform/#option-2-leveraging-langgraph-platform-for-complex-deployments) essential for production-level deployment of large language model (LLM) applications.

In contrast to LangServe, LangGraph Platform provides comprehensive, out-of-the-box support for [persistence](https://langchain-ai.github.io/langgraph/concepts/application_structure/), [memory](https://langchain-ai.github.io/langgraph/concepts/assistants/), [double-texting handling](https://langchain-ai.github.io/langgraph/concepts/double_texting/), [human-in-the-loop workflows](https://langchain-ai.github.io/langgraph/concepts/assistants/), [cron job scheduling](https://langchain-ai.github.io/langgraph/concepts/langgraph_server/#cron-jobs), [webhooks](https://langchain-ai.github.io/langgraph/concepts/langgraph_server/#webhooks), high-load management, advanced streaming, support for long-running tasks, background task processing, and much more.

The LangGraph Platform ecosystem includes the following components:

- [LangGraph Server](https://langchain-ai.github.io/langgraph/concepts/langgraph_server/): Provides an [Assistants API](https://langchain-ai.github.io/langgraph/cloud/reference/api/api_ref.html) for LLM applications (graphs) built with [LangGraph](https://langchain-ai.github.io/langgraph/). Available in both Python and JavaScript/TypeScript.
- [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/): A specialized IDE for real-time visualization, debugging, and interaction via a graphical interface. Available as a web application or macOS desktop app, it's a substantial improvement over LangServe's playground.
- [SDK](https://langchain-ai.github.io/langgraph/concepts/sdk/): Enables programmatic interaction with the server, available in Python and JavaScript/TypeScript.
- [RemoteGraph](https://langchain-ai.github.io/langgraph/how-tos/use-remote-graph/): Allows interaction with a remote graph as if it were running locally, serving as LangGraph's equivalent to LangServe's RemoteRunnable. Available in both Python and JavaScript/TypeScript.
 
## Context 

LangServe was built as a deployment solution for LangChain Runnables created using the [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/concepts/lcel). In LangServe, the LCEL was the orchestration layer that managed the execution of the Runnable.

[LangGraph](https://langchain-ai.github.io/langgraph/) is an open source library created by the LangChain team that provides a more flexible orchestration layer that's better suited for creating more complex LLM applications. LangGraph Platform
is the deployment solution for LangGraph applications.

## LangServe Support

We recommend using LangGraph Platform rather than LangServe for new projects.

We will continue to accept bug fixes for LangServe from the community; however, we will not be accepting new feature contributions.

## Migration

If you would like to migrate an existing LangServe application to LangGraph Platform, you have two options:

1. You can wrap the existing `Runnable` that you expose in the LangServe application via `add_routes` in a `LangGraph` node. This is the quickest way to migrate your application to LangGraph Platform.
2. You can do a larger refactor to break up the existing LCEL into appropriate `LangGraph` nodes. This is recommended if you want to take advantage of more advanced features in LangGraph Platform.

### Option 1: Wrap Runnable in LangGraph Node

This option is the quickest way to migrate your application to LangGraph Platform. You can wrap the existing `Runnable` that you expose in the LangServe application via `add_routes` in a `LangGraph` node.


Original LangServe code:

```python
from langserve import add_routes

app = FastAPI()

# Some input schema
class Input(BaseModel):
    input: str
    foo: Optional[str]

# Some output schema
class Output(BaseModel):
    output: Any
    
    
runnable = .... # Your existing Runnable
runnable_with_types = runnable.with_types(input_type=Input, output_type=Output)

# Adds routes to the app for using the chain under:
add_routes(
    app,
    runnable_with_types,
)
```

Migrated LangGraph Platform code:

```python

@dataclass
class InputState: # Equivalent to Input in the original code
    """Defines the input state, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    for more information.
    """

    input: str
    foo: Optional[str] = None


@dataclass
class OutputState: # Equivalent to Output in the original code
    """Defines the output state, representing a narrower interface to the outside world.
    
    https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """
    output: Any

@dataclass
class SharedState:
    """The full graph state.
    
    https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """
    input: str
    foo: Optional[str] = None
    output: Any
    
runnable = ... # Same code as before

async def my_node(state: InputState, config: RunnableConfig) -> OutputState:
    """Each node does work."""
    return await runnable.ainvoke({"input": state.input, "foo": state.foo})


# Define a new graph
builder = StateGraph(
    SharedState, config_schema=Configuration, input=InputState, output=OutputState
)

# Add the node to the graph
builder.add_node("my_node", my_node)

# Set the entrypoint as `call_model`
builder.add_edge("__start__", "my_node")

# Compile the workflow into an executable graph
graph = builder.compile()
graph.name = "New Graph"  # This defines the custom name in LangSmith
```

### 2. Refactor LCEL into LangGraph Nodes

This option is recommended if you want to take advantage of more advanced features in LangGraph Platform.

#### Memory

For example, LangGraph comes with built-in persistence that is more general than LangChain's `RunnableWithMessageHistory`.

Please refer to the guide on [upgrading to LangGraph memory](https://python.langchain.com/docs/versions/migrating_memory/) for more details.

#### Agents

If you're relying on legacy LangChain agents, you can migrate them into the pre-built
LangGraph agents. Please refer to the guide on [migrating agents](https://python.langchain.com/docs/how_to/migrate_agent/) for more details.

#### Custom Chains

If you created a custom chain and used LCEL to orchestrate it, you will usually be able to refactor it into a LangGraph without too much difficulty.

There isn't a one-size-fits-all guide for this, but generally speaking, consider creating
a separate node for any long-running step in your LCEL chain or any step that you would
want to be able to monitor or debug separately.

For example, if you have a simple Retrieval Augmented Generation (RAG) pipeline, you might have a node for the retrieval step and a node for the generation step.

Original LCEL code:

```python
...
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
rag_chain.with_types(input_type=Input, output_type=Output)
```

Using LangGraph for the same pipeline:


```python

@dataclass
class InputState: # Equivalent to Input in the original code
    """Input question from the user."""
    question: str
   
@dataclass 
class OutputState: # Equivalent to Output in the original code
    """The output from the graph."""
    answer: str

@dataclass 
class SharedState:
    question: str
    docs: List[str]
    response: str
   
async def retriever_node(state: InputState) -> SharedState:
    """Rettrieve documents based on the user's question."""
    documents = await retriever.ainvoke({"context": state.question})
    return {
        "docs": documents
    }

async def generator_node(state: SharedState) -> OutputState:
    """Generate an answer using an LLM based on the retrieved documents and question."""
    context = " -- DOCUMENT -- ".join(state.docs)
    prompt = [
        SystemMessage(
            content=(
                "Answer the user's question based on the list of documents "
                "that were retrieved. Here are the documents: \n\n"
                f"{context}"
            )
        ),
        HumanMessage(content=state.question),
    ]
    ai_message = await llm.ainvoke(prompt)
    return {"answer": ai_message.content}
    
# Define a new graph
builder = StateGraph(
    SharedState, config_schema=Configuration, input=InputState, output=OutputState
)
builder.add_node("retriever", retriever_node)
builder.add_node("generator", generator_node)
builder.add_edge("__start__", "retriever")
builder.add_edge("retriever", "generator")
graph = builder.compile()
graph.name = "RAG Graph"
```

Please see the [LangGraph tutorials](https://langchain-ai.github.io/langgraph/tutorials/)
for tutorials and examples that will help you get started with LangGraph
and LangGraph Platform.