# LangServe 🦜️🔗 

## Overview

`LangServe` helps developers deploy `LangChain` [runnables and chains](https://python.langchain.com/docs/expression_language/) as a REST API.

This library is integrated with [FastAPI](https://fastapi.tiangolo.com/) and uses [pydantic](https://docs.pydantic.dev/latest/) for data validation.

In addition, it provides a client that can be used to call into runnables deployed on a server.
A javascript client is available in [LangChainJS](https://js.langchain.com/docs/api/runnables_remote/classes/RemoteRunnable).

## Features

- Input and Output schemas automatically inferred from your LangChain object, and enforced on every API call, with rich error messages
- API docs page with JSONSchema and Swagger (insert example link)
- Efficient `/invoke`, `/batch` and `/stream` endpoints with support for many concurrent requests on a single server
- `/stream_log` endpoint for streaming all (or some) intermediate steps from your chain/agent
- Built-in (optional) tracing to [LangSmith](https://www.langchain.com/langsmith), just add your API key (see [Instructions](https://docs.smith.langchain.com/)])
- All built with battle-tested open-source Python libraries like FastAPI, Pydantic, uvloop and asyncio.
- Use the client SDK to call a LangServe server as if it was a Runnable running locally (or call the HTTP API directly)

### Limitations

- Client callbacks are not yet supported for events that originate on the server
- Does not work with [pydantic v2 yet](https://github.com/tiangolo/fastapi/issues/10360)

## LangChain CLI  🛠️

Use the `LangChain` CLI to bootstrap a `LangServe` project quickly.

To use the langchain CLI make sure that you have a recent version of `langchain` installed
and also `typer`. (`pip install langchain typer` or `pip install "langchain[cli]"`)

```sh
langchain ../path/to/directory
```

And follow the instructions...

## Examples

For more examples, see the [examples](./examples) directory.


### Server

Here's a server that deploys an OpenAI chat model, an Anthropic chat model, and a chain that uses
the Anthropic model to tell a joke about a topic.

```python
#!/usr/bin/env python
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langserve import add_routes


app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    ChatOpenAI(),
    path="/openai",
)

add_routes(
    app,
    ChatAnthropic(),
    path="/anthropic",
)

model = ChatAnthropic()
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
add_routes(
    app,
    prompt | model,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
```

### Docs

If you've deployed the server above, you can view the generated OpenAPI docs using:

```sh
curl localhost:8000/docs
```

### Client

Python SDK

```python

from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from langserve import RemoteRunnable

openai = RemoteRunnable("http://localhost:8000/openai/")
anthropic = RemoteRunnable("http://localhost:8000/anthropic/")
joke_chain = RemoteRunnable("http://localhost:8000/chain/")

joke_chain.invoke({"topic": "parrots"})

# or async
await joke_chain.ainvoke({"topic": "parrots"})

prompt = [
    SystemMessage(content='Act like either a cat or a parrot.'), 
    HumanMessage(content='Hello!')
]

# Supports astream
async for msg in anthropic.astream(prompt):
    print(msg, end="", flush=True)
    
prompt = ChatPromptTemplate.from_messages(
    [("system", "Tell me a long story about {topic}")]
)
    
# Can define custom chains
chain = prompt | RunnableMap({
    "openai": openai,
    "anthropic": anthropic,
})

chain.batch([{ "topic": "parrots" }, { "topic": "cats" }])
```

In TypeScript (requires LangChain.js version 0.0.166 or later):

```typescript
import { RemoteRunnable } from "langchain/runnables/remote";

const chain = new RemoteRunnable({ url: `http://localhost:8000/chain/invoke/` });
const result = await chain.invoke({
  "topic": "cats", 
});
```

Python using `requests`:

```python
import requests
response = requests.post(
    "http://localhost:8000/chain/invoke/",
    json={'input': {'topic': 'cats'}}
)
response.json()
```

You can also use `curl`:

```sh
curl --location --request POST 'http://localhost:8000/chain/invoke/' \
    --header 'Content-Type: application/json' \
    --data-raw '{
        "input": {
            "topic": "cats"
        }
    }'
```


## Endpoints 

The following code:

```python
...
add_routes(
  app,
  runnable,
  path="/my_runnable",
)
```

adds of these endpoints to the server:

- `POST /my_runnable/invoke` - invoke the runnable on a single input
- `POST /my_runnable/batch` - invoke the runnable on a batch of inputs
- `POST /my_runnable/stream` - invoke on a single input and stream the output
- `POST /my_runnable/stream_log` - invoke on a single input and stream the output, including output of intermediate steps as it's generated
- `GET /my_runnable/input_schema` - json schema for input to the runnable
- `GET /my_runnable/output_schema` - json schema for output of the runnable
- `GET /my_runnable/config_schema` - json schema for config of the runnable

## Installation

For both client and server:

```bash
pip install "langserve[all]"
```

or `pip install "langserve[client]"` for client code, and `pip install "langserve[server]"` for server code.

## Legacy Chains

LangServe works with both Runnables (constructed via [LangChain Expression Language](https://python.langchain.com/docs/expression_language/)) and legacy chains (inheriting from `Chain`).
However, some of the input schemas for legacy chains may be incomplete/incorrect, leading to errors.
This can be fixed by updating the `input_schema` property of those chains in LangChain.
If you encounter any errors, please open an issue on THIS repo, and we will work to address it.


## Handling Authentication

If you need to add authentication to your server, 
please reference FastAPI's [security documentation](https://fastapi.tiangolo.com/tutorial/security/)
and [middleware documentation](https://fastapi.tiangolo.com/tutorial/middleware/).

## Deployment

### Deploy to GCP

You can deploy to GCP Cloud Run using the following command:

```
gcloud run deploy [your-service-name] --source . --port 8001 --allow-unauthenticated --region us-central1 --set-env-vars=OPENAI_API_KEY=your_key
```
