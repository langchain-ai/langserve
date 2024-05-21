#!/usr/bin/env python
"""Example LangChain Server that uses a Fast API Router.

When applications grow, it becomes useful to use FastAPI's Router to organize
the routes.

See more documentation at:

https://fastapi.tiangolo.com/tutorial/bigger-applications/
"""
from fastapi import APIRouter, FastAPI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from langserve import add_routes

app = FastAPI()

router = APIRouter(prefix="/models")

# Invocations to this router will appear in trace logs as /models/openai
add_routes(
    router,
    ChatOpenAI(model="gpt-3.5-turbo-0125"),
    path="/openai",
)
# Invocations to this router will appear in trace logs as /models/anthropic
add_routes(
    router,
    ChatAnthropic(model="claude-3-haiku-20240307"),
    path="/anthropic",
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
