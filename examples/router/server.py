#!/usr/bin/env python
"""Example LangChain Server that uses a Fast API Router.

When applications grow, it becomes useful to use FastAPI's Router to organize
the routes.

See more documentation at:

https://fastapi.tiangolo.com/tutorial/bigger-applications/
"""
from fastapi import APIRouter, FastAPI
from langchain.chat_models import ChatAnthropic, ChatOpenAI

from langserve import add_routes

app = FastAPI()

router = APIRouter(prefix="/models")

# Invocations to this router will appear in trace logs as /models/openai
add_routes(
    router,
    ChatOpenAI(),
    path="/openai",
)
# Invocations to this router will appear in trace logs as /models/anthropic
add_routes(
    router,
    ChatAnthropic(),
    path="/anthropic",
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
