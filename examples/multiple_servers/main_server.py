"""Main server that exposes one or more chains as HTTP endpoints."""
from fastapi import FastAPI
from langchain_openai import ChatOpenAI

from langserve import add_routes

app = FastAPI()

# Let's add an example chain
add_routes(
    app,
    ChatOpenAI(),
    path="/chat_model",
)

if __name__ == "__main__":
    import uvicorn

    # Running on port 8123
    uvicorn.run(app, port=8123)
