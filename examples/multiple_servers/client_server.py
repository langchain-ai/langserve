"""Client server that interacts with the main server via a remote runnable.

This server sets up a simple proxy to the main server. It uses the RemoteRunnable
to interact with the main server. The main server is expected to be running at
http://localhost:8123.

A client server will likely end up doing something more clever rather than
just being a proxy.
"""
from fastapi import FastAPI

from langserve import RemoteRunnable, add_routes

app = FastAPI()

MAIN_SERVER_URL = (
    "http://localhost:8123/chat_model/"  # <-- URL of the RUNNABLE on the main server
)
# Type inference is not automatic for remote runnables at the moment,
# so you must specify which types are used for the playground to work.
remote_runnable = RemoteRunnable(MAIN_SERVER_URL).with_types(input_type=str)


# Let's add an example chain
add_routes(
    app,
    remote_runnable,
    path="/proxied",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
