#!/usr/bin/env python
"""An example that shows how to use path dependencies for authentication.

The path dependencies are applied to all the routes added by the `add_routes`.

To keep this example brief, we're providing a placeholder verify_token function
that shows how to use path dependencies.

To implement proper auth, please see the FastAPI docs:

* https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-in-path-operation-decorators/
* https://fastapi.tiangolo.com/tutorial/dependencies/
* https://fastapi.tiangolo.com/tutorial/security/
"""  # noqa: E501

from fastapi import Depends, FastAPI, Header, HTTPException
from langchain_core.runnables import RunnableLambda
from typing_extensions import Annotated

from langserve import add_routes


async def verify_token(x_token: Annotated[str, Header()]) -> None:
    """Verify the token is valid."""
    # Replace this with your actual authentication logic
    if x_token != "secret-token":
        raise HTTPException(status_code=400, detail="X-Token header invalid")


app = FastAPI()


def add_one(x: int) -> int:
    """Add one to an integer."""
    return x + 1


chain = RunnableLambda(add_one)


add_routes(
    app,
    chain,
    dependencies=[Depends(verify_token)],
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
