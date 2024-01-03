#!/usr/bin/env python
"""An example that uses Fast API global dependencies.

This approach can be used if the same authentication logic can be used
for all endpoints in the application.

This may be a reasonable approach for simple applications.

See:

* https://fastapi.tiangolo.com/tutorial/dependencies/global-dependencies/
* https://fastapi.tiangolo.com/tutorial/dependencies/
* https://fastapi.tiangolo.com/tutorial/security/
"""

from fastapi import Depends, FastAPI, Header, HTTPException
from langchain_core.runnables import RunnableLambda
from typing_extensions import Annotated

from langserve import add_routes


async def verify_token(x_token: Annotated[str, Header()]) -> None:
    """Verify the token is valid."""
    # Replace this with your actual authentication logic
    if x_token != "secret-token":
        raise HTTPException(status_code=400, detail="X-Token header invalid")


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    dependencies=[Depends(verify_token)],
)


def add_one(x: int) -> int:
    """Add one to an integer."""
    return x + 1


chain = RunnableLambda(add_one)


add_routes(
    app,
    chain,
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
