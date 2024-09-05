#!/usr/bin/env python
"""Example that shows how to use the underlying APIHandler class directly with Auth.

This example shows how to apply logic based on the user's identity.

You can build on these concepts to implement a more complex app:
* Add endpoints that allow users to manage their documents.
* Make a more complex runnable that does something with the retrieved documents; e.g.,
  a conversational agent that responds to the user's input with the retrieved documents
  (which are user specific documents).

For authentication, we use a fake token that's the same as the username, adapting
the following example from the FastAPI docs:

https://fastapi.tiangolo.com/tutorial/security/simple-oauth2/

**ATTENTION**

This example is not actually secure and should not be used in production.

Once you understand how to use `per_req_config_modifier`, read through
the FastAPI docs and implement proper auth:
https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/


**ATTENTION**

This example does not integrate auth with OpenAPI, so the OpenAPI docs won't
be able to help with authentication. This is currently a limitation
if using `add_routes`. If you need this functionality, you can use
the underlying `APIHandler` class directly, which affords maximal flexibility.
"""
from importlib import metadata
from typing import Any, List, Optional, Union

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import (
    ConfigurableField,
    RunnableConfig,
    RunnableSerializable,
)
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from typing_extensions import Annotated

from langserve import APIHandler
from langserve.pydantic_v1 import BaseModel


class User(BaseModel):
    username: str
    email: Union[str, None] = None
    full_name: Union[str, None] = None
    disabled: Union[bool, None] = None


class UserInDB(User):
    hashed_password: str


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()

FAKE_USERS_DB = {
    "alice": {
        "username": "alice",
        "full_name": "Alice Wonderson",
        "email": "alice@example.com",
        "hashed_password": "fakehashedsecret1",
        "disabled": False,
    },
    "john": {
        "username": "john",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "fakehashedsecret2",
        "disabled": False,
    },
    "bob": {
        "username": "john",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "fakehashedsecret3",
        "disabled": True,
    },
}


def _fake_hash_password(password: str) -> str:
    """Fake a hashed password."""
    return "fakehashed" + password


def _get_user(db: dict, username: str) -> Union[UserInDB, None]:
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


def _fake_decode_token(token: str) -> Union[User, None]:
    # This doesn't provide any security at all
    # Check the next version
    user = _get_user(FAKE_USERS_DB, token)
    return user


@app.post("/token")
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user_dict = FAKE_USERS_DB.get(form_data.username)
    if not user_dict:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    user = UserInDB(**user_dict)
    hashed_password = _fake_hash_password(form_data.password)
    if not hashed_password == user.hashed_password:
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    return {"access_token": user.username, "token_type": "bearer"}


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    user = _fake_decode_token(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


class PerUserVectorstore(RunnableSerializable):
    """A custom runnable that returns a list of documents for the given user.

    The runnable is configurable by the user, and the search results are
    filtered by the user ID.
    """

    user_id: Optional[str]
    vectorstore: VectorStore

    class Config:
        # Allow arbitrary types since VectorStore is an abstract interface
        # and not a pydantic model
        arbitrary_types_allowed = True

    def _invoke(
        self, input: str, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> List[Document]:
        """Invoke the retriever."""
        # WARNING: Verify documentation of underlying vectorstore to make
        # sure that it actually uses filters.
        # Highly recommended to use unit-tests to verify this behavior, as
        # implementations can be different depending on the underlying vectorstore.
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"filter": {"owner_id": self.user_id}}
        )
        return retriever.invoke(input, config=config)

    def invoke(
        self, input: str, config: Optional[RunnableConfig] = None, **kwargs
    ) -> List[Document]:
        """Add one to an integer."""
        return self._call_with_config(self._invoke, input, config, **kwargs)


vectorstore = Chroma(
    collection_name="some_collection",
    embedding_function=OpenAIEmbeddings(),
)

vectorstore.add_documents(
    [
        Document(
            page_content="cats like cheese",
            metadata={"owner_id": "alice"},
        ),
        Document(
            page_content="cats like mice",
            metadata={"owner_id": "alice"},
        ),
        Document(
            page_content="dogs like sticks",
            metadata={"owner_id": "john"},
        ),
        Document(
            page_content="my favorite food is cheese",
            metadata={"owner_id": "john"},
        ),
        Document(
            page_content="i like walks by the ocean",
            metadata={"owner_id": "john"},
        ),
        Document(
            page_content="dogs like grass",
            metadata={"owner_id": "bob"},
        ),
    ]
)

per_user_retriever = PerUserVectorstore(
    user_id=None,  # Placeholder ID that will be replaced by the per_req_config_modifier
    vectorstore=vectorstore,
).configurable_fields(
    # Attention: Make sure to override the user ID for each request in the
    # per_req_config_modifier. This should not be client configurable.
    user_id=ConfigurableField(
        id="user_id",
        name="User ID",
        description="The user ID to use for the retriever.",
    )
)


# Let's define the API Handler
api_handler = APIHandler(
    per_user_retriever,
    # Namespace for the runnable.
    # Endpoints like batch / invoke should be under /my_runnable/invoke
    # and /my_runnable/batch etc.
    path="/my_runnable",
)


PYDANTIC_VERSION = metadata.version("pydantic")
_PYDANTIC_MAJOR_VERSION: int = int(PYDANTIC_VERSION.split(".")[0])


# **ATTENTION** Your code does not need to include both versions.
# Use whichever version is appropriate given the pydantic version you are using.
# Both versions are included here for demonstration purposes.
#
# If using pydantic <2, everything works as expected.
# However, when using pydantic >=2 is installed, things are a bit
# more complicated because LangChain uses the pydantic.v1 namespace
# But the pydantic.v1 namespace is not supported by FastAPI.
# See this issue: https://github.com/tiangolo/fastapi/issues/10360
# So when using pydantic >=2, we need to use a vanilla starlette request
# and response, and we will not have documentation.
# Or we can create custom models for the request and response.
# The underlying API Handler will still validate the request
# correctly even if vanilla requests are used.
if _PYDANTIC_MAJOR_VERSION == 1:

    @app.post("/my_runnable/invoke")
    async def invoke_with_auth(
        # Included for documentation purposes
        invoke_request: api_handler.InvokeRequest,
        request: Request,
        current_user: Annotated[User, Depends(get_current_active_user)],
    ) -> Response:
        """Handle a request."""
        # The API Handler validates the parts of the request
        # that are used by the runnnable (e.g., input, config fields)
        config = {"configurable": {"user_id": current_user.username}}
        return await api_handler.invoke(request, server_config=config)
else:

    @app.post("/my_runnable/invoke")
    async def invoke_with_auth(
        request: Request,
        current_user: Annotated[User, Depends(get_current_active_user)],
    ) -> Response:
        """Handle a request."""
        # The API Handler validates the parts of the request
        # that are used by the runnnable (e.g., input, config fields)
        config = {"configurable": {"user_id": current_user.username}}
        return await api_handler.invoke(request, server_config=config)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
