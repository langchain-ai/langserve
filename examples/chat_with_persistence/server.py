#!/usr/bin/env python
"""Example of a chat server with persistence handled on the backend.

For simplicity, we're using file storage here -- to avoid the need to set up
a database. This is obviously not a good idea for a production environment,
but will help us to demonstrate the RunnableWithMessageHistory interface.

We'll use cookies to identify the user. This will help illustrate how to
fetch configuration from the request.
"""
import re
from pathlib import Path
from typing import Any, Callable, Dict, Union

from fastapi import FastAPI, HTTPException, Request
from langchain.chat_models import ChatAnthropic
from langchain.memory import FileChatMessageHistory
from langchain.schema.runnable.utils import ConfigurableFieldSpec
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from typing_extensions import TypedDict

from langserve import add_routes


def _is_valid_identifier(value: str) -> bool:
    """Check if the session ID is in a valid format."""
    # Use a regular expression to match the allowed characters
    valid_characters = re.compile(r"^[a-zA-Z0-9-_]+$")
    return bool(valid_characters.match(value))


def create_session_factory(
    base_dir: Union[str, Path]
) -> Callable[[str], BaseChatMessageHistory]:
    """Create a session ID factory that creates session IDs from a base dir.

    Args:
        base_dir: Base directory to use for storing the chat histories.

    Returns:
        A session ID factory that creates session IDs from a base path.
    """
    base_dir_ = Path(base_dir) if isinstance(base_dir, str) else base_dir
    if not base_dir_.exists():
        base_dir_.mkdir(parents=True)

    def get_chat_history(user_id: str, conversation_id: str) -> FileChatMessageHistory:
        """Get a chat history from a session ID."""
        if not _is_valid_identifier(user_id):
            raise ValueError(
                f"User ID {user_id} is not in a valid format. "
                "User ID must only contain alphanumeric characters, "
                "hyphens, and underscores."
            )
        if not _is_valid_identifier(conversation_id):
            raise ValueError(
                f"Session ID {conversation_id} is not in a valid format. "
                "Session ID must only contain alphanumeric characters, "
                "hyphens, and underscores."
            )

        user_dir = base_dir_ / user_id
        if not user_dir.exists():
            user_dir.mkdir(parents=True)
        file_path = user_dir / f"{conversation_id}.json"
        return FileChatMessageHistory(file_path)

    return get_chat_history


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)


def _per_request_config_modifier(
    config: Dict[str, Any], request: Request
) -> Dict[str, Any]:
    """Update the config"""
    config = config.copy()
    configurable = config.get("configurable", {})
    # Look for a cookie named "user_id"
    user_id = request.cookies.get("user_id", None)

    if user_id is None:
        raise HTTPException(
            status_code=400,
            detail="No session ID found. Please set a cookie named 'session_id'.",
        )

    configurable["user_id"] = user_id
    config["configurable"] = configurable
    return config


# Declare a chain
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You're an assistant by the name of Bob."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{human_input}"),
    ]
)

chain = prompt | ChatAnthropic(model="claude-2")


class InputChat(TypedDict):
    """Input for the chat endpoint."""

    human_input: str
    """Human input"""


chain_with_history = RunnableWithMessageHistory(
    chain,
    create_session_factory("chat_histories"),
    input_messages_key="human_input",
    history_messages_key="history",
    session_history_config_specs=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="Unique identifier for the user.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="Unique identifier for the conversation.",
            # None means that the conversation ID will be generated automatically
            default=None,
            is_shared=True,
        ),
    ],
).with_types(input_type=InputChat)


add_routes(
    app,
    chain_with_history,
    per_req_config_modifier=_per_request_config_modifier,
    # Disable playground and batch
    # 1) Playground we're passing information via headers, which is not supported via
    #    the playground right now.
    # 2) Disable batch to avoid users being confused. Batch will work fine
    #    as long as users invoke it with multiple configs appropriately, but
    #    without validation users are likely going to forget to do that.
    #    In addition, there's likely little sense in support batch for a chatbot.
    disabled_endpoints=["playground", "batch"],
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
