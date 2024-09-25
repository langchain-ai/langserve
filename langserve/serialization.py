"""Serialization for well known objects and callback events.

Specialized JSON serialization for well known LangChain objects that
can be expected to be frequently transmitted between chains.

Callback events handle well known objects together with a few other
common types like UUIDs and Exceptions that might appear in the callback.

By default, exceptions are serialized as a generic exception without
any information about the exception. This is done to prevent leaking
sensitive information from the server to the client.
"""
import abc
import logging
from functools import lru_cache
from typing import Annotated, Any, Dict, List, Union

import orjson
from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    Generation,
    LLMResult,
)
from langchain_core.prompt_values import ChatPromptValueConcrete
from langchain_core.prompts.base import StringPromptValue
from pydantic import BaseModel, Discriminator, Field, RootModel, Tag, ValidationError

from langserve.validation import CallbackEvent

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1_000)  # Will accommodate up to 1_000 different error messages
def _log_error_message_once(error_message: str) -> None:
    """Log an error message once."""
    logger.error(error_message)


def _get_type(v: Any) -> str:
    """Get the type associated with the object for serialization purposes."""
    if isinstance(v, dict) and "type" in v:
        return v["type"]
    elif hasattr(v, "type"):
        return v.type
    else:
        raise TypeError(
            f"Expected either a dictionary with a 'type' key or an object "
            f"with a 'type' attribute. Instead got type {type(v)}."
        )


# A well known LangChain object.
# A pydantic model that defines what constitutes a well known LangChain object.
# All well-known objects are allowed to be serialized and de-serialized.

WellKnownLCObject = RootModel[
    Annotated[
        Union[
            Annotated[AIMessage, Tag(tag="ai")],
            Annotated[HumanMessage, Tag(tag="human")],
            Annotated[ChatMessage, Tag(tag="chat")],
            Annotated[SystemMessage, Tag(tag="system")],
            Annotated[FunctionMessage, Tag(tag="function")],
            Annotated[ToolMessage, Tag(tag="tool")],
            Annotated[AIMessageChunk, Tag(tag="AIMessageChunk")],
            Annotated[HumanMessageChunk, Tag(tag="HumanMessageChunk")],
            Annotated[ChatMessageChunk, Tag(tag="ChatMessageChunk")],
            Annotated[SystemMessageChunk, Tag(tag="SystemMessageChunk")],
            Annotated[FunctionMessageChunk, Tag(tag="FunctionMessageChunk")],
            Annotated[ToolMessageChunk, Tag(tag="ToolMessageChunk")],
            Annotated[Document, Tag(tag="Document")],
            Annotated[StringPromptValue, Tag(tag="StringPromptValue")],
            Annotated[ChatPromptValueConcrete, Tag(tag="ChatPromptValueConcrete")],
            Annotated[AgentAction, Tag(tag="AgentAction")],
            Annotated[AgentFinish, Tag(tag="AgentFinish")],
            Annotated[AgentActionMessageLog, Tag(tag="AgentActionMessageLog")],
            Annotated[ChatGeneration, Tag(tag="ChatGeneration")],
            Annotated[Generation, Tag(tag="Generation")],
            Annotated[ChatGenerationChunk, Tag(tag="ChatGenerationChunk")],
            Annotated[LLMResult, Tag(tag="LLMResult")],
        ],
        Field(discriminator=Discriminator(_get_type)),
    ]
]


def default(obj) -> Any:
    """Default serialization for well known objects."""
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    return super().default(obj)


def _decode_lc_objects(value: Any) -> Any:
    """Decode the value."""
    if isinstance(value, dict):
        v = {key: _decode_lc_objects(v) for key, v in value.items()}

        try:
            obj = WellKnownLCObject.model_validate(v)
            parsed = obj.root
            return parsed
        except (ValidationError, ValueError, TypeError):
            return v
    elif isinstance(value, list):
        return [_decode_lc_objects(item) for item in value]
    else:
        return value


class ServerSideException(Exception):
    """Exception raised when a server side exception occurs.

    The goal of this exception is to provide a way to communicate
    to the client that a server side exception occurred without
    revealing too much information about the exception as it may contain
    sensitive information.
    """


def _decode_event_data(value: Any) -> Any:
    """Decode the event data from a JSON object representation."""
    if isinstance(value, dict):
        try:
            obj = CallbackEvent.model_validate(value)
            return obj.root
        except ValidationError:
            try:
                obj = WellKnownLCObject.model_validate(value)
                return obj.root
            except ValidationError:
                return {key: _decode_event_data(v) for key, v in value.items()}
    elif isinstance(value, list):
        return [_decode_event_data(item) for item in value]
    else:
        return value


# PUBLIC API


class Serializer(abc.ABC):
    def dumpd(self, obj: Any) -> Any:
        """Convert the given object to a JSON serializable object."""
        return orjson.loads(self.dumps(obj))

    def loads(self, s: bytes) -> Any:
        """Load the given JSON string."""
        return self.loadd(orjson.loads(s))

    @abc.abstractmethod
    def dumps(self, obj: Any) -> bytes:
        """Dump the given object to a JSON byte string."""

    @abc.abstractmethod
    def loadd(self, s: bytes) -> Any:
        """Given a python object, load it into a well known object.

        The obj represents content that was json loaded from a string, but
        not yet validated or converted into a well known object.
        """


class WellKnownLCSerializer(Serializer):
    """A pre-defined serializer for well known LangChain objects.

    This is the default serialized used by LangServe for serializing and
    de-serializing well known LangChain objects.

    If you need to extend the serialization capabilities for your own application,
    feel free to create a new instance of the Serializer class and implement
    the abstract methods dumps and loadd.
    """

    def dumps(self, obj: Any) -> bytes:
        """Dump the given object to a JSON byte string."""
        return orjson.dumps(obj, default=default)

    def loadd(self, obj: Any) -> Any:
        """Given a python object, load it into a well known object.

        The obj represents content that was json loaded from a string, but
        not yet validated or converted into a well known object.
        """
        return _decode_lc_objects(obj)


def _project_top_level(model: BaseModel) -> Dict[str, Any]:
    """Project the top level of the model as dict."""
    return {key: getattr(model, key) for key in model.model_fields}


def load_events(events: Any) -> List[Dict[str, Any]]:
    """Load and validate the event.

    Args:
        events: The events to load and validate.

    Returns:
        The loaded and validated events.
    """
    if not isinstance(events, list):
        _log_error_message_once(f"Expected a list got {type(events)}")
        return []

    decoded_events = []

    for event in events:
        if not isinstance(event, dict):
            _log_error_message_once(f"Expected a dict got {type(event)}")
            # Discard the event / potentially error
            continue

        # First load all inner objects
        decoded_event_data = {
            key: _decode_lc_objects(value) for key, value in event.items()
        }

        # Then validate the event
        try:
            full_event = CallbackEvent.model_validate(decoded_event_data)
        except ValidationError as e:
            msg = f"Encountered an invalid event: {e}"
            if "type" in decoded_event_data:
                msg += f' of type {repr(decoded_event_data["type"])}'
            _log_error_message_once(msg)
            continue

        decoded_event_data = _project_top_level(full_event.root)

        if decoded_event_data["type"].endswith("_error"):
            # Data is validated by this point, so we can assume that the shape
            # of the data is correct
            error = decoded_event_data["error"]
            msg = f"{error['status_code']}: {error['message']}"
            decoded_event_data["error"] = ServerSideException(msg)

        decoded_events.append(decoded_event_data)

    return decoded_events
