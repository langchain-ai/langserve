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
from typing import Any, Dict, List, Union

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
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    Generation,
    LLMResult,
)
from langchain_core.prompt_values import ChatPromptValueConcrete
from langchain_core.prompts.base import StringPromptValue

from langserve.pydantic_v1 import BaseModel, ValidationError
from langserve.validation import CallbackEvent

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1_000)  # Will accommodate up to 1_000 different error messages
def _log_error_message_once(error_message: str) -> None:
    """Log an error message once."""
    logger.error(error_message)


class WellKnownLCObject(BaseModel):
    """A well known LangChain object.

    A pydantic model that defines what constitutes a well known LangChain object.

    All well-known objects are allowed to be serialized and de-serialized.
    """

    __root__: Union[
        Document,
        HumanMessage,
        SystemMessage,
        ChatMessage,
        FunctionMessage,
        AIMessage,
        HumanMessageChunk,
        SystemMessageChunk,
        ChatMessageChunk,
        FunctionMessageChunk,
        AIMessageChunk,
        StringPromptValue,
        ChatPromptValueConcrete,
        AgentAction,
        AgentFinish,
        AgentActionMessageLog,
        LLMResult,
        ChatGeneration,
        Generation,
        ChatGenerationChunk,
    ]


def default(obj) -> Any:
    """Default serialization for well known objects."""
    if isinstance(obj, BaseModel):
        return obj.dict()
    return super().default(obj)


def _decode_lc_objects(value: Any) -> Any:
    """Decode the value."""
    if isinstance(value, dict):
        v = {key: _decode_lc_objects(v) for key, v in value.items()}

        try:
            obj = WellKnownLCObject.parse_obj(v)
            parsed = obj.__root__
            if set(parsed.dict()) != set(value):
                raise ValueError("Invalid object")
            return parsed
        except (ValidationError, ValueError):
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
            obj = CallbackEvent.parse_obj(value)
            return obj.__root__
        except ValidationError:
            try:
                obj = WellKnownLCObject.parse_obj(value)
                return obj.__root__
            except ValidationError:
                return {key: _decode_event_data(v) for key, v in value.items()}
    elif isinstance(value, list):
        return [_decode_event_data(item) for item in value]
    else:
        return value


# PUBLIC API


class Serializer(abc.ABC):
    @abc.abstractmethod
    def dumpd(self, obj: Any) -> Any:
        """Convert the given object to a JSON serializable object."""

    @abc.abstractmethod
    def dumps(self, obj: Any) -> bytes:
        """Dump the given object as a JSON string."""

    @abc.abstractmethod
    def loads(self, s: bytes) -> Any:
        """Load the given JSON string."""

    @abc.abstractmethod
    def loadd(self, obj: Any) -> Any:
        """Load the given object."""


class WellKnownLCSerializer(Serializer):
    def dumpd(self, obj: Any) -> Any:
        """Convert the given object to a JSON serializable object."""
        return orjson.loads(orjson.dumps(obj, default=default))

    def dumps(self, obj: Any) -> bytes:
        """Dump the given object as a JSON string."""
        return orjson.dumps(obj, default=default)

    def loadd(self, obj: Any) -> Any:
        """Load the given object."""
        return _decode_lc_objects(obj)

    def loads(self, s: bytes) -> Any:
        """Load the given JSON string."""
        return self.loadd(orjson.loads(s))


def _project_top_level(model: BaseModel) -> Dict[str, Any]:
    """Project the top level of the model as dict."""
    return {key: getattr(model, key) for key in model.__fields__}


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
            full_event = CallbackEvent.parse_obj(decoded_event_data)
        except ValidationError as e:
            msg = f"Encountered an invalid event: {e}"
            if "type" in decoded_event_data:
                msg += f' of type {repr(decoded_event_data["type"])}'
            _log_error_message_once(msg)
            continue

        decoded_event_data = _project_top_level(full_event.__root__)

        if decoded_event_data["type"].endswith("_error"):
            # Data is validated by this point, so we can assume that the shape
            # of the data is correct
            error = decoded_event_data["error"]
            msg = f"{error['status_code']}: {error['message']}"
            decoded_event_data["error"] = ServerSideException(msg)

        decoded_events.append(decoded_event_data)

    return decoded_events
