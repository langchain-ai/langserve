"""Serialization module for Well Known LangChain objects.

Specialized JSON serialization for well known LangChain objects that
can be expected to be frequently transmitted between chains.
"""
import abc
import json
from typing import Any, Union
from uuid import UUID

from langchain.prompts.base import StringPromptValue
from langchain.prompts.chat import ChatPromptValueConcrete
from langchain.schema.agent import AgentAction, AgentActionMessageLog, AgentFinish
from langchain.schema.document import Document
from langchain.schema.messages import (
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

try:
    from pydantic.v1 import BaseModel, ValidationError
except ImportError:
    from pydantic import BaseModel, ValidationError


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
    ]


# Custom JSON Encoder
class _LangChainEncoder(json.JSONEncoder):
    """Custom JSON Encoder that can encode pydantic objects as well."""

    def default(self, obj) -> Any:
        if isinstance(obj, BaseModel):
            return obj.dict()
        return super().default(obj)


# Custom JSON Decoder
class _LangChainDecoder(json.JSONDecoder):
    """Custom JSON Decoder that handles well known LangChain objects."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the LangChainDecoder."""
        super().__init__(object_hook=self.decoder, *args, **kwargs)

    def decoder(self, value) -> Any:
        """Decode the value."""
        if isinstance(value, dict):
            try:
                obj = WellKnownLCObject.parse_obj(value)
                return obj.__root__
            except ValidationError:
                return {key: self.decoder(v) for key, v in value.items()}
        elif isinstance(value, list):
            return [self.decoder(item) for item in value]
        else:
            return value


class ServerSideException(Exception):
    """Exception raised when a server side exception occurs.

    The goal of this exception is to provide a way to communicate
    to the client that a server side exception occurred without
    revealing too much information about the exception as it may contain
    sensitive information.
    """


class _EventEncoder(json.JSONEncoder):
    """Custom JSON Encoder for serializing callback events."""

    def default(self, o: Any) -> Any:
        if isinstance(o, UUID):
            return {
                "__typename": "UUID",
                "value": str(o),
            }
        elif isinstance(o, Exception):
            return {
                "__typename": "ServerSideException",
                # We do not want to expose much information about the exception
                # Until we can do it safely (without revealing sensitive)
                "value": "A server exception occurred.",
            }
        return super().default(o)


class _EventDecoder(json.JSONDecoder):
    """Custom JSON Decoder for deserializing callback events."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the LangChainDecoder."""
        super().__init__(object_hook=self.decoder, *args, **kwargs)

    def decoder(self, value: Any) -> Any:
        """Decode the value"""
        if isinstance(value, dict):
            if "__typename" in value:
                if value["__typename"] == "UUID":
                    return UUID(value["value"])
                elif value["__typename"] == "ServerSideException":
                    return ServerSideException(value["value"])
            else:
                try:
                    obj = WellKnownLCObject.parse_obj(value)
                    return obj.__root__
                except ValidationError:
                    return {key: self.decoder(v) for key, v in value.items()}
        elif isinstance(value, list):
            return [self.decoder(item) for item in value]
        else:
            return value


# PUBLIC API


class Serializer(abc.ABC):
    @abc.abstractmethod
    def dumpd(self, obj: Any) -> Any:
        """Convert the given object to a JSON serializable object."""

    @abc.abstractmethod
    def dumps(self, obj: Any) -> str:
        """Dump the given object as a JSON string."""

    @abc.abstractmethod
    def loads(self, s: str) -> Any:
        """Load the given JSON string."""


class WellKnownLCSerializer(Serializer):
    def dumpd(self, obj: Any) -> Any:
        """Convert the given object to a JSON serializable object."""
        return json.loads(json.dumps(obj, cls=_LangChainEncoder))

    def dumps(self, obj: Any) -> str:
        """Dump the given object as a JSON string."""
        return json.dumps(obj, cls=_LangChainEncoder)

    def loads(self, s: str) -> Any:
        """Load the given JSON string."""
        return json.loads(s, cls=_LangChainDecoder)


class CallbackEventSerializer(Serializer):
    """Default implementation of a callback event serializer."""

    def dumpd(self, obj: Any) -> Any:
        """Convert the given object to a JSON serializable object."""
        return json.loads(json.dumps(obj, cls=_EventEncoder))

    def dumps(self, obj: Any) -> str:
        """Dump the given object as a JSON string."""
        return json.dumps(obj, cls=_EventEncoder)

    def loads(self, s: str) -> Any:
        """Load the given JSON string."""
        return json.loads(s, cls=_EventDecoder)
