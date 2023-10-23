"""Serialization module for Well Known LangChain objects.

Specialized JSON serialization for well known LangChain objects that
can be expected to be frequently transmitted between chains.
"""
import abc
import json
from typing import Any, Union

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


def _decode_lc_objects(value: Any) -> Any:
    """Decode the value."""
    if isinstance(value, dict):
        try:
            obj = WellKnownLCObject.parse_obj(value)
            return obj.__root__
        except ValidationError:
            return {key: _decode_lc_objects(v) for key, v in value.items()}
    elif isinstance(value, list):
        return [_decode_lc_objects(item) for item in value]
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

    @abc.abstractmethod
    def loadd(self, obj: Any) -> Any:
        """Load the given object."""


class WellKnownLCSerializer(Serializer):
    def dumpd(self, obj: Any) -> Any:
        """Convert the given object to a JSON serializable object."""
        return json.loads(json.dumps(obj, cls=_LangChainEncoder))  # :*(

    def dumps(self, obj: Any) -> str:
        """Dump the given object as a JSON string."""
        return json.dumps(obj, cls=_LangChainEncoder)

    def loadd(self, obj: Any) -> Any:
        return _decode_lc_objects(obj)

    def loads(self, s: str) -> Any:
        """Load the given JSON string."""
        return self.loadd(json.loads(s))
