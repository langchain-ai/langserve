"""Serialization for LangChain event data."""
import json
from typing import Any, Union

from langchain.prompts.base import StringPromptValue
from langchain.prompts.chat import ChatPromptValueConcrete
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
    """A well known LangChain object."""

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
    ]

from uuid import UUID

# Custom JSON Encoder
class _LangChainEncoder(json.JSONEncoder):
    """Custom JSON Encoder that can encode pydantic objects as well."""

    def default(self, obj) -> Any:
        if isinstance(obj, BaseModel):
            return obj.dict()
        elif isinstance(obj, UUID):
            return {
                "__typna"
            }
        return super().default(obj)


def _decoder(value: Any) -> Any:
    """Decode the value."""
    if isinstance(value, dict):
        try:
            obj = WellKnownLCObject.parse_obj(value)
            return obj.__root__
        except ValidationError:
            return {key: _decoder(v) for key, v in value.items()}
    elif isinstance(value, list):
        return [_decoder(item) for item in value]
    else:
        return value


# Custom JSON Decoder
class _LangChainDecoder(json.JSONDecoder):
    """Custom JSON Decoder that handles well known LangChain objects."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the LangChainDecoder."""
        super().__init__(object_hook=self.decoder, *args, **kwargs)

    def decoder(self, value: Any) -> Any:
        """Decode the value."""
        return _decoder(value)


# PUBLIC API


def simple_dumpd(obj: Any) -> Any:
    """Convert the given object to a JSON serializable object."""
    return json.loads(json.dumps(obj, cls=_LangChainEncoder))


def simple_dumps(obj: Any) -> str:
    """Dump the given object as a JSON string."""
    return json.dumps(obj, cls=_LangChainEncoder)


def simple_loads(s: str) -> Any:
    """Load the given JSON string."""
    return json.loads(s, cls=_LangChainDecoder)
