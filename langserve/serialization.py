"""Serialization module for Well Known LangChain objects.

Specialized JSON serialization for well known LangChain objects that
can be expected to be frequently transmitted between chains.
"""
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


def _recurse_obj(obj: Any) -> Any:
    """Recursively convert a pydantic object to a dict."""
    if isinstance(obj, BaseModel):
        d = {
            field_name: _recurse_obj(getattr(obj, field_name))
            for field_name in obj.__fields__
        }
        d["__typename"] = obj.__class__.__name__
        return d
    elif isinstance(obj, list):
        return [_recurse_obj(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: _recurse_obj(value) for key, value in obj.items()}
    else:
        return obj


# Custom JSON Encoder
class _LangChainEncoder(json.JSONEncoder):
    """Custom JSON Encoder that can encode pydantic objects as well."""

    def default(self, obj) -> Any:
        return _recurse_obj(obj)


WellKnownTypes = [
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

NAME_TO_TYPE = {type_.__name__: type_ for type_ in WellKnownTypes}


# Custom JSON Decoder
class _LangChainDecoder(json.JSONDecoder):
    """Custom JSON Decoder that handles well known LangChain objects."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the LangChainDecoder."""
        super().__init__(object_hook=self.decoder, *args, **kwargs)

    def decoder(self, value) -> Any:
        """Decode the value."""
        if isinstance(value, dict):
            new_value = {
                key: self.decoder(v) for key, v in value.items() if key != "__typename"
            }

            if "__typename" in value:
                type_ = NAME_TO_TYPE[value["__typename"]]
                return type_.parse_obj(new_value)
            else:
                return new_value
        elif isinstance(value, list):
            return [self.decoder(item) for item in value]
        else:
            return value


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
