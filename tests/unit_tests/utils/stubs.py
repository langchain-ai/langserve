from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk


class AnyStr(str):
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, str)


def _AnyIdAIMessage(**kwargs: Any) -> AIMessage:
    """Create ai message with an any id field."""
    message = AIMessage(**kwargs)
    message.id = AnyStr()
    return message


def _AnyIdAIMessageChunk(**kwargs: Any) -> AIMessageChunk:
    """Create ai message with an any id field."""
    message = AIMessageChunk(**kwargs)
    message.id = AnyStr()
    return message
