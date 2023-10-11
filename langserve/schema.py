from typing import Dict

from typing_extensions import TypedDict


class CallbackEvent(TypedDict):
    """Serialized representation of a callback event."""
    type: str
    data: Dict
