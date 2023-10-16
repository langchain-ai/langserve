import uuid
from typing import Optional

from typing_extensions import TypedDict


class EventData(TypedDict, total=False):
    """Event data for a callback event."""

    run_id: uuid.UUID
    parent_run_id: Optional[uuid.UUID]


class CallbackEvent(TypedDict):
    """Dict representation of a callback event."""

    name: str
    data: EventData
