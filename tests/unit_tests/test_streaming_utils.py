"""Test utilities for streaming."""
import datetime
import json
import uuid

from langsmith.schemas import FeedbackIngestToken

from langserve.api_handler import _create_metadata_event


def test_create_metadata_event() -> None:
    """Test that the metadata event is created correctly."""
    run_id = uuid.UUID(int=7)
    event = _create_metadata_event(run_id, feedback_ingest_token=None)
    assert event == {
        "data": '{"run_id": "00000000-0000-0000-0000-000000000007"}',
        "event": "metadata",
    }

    # Test with feedback ingest token
    feedback_ingest_token = FeedbackIngestToken(
        id=uuid.UUID(int=8), expires_at=datetime.datetime(2022, 1, 1), url="ingest-url"
    )
    event = _create_metadata_event(
        run_id, feedback_ingest_token=feedback_ingest_token, feedback_key="key"
    )
    data = json.loads(event.pop("data"))
    assert event == {
        "event": "metadata",
    }
    assert data == {
        "feedback_tokens": [
            {
                "expires_at": "2022-01-01T00:00:00",
                "key": "key",
                "token_url": "ingest-url",
            }
        ],
        "run_id": "00000000-0000-0000-0000-000000000007",
    }
