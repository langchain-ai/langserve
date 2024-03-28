from datetime import datetime
from typing import Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel  # Floats between v1 and v2

from langserve.pydantic_v1 import BaseModel as BaseModelV1


class CustomUserType(BaseModelV1):
    """Inherit from this class to create a custom user type.

    Use a custom user type if you want the data to de-serialize
    into a pydantic model rather than the equivalent dict representation.

    In general, make sure to add a `type` attribute to your class
    to help pydantic to discriminate unions.

    https://docs.pydantic.dev/1.10/usage/types/#discriminated-unions-aka-tagged-unions

    Limitations:
        At the moment, this type only works SERVER side and is used
        to specify desired DECODING behavior. If inheriting from this type
        the server will keep the decoded type as a pydantic model instead
        of converting it into a dict.
    """


class SharedResponseMetadata(BaseModelV1):
    """
    Any response metadata should inherit from this class. Response metadata
    represents non-output data that may be useful to some clients, but
    ignorable to most. For example, the run_ids associated with each run
    kicked off by the associated request.

    SharedResponseMetadata is an abstraction to represent any metadata
    representing a LangServe response shared across all outputs in said
    response.
    """

    pass


class InvokeResponseMetadata(SharedResponseMetadata):
    """Represents response metadata used for just single input/output LangServe
    responses.
    """

    # Represents the parent run id for a given request
    run_id: UUID
    feedback_token_url: Optional[str] = None
    feedback_token_expires_at: Optional[datetime] = None


class BatchResponseMetadata(SharedResponseMetadata):
    """
    Represents response metadata used for batches of input/output LangServe
    responses.
    """

    # This namespace can include any additional metadata that is shared
    # across all responses in the batch (e.g., if a batch run
    # ID was a thing, it would go here)

    # metadata for each individual response in the batch
    metadatas: List[InvokeResponseMetadata]

    # A list of UUIDs
    # Represents each parent run id for a given request, in
    # the same order in which they were received
    run_ids: List[UUID]  # For backwards compatibility, clients should not use


class BaseFeedback(BaseModel):
    """
    Shared information between create requests of feedback and feedback objects
    """

    run_id: Optional[UUID]
    """The associated run ID this feedback is logged for."""

    token_or_url: Union[str, UUID]
    """The token or URL associated with the feedback.
    
    This is a new way to associate feedback with a run: the token or URL
    are scoped to a specific run and can be associated with expiration times.
    """

    key: str
    """The metric name, tag, or aspect to provide feedback on."""

    score: Optional[Union[float, int, bool]] = None
    """Value or score to assign the run."""

    value: Optional[Union[float, int, bool, str, Dict]] = None
    """The display value for the feedback if not a metric."""

    comment: Optional[str] = None
    """Comment or explanation for the feedback."""


class FeedbackCreateRequest(BaseFeedback):
    """Represents a request that creates feedback for an individual run"""


class Feedback(BaseFeedback):
    """Represents feedback given on an individual run"""

    id: UUID
    """The unique ID of the feedback that was created."""

    created_at: datetime
    """The time the feedback was created."""

    modified_at: datetime
    """The time the feedback was last modified."""

    correction: Optional[Dict] = None
    """Correction for the run."""


class PublicTraceLinkCreateRequest(BaseModel):
    """Represents a request that creates a public trace for an individual run."""

    run_id: UUID
    """The unique ID of the run to share."""


class PublicTraceLink(BaseModel):
    """
    Represents a public trace for an individual run
    """

    public_url: str
    """Public URL for the trace."""
