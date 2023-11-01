from typing import List

try:
    from pydantic.v1 import BaseModel
except ImportError:
    from pydantic import BaseModel


class CustomUserType(BaseModel):
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


class SharedResponseMetadata(BaseModel):
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


class SingletonResponseMetadata(SharedResponseMetadata):
    """
    Represents response metadata used for just single input/output LangServe
    responses.
    """

    # Represents the parent run id for a given request
    run_id: str


class BatchResponseMetadata(SharedResponseMetadata):
    """
    Represents response metadata used for batches of input/output LangServe
    responses.
    """

    # Represents each parent run id for a given request, in
    # the same order in which they were received
    run_ids: List[str]
