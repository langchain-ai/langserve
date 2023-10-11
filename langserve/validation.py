"""Code to dynamically create pydantic models for validating requests and responses.

Requests share the same basic shape of input, config, and kwargs.

Invoke and Batch responses use an `output` key for the output, other keys may
be added to the response at a later date.

Responses for stream and stream_log are specified as those endpoints use
a streaming response.

Type information for input, config and output can be specified by the user
per runnable. This type information will be used for validation of the input and
output and will appear in the OpenAPI spec for the corresponding endpoint.

Models are created with a namespace to avoid name collisions when hosting
multiple runnables. When present the name collisions prevent fastapi from
generating OpenAPI specs.
"""
from typing import List, Optional, Sequence, Union

try:
    from pydantic.v1 import BaseModel, Field, create_model
except ImportError:
    from pydantic import BaseModel, Field, create_model

from typing_extensions import Type, TypedDict

# Type that is either a python annotation or a pydantic model that can be
# used to validate the input or output of a runnable.
Validator = Union[Type[BaseModel], type]

# PUBLIC API


def create_invoke_request_model(
    namespace: str,
    input_type: Validator,
    config: TypedDict,
) -> Type[BaseModel]:
    """Create a pydantic model for the invoke request."""
    invoke_request_type = create_model(
        f"{namespace}InvokeRequest",
        input=(input_type, ...),
        config=(config, Field(default_factory=dict)),
        kwargs=(dict, Field(default_factory=dict)),
    )
    invoke_request_type.update_forward_refs()
    return invoke_request_type


def create_stream_request_model(
    namespace: str,
    input_type: Validator,
    config: TypedDict,
) -> Type[BaseModel]:
    """Create a pydantic model for the stream request."""
    stream_request_model = create_model(
        f"{namespace}StreamRequest",
        input=(input_type, ...),
        config=(config, Field(default_factory=dict)),
        kwargs=(dict, Field(default_factory=dict)),
    )
    stream_request_model.update_forward_refs()
    return stream_request_model


def create_batch_request_model(
    namespace: str,
    input_type: Validator,
    config: TypedDict,
) -> Type[BaseModel]:
    """Create a pydantic model for the batch request."""
    batch_request_type = create_model(
        f"{namespace}BatchRequest",
        inputs=(List[input_type], ...),
        config=(Union[config, List[config]], Field(default_factory=dict)),
        kwargs=(dict, Field(default_factory=dict)),
    )
    batch_request_type.update_forward_refs()
    return batch_request_type


def create_stream_log_request_model(
    namespace: str,
    input_type: Validator,
    config: TypedDict,
) -> Type[BaseModel]:
    """Create a pydantic model for the invoke request."""
    stream_log_request = create_model(
        f"{namespace}StreamLogRequest",
        input=(input_type, ...),
        config=(config, Field(default_factory=dict)),
        diff=(Optional[bool], False),
        include_names=(Optional[Sequence[str]], None),
        include_types=(Optional[Sequence[str]], None),
        include_tags=(Optional[Sequence[str]], None),
        exclude_names=(Optional[Sequence[str]], None),
        exclude_types=(Optional[Sequence[str]], None),
        exclude_tags=(Optional[Sequence[str]], None),
        kwargs=(dict, Field(default_factory=dict)),
    )
    stream_log_request.update_forward_refs()
    return stream_log_request


def create_invoke_response_model(
    namespace: str,
    output_type: Validator,
) -> Type[BaseModel]:
    """Create a pydantic model for the invoke response."""
    # The invoke response uses a key called `output` for the output, so
    # other information can be added to the response at a later date.
    invoke_response_type = create_model(
        f"{namespace}InvokeResponse",
        output=(output_type, ...),
    )
    invoke_response_type.update_forward_refs()
    return invoke_response_type


def create_batch_response_model(
    namespace: str,
    output_type: Validator,
) -> Type[BaseModel]:
    """Create a pydantic model for the batch response."""
    # The response uses a key called `output` for the output, so
    # other information can be added to the response at a later date.
    batch_response_type = create_model(
        f"{namespace}BatchResponse",
        output=(List[output_type], ...),
    )
    batch_response_type.update_forward_refs()
    return batch_response_type
