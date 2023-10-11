from typing import List, Optional, Sequence, Union

try:
    from pydantic.v1 import BaseModel, Field, create_model
except ImportError:
    from pydantic import BaseModel, Field, create_model

from typing_extensions import Type, TypedDict

InputValidator = Union[Type[BaseModel], type]
# The following langchain objects are considered to be safe to load.

# PUBLIC API


def create_invoke_request_model(
    namespace: str,
    input_type: InputValidator,
    config: TypedDict,
) -> Type[BaseModel]:
    """Create a pydantic model for the invoke request."""
    invoke_request_type = create_model(
        f"{namespace}InvokeRequest",
        input=(input_type, Field(..., description="The input to the runnable.")),
        config=(
            config,
            Field(
                default_factory=dict,
                description=(
                    "Subset of RunnableConfig object in LangChain. "
                    "Useful for passing information like tags, metadata etc."
                ),
            ),
        ),
        kwargs=(
            dict,
            Field(
                default_factory=dict,
                description="Keyword arguments to the runnable. Currently ignored.",
            ),
        ),
    )
    invoke_request_type.update_forward_refs()
    return invoke_request_type


def create_stream_request_model(
    namespace: str,
    input_type: InputValidator,
    config: TypedDict,
) -> Type[BaseModel]:
    """Create a pydantic model for the invoke request."""
    stream_request_model = create_model(
        f"{namespace}StreamRequest",
        input=(input_type, Field(..., description="The input to the runnable.")),
        config=(
            config,
            Field(
                default_factory=dict,
                description=(
                    "Subset of RunnableConfig object in LangChain. "
                    "Useful for passing information like tags, metadata etc."
                ),
            ),
        ),
        kwargs=(
            dict,
            Field(
                default_factory=dict,
                description="Keyword arguments to the runnable. Currently ignored.",
            ),
        ),
    )
    stream_request_model.update_forward_refs()
    return stream_request_model


def create_batch_request_model(
    namespace: str,
    input_type: InputValidator,
    config: TypedDict,
) -> Type[BaseModel]:
    """Create a pydantic model for the batch request."""
    batch_request_type = create_model(
        f"{namespace}BatchRequest",
        inputs=(List[input_type], ...),
        config=(
            Union[config, List[config]],
            Field(
                default_factory=dict,
                description=(
                    "Subset of RunnableConfig object in LangChain. Either specify one "
                    "config for all inputs or a list of configs with one per input. "
                    "Useful for passing information like tags, metadata etc."
                ),
            ),
        ),
        kwargs=(
            dict,
            Field(
                default_factory=dict,
                description="Keyword arguments to the runnable. Currently ignored.",
            ),
        ),
    )
    batch_request_type.update_forward_refs()
    return batch_request_type


def create_stream_log_request_model(
    namespace: str,
    input_type: InputValidator,
    config: TypedDict,
) -> Type[BaseModel]:
    """Create a pydantic model for the invoke request."""
    stream_log_request = create_model(
        f"{namespace}StreamLogRequest",
        input=(input_type, ...),
        config=(config, Field(default_factory=dict)),
        diff=(Optional[bool], False),
        include_names=(
            Optional[Sequence[str]],
            Field(
                None,
                description="If specified, filter to runnables with matching names",
            ),
        ),
        include_types=(
            Optional[Sequence[str]],
            Field(
                None,
                description="If specified, filter to runnables with matching types",
            ),
        ),
        include_tags=(
            Optional[Sequence[str]],
            Field(
                None,
                description="If specified, filter to runnables with matching tags",
            ),
        ),
        exclude_names=(
            Optional[Sequence[str]],
            Field(
                None,
                description="If specified, exclude runnables with matching names",
            ),
        ),
        exclude_types=(
            Optional[Sequence[str]],
            Field(
                None,
                description="If specified, exclude runnables with matching types",
            ),
        ),
        exclude_tags=(
            Optional[Sequence[str]],
            Field(
                None,
                description="If specified, exclude runnables with matching tags",
            ),
        ),
        kwargs=(dict, Field(default_factory=dict)),
    )
    stream_log_request.update_forward_refs()
    return stream_log_request


def create_invoke_response_model(
    namespace: str,
    output_type: InputValidator,
) -> Type[BaseModel]:
    """Create a pydantic model for the invoke response."""
    invoke_response_type = create_model(
        f"{namespace}InvokeResponse",
        output=(output_type, Field(..., description="The output of the invocation.")),
    )
    invoke_response_type.update_forward_refs()
    return invoke_response_type


def create_batch_response_model(
    namespace: str,
    output_type: InputValidator,
) -> Type[BaseModel]:
    """Create a pydantic model for the batch response."""
    batch_response_type = create_model(
        f"{namespace}BatchResponse",
        output=(
            List[output_type],
            Field(
                ...,
                description=(
                    "The outputs corresponding to the inputs the batch request."
                ),
            ),
        ),
    )
    batch_response_type.update_forward_refs()
    return batch_response_type
