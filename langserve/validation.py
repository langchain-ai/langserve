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
from typing import Any, Dict, List, Literal, Optional, Sequence, Union
from uuid import UUID

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, Generation, RunInfo
from pydantic import BaseModel, Field, RootModel, create_model
from typing_extensions import Type

from langserve.schema import BatchResponseMetadata, InvokeResponseMetadata

# Type that is either a python annotation or a pydantic model that can be
# used to validate the input or output of a runnable.
Validator = Union[Type[BaseModel], type]

# PUBLIC API


def create_invoke_request_model(
    namespace: str,
    input_type: Validator,
    config: Type[BaseModel],
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
    invoke_request_type.model_rebuild()
    return invoke_request_type


def create_stream_request_model(
    namespace: str,
    input_type: Validator,
    config: Type[BaseModel],
) -> Type[BaseModel]:
    """Create a pydantic model for the stream request."""
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
    stream_request_model.model_rebuild()
    return stream_request_model


def create_batch_request_model(
    namespace: str,
    input_type: Validator,
    config: Type[BaseModel],
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
    batch_request_type.model_rebuild()
    return batch_request_type


def create_stream_log_request_model(
    namespace: str,
    input_type: Validator,
    config: Type[BaseModel],
) -> Type[BaseModel]:
    """Create a pydantic model for the invoke request."""
    stream_log_request = create_model(
        f"{namespace}StreamLogRequest",
        input=(input_type, ...),
        config=(config, Field(default_factory=dict)),
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
    stream_log_request.model_rebuild()
    return stream_log_request


def create_stream_events_request_model(
    namespace: str,
    input_type: Validator,
    config: Type[BaseModel],
) -> Type[BaseModel]:
    """Create a pydantic model for the stream events request."""
    stream_events_request = create_model(
        f"{namespace}StreamEventsRequest",
        input=(input_type, ...),
        config=(config, Field(default_factory=dict)),
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
    stream_events_request.model_rebuild()
    return stream_events_request


class InvokeBaseResponse(BaseModel):
    """Base class for invoke request."""


class BatchBaseResponse(BaseModel):
    """Base class for batch response."""


def create_invoke_response_model(
    namespace: str,
    output_type: Validator,
    include_callbacks: bool,
) -> Type[BaseModel]:
    """Create a pydantic model for the invoke response."""
    # The invoke response uses a key called `output` for the output, so
    # other information can be added to the response at a later date.

    fields = {
        "output": (
            output_type,
            Field(..., description="The output of the invocation."),
        ),
        "metadata": (
            InvokeResponseMetadata,
            Field(
                ...,
                description=(
                    "Metadata about the response that may be useful to "
                    "specific clients"
                ),
            ),
        ),
    }

    if include_callbacks:
        fields["callback_events"] = (
            List[CallbackEvent],
            Field(
                ...,
                description=("Callback events generated by the server side."),
            ),
        )

    invoke_response_type = create_model(
        f"{namespace}InvokeResponse",
        __base__=InvokeBaseResponse,
        **fields,
    )
    invoke_response_type.model_rebuild()
    return invoke_response_type


def create_batch_response_model(
    namespace: str,
    output_type: Validator,
    include_callbacks: bool,
) -> Type[BaseModel]:
    """Create a pydantic model for the batch response."""
    # The response uses a key called `output` for the output, so
    # other information can be added to the response at a later date.
    fields = {
        "output": (
            List[output_type],
            Field(
                ...,
                description="The outputs corresponding to the inputs the "
                "batch request.",
            ),
        ),
        "metadata": (
            BatchResponseMetadata,
            Field(
                ...,
                description=(
                    "Metadata about the response that may be useful to specific clients"
                ),
            ),
        ),
    }

    if include_callbacks:
        fields["callback_events"] = (
            List[List[CallbackEvent]],
            Field(
                ...,
                description=(
                    "Callback events generated by the server side."
                    "The outer list corresponds to the inputs and the inner "
                    "list corresponds to the callbacks generated for that input."
                ),
            ),
        )

    batch_response_type = create_model(
        f"{namespace}BatchResponse",
        __base__=BatchBaseResponse,
        **fields,
    )
    batch_response_type.model_rebuild()
    return batch_response_type


class InvokeRequestShallowValidator(BaseModel):
    """Shallow validator for Invoke Request.

    Validate basic shape of invoke request, downstream code
    is expected to do further validation.
    """

    input: Any = Field(..., description="The input to the runnable.")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)


class BatchRequestShallowValidator(BaseModel):
    """Shallow validator for Batch Request."""

    inputs: Any = Field(..., description="The inputs to the runnable.")
    config: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = Field(
        default_factory=dict
    )


class StreamLogParameters(BaseModel):
    """Shallow validator for Stream Log Request"""

    include_names: Optional[Sequence[str]] = None
    include_types: Optional[Sequence[str]] = None
    include_tags: Optional[Sequence[str]] = None
    exclude_names: Optional[Sequence[str]] = None
    exclude_types: Optional[Sequence[str]] = None
    exclude_tags: Optional[Sequence[str]] = None


class StreamEventsParameters(BaseModel):
    """Shallow validator for Stream Events Request."""

    include_names: Optional[Sequence[str]] = None
    include_types: Optional[Sequence[str]] = None
    include_tags: Optional[Sequence[str]] = None
    exclude_names: Optional[Sequence[str]] = None
    exclude_types: Optional[Sequence[str]] = None
    exclude_tags: Optional[Sequence[str]] = None


# Pydantic validators for callback events
# These objects may have a slightly different shape than the callback events
# used internally in langchain because they represent a serialized version
# of the callback event.
# For example, exceptions are replaced by error objects consisting of a
# status code and a message.


class BaseCallback(BaseModel):
    """Base class for all callback events."""

    run_id: UUID
    parent_run_id: Optional[UUID] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class OnChainStart(BaseCallback):
    """On Chain Start Callback Event."""

    serialized: Optional[Dict[str, Any]] = None
    inputs: Any
    kwargs: Optional[Dict[str, Any]] = None
    type: Literal["on_chain_start"] = "on_chain_start"


class OnChainEnd(BaseCallback):
    """On Chain End Callback Event."""

    outputs: Any
    kwargs: Optional[Dict[str, Any]] = None
    type: Literal["on_chain_end"] = "on_chain_end"


class Error(BaseModel):
    """Error object that is modeled after an HTTP error format."""

    status_code: int
    message: str
    type: Literal["error"] = "error"


class OnChainError(BaseCallback):
    """On Chain Error Callback Event."""

    error: Error
    kwargs: Optional[Dict[str, Any]] = None
    type: Literal["on_chain_error"] = "on_chain_error"


class OnToolStart(BaseCallback):
    """On Tool Start Callback Event."""

    serialized: Optional[Dict[str, Any]] = None
    input_str: str
    run_id: UUID
    parent_run_id: Optional[UUID] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    kwargs: Optional[Dict[str, Any]] = None
    type: Literal["on_tool_start"] = "on_tool_start"


class OnToolEnd(BaseCallback):
    """On Tool End Callback Event."""

    output: str
    run_id: UUID
    parent_run_id: Optional[UUID] = None
    tags: Optional[List[str]] = None
    kwargs: Optional[Dict[str, Any]] = None
    type: Literal["on_tool_end"] = "on_tool_end"


class OnToolError(BaseModel):
    """On Tool Error Callback Event."""

    error: Error
    kwargs: Optional[Dict[str, Any]] = None
    type: Literal["on_tool_error"] = "on_tool_error"


class OnChatModelStart(BaseCallback):
    """On Chat Model Start Callback Event."""

    serialized: Optional[Dict[str, Any]] = None
    messages: List[List[BaseMessage]]
    kwargs: Optional[Dict[str, Any]] = None
    type: Literal["on_chat_model_start"] = "on_chat_model_start"


class OnLLMStart(BaseCallback):
    """On LLM Start Callback Event."""

    serialized: Optional[Dict[str, Any]] = None
    prompts: List[str]
    run_id: UUID
    parent_run_id: Optional[UUID] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    kwargs: Optional[Dict[str, Any]] = None
    type: Literal["on_llm_start"] = "on_llm_start"


class LLMResult(BaseModel):
    """Concrete instance of LLMResult for validation only.

    Must be kept in sync with langchain.schema.llm.LLMResult.
    """

    generations: List[List[Union[Generation, ChatGeneration]]]
    """List of generated outputs. This is a List[List[]] because
    each input could have multiple candidate generations."""
    llm_output: Optional[dict] = None
    """Arbitrary LLM provider-specific output."""
    run: Optional[List[RunInfo]] = None
    """List of metadata info for model call for each input."""


class OnLLMEnd(BaseCallback):
    """On LLM End Callback Event."""

    response: LLMResult
    kwargs: Optional[Dict[str, Any]] = None
    type: Literal["on_llm_end"] = "on_llm_end"


class OnRetrieverStart(BaseCallback):
    """On Retriever Start Callback Event."""

    serialized: Optional[Dict[str, Any]] = None
    query: str
    kwargs: Optional[Dict[str, Any]] = None
    type: Literal["on_retriever_start"] = "on_retriever_start"


class OnRetrieverError(BaseCallback):
    """On Retriever Error Callback Event."""

    error: Error
    run_id: UUID
    parent_run_id: Optional[UUID] = None
    tags: Optional[List[str]] = None
    kwargs: Optional[Dict[str, Any]] = None
    type: Literal["on_retriever_error"] = "on_retriever_error"


class OnRetrieverEnd(BaseCallback):
    """On Retriever End Callback Event."""

    documents: Sequence[Document]
    kwargs: Optional[Dict[str, Any]] = None
    type: Literal["on_retriever_end"] = "on_retriever_end"


CallbackEvent = RootModel[
    Union[
        OnChainStart,
        OnChainEnd,
        OnChainError,
        OnChatModelStart,
        OnLLMStart,
        OnLLMEnd,
        OnToolStart,
        OnToolEnd,
        OnToolError,
        OnRetrieverStart,
        OnRetrieverEnd,
        OnRetrieverError,
    ]
]
