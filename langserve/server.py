"""FastAPI integration for langchain runnables.

This code contains integration for langchain runnables with FastAPI.

The main entry point is the `add_routes` function which adds the routes to an existing
FastAPI app or APIRouter.
"""
import re
from inspect import isclass
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Literal,
    Mapping,
    Sequence,
    Type,
    Union,
)

from fastapi import Request
from langchain.callbacks.tracers.log_stream import RunLog, RunLogPatch
from langchain.load.serializable import Serializable
from langchain.schema.runnable import Runnable
from typing_extensions import Annotated

from langserve.version import __version__

try:
    from pydantic.v1 import BaseModel, create_model
except ImportError:
    from pydantic import BaseModel, Field, create_model

from langserve.serialization import simple_dumpd, simple_dumps
from langserve.validation import (
    create_batch_request_model,
    create_batch_response_model,
    create_invoke_request_model,
    create_invoke_response_model,
    create_stream_log_request_model,
    create_stream_request_model,
)

try:
    from fastapi import APIRouter, FastAPI
except ImportError:
    # [server] extra not installed
    APIRouter = FastAPI = Any


def _unpack_config(d: Union[BaseModel, Mapping], keys: Sequence[str]) -> Dict[str, Any]:
    """Project the given keys from the given dict."""
    _d = d.dict() if isinstance(d, BaseModel) else d
    return {k: _d[k] for k in keys if k in _d}


def _unpack_input(validated_model: BaseModel) -> Any:
    """Unpack the decoded input from the validated model."""
    if hasattr(validated_model, "__root__"):
        model = validated_model.__root__
    else:
        model = validated_model

    if isinstance(model, BaseModel) and not isinstance(model, Serializable):
        # If the model is a pydantic model, but not a Serializable, then
        # it was created by the server as part of validation and isn't expected
        # to be accepted by the runnables as input as a pydantic model,
        # instead we need to convert it into a corresponding python dict.
        return model.dict()

    return model


def _rename_pydantic_model(model: Type[BaseModel], name: str) -> Type[BaseModel]:
    """Rename the given pydantic model to the given name."""
    return create_model(
        name,
        __config__=model.__config__,
        **{
            fieldname: (
                field.annotation,
                Field(
                    field.default,
                    title=fieldname,
                    description=field.field_info.description,
                ),
            )
            for fieldname, field in model.__fields__.items()
        },
    )


# This is a global registry of models to avoid creating the same model
# multiple times.
# Duplicated model names break fastapi's openapi generation.
_MODEL_REGISTRY = {}
_SEEN_NAMES = set()


def _replace_non_alphanumeric_with_underscores(s: str) -> str:
    """Replace non-alphanumeric characters with underscores."""
    return re.sub(r"[^a-zA-Z0-9]", "_", s)


def _resolve_model(
    type_: Union[Type, BaseModel], default_name: str, namespace: str
) -> Type[BaseModel]:
    """Resolve the input type to a BaseModel."""
    if isclass(type_) and issubclass(type_, BaseModel):
        model = type_
    else:
        model = create_model(default_name, __root__=(type_, ...))

    hash_ = model.schema_json()

    if model.__name__ in _SEEN_NAMES and hash_ not in _MODEL_REGISTRY:
        # If the model name has been seen before, but the model itself is different
        # generate a new name for the model.
        model_to_use = _rename_pydantic_model(model, f"{namespace}{model.__name__}")
        hash_ = model_to_use.schema_json()
    else:
        model_to_use = model

    if hash_ not in _MODEL_REGISTRY:
        _SEEN_NAMES.add(model_to_use.__name__)
        _MODEL_REGISTRY[hash_] = model_to_use

    return _MODEL_REGISTRY[hash_]


def _add_namespace_to_model(namespace: str, model: Type[BaseModel]) -> Type[BaseModel]:
    """Prefix the name of the given model with the given namespace.

    Code is used to help avoid name collisions when hosting multiple runnables
    that may use the same underlying models.

    Args:
        namespace: The namespace to use for the model.
        model: The model to create a unique name for.

    Returns:
        A new model with name prepended with the given namespace.
    """
    model_with_unique_name = _rename_pydantic_model(
        model,
        f"{namespace}{model.__name__}",
    )
    model_with_unique_name.update_forward_refs()
    return model_with_unique_name


def _add_tracing_info_to_metadata(config: Dict[str, Any], request: Request) -> None:
    """Add information useful for tracing and debugging purposes.

    Args:
        config: The config to expand with tracing information.
        request: The request to use for expanding the metadata.
    """

    metadata = config["metadata"] if "metadata" in config else {}

    info = {
        "__useragent": request.headers.get("user-agent"),
        "__langserve_version": __version__,
    }
    metadata.update(info)
    config["metadata"] = metadata


# PUBLIC API


def add_routes(
    app: Union[FastAPI, APIRouter],
    runnable: Runnable,
    *,
    path: str = "",
    input_type: Union[Type, Literal["auto"], BaseModel] = "auto",
    output_type: Union[Type, Literal["auto"], BaseModel] = "auto",
    config_keys: Sequence[str] = (),
) -> None:
    """Register the routes on the given FastAPI app or APIRouter.


    The following routes are added per runnable under the specified `path`:

    * /invoke - for invoking a runnable with a single input
    * /batch - for invoking a runnable with multiple inputs
    * /stream - for streaming the output of a runnable
    * /stream_log - for streaming intermediate outputs for a runnable
    * /input_schema - for returning the input schema of the runnable
    * /output_schema - for returning the output schema of the runnable
    * /config_schema - for returning the config schema of the runnable

    Args:
        app: The FastAPI app or APIRouter to which routes should be added.
        runnable: The runnable to wrap, must not be stateful.
        path: A path to prepend to all routes.
        input_type: type to use for input validation.
            Default is "auto" which will use the InputType of the runnable.
            User is free to provide a custom type annotation.
        output_type: type to use for output validation.
            Default is "auto" which will use the OutputType of the runnable.
            User is free to provide a custom type annotation.
        config_keys: list of config keys that will be accepted, by default
                     no config keys are accepted.
    """
    try:
        from sse_starlette import EventSourceResponse
    except ImportError:
        raise ImportError(
            "sse_starlette must be installed to implement the stream and "
            "stream_log endpoints. "
            "Use `pip install sse_starlette` to install."
        )

    namespace = path or ""

    model_namespace = _replace_non_alphanumeric_with_underscores(path.strip("/"))

    input_type_ = _resolve_model(
        runnable.input_schema if input_type == "auto" else input_type,
        "Input",
        model_namespace,
    )

    output_type_ = _resolve_model(
        runnable.output_schema if output_type == "auto" else output_type,
        "Output",
        model_namespace,
    )

    config = _add_namespace_to_model(
        model_namespace, runnable.config_schema(include=config_keys)
    )
    InvokeRequest = create_invoke_request_model(model_namespace, input_type_, config)
    BatchRequest = create_batch_request_model(model_namespace, input_type_, config)
    StreamRequest = create_stream_request_model(model_namespace, input_type_, config)
    StreamLogRequest = create_stream_log_request_model(
        model_namespace, input_type_, config
    )
    # Generate the response models
    InvokeResponse = create_invoke_response_model(model_namespace, output_type_)
    BatchResponse = create_batch_response_model(model_namespace, output_type_)

    @app.post(
        f"{namespace}/invoke",
        response_model=InvokeResponse,
    )
    async def invoke(
        invoke_request: Annotated[InvokeRequest, InvokeRequest],
        request: Request,
    ) -> InvokeResponse:
        """Invoke the runnable with the given input and config."""
        # Request is first validated using InvokeRequest which takes into account
        # config_keys as well as input_type.
        config = _unpack_config(invoke_request.config, config_keys)
        _add_tracing_info_to_metadata(config, request)
        output = await runnable.ainvoke(
            _unpack_input(invoke_request.input), config=config
        )

        return InvokeResponse(output=simple_dumpd(output))

    #
    @app.post(f"{namespace}/batch", response_model=BatchResponse)
    async def batch(
        batch_request: Annotated[BatchRequest, BatchRequest],
        request: Request,
    ) -> BatchResponse:
        """Invoke the runnable with the given inputs and config."""
        if isinstance(batch_request.config, list):
            config = [
                _unpack_config(config, config_keys) for config in batch_request.config
            ]

            for c in config:
                _add_tracing_info_to_metadata(c, request)
        else:
            config = _unpack_config(batch_request.config, config_keys)
            _add_tracing_info_to_metadata(config, request)
        inputs = [_unpack_input(input_) for input_ in batch_request.inputs]
        output = await runnable.abatch(inputs, config=config)

        return BatchResponse(output=simple_dumpd(output))

    @app.post(f"{namespace}/stream")
    async def stream(
        stream_request: Annotated[StreamRequest, StreamRequest],
        request: Request,
    ) -> EventSourceResponse:
        """Invoke the runnable stream the output.

        This endpoint allows to stream the output of the runnable.

        The endpoint uses a server sent event stream to stream the output.

        https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events

        Important: Set the "text/event-stream" media type for request headers if
                   not using an existing SDK.

        This endpoint uses two different types of events:

        * data - for streaming the output of the runnable

            {
                "event": "data",
                "data": {
                ...
                }
            }

        * end - for signaling the end of the stream.

            This helps the client to know when to stop listening for events and
            know that the streaming has ended successfully.

            {
                "event": "end",
            }
        """
        # Request is first validated using InvokeRequest which takes into account
        # config_keys as well as input_type.
        # After validation, the input is loaded using LangChain's load function.
        input_ = _unpack_input(stream_request.input)
        config = _unpack_config(stream_request.config, config_keys)
        _add_tracing_info_to_metadata(config, request)

        async def _stream() -> AsyncIterator[dict]:
            """Stream the output of the runnable."""
            async for chunk in runnable.astream(
                input_,
                config=config,
            ):
                yield {"data": simple_dumps(chunk), "event": "data"}
            yield {"event": "end"}

        return EventSourceResponse(_stream())

    @app.post(f"{namespace}/stream_log")
    async def stream_log(
        stream_log_request: Annotated[StreamLogRequest, StreamLogRequest],
        request: Request,
    ) -> EventSourceResponse:
        """Invoke the runnable stream_log the output.

        This endpoint allows to stream the output of the runnable, including
        the output of all intermediate steps.

        The endpoint uses a server sent event stream to stream the output.

        https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events

        Important: Set the "text/event-stream" media type for request headers if
                   not using an existing SDK.

        This endpoint uses two different types of events:

        * data - for streaming the output of the runnable

            {
                "event": "data",
                "data": {
                ...
                }
            }

        * end - for signaling the end of the stream.

            This helps the client to know when to stop listening for events and
            know that the streaming has ended successfully.

            {
                "event": "end",
            }
        """
        # Request is first validated using InvokeRequest which takes into account
        # config_keys as well as input_type.
        # After validation, the input is loaded using LangChain's load function.
        input_ = _unpack_input(stream_log_request.input)
        config = _unpack_config(stream_log_request.config, config_keys)
        _add_tracing_info_to_metadata(config, request)

        async def _stream_log() -> AsyncIterator[dict]:
            """Stream the output of the runnable."""
            async for chunk in runnable.astream_log(
                input_,
                config=config,
                diff=stream_log_request.diff,
                include_names=stream_log_request.include_names,
                include_types=stream_log_request.include_types,
                include_tags=stream_log_request.include_tags,
                exclude_names=stream_log_request.exclude_names,
                exclude_types=stream_log_request.exclude_types,
                exclude_tags=stream_log_request.exclude_tags,
            ):
                if stream_log_request.diff:  # Run log patch
                    if not isinstance(chunk, RunLogPatch):
                        raise AssertionError(
                            f"Expected a RunLog instance got {type(chunk)}"
                        )
                    data = {
                        "ops": chunk.ops,
                    }
                else:
                    # Then it's a run log
                    if not isinstance(chunk, RunLog):
                        raise AssertionError(
                            f"Expected a RunLog instance got {type(chunk)}"
                        )
                    data = {
                        "state": chunk.state,
                        "ops": chunk.ops,
                    }

                # Temporary adapter
                yield {
                    "data": simple_dumps(data),
                    "event": "data",
                }
            yield {"event": "end"}

        return EventSourceResponse(_stream_log())

    @app.get(f"{namespace}/input_schema")
    async def input_schema() -> Any:
        """Return the input schema of the runnable."""
        return runnable.input_schema.schema()

    @app.get(f"{namespace}/output_schema")
    async def output_schema() -> Any:
        """Return the output schema of the runnable."""
        return runnable.output_schema.schema()

    @app.get(f"{namespace}/config_schema")
    async def config_schema() -> Any:
        """Return the config schema of the runnable."""
        return runnable.config_schema(include=config_keys).schema()
