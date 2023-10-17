"""FastAPI integration for langchain runnables.

This code contains integration for langchain runnables with FastAPI.

The main entry point is the `add_routes` function which adds the routes to an existing
FastAPI app or APIRouter.
"""
import json
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

from fastapi import HTTPException, Request
from langchain.callbacks.tracers.log_stream import RunLog, RunLogPatch
from langchain.load.serializable import Serializable
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import merge_configs
from typing_extensions import Annotated

from langserve.version import __version__
from langserve.lzstring import LZString

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


def _config_from_hash(config_hash: str) -> Dict[str, Any]:
    try:
        if not config_hash:
            return {}

        uncompressed = LZString.decompressFromEncodedURIComponent(config_hash)
        parsed = json.loads(uncompressed)
        if isinstance(parsed, dict):
            return parsed
        else:
            raise HTTPException(400, "Invalid config hash")
    except Exception:
        raise HTTPException(400, "Invalid config hash")


def _unpack_config(
    *configs: Union[BaseModel, Mapping, str], keys: Sequence[str]
) -> Dict[str, Any]:
    """Merge configs, and project the given keys from the merged dict."""
    config_dicts = []
    for config in configs:
        if isinstance(config, str):
            config_dicts.append(_config_from_hash(config))
        elif isinstance(config, BaseModel):
            config_dicts.append(config.dict())
        else:
            config_dicts.append(config)
    config = merge_configs(*config_dicts)
    return {k: config[k] for k in keys if k in config}


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


# This is a global registry of models to avoid creating the same model
# multiple times.
# Duplicated model names break fastapi's openapi generation.
_MODEL_REGISTRY = {}


def _resolve_model(type_: Union[Type, BaseModel], default_name: str) -> Type[BaseModel]:
    """Resolve the input type to a BaseModel."""
    if isclass(type_) and issubclass(type_, BaseModel):
        model = type_
    else:
        model = create_model(default_name, __root__=(type_, ...))

    hash_ = model.schema_json()

    if hash_ not in _MODEL_REGISTRY:
        _MODEL_REGISTRY[hash_] = model

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

    class Config:
        arbitrary_types_allowed = True

    model_with_unique_name = create_model(
        f"{namespace}{model.__name__}",
        config=Config,
        **{
            name: (
                field.annotation,
                Field(
                    field.default,
                    title=name,
                    description=field.field_info.description,
                ),
            )
            for name, field in model.__fields__.items()
        },
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

    input_type_ = _resolve_model(
        runnable.input_schema if input_type == "auto" else input_type, "Input"
    )

    output_type_ = _resolve_model(
        runnable.output_schema if output_type == "auto" else output_type, "Output"
    )

    namespace = path or ""

    model_namespace = path.strip("/").replace("/", "_")

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

    @app.post(namespace + "/h{config_hash}/invoke", response_model=InvokeResponse)
    @app.post(f"{namespace}/invoke", response_model=InvokeResponse)
    async def invoke(
        invoke_request: Annotated[InvokeRequest, InvokeRequest],
        request: Request,
        config_hash: str = "",
    ) -> InvokeResponse:
        """Invoke the runnable with the given input and config."""
        # Request is first validated using InvokeRequest which takes into account
        # config_keys as well as input_type.
        config = _unpack_config(config_hash, invoke_request.config, keys=config_keys)
        _add_tracing_info_to_metadata(config, request)
        output = await runnable.ainvoke(
            _unpack_input(invoke_request.input), config=config
        )

        return InvokeResponse(output=simple_dumpd(output))

    @app.post(namespace + "/h{config_hash}/batch", response_model=BatchResponse)
    @app.post(f"{namespace}/batch", response_model=BatchResponse)
    async def batch(
        batch_request: Annotated[BatchRequest, BatchRequest],
        request: Request,
        config_hash: str = "",
    ) -> BatchResponse:
        """Invoke the runnable with the given inputs and config."""
        if isinstance(batch_request.config, list):
            config = [
                _unpack_config(config, keys=config_keys)
                for config in batch_request.config
            ]

            for c in config:
                _add_tracing_info_to_metadata(c, request)
        else:
            config = _unpack_config(config_hash, batch_request.config, keys=config_keys)
            _add_tracing_info_to_metadata(config, request)
        inputs = [_unpack_input(input_) for input_ in batch_request.inputs]
        output = await runnable.abatch(inputs, config=config)

        return BatchResponse(output=simple_dumpd(output))

    @app.post(namespace + "/h{config_hash}/stream")
    @app.post(f"{namespace}/stream")
    async def stream(
        stream_request: Annotated[StreamRequest, StreamRequest],
        request: Request,
        config_hash: str = "",
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
        config = _unpack_config(config_hash, stream_request.config, keys=config_keys)
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

    @app.post(namespace + "/h{config_hash}/stream_log")
    @app.post(f"{namespace}/stream_log")
    async def stream_log(
        stream_log_request: Annotated[StreamLogRequest, StreamLogRequest],
        request: Request,
        config_hash: str = "",
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
        config = _unpack_config(
            config_hash, stream_log_request.config, keys=config_keys
        )
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

    @app.get(namespace + "/h{config_hash}/input_schema")
    @app.get(f"{namespace}/input_schema")
    async def input_schema(config_hash: str = "") -> Any:
        """Return the input schema of the runnable."""
        return runnable.input_schema.schema()

    @app.get(namespace + "/h{config_hash}/output_schema")
    @app.get(f"{namespace}/output_schema")
    async def output_schema(config_hash: str = "") -> Any:
        """Return the output schema of the runnable."""
        return runnable.output_schema.schema()

    @app.get(namespace + "/h{config_hash}/config_schema")
    @app.get(f"{namespace}/config_schema")
    async def config_schema(config_hash: str = "") -> Any:
        """Return the config schema of the runnable."""
        return runnable.config_schema(include=config_keys).schema()
