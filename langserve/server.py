"""FastAPI integration for langchain runnables.

This code contains integration for langchain runnables with FastAPI.

The main entry point is the `add_routes` function which adds the routes to an existing
FastAPI app or APIRouter.
"""
import contextlib
import json
import re
import weakref
from inspect import isclass
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generator,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from fastapi import HTTPException, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from langchain.callbacks.tracers.log_stream import RunLogPatch
from langchain.load.serializable import Serializable
from langchain.schema.runnable import Runnable, RunnableConfig
from langchain.schema.runnable.config import get_config_list, merge_configs
from langsmith import client as ls_client
from langsmith.utils import LangSmithNotFoundError, tracing_is_enabled
from typing_extensions import Annotated

try:
    from pydantic.v1 import BaseModel, Field, ValidationError, create_model
except ImportError:
    from pydantic import BaseModel, Field, ValidationError, create_model

from langserve.callbacks import AsyncEventAggregatorCallback, CallbackEventDict
from langserve.lzstring import LZString
from langserve.playground import serve_playground
from langserve.pydantic_v1 import _PYDANTIC_MAJOR_VERSION
from langserve.schema import (
    BatchResponseMetadata,
    CustomUserType,
    Feedback,
    FeedbackCreateRequest,
    SingletonResponseMetadata,
)
from langserve.serialization import WellKnownLCSerializer
from langserve.validation import (
    BatchBaseResponse,
    BatchRequestShallowValidator,
    InvokeBaseResponse,
    InvokeRequestShallowValidator,
    StreamLogParameters,
    create_batch_request_model,
    create_batch_response_model,
    create_invoke_request_model,
    create_invoke_response_model,
    create_stream_log_request_model,
    create_stream_request_model,
)
from langserve.version import __version__

try:
    from fastapi import APIRouter, FastAPI
except ImportError:
    # [server] extra not installed
    APIRouter = FastAPI = Any

# A function that that takes a config and a raw request
# and updates the config based on the request.
PerRequestConfigModifier = Callable[[Dict[str, Any], Request], Dict[str, Any]]


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


def _unpack_request_config(
    *configs: Union[BaseModel, Mapping, str],
    keys: Sequence[str],
    model: Type[BaseModel],
    request: Request,
    per_req_config_modifier: Optional[PerRequestConfigModifier],
) -> Dict[str, Any]:
    """Merge configs, and project the given keys from the merged dict."""
    config_dicts = []
    for config in configs:
        if isinstance(config, str):
            config_dicts.append(model(**_config_from_hash(config)).dict())
        elif isinstance(config, BaseModel):
            config_dicts.append(config.dict())
        elif isinstance(config, Mapping):
            config_dicts.append(model(**config).dict())
        else:
            raise TypeError(f"Expected a string, dict or BaseModel got {type(config)}")
    config = merge_configs(*config_dicts)
    projected_config = {k: config[k] for k in keys if k in config}
    return (
        per_req_config_modifier(projected_config, request)
        if per_req_config_modifier
        else projected_config
    )


def _unpack_input(validated_model: BaseModel) -> Any:
    """Unpack the decoded input from the validated model."""
    if hasattr(validated_model, "__root__"):
        model = validated_model.__root__
    else:
        model = validated_model

    if isinstance(model, BaseModel) and not isinstance(
        model, (Serializable, CustomUserType)
    ):
        # If the model is a pydantic model, but not a Serializable, then
        # it was created by the server as part of validation and isn't expected
        # to be accepted by the runnables as input as a pydantic model,
        # instead we need to convert it into a corresponding python dict.
        # This logic should be applied recursively to nested models.
        return {
            fieldname: _unpack_input(getattr(model, fieldname))
            for fieldname in model.__fields__.keys()
        }

    return model


def _rename_pydantic_model(model: Type[BaseModel], prefix: str) -> Type[BaseModel]:
    """Rename the given pydantic model to the given name."""
    return create_model(
        prefix + model.__name__,
        __config__=model.__config__,
        **{
            fieldname: (
                _rename_pydantic_model(field.annotation, prefix)
                if isclass(field.annotation) and issubclass(field.annotation, BaseModel)
                else field.annotation,
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
        model_to_use = _rename_pydantic_model(model, namespace)
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
    model_with_unique_name = _rename_pydantic_model(model, namespace)
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


def _scrub_exceptions_in_event(event: CallbackEventDict) -> CallbackEventDict:
    """Scrub exceptions and change to a serializable format."""
    type_ = event["type"]
    # Check if the event type is one that could contain an error key
    # for example, on_chain_error, on_tool_error, etc.
    if "error" not in type_:
        return event

    # This is not scrubbing -- it's doing serialization
    if "error" not in event:  # if there is an error key, scrub it
        return event

    if isinstance(event["error"], BaseException):
        event.copy()
        event["error"] = {"status_code": 500, "message": "Internal Server Error"}
        return event

    raise AssertionError(f"Expected an exception got {type(event['error'])}")


_APP_SEEN = weakref.WeakSet()
_APP_TO_PATHS = weakref.WeakKeyDictionary()


def _setup_global_app_handlers(app: Union[FastAPI, APIRouter]) -> None:
    @app.on_event("startup")
    async def startup_event():
        # ruff: noqa: E501
        LANGSERVE = """
 __          ___      .__   __.   _______      _______. _______ .______     ____    ____  _______ 
|  |        /   \     |  \ |  |  /  _____|    /       ||   ____||   _  \    \   \  /   / |   ____|
|  |       /  ^  \    |   \|  | |  |  __     |   (----`|  |__   |  |_)  |    \   \/   /  |  |__   
|  |      /  /_\  \   |  . `  | |  | |_ |     \   \    |   __|  |      /      \      /   |   __|  
|  `----./  _____  \  |  |\   | |  |__| | .----)   |   |  |____ |  |\  \----.  \    /    |  |____ 
|_______/__/     \__\ |__| \__|  \______| |_______/    |_______|| _| `._____|   \__/     |_______|
"""

        def green(text):
            return "\x1b[1;32;40m" + text + "\x1b[0m"

        paths = _APP_TO_PATHS[app]
        print(LANGSERVE)
        for path in paths:
            print(
                f'{green("LANGSERVE:")} Playground for chain "{path or ""}/" is live at:'
            )
            print(f'{green("LANGSERVE:")}  │')
            print(f'{green("LANGSERVE:")}  └──> {path}/playground/')
            print(f'{green("LANGSERVE:")}')
        print(f'{green("LANGSERVE:")} See all available routes at {app.docs_url}/')
        print()


def _register_path_for_app(app: Union[FastAPI, APIRouter], path: str) -> None:
    """Register a path when its added to app. Raise if path already seen."""
    if app in _APP_TO_PATHS:
        seen_paths = _APP_TO_PATHS.get(app)
        if path in seen_paths:
            raise ValueError(
                f"A runnable already exists at path: {path}. If adding "
                f"multiple runnables make sure they have different paths."
            )
        seen_paths.add(path)
    else:
        _setup_global_app_handlers(app)
        _APP_TO_PATHS[app] = {path}


@contextlib.contextmanager
def _with_validation_error_translation() -> Generator[None, None, None]:
    """Context manager to translate validation errors to request validation errors.

    This makes sure that validation errors are surfaced as client side errors.
    """
    try:
        yield
    except ValidationError as e:
        raise RequestValidationError(e.errors(), body=e.model)


def _get_base_run_id_as_str(
    event_aggregator: AsyncEventAggregatorCallback,
) -> Optional[str]:
    """
    Uses `event_aggregator` to determine the base run ID for a given run. Returns
    the run_id as a string, or None if it does not exist.
    """
    # The first run in the callback_events list corresponds to the
    # overall trace for request
    if event_aggregator.callback_events and event_aggregator.callback_events[0].get(
        "run_id"
    ):
        return str(event_aggregator.callback_events[0].get("run_id"))
    else:
        raise AssertionError("No run_id found for the given run")


def _json_encode_response(model: BaseModel) -> JSONResponse:
    """Return a JSONResponse with the given content.

    We're doing the encoding manually here as a workaround to fastapi
    not supporting models from pydantic v1 when pydantic
    v2 is imported.

    Args:
        obj: The object to encode; either an invoke response or a batch response.

    Returns:
        A JSONResponse with the given content.
    """
    obj = jsonable_encoder(model)

    if isinstance(model, InvokeBaseResponse):
        # Invoke Response
        # Collapse '__root__' from output field if it exists. This is done
        # automatically by fastapi when annotating request and response with
        # We need to do this manually since we're using vanilla JSONResponse
        if isinstance(obj["output"], dict) and "__root__" in obj["output"]:
            obj["output"] = obj["output"]["__root__"]

        if "callback_events" in obj:
            for idx, callback_event in enumerate(obj["callback_events"]):
                if isinstance(callback_event, dict) and "__root__" in callback_event:
                    obj["callback_events"][idx] = callback_event["__root__"]
    elif isinstance(model, BatchBaseResponse):
        if not isinstance(obj["output"], list):
            raise AssertionError("Expected output to be a list")

        # Collapse '__root__' from output field if it exists. This is done
        # automatically by fastapi when annotating request and response with
        # We need to do this manually since we're using vanilla JSONResponse
        outputs = obj["output"]
        for idx, output in enumerate(outputs):
            if isinstance(output, dict) and "__root__" in output:
                outputs[idx] = output["__root__"]

        if "callback_events" in obj:
            if not isinstance(obj["callback_events"], list):
                raise AssertionError("Expected callback_events to be a list")

            for callback_events in obj["callback_events"]:
                for idx, callback_event in enumerate(callback_events):
                    if (
                        isinstance(callback_event, dict)
                        and "__root__" in callback_event
                    ):
                        callback_events[idx] = callback_event["__root__"]
    else:
        raise AssertionError(
            f"Expected a InvokeBaseResponse or BatchBaseResponse got: {type(model)}"
        )

    return JSONResponse(content=obj)


# PUBLIC API


def add_routes(
    app: Union[FastAPI, APIRouter],
    runnable: Runnable,
    *,
    path: str = "",
    input_type: Union[Type, Literal["auto"], BaseModel] = "auto",
    output_type: Union[Type, Literal["auto"], BaseModel] = "auto",
    config_keys: Sequence[str] = (),
    include_callback_events: bool = False,
    enable_feedback_endpoint: bool = False,
    per_req_config_modifier: Optional[PerRequestConfigModifier] = None,
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
            Favor using runnable.with_types(input_type=..., output_type=...) instead.
            This parameter may get deprecated!
        output_type: type to use for output validation.
            Default is "auto" which will use the OutputType of the runnable.
            User is free to provide a custom type annotation.
            Favor using runnable.with_types(input_type=..., output_type=...) instead.
            This parameter may get deprecated!
        config_keys: list of config keys that will be accepted, by default
                     no config keys are accepted.
        include_callback_events: Whether to include callback events in the response.
            If true, the client will be able to show trace information
            including events that occurred on the server side.
            Be sure not to include any sensitive information in the callback events.
        enable_feedback_endpoint: Whether to enable an endpoint for logging feedback
            to LangSmith. Enabled by default. If this flag is disabled or LangSmith
            tracing is not enabled for the runnable, then 400 errors will be thrown
            when accessing the feedback endpoint
        per_req_config_modifier: optional function that can be used to update the
            RunnableConfig for a given run based on the raw request. This is useful,
            for example, if the user wants to pass in a header containing credentials
            to a runnable. The RunnableConfig is presented in its dictionary form.
            Note that only keys in `config_keys` will be modifiable by this function.
    """
    try:
        from sse_starlette import EventSourceResponse
    except ImportError:
        raise ImportError(
            "sse_starlette must be installed to implement the stream and "
            "stream_log endpoints. "
            "Use `pip install sse_starlette` to install."
        )

    if isinstance(app, FastAPI):  # type: ignore
        # Cannot do this checking logic for a router since
        # API routers are not hashable
        _register_path_for_app(app, path)
    well_known_lc_serializer = WellKnownLCSerializer()

    if path and not path.startswith("/"):
        raise ValueError(
            f"Got an invalid path: {path}. "
            f"If specifying path please start it with a `/`"
        )

    namespace = path or ""

    model_namespace = _replace_non_alphanumeric_with_underscores(path.strip("/"))

    route_tags = [path.strip("/")] if path else None
    route_tags_with_config = [f"{path.strip('/')}/config"] if path else ["config"]

    # Please do not change the naming on ls_client. It is used with mocking
    # in our unit tests for langsmith integrations.
    langsmith_client = (
        ls_client.Client()
        if tracing_is_enabled() and enable_feedback_endpoint
        else None
    )

    def _route_name(name: str) -> str:
        """Return the route name with the given name."""
        return f"{path.strip('/')} {name}" if path else name

    def _route_name_with_config(name: str) -> str:
        """Return the route name with the given name."""
        return (
            f"{path.strip('/')} {name} with config" if path else f"{name} with config"
        )

    if hasattr(app, "openapi_tags") and (path or (app not in _APP_SEEN)):
        if not path:
            _APP_SEEN.add(app)
        app.openapi_tags = [
            *(getattr(app, "openapi_tags", []) or []),
            {
                "name": route_tags[0] if route_tags else "default",
            },
            {
                "name": route_tags_with_config[0],
                "description": (
                    "Endpoints with a default configuration "
                    "set by `config_hash` path parameter."
                ),
            },
        ]

    with_types = {}

    if input_type != "auto":
        with_types["input_type"] = input_type
    if output_type != "auto":
        with_types["output_type"] = output_type

    if with_types:
        runnable = runnable.with_types(**with_types)

    input_type_ = _resolve_model(runnable.get_input_schema(), "Input", model_namespace)

    output_type_ = _resolve_model(
        runnable.get_output_schema(),
        "Output",
        model_namespace,
    )

    ConfigPayload = _add_namespace_to_model(
        model_namespace, runnable.config_schema(include=config_keys)
    )

    InvokeRequest = create_invoke_request_model(
        model_namespace, input_type_, ConfigPayload
    )
    BatchRequest = create_batch_request_model(
        model_namespace, input_type_, ConfigPayload
    )
    StreamRequest = create_stream_request_model(
        model_namespace, input_type_, ConfigPayload
    )
    StreamLogRequest = create_stream_log_request_model(
        model_namespace, input_type_, ConfigPayload
    )
    # Generate the response models
    InvokeResponse = create_invoke_response_model(model_namespace, output_type_)
    BatchResponse = create_batch_response_model(model_namespace, output_type_)

    async def _get_config_and_input(
        request: Request, config_hash: str
    ) -> Tuple[RunnableConfig, Any]:
        """Extract the config and input from the request, validating the request."""
        try:
            body = await request.json()
        except json.JSONDecodeError:
            raise RequestValidationError(errors=["Invalid JSON body"])
        try:
            body = InvokeRequestShallowValidator.validate(body)

            # Merge the config from the path with the config from the body.
            config = _unpack_request_config(
                config_hash,
                body.config,
                keys=config_keys,
                model=ConfigPayload,
                request=request,
                per_req_config_modifier=per_req_config_modifier,
            )
            # Unpack the input dynamically using the input schema of the runnable.
            # This takes into account changes in the input type when
            # using configuration.
            schema = runnable.with_config(config).input_schema
            input_ = schema.validate(body.input)
            return config, _unpack_input(input_)
        except ValidationError as e:
            raise RequestValidationError(e.errors(), body=body)

    @app.post(
        namespace + "/c/{config_hash}/invoke",
        include_in_schema=False,
    )
    @app.post(f"{namespace}/invoke", include_in_schema=False)
    async def invoke(
        request: Request,
        config_hash: str = "",
    ) -> Response:
        """Invoke the runnable with the given input and config."""
        # We do not use the InvokeRequest model here since configurable runnables
        # have dynamic schema -- so the validation below is a bit more involved.
        config, input_ = await _get_config_and_input(request, config_hash)

        event_aggregator = AsyncEventAggregatorCallback()
        _add_tracing_info_to_metadata(config, request)
        config["callbacks"] = [event_aggregator]
        output = await runnable.ainvoke(input_, config=config)

        if include_callback_events:
            callback_events = [
                _scrub_exceptions_in_event(event)
                for event in event_aggregator.callback_events
            ]
        else:
            callback_events = []

        return _json_encode_response(
            InvokeResponse(
                output=well_known_lc_serializer.dumpd(output),
                # Callbacks are scrubbed and exceptions are converted to serializable format
                # before returned in the response.
                callback_events=callback_events,
                metadata=SingletonResponseMetadata(
                    run_id=_get_base_run_id_as_str(event_aggregator)
                ),
            ),
        )

    @app.post(
        namespace + "/c/{config_hash}/batch",
        include_in_schema=False,
    )
    @app.post(f"{namespace}/batch", include_in_schema=False)
    async def batch(
        request: Request,
        config_hash: str = "",
    ) -> Response:
        """Invoke the runnable with the given inputs and config."""
        try:
            body = await request.json()
        except json.JSONDecodeError:
            raise RequestValidationError(errors=["Invalid JSON body"])

        with _with_validation_error_translation():
            body = BatchRequestShallowValidator.validate(body)
            config = body.config

            # First unpack the config
            if isinstance(config, list):
                if len(config) != len(body.inputs):
                    raise HTTPException(
                        422,
                        f"Number of configs ({len(config)}) must "
                        f"match number of inputs ({len(body.inputs)})",
                    )

                configs = [
                    _unpack_request_config(
                        config_hash,
                        config,
                        keys=config_keys,
                        model=ConfigPayload,
                        request=request,
                        per_req_config_modifier=per_req_config_modifier,
                    )
                    for config in config
                ]
            elif isinstance(config, dict):
                configs = _unpack_request_config(
                    config_hash,
                    config,
                    keys=config_keys,
                    model=ConfigPayload,
                    request=request,
                    per_req_config_modifier=per_req_config_modifier,
                )
            else:
                raise HTTPException(
                    422, "Value for 'config' key must be a dict or list if provided"
                )

        inputs_ = body.inputs

        # Make sure that the number of configs matches the number of inputs
        # Since we'll be adding callbacks to the configs.

        configs_ = [
            {k: v for k, v in config_.items() if k in config_keys}
            for config_ in get_config_list(configs, len(inputs_))
        ]

        inputs = [
            _unpack_input(runnable.with_config(config_).input_schema.validate(input_))
            for config_, input_ in zip(configs_, inputs_)
        ]

        # Update the configuration with callbacks
        aggregators = [AsyncEventAggregatorCallback() for _ in range(len(inputs))]

        for config_, aggregator in zip(configs_, aggregators):
            _add_tracing_info_to_metadata(config_, request)
            config_["callbacks"] = [aggregator]

        output = await runnable.abatch(inputs, config=configs_)

        if include_callback_events:
            callback_events = [
                # Scrub sensitive information and convert
                # exceptions to serializable format
                [
                    _scrub_exceptions_in_event(event)
                    for event in aggregator.callback_events
                ]
                for aggregator in aggregators
            ]
        else:
            callback_events = []

        obj = BatchResponse(
            output=well_known_lc_serializer.dumpd(output),
            callback_events=callback_events,
            metadata=BatchResponseMetadata(
                run_ids=[_get_base_run_id_as_str(agg) for agg in aggregators]
            ),
        )
        return _json_encode_response(obj)

    @app.post(namespace + "/c/{config_hash}/stream", include_in_schema=False)
    @app.post(f"{namespace}/stream", include_in_schema=False)
    async def stream(
        request: Request,
        config_hash: str = "",
    ) -> EventSourceResponse:
        """Invoke the runnable stream the output.

        See documentation for endpoint at the end of the file.
        It's attached to _stream_docs endpoint.
        """
        err_event = {}
        validation_exception: Optional[BaseException] = None
        try:
            config, input_ = await _get_config_and_input(request, config_hash)
        except BaseException as e:
            validation_exception = e
            if isinstance(e, RequestValidationError):
                err_event = {
                    "event": "error",
                    "data": json.dumps(
                        {"status_code": 422, "message": repr(e.errors())}
                    ),
                }
            else:
                err_event = {
                    "event": "error",
                    # Do not expose the error message to the client since
                    # the message may contain sensitive information.
                    "data": json.dumps(
                        {"status_code": 500, "message": "Internal Server Error"}
                    ),
                }

        async def _stream() -> AsyncIterator[dict]:
            """Stream the output of the runnable."""
            if validation_exception:
                yield err_event
                if isinstance(validation_exception, RequestValidationError):
                    return
                else:
                    raise AssertionError(
                        "Internal server error"
                    ) from validation_exception

            try:
                config_w_callbacks = config.copy()
                event_aggregator = AsyncEventAggregatorCallback()
                config_w_callbacks["callbacks"] = [event_aggregator]
                has_sent_metadata = False
                async for chunk in runnable.astream(
                    input_,
                    config=config_w_callbacks,
                ):
                    if not has_sent_metadata and event_aggregator.callback_events:
                        yield {
                            "event": "metadata",
                            "data": json.dumps(
                                {
                                    "run_id": _get_base_run_id_as_str(event_aggregator),
                                }
                            ),
                        }
                        has_sent_metadata = True

                    yield {
                        # EventSourceResponse expects a string for data
                        # so after serializing into bytes, we decode into utf-8
                        # to get a string.
                        "data": well_known_lc_serializer.dumps(chunk).decode("utf-8"),
                        "event": "data",
                    }
                yield {"event": "end"}
            except BaseException:
                yield {
                    "event": "error",
                    # Do not expose the error message to the client since
                    # the message may contain sensitive information.
                    # We'll add client side errors for validation as well.
                    "data": json.dumps(
                        {"status_code": 500, "message": "Internal Server Error"}
                    ),
                }
                raise

        return EventSourceResponse(_stream())

    @app.post(
        namespace + "/c/{config_hash}/stream_log",
        include_in_schema=False,
    )
    @app.post(
        f"{namespace}/stream_log",
        include_in_schema=False,
    )
    async def stream_log(
        request: Request,
        config_hash: str = "",
    ) -> EventSourceResponse:
        """Invoke the runnable stream_log the output.

        View documentation for endpoint at the end of the file.
        It's attached to _stream_log_docs endpoint.
        """
        err_event = {}
        validation_exception: Optional[BaseException] = None
        try:
            config, input_ = await _get_config_and_input(request, config_hash)
        except BaseException as e:
            validation_exception = e
            if isinstance(e, RequestValidationError):
                err_event = {
                    "event": "error",
                    "data": json.dumps(
                        {"status_code": 422, "message": repr(e.errors())}
                    ),
                }
            else:
                err_event = {
                    "event": "error",
                    # Do not expose the error message to the client since
                    # the message may contain sensitive information.
                    "data": json.dumps(
                        {"status_code": 500, "message": "Internal Server Error"}
                    ),
                }

        try:
            body = await request.json()
            with _with_validation_error_translation():
                stream_log_request = StreamLogParameters(**body)
        except json.JSONDecodeError:
            # Body as text
            validation_exception = RequestValidationError(errors=["Invalid JSON body"])
            err_event = {
                "event": "error",
                "data": json.dumps(
                    {"status_code": 422, "message": "Invalid JSON body"}
                ),
            }
        except RequestValidationError as e:
            validation_exception = e
            err_event = {
                "event": "error",
                "data": json.dumps({"status_code": 422, "message": repr(e.errors())}),
            }

        async def _stream_log() -> AsyncIterator[dict]:
            """Stream the output of the runnable."""
            if validation_exception:
                yield err_event
                if isinstance(validation_exception, RequestValidationError):
                    return
                else:
                    raise AssertionError(
                        "Internal server error"
                    ) from validation_exception

            try:
                async for chunk in runnable.astream_log(
                    input_,
                    config=config,
                    diff=True,
                    include_names=stream_log_request.include_names,
                    include_types=stream_log_request.include_types,
                    include_tags=stream_log_request.include_tags,
                    exclude_names=stream_log_request.exclude_names,
                    exclude_types=stream_log_request.exclude_types,
                    exclude_tags=stream_log_request.exclude_tags,
                ):
                    if not isinstance(chunk, RunLogPatch):
                        raise AssertionError(
                            f"Expected a RunLog instance got {type(chunk)}"
                        )
                    data = {
                        "ops": chunk.ops,
                    }

                    # Temporary adapter
                    yield {
                        # EventSourceResponse expects a string for data
                        # so after serializing into bytes, we decode into utf-8
                        # to get a string.
                        "data": well_known_lc_serializer.dumps(data).decode("utf-8"),
                        "event": "data",
                    }
                yield {"event": "end"}
            except BaseException:
                yield {
                    "event": "error",
                    # Do not expose the error message to the client since
                    # the message may contain sensitive information.
                    # We'll add client side errors for validation as well.
                    "data": json.dumps(
                        {"status_code": 500, "message": "Internal Server Error"}
                    ),
                }
                raise

        return EventSourceResponse(_stream_log())

    @app.get(
        namespace + "/c/{config_hash}/input_schema",
        tags=route_tags_with_config,
        name=_route_name_with_config("input_schema"),
    )
    @app.get(
        f"{namespace}/input_schema", tags=route_tags, name=_route_name("input_schema")
    )
    async def input_schema(request: Request, config_hash: str = "") -> Any:
        """Return the input schema of the runnable."""
        with _with_validation_error_translation():
            config = _unpack_request_config(
                config_hash,
                keys=config_keys,
                model=ConfigPayload,
                request=request,
                per_req_config_modifier=per_req_config_modifier,
            )

        return runnable.get_input_schema(config).schema()

    @app.get(
        namespace + "/c/{config_hash}/output_schema",
        tags=route_tags_with_config,
        name=_route_name_with_config("output_schema"),
    )
    @app.get(
        f"{namespace}/output_schema",
        tags=route_tags,
        name=_route_name("output_schema"),
    )
    async def output_schema(request: Request, config_hash: str = "") -> Any:
        """Return the output schema of the runnable."""
        with _with_validation_error_translation():
            config = _unpack_request_config(
                config_hash,
                keys=config_keys,
                model=ConfigPayload,
                request=request,
                per_req_config_modifier=per_req_config_modifier,
            )
        return runnable.get_output_schema(config).schema()

    @app.get(
        namespace + "/c/{config_hash}/config_schema",
        tags=route_tags_with_config,
        name=_route_name_with_config("config_schema"),
    )
    @app.get(
        f"{namespace}/config_schema", tags=route_tags, name=_route_name("config_schema")
    )
    async def config_schema(request: Request, config_hash: str = "") -> Any:
        """Return the config schema of the runnable."""
        with _with_validation_error_translation():
            config = _unpack_request_config(
                config_hash,
                keys=config_keys,
                model=ConfigPayload,
                request=request,
                per_req_config_modifier=per_req_config_modifier,
            )
        return runnable.with_config(config).config_schema(include=config_keys).schema()

    @app.get(
        namespace + "/c/{config_hash}/playground/{file_path:path}",
        include_in_schema=False,
    )
    @app.get(namespace + "/playground/{file_path:path}", include_in_schema=False)
    async def playground(
        file_path: str, request: Request, config_hash: str = ""
    ) -> Any:
        """Return the playground of the runnable."""
        with _with_validation_error_translation():
            config = _unpack_request_config(
                config_hash,
                keys=config_keys,
                model=ConfigPayload,
                request=request,
                per_req_config_modifier=per_req_config_modifier,
            )
        return await serve_playground(
            runnable.with_config(config),
            runnable.with_config(config).input_schema,
            config_keys,
            f"{namespace}/playground",
            file_path,
        )

    @app.post(namespace + "/feedback")
    async def feedback(feedback_create_req: FeedbackCreateRequest) -> Feedback:
        """
        Send feedback on an individual run to langsmith
        """

        if not tracing_is_enabled() or not enable_feedback_endpoint:
            raise HTTPException(
                400,
                "The feedback endpoint is only accessible when LangSmith is "
                + "enabled on your LangServe server.",
            )

        try:
            feedback_from_langsmith = langsmith_client.create_feedback(
                feedback_create_req.run_id,
                feedback_create_req.key,
                score=feedback_create_req.score,
                value=feedback_create_req.value,
                comment=feedback_create_req.comment,
                source_info={
                    "from_langserve": True,
                },
                # We execute eagerly, meaning we confirm the run exists in
                # LangSmith before returning a response to the user. This ensures
                # that clients of the UI know that the feedback was successfully
                # recorded before they receive a 200 response
                eager=True,
                # We lower the number of attempts to 3 to ensure we have time
                # to wait for a run to show up, but do not take forever in cases
                # of bad input
                stop_after_attempt=3,
            )
        except LangSmithNotFoundError:
            raise HTTPException(404, "No run with the given run_id exists")

        # We purposefully select out fields from langsmith so that we don't
        # fail validation if langsmith adds extra fields. We prefer this over
        # using "Extra.allow" in pydantic since syntax changes between pydantic
        # 1.x and 2.x for this functionality
        return Feedback(
            run_id=str(feedback_from_langsmith.run_id),
            created_at=str(feedback_from_langsmith.created_at),
            modified_at=str(feedback_from_langsmith.modified_at),
            key=str(feedback_from_langsmith.key),
            score=feedback_from_langsmith.score,
            value=feedback_from_langsmith.value,
            comment=feedback_from_langsmith.comment,
        )

    #######################################
    # Documentation variants of end points.
    #######################################
    # At the moment, we only support pydantic 1.x for documentation
    if _PYDANTIC_MAJOR_VERSION == 1:

        @app.post(
            namespace + "/c/{config_hash}/invoke",
            response_model=InvokeResponse,
            tags=route_tags_with_config,
            name=_route_name_with_config("invoke"),
        )
        @app.post(
            f"{namespace}/invoke",
            response_model=InvokeResponse,
            tags=route_tags,
            name=_route_name("invoke"),
        )
        async def _invoke_docs(
            invoke_request: Annotated[InvokeRequest, InvokeRequest],
            config_hash: str = "",
        ) -> InvokeResponse:
            """Invoke the runnable with the given input and config."""
            raise AssertionError("This endpoint should not be reachable.")

        @app.post(
            namespace + "/c/{config_hash}/batch",
            response_model=BatchResponse,
            tags=route_tags_with_config,
            name=_route_name_with_config("batch"),
        )
        @app.post(
            f"{namespace}/batch",
            response_model=BatchResponse,
            tags=route_tags,
            name=_route_name("batch"),
        )
        async def _batch_docs(
            batch_request: Annotated[BatchRequest, BatchRequest],
            config_hash: str = "",
        ) -> BatchResponse:
            """Batch invoke the runnable with the given inputs and config."""
            raise AssertionError("This endpoint should not be reachable.")

        @app.post(
            namespace + "/c/{config_hash}/stream",
            include_in_schema=True,
            tags=route_tags_with_config,
            name=_route_name_with_config("stream"),
        )
        @app.post(
            f"{namespace}/stream",
            include_in_schema=True,
            tags=route_tags,
            name=_route_name("stream"),
        )
        async def _stream_docs(
            stream_request: Annotated[StreamRequest, StreamRequest],
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

            * error - for signaling an error in the stream, also ends the stream.

            {
                "event": "error",
                "data": {
                    "status_code": 500,
                    "message": "Internal Server Error"
                }
            }

            * end - for signaling the end of the stream.

                This helps the client to know when to stop listening for events and
                know that the streaming has ended successfully.

                {
                    "event": "end",
                }
            """
            raise AssertionError("This endpoint should not be reachable.")

        @app.post(
            namespace + "/c/{config_hash}/stream_log",
            include_in_schema=True,
            tags=route_tags_with_config,
            name=_route_name_with_config("stream_log"),
        )
        @app.post(
            f"{namespace}/stream_log",
            include_in_schema=True,
            tags=route_tags,
            name=_route_name("stream_log"),
        )
        async def _stream_log_docs(
            stream_log_request: Annotated[StreamLogRequest, StreamLogRequest],
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

            * error - for signaling an error in the stream, also ends the stream.

            {
                "event": "error",
                "data": {
                    "status_code": 500,
                    "message": "Internal Server Error"
                }
            }

            * end - for signaling the end of the stream.

                This helps the client to know when to stop listening for events and
                know that the streaming has ended successfully.

                {
                    "event": "end",
                }
            """
            raise AssertionError("This endpoint should not be reachable.")
