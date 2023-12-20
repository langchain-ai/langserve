import contextlib
import importlib
import json
import os
import re
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

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.callbacks.tracers.log_stream import RunLogPatch
from langchain.load.serializable import Serializable
from langchain.schema.runnable import Runnable, RunnableConfig
from langchain.schema.runnable.config import get_config_list, merge_configs
from langsmith import client as ls_client
from langsmith.utils import tracing_is_enabled
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from langserve.callbacks import AsyncEventAggregatorCallback, CallbackEventDict
from langserve.lzstring import LZString
from langserve.playground import serve_playground
from langserve.pydantic_v1 import BaseModel, Field, ValidationError, create_model
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
    from sse_starlette import EventSourceResponse
except ImportError:
    EventSourceResponse = Any


def _is_hosted() -> bool:
    return os.environ.get("HOSTED_LANGSERVE_ENABLED", "false").lower() == "true"


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


PerRequestConfigModifier = Callable[[Dict[str, Any], Request], Dict[str, Any]]


def _unpack_request_config(
    *client_sent_configs: Union[BaseModel, Mapping, str],
    config_keys: Sequence[str],
    model: Type[BaseModel],
    request: Request,
    per_req_config_modifier: Optional[PerRequestConfigModifier],
    server_config: Optional[RunnableConfig],
) -> Dict[str, Any]:
    """Merge configs, and project the given keys from the merged dict.

    Args:
        client_sent_configs: configs sent by the client. These will be
            merged together with last writer wins semantics.
            Each of these configs will be validated.
        config_keys: keys to project from the merged config.
            Used to make sure that a client can't override any configuration
            that the server doesn't want to allow overriding.
        model: pydantic model to use for validation of the client sent configs.
        request: The request object.
        server_config: optional server configuration that will be merged
           into the configuration sent by the client. It's written after
           any client configuration, and can override any client configuration.
           This runs before the per_req_config_modifier.
        per_req_config_modifier: optional function that can be used to update the
            RunnableConfig for a given run based on the raw request. This is useful,
            for example, if the user wants to pass in a header containing
            credentials to a runnable. The RunnableConfig is presented in its
            dictionary form. Note that only keys in `config_keys` will be
            modifiable by this function.
    """
    config_dicts = []
    for config in client_sent_configs:
        if isinstance(config, str):
            config_dicts.append(model(**_config_from_hash(config)).dict())
        elif isinstance(config, BaseModel):
            config_dicts.append(config.dict())
        elif isinstance(config, Mapping):
            config_dicts.append(model(**config).dict())
        else:
            raise TypeError(f"Expected a string, dict or BaseModel got {type(config)}")
    config = merge_configs(*config_dicts)
    if "configurable" in config and config["configurable"]:
        if "configurable" not in config_keys:
            raise HTTPException(
                422,
                "The config field `configurable` has been disallowed by the server. "
                "This can be modified server side by adding `configurable` to the list "
                "of `config_keys` argument in `add_routes`",
            )
    projected_config = {k: config[k] for k in config_keys if k in config}

    if server_config:
        # Merge the server provided config last, so that it overrides any
        # client provided config.
        projected_config = merge_configs(projected_config, server_config)

    # Finally apply the per_req_config_modifier if it was provided.
    return (
        per_req_config_modifier(projected_config, request)
        if per_req_config_modifier
        else projected_config
    )


def _update_config_with_defaults(
    path: str,
    incoming_config: RunnableConfig,
    request: Request,
    *,
    endpoint: Optional[str] = None,
) -> RunnableConfig:
    """Set up some baseline configuration for the underlying runnable."""

    # Currently all defaults are non-overridable
    overridable_default_config = RunnableConfig()

    metadata = {
        "__useragent": request.headers.get("user-agent"),
        "__langserve_version": __version__,
    }

    if endpoint:
        metadata["__langserve_endpoint"] = endpoint

    if _is_hosted():
        hosted_metadata = {
            "__langserve_hosted_git_commit_sha": os.environ.get(
                "HOSTED_LANGSERVE_GIT_COMMIT", ""
            ),
            "__langserve_hosted_repo_subdirectory_path": os.environ.get(
                "HOSTED_LANGSERVE_GIT_REPO_PATH", ""
            ),
            "__langserve_hosted_repo_url": os.environ.get(
                "HOSTED_LANGSERVE_GIT_REPO", ""
            ),
            "__langserve_hosted_is_hosted": "true",
        }
        metadata.update(hosted_metadata)

    non_overridable_default_config = RunnableConfig(
        run_name=path,
        metadata=metadata,
    )

    # merge_configs is last-writer-wins, so we specifically pass in the
    # overridable configs first, then the user provided configs, then
    # finally the non-overridable configs
    return merge_configs(
        overridable_default_config,
        incoming_config,
        non_overridable_default_config,
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


def _add_callbacks(
    config: RunnableConfig, callbacks: Sequence[AsyncCallbackHandler]
) -> None:
    """Add the callback aggregator to the config."""
    if "callbacks" not in config:
        config["callbacks"] = []
    config["callbacks"].extend(callbacks)


_MODEL_REGISTRY = {}
_SEEN_NAMES = set()


# PUBLIC API


class APIHandler:
    """Implementation of the various API endpoints for a runnable server.

    This is a private class whose API is expected to change.

    Currently, the sole role of the handler at the moment is to separate the
    implementation of the endpoints from the logic that registers them on
    a FastAPI app and logic that adds them to the FastAPI OpenAPI docs.
    """

    def __init__(
        self,
        runnable: Runnable,
        *,
        # Named arguments below to make it easier to do gracious deprecation
        # of features in the future if need be.
        path: str,
        prefix: str = "",
        input_type: Union[Type, Literal["auto"], BaseModel] = "auto",
        output_type: Union[Type, Literal["auto"], BaseModel] = "auto",
        config_keys: Sequence[str] = ("configurable",),
        include_callback_events: bool = False,
        enable_feedback_endpoint: bool = False,
        per_req_config_modifier: Optional[PerRequestConfigModifier] = None,
        stream_log_name_allow_list: Optional[Sequence[str]] = None,
    ) -> None:
        """Create an API handler for the given runnable.

        The API Handler has implementations for the various endpoints
        associated with the runnable. The endpoints use Request and Response
        objects from Starlette (same as fast API). The API handler
        does complete validation of the request objects.

        Response validation can be added by the user.

        Args:
            runnable: The runnable to serve.
            path: The path to serve the runnable under.
            prefix: Any prefix that may need to be added to the path
                to get the full URL for the runnable.
                This is relevant when the runnable is added to an APIRouter.
            input_type: type to use for input validation.
                Default is "auto" which will use the InputType of the runnable.
                User is free to provide a custom type annotation.
                Favor using runnable.with_types(input_type=..., output_type=...)
                instead. This parameter may get deprecated!
            output_type: type to use for output validation.
                Default is "auto" which will use the OutputType of the runnable.
                User is free to provide a custom type annotation.
                Favor using runnable.with_types(input_type=..., output_type=...)
                instead. This parameter may get deprecated!
            config_keys: list of config keys that will be accepted, by default
                will accept `configurable` key in the config. Will only be used
                if the runnable is configurable. Cannot configure run_name,
                which is set by default to the path of the API.
            include_callback_events: Whether to include callback events in the response.
                If true, the client will be able to show trace information
                including events that occurred on the server side.
                Be sure not to include any sensitive information in the callback events.
            enable_feedback_endpoint: Whether to enable an endpoint for logging feedback
                to LangSmith. Disabled by default. If this flag is disabled or LangSmith
                tracing is not enabled for the runnable, then 4xx errors will be thrown
                when accessing the feedback endpoint
            per_req_config_modifier: optional function that can be used to update the
                RunnableConfig for a given run based on the raw request. This is useful,
                for example, if the user wants to pass in a header containing
                credentials to a runnable. The RunnableConfig is presented in its
                dictionary form. Note that only keys in `config_keys` will be
                modifiable by this function.
        """
        if importlib.util.find_spec("sse_starlette") is None:
            raise ImportError(
                "sse_starlette must be installed to implement the stream and "
                "stream_log endpoints. "
                "Use `pip install sse_starlette` to install."
            )

        if "run_name" in config_keys:
            raise ValueError(
                "Cannot configure run_name. Please remove it from config_keys."
            )

        if path and not path.startswith("/"):
            raise ValueError(
                f"Got an invalid path: {path}. "
                f"If specifying path please start it with a `/`"
            )

        self._config_keys = config_keys

        self._path = path
        self._base_url = prefix + path
        self._include_callback_events = include_callback_events
        self._per_req_config_modifier = per_req_config_modifier
        self._serializer = WellKnownLCSerializer()
        self._enable_feedback_endpoint = enable_feedback_endpoint
        self._stream_log_name_allow_list = stream_log_name_allow_list

        # Client is patched using mock.patch, if changing the names
        # remember to make relevant updates in the unit tests.
        self._langsmith_client = (
            ls_client.Client()
            if tracing_is_enabled() and enable_feedback_endpoint
            else None
        )

        with_types = {}

        if input_type != "auto":
            with_types["input_type"] = input_type
        if output_type != "auto":
            with_types["output_type"] = output_type

        if with_types:
            runnable = runnable.with_types(**with_types)

        self._runnable = runnable

        model_namespace = _replace_non_alphanumeric_with_underscores(path.strip("/"))

        input_type_ = _resolve_model(
            runnable.get_input_schema(), "Input", model_namespace
        )

        output_type_ = _resolve_model(
            runnable.get_output_schema(),
            "Output",
            model_namespace,
        )

        self._ConfigPayload = _add_namespace_to_model(
            model_namespace, runnable.config_schema(include=config_keys)
        )

        self._InvokeRequest = create_invoke_request_model(
            model_namespace, input_type_, self._ConfigPayload
        )

        self._BatchRequest = create_batch_request_model(
            model_namespace, input_type_, self._ConfigPayload
        )
        self._StreamRequest = create_stream_request_model(
            model_namespace, input_type_, self._ConfigPayload
        )
        self._StreamLogRequest = create_stream_log_request_model(
            model_namespace, input_type_, self._ConfigPayload
        )
        # Generate the response models
        self._InvokeResponse = create_invoke_response_model(
            model_namespace, output_type_
        )
        self._BatchResponse = create_batch_response_model(model_namespace, output_type_)

    @property
    def InvokeRequest(self) -> Type[BaseModel]:
        """Return the invoke request model."""
        return self._InvokeRequest

    @property
    def BatchRequest(self) -> Type[BaseModel]:
        """Return the batch request model."""
        return self._BatchRequest

    @property
    def StreamRequest(self) -> Type[BaseModel]:
        """Return the stream request model."""
        return self._StreamRequest

    @property
    def StreamLogRequest(self) -> Type[BaseModel]:
        """Return the stream log request model."""
        return self._StreamLogRequest

    @property
    def InvokeResponse(self) -> Type[BaseModel]:
        """Return the invoke response model."""
        return self._InvokeResponse

    @property
    def BatchResponse(self) -> Type[BaseModel]:
        """Return the batch response model."""
        return self._BatchResponse

    async def _get_config_and_input(
        self,
        request: Request,
        config_hash: str,
        *,
        endpoint: Optional[str] = None,
        server_config: Optional[RunnableConfig] = None,
    ) -> Tuple[RunnableConfig, Any]:
        """Extract the config and input from the request, validating the request."""
        try:
            body = await request.json()
        except json.JSONDecodeError:
            raise RequestValidationError(errors=["Invalid JSON body"])
        try:
            body = InvokeRequestShallowValidator.validate(body)

            # Merge the config from the path with the config from the body.
            user_provided_config = _unpack_request_config(
                config_hash,
                body.config,
                config_keys=self._config_keys,
                model=self._ConfigPayload,
                request=request,
                per_req_config_modifier=self._per_req_config_modifier,
                server_config=server_config,
            )
            config = _update_config_with_defaults(
                self._path,
                user_provided_config,
                request,
                endpoint=endpoint,
            )
            # Unpack the input dynamically using the input schema of the runnable.
            # This takes into account changes in the input type when
            # using configuration.
            schema = self._runnable.with_config(config).input_schema
            input_ = schema.validate(body.input)
            return config, _unpack_input(input_)
        except ValidationError as e:
            raise RequestValidationError(e.errors(), body=body)

    async def invoke(
        self,
        request: Request,
        *,
        config_hash: str = "",
        server_config: Optional[RunnableConfig] = None,
    ) -> Response:
        """Invoke the runnable with the given input and config.

        Args:
            request: The request object.
            config_hash: A compressed representation of a config. Optionally
                         sent from the client side. This config must be validated.
            server_config: optional server configuration that will be merged
                with any other configuration. It's the last to be written, so
                it will override any other configuration.
        """
        # We do not use the InvokeRequest model here since configurable runnables
        # have dynamic schema -- so the validation below is a bit more involved.
        config, input_ = await self._get_config_and_input(
            request,
            config_hash,
            endpoint="invoke",
            server_config=server_config,
        )

        event_aggregator = AsyncEventAggregatorCallback()
        _add_callbacks(config, [event_aggregator])
        output = await self._runnable.ainvoke(input_, config=config)

        if self._include_callback_events:
            callback_events = [
                _scrub_exceptions_in_event(event)
                for event in event_aggregator.callback_events
            ]
        else:
            callback_events = []

        return _json_encode_response(
            self._InvokeResponse(
                output=self._serializer.dumpd(output),
                # Callbacks are scrubbed and exceptions are converted to
                # serializable format before returned in the response.
                callback_events=callback_events,
                metadata=SingletonResponseMetadata(
                    run_id=_get_base_run_id_as_str(event_aggregator)
                ),
            ),
        )

    async def batch(
        self,
        request: Request,
        *,
        config_hash: str = "",
        server_config: Optional[RunnableConfig] = None,
    ) -> Response:
        """Invoke the runnable with the given inputs and config.

        Args:
            request: The request object.
            config_hash: A compressed representation of a config.
                Originates from the client side. This config must be validated.
            server_config: optional server configuration that will be merged
                with any other configuration. It's the last to be written, so
                it will override any other configuration.
        """
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
                        config_keys=self._config_keys,
                        model=self._ConfigPayload,
                        request=request,
                        per_req_config_modifier=self._per_req_config_modifier,
                        server_config=server_config,
                    )
                    for config in config
                ]
            elif isinstance(config, dict):
                configs = _unpack_request_config(
                    config_hash,
                    config,
                    config_keys=self._config_keys,
                    model=self._ConfigPayload,
                    request=request,
                    per_req_config_modifier=self._per_req_config_modifier,
                    server_config=server_config,
                )
            else:
                raise HTTPException(
                    422, "Value for 'config' key must be a dict or list if provided"
                )

        inputs_ = body.inputs

        # Make sure that the number of configs matches the number of inputs
        # Since we'll be adding callbacks to the configs.

        configs_ = [
            {k: v for k, v in config_.items() if k in self._config_keys}
            for config_ in get_config_list(configs, len(inputs_))
        ]

        inputs = [
            _unpack_input(
                self._runnable.with_config(config_).input_schema.validate(input_)
            )
            for config_, input_ in zip(configs_, inputs_)
        ]

        # Update the configuration with callbacks
        aggregators = [AsyncEventAggregatorCallback() for _ in range(len(inputs))]

        final_configs = []
        for config_, aggregator in zip(configs_, aggregators):
            _add_callbacks(config_, [aggregator])
            final_configs.append(
                _update_config_with_defaults(
                    self._path, config_, request, endpoint="batch"
                )
            )

        output = await self._runnable.abatch(inputs, config=final_configs)

        if self._include_callback_events:
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

        obj = self._BatchResponse(
            output=self._serializer.dumpd(output),
            callback_events=callback_events,
            metadata=BatchResponseMetadata(
                run_ids=[_get_base_run_id_as_str(agg) for agg in aggregators]
            ),
        )
        return _json_encode_response(obj)

    async def stream(
        self,
        request: Request,
        *,
        config_hash: str = "",
        server_config: Optional[RunnableConfig] = None,
    ) -> EventSourceResponse:
        """Invoke the runnable stream the output.

        This endpoint allows to stream the output of the runnable.

        The endpoint uses a server sent event stream to stream the output.

        https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events

        Important: Set the "text/event-stream" media type for request headers if
            not using an existing SDK.

        The events that the endpoint uses are the following:
        * "data" -- used for streaming the output of the runnale
        * "error" -- used for signaling an error in the stream, also ends the stream.
        * "end" -- used for signaling the end of the stream
        * "metadata" -- used for sending metadata about the run; e.g., run id.

        The event type is in the "event" field of the event.
        The payload associated with the event is in the "data" field of the event,
        and it is JSON encoded.

        Here are some examples of events that the endpoint can send:

        Regular streaming event:
        {
            "event": "data",
            "data": {
                ...
            }
        }

        Internal server error:
        {
            "event": "error",
            "data": {
                "status_code": 500,
                "message": "Internal Server Error"
            }
        }

        Streaming ended so client should stop listening for events:
        {
            "event": "end",
        }

        Args:
            request: The request object.
            config_hash: A compressed representation of a config.
                Originates from the client side. This config must be validated.
            server_config: optional server configuration that will be merged
        """
        err_event = {}
        validation_exception: Optional[BaseException] = None
        try:
            config, input_ = await self._get_config_and_input(
                request,
                config_hash,
                endpoint="stream",
                server_config=server_config,
            )
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
                _add_callbacks(config_w_callbacks, [event_aggregator])
                has_sent_metadata = False
                async for chunk in self._runnable.astream(
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
                        "data": self._serializer.dumps(chunk).decode("utf-8"),
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

    async def stream_log(
        self,
        request: Request,
        *,
        config_hash: str = "",
        server_config: Optional[RunnableConfig] = None,
    ) -> EventSourceResponse:
        """Invoke the runnable stream_log the output.

        View documentation for endpoint at the end of the file.
        It's attached to _stream_log_docs endpoint.
        """
        err_event = {}
        validation_exception: Optional[BaseException] = None
        try:
            config, input_ = await self._get_config_and_input(
                request,
                config_hash,
                endpoint="stream_log",
                server_config=server_config,
            )
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
                async for chunk in self._runnable.astream_log(
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
                    if (
                        self._stream_log_name_allow_list is None
                        or self._runnable.config.get("run_name")
                        in self._stream_log_name_allow_list
                    ):
                        data = {
                            "ops": chunk.ops,
                        }

                        # Temporary adapter
                        yield {
                            # EventSourceResponse expects a string for data
                            # so after serializing into bytes, we decode into utf-8
                            # to get a string.
                            "data": self._serializer.dumps(data).decode("utf-8"),
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

    async def input_schema(
        self,
        request: Request,
        *,
        config_hash: str = "",
        server_config: Optional[RunnableConfig] = None,
    ) -> Any:
        """Return the input schema of the runnable."""
        with _with_validation_error_translation():
            user_provided_config = _unpack_request_config(
                config_hash,
                config_keys=self._config_keys,
                model=self._ConfigPayload,
                request=request,
                # Do not use per request config modifier for output schema
                # since it's unclear why it would make sense to modify
                # this using a per request config modifier.
                # If this is needed, for some reason please file an issue explaining
                # the user case.
                per_req_config_modifier=None,
                server_config=server_config,
            )
            config = _update_config_with_defaults(
                self._path, user_provided_config, request
            )

        return self._runnable.get_input_schema(config).schema()

    async def output_schema(
        self,
        request: Request,
        *,
        config_hash: str = "",
        server_config: Optional[RunnableConfig] = None,
    ) -> Any:
        """Return the output schema of the runnable."""
        with _with_validation_error_translation():
            user_provided_config = _unpack_request_config(
                config_hash,
                config_keys=self._config_keys,
                model=self._ConfigPayload,
                request=request,
                # Do not use per request config modifier for output schema
                # since it's unclear why it would make sense to modify
                # this using a per request config modifier.
                # If this is needed, for some reason please file an issue explaining
                # the user case.
                per_req_config_modifier=None,
                server_config=server_config,
            )
            config = _update_config_with_defaults(
                self._path, user_provided_config, request
            )
        return self._runnable.get_output_schema(config).schema()

    async def config_schema(
        self,
        request: Request,
        *,
        config_hash: str = "",
        server_config: Optional[RunnableConfig] = None,
    ) -> Any:
        """Return the config schema of the runnable."""
        with _with_validation_error_translation():
            user_provided_config = _unpack_request_config(
                config_hash,
                config_keys=self._config_keys,
                model=self._ConfigPayload,
                request=request,
                # Do not use per request config modifier for output schema
                # since it's unclear why it would make sense to modify
                # this using a per request config modifier.
                # If this is needed, for some reason please file an issue explaining
                # the user case.
                per_req_config_modifier=None,
                server_config=server_config,
            )
            config = _update_config_with_defaults(
                self._path, user_provided_config, request
            )
        return (
            self._runnable.with_config(config)
            .config_schema(include=self._config_keys)
            .schema()
        )

    async def playground(
        self,
        file_path: str,
        request: Request,
        *,
        config_hash: str = "",
        server_config: Optional[RunnableConfig] = None,
    ) -> Any:
        """Return the playground of the runnable."""
        with _with_validation_error_translation():
            user_provided_config = _unpack_request_config(
                config_hash,
                config_keys=self._config_keys,
                model=self._ConfigPayload,
                request=request,
                # Do not use per request config modifier for output schema
                # since it's unclear why it would make sense to modify
                # this using a per request config modifier.
                # If this is needed, for some reason please file an issue explaining
                # the user case.
                per_req_config_modifier=None,
                server_config=server_config,
            )

            config = _update_config_with_defaults(
                self._path, user_provided_config, request
            )

        feedback_enabled = tracing_is_enabled() and self._enable_feedback_endpoint

        if self._base_url.endswith("/"):
            playground_url = self._base_url + "playground"
        else:
            playground_url = self._base_url + "/playground"

        return await serve_playground(
            self._runnable.with_config(config),
            self._runnable.with_config(config).input_schema,
            self._config_keys,
            playground_url,
            file_path,
            feedback_enabled,
        )

    async def create_feedback(
        self, feedback_create_req: FeedbackCreateRequest
    ) -> Feedback:
        """Send feedback on an individual run to langsmith

        Note that a successful response means that feedback was successfully
        submitted. It does not guarantee that the feedback is recorded by
        langsmith. Requests may be silently rejected if they are
        unauthenticated or invalid by the server.
        """

        if not tracing_is_enabled() or not self._enable_feedback_endpoint:
            raise HTTPException(
                400,
                "The feedback endpoint is only accessible when LangSmith is "
                + "enabled on your LangServe server.",
            )

        feedback_from_langsmith = self._langsmith_client.create_feedback(
            feedback_create_req.run_id,
            feedback_create_req.key,
            score=feedback_create_req.score,
            value=feedback_create_req.value,
            comment=feedback_create_req.comment,
            source_info={
                "from_langserve": True,
            },
        )

        # We purposefully select out fields from langsmith so that we don't
        # fail validation if langsmith adds extra fields. We prefer this over
        # using "Extra.allow" in pydantic since syntax changes between pydantic
        # 1.x and 2.x for this functionality
        return Feedback(
            id=str(feedback_from_langsmith.id),
            run_id=str(feedback_from_langsmith.run_id),
            created_at=str(feedback_from_langsmith.created_at),
            modified_at=str(feedback_from_langsmith.modified_at),
            key=str(feedback_from_langsmith.key),
            score=feedback_from_langsmith.score,
            value=feedback_from_langsmith.value,
            comment=feedback_from_langsmith.comment,
        )

    async def _check_feedback_enabled(self) -> None:
        """Check if feedback is enabled for the runnable.

        This endpoint is private since it will be deprecated in the future.
        """
        if not (await self.check_feedback_enabled()):
            raise HTTPException(
                400,
                "The feedback endpoint is only accessible when LangSmith is "
                + "enabled on your LangServe server.",
            )

    async def check_feedback_enabled(self) -> bool:
        """Check if feedback is enabled for the runnable."""
        return self._enable_feedback_endpoint or not tracing_is_enabled()
