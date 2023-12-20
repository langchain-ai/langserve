"""FastAPI integration for langchain runnables.

This code contains integration for langchain runnables with FastAPI.

The main entry point is the `add_routes` function which adds the routes to an existing
FastAPI app or APIRouter.
"""
import weakref
from typing import (
    Any,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
)

from langchain.schema.runnable import Runnable
from typing_extensions import Annotated

from langserve.api_handler import APIHandler, PerRequestConfigModifier, _is_hosted
from langserve.pydantic_v1 import (
    _PYDANTIC_MAJOR_VERSION,
    PYDANTIC_VERSION,
    BaseModel,
)

try:
    from fastapi import APIRouter, FastAPI, Request, Response
except ImportError:
    # [server] extra not installed
    APIRouter = FastAPI = Request = Response = Any

# A function that that takes a config and a raw request
# and updates the config based on the request.

# This is a global registry of models to avoid creating the same model
# multiple times.
# Duplicated model names break fastapi's openapi generation.

_APP_SEEN = weakref.WeakSet()
_APP_TO_PATHS = weakref.WeakKeyDictionary()


def _setup_global_app_handlers(app: Union[FastAPI, APIRouter]) -> None:
    @app.on_event("startup")
    async def startup_event():
        LANGSERVE = r"""
 __          ___      .__   __.   _______      _______. _______ .______     ____    ____  _______
|  |        /   \     |  \ |  |  /  _____|    /       ||   ____||   _  \    \   \  /   / |   ____|
|  |       /  ^  \    |   \|  | |  |  __     |   (----`|  |__   |  |_)  |    \   \/   /  |  |__
|  |      /  /_\  \   |  . `  | |  | |_ |     \   \    |   __|  |      /      \      /   |   __|
|  `----./  _____  \  |  |\   | |  |__| | .----)   |   |  |____ |  |\  \----.  \    /    |  |____
|_______/__/     \__\ |__| \__|  \______| |_______/    |_______|| _| `._____|   \__/     |_______|
"""  # noqa: E501

        def green(text: str) -> str:
            """Return the given text in green."""
            return "\x1b[1;32;40m" + text + "\x1b[0m"

        def orange(text: str) -> str:
            """Return the given text in orange."""
            return "\x1b[1;31;40m" + text + "\x1b[0m"

        paths = _APP_TO_PATHS[app]
        print(LANGSERVE)
        for path in paths:
            print(
                f'{green("LANGSERVE:")} Playground for chain "{path or ""}/" is '
                f"live at:"
            )
            print(f'{green("LANGSERVE:")}  │')
            print(f'{green("LANGSERVE:")}  └──> {path}/playground/')
            print(f'{green("LANGSERVE:")}')
        print(f'{green("LANGSERVE:")} See all available routes at {app.docs_url}/')

        if _PYDANTIC_MAJOR_VERSION == 2:
            print()
            print(f'{orange("LANGSERVE:")} ', end="")
            print(
                f"⚠️ Using pydantic {PYDANTIC_VERSION}. "
                f"OpenAPI docs for invoke, batch, stream, stream_log "
                f"endpoints will not be generated. API endpoints and playground "
                f"should work as expected. "
                f"If you need to see the docs, you can downgrade to pydantic 1. "
                "For example, `pip install pydantic==1.10.13`. "
                f"See https://github.com/tiangolo/fastapi/issues/10360 for details."
            )
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


EndpointName = Literal[
    "invoke",
    "batch",
    "stream",
    "stream_log",
    "playground",
    "feedback",
    "input_schema",
    "config_schema",
    "output_schema",
    "config_hashes",
]

KNOWN_ENDPOINTS = {
    "invoke",
    "batch",
    "stream",
    "stream_log",
    "playground",
    "feedback",
    "input_schema",
    "config_schema",
    "output_schema",
    "config_hashes",
}


class _EndpointConfiguration:
    """Logic for enabling/disabling endpoints."""

    def __init__(
        self,
        *,
        enabled_endpoints: Optional[Sequence[EndpointName]] = None,
        disabled_endpoints: Optional[Sequence[EndpointName]] = None,
        enable_feedback_endpoint: bool = False,
    ) -> None:
        """Initialize the endpoint configuration."""
        if enabled_endpoints and disabled_endpoints:
            raise ValueError(
                f'Cannot specify both "enabled_endpoints" and "disabled_endpoints".'
                f"Got enabled_endpoints={enabled_endpoints} and disabled_endpoints="
                f"{disabled_endpoints}."
            )

        if enabled_endpoints and not isinstance(enabled_endpoints, Sequence):
            raise ValueError(
                f"Expected enabled_endpoints to be a sequence (e.g., list or tuple), "
                f"got {type(enabled_endpoints)}"
            )

        if disabled_endpoints and not isinstance(disabled_endpoints, Sequence):
            raise ValueError(
                f"Expected disabled_endpoints to be a sequence (e.g., list or tuple), "
                f"got {type(disabled_endpoints)}"
            )

        if enabled_endpoints is None:
            if disabled_endpoints is None:
                is_invoke_enabled = True
                is_batch_enabled = True
                is_stream_enabled = True
                is_stream_log_enabled = True
                is_playground_enabled = True
                is_input_schema_enabled = True
                is_output_schema_enabled = True
                is_config_schema_enabled = True
                is_config_hash_enabled = True
            else:
                disabled_endpoints_ = set(name.lower() for name in disabled_endpoints)
                if disabled_endpoints_ - KNOWN_ENDPOINTS:
                    raise ValueError(
                        f"Got unknown endpoint "
                        f"names: {disabled_endpoints_ - KNOWN_ENDPOINTS}"
                    )
                is_invoke_enabled = "invoke" not in disabled_endpoints_
                is_batch_enabled = "batch" not in disabled_endpoints_
                is_stream_enabled = "stream" not in disabled_endpoints_
                is_stream_log_enabled = "stream_log" not in disabled_endpoints_
                is_playground_enabled = "playground" not in disabled_endpoints_
                is_input_schema_enabled = "input_schema" not in disabled_endpoints_
                is_output_schema_enabled = "output_schema" not in disabled_endpoints_
                is_config_schema_enabled = "config_schema" not in disabled_endpoints_
                is_config_hash_enabled = "config_hashes" not in disabled_endpoints_
        else:
            enabled_endpoints_ = set(name.lower() for name in enabled_endpoints)
            if enabled_endpoints_ - KNOWN_ENDPOINTS:
                raise ValueError(
                    f"Got unknown endpoint names: {enabled_endpoints_- KNOWN_ENDPOINTS}"
                )
            is_invoke_enabled = "invoke" in enabled_endpoints_
            is_batch_enabled = "batch" in enabled_endpoints_
            is_stream_enabled = "stream" in enabled_endpoints_
            is_stream_log_enabled = "stream_log" in enabled_endpoints_
            is_playground_enabled = "playground" in enabled_endpoints_
            is_input_schema_enabled = "input_schema" in enabled_endpoints_
            is_output_schema_enabled = "output_schema" in enabled_endpoints_
            is_config_schema_enabled = "config_schema" in enabled_endpoints_
            is_config_hash_enabled = "config_hashes" in enabled_endpoints_

        self.is_invoke_enabled = is_invoke_enabled
        self.is_batch_enabled = is_batch_enabled
        self.is_stream_enabled = is_stream_enabled
        self.is_stream_log_enabled = is_stream_log_enabled
        self.is_playground_enabled = is_playground_enabled
        self.is_input_schema_enabled = is_input_schema_enabled
        self.is_output_schema_enabled = is_output_schema_enabled
        self.is_config_schema_enabled = is_config_schema_enabled
        self.is_config_hash_enabled = is_config_hash_enabled
        self.is_feedback_enabled = enable_feedback_endpoint


# PUBLIC API


def add_routes(
    app: Union[FastAPI, APIRouter],
    runnable: Runnable,
    *,
    path: str = "",
    input_type: Union[Type, Literal["auto"], BaseModel] = "auto",
    output_type: Union[Type, Literal["auto"], BaseModel] = "auto",
    config_keys: Sequence[str] = ("configurable",),
    include_callback_events: bool = False,
    per_req_config_modifier: Optional[PerRequestConfigModifier] = None,
    enable_feedback_endpoint: bool = _is_hosted(),
    disabled_endpoints: Optional[Sequence[EndpointName]] = None,
    stream_log_name_allow_list: Optional[Sequence[str]] = None,
    enabled_endpoints: Optional[Sequence[EndpointName]] = None,
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
            will accept `configurable` key in the config. Will only be used
            if the runnable is configurable. Cannot configure run_name,
            which is set by default to the path of the API.
        include_callback_events: Whether to include callback events in the response.
            If true, the client will be able to show trace information
            including events that occurred on the server side.
            Be sure not to include any sensitive information in the callback events.
        per_req_config_modifier: optional function that can be used to update the
            RunnableConfig for a given run based on the raw request. This is useful,
            for example, if the user wants to pass in a header containing credentials
            to a runnable. The RunnableConfig is presented in its dictionary form.
            Note that only keys in `config_keys` will be modifiable by this function.
            As of 0.0.37, this function is only called for the invoke, batch, stream,
            and stream_log endpoints. This function is not called for the playground,
            input_schema, output_schema, and config_schema endpoints etc.
        enable_feedback_endpoint: Whether to enable an endpoint for logging feedback
            to LangSmith. Enabled by default. If this flag is disabled or LangSmith
            tracing is not enabled for the runnable, then 400 errors will be thrown
            when accessing the feedback endpoint
        enabled_endpoints: A list of endpoints which should be enabled. If not
            specified, all associated endpoints will be enabled. The list can contain
            the following values: *invoke*, *batch*, *stream*, *stream_log*,
            *playground*, *input_schema*, *output_schema*, *config_schema*,
            *config_hashes*.

            *config_hashes* represents the config hash variant (when it exists)
            of each endpoint. Enabling this is useful when working with configurable
            runnables and sharing playground configuration links to the runnables.

            For example, if we want to enable regular invoke and batch endpoints
            and their config hash variants, we can do:

            ```python

            add_routes(
                ...,
                enabled_endpoints=("invoke", "batch", "config_hashes"),
            )
            ```

            Please note that the feedback endpoint is not included in this list
            and is controlled by the `enable_feedback_endpoint` flag.
        disabled_endpoints: A list of endpoints which should be disabled. If not
            specified, all associated endpoints will be enabled. The list can contain
            the following values: *invoke*, *batch*, *stream*, *stream_log*,
            *playground*, *input_schema*, *output_schema*, *config_schema*,
            *config_hashes*.

            *config_hashes* represents the config hash variant (when it exists)
            of each endpoint. Enabling this is useful when working with configurable
            runnables and sharing playground configuration links to the runnables.

            For example, if we want to enable regular invoke and batch endpoints
            and their config hash variants, we can do:

            ```python

            add_routes(
                ...,
                disabled_endpoints=["playground"],
            )
            ```
        stream_log_name_allow_list: list of run names that the client can
            stream as intermediate steps

    """
    endpoint_configuration = _EndpointConfiguration(
        enabled_endpoints=enabled_endpoints,
        disabled_endpoints=disabled_endpoints,
        enable_feedback_endpoint=enable_feedback_endpoint,
    )

    try:
        from sse_starlette import EventSourceResponse
    except ImportError:
        raise ImportError(
            "sse_starlette must be installed to implement the stream and "
            "stream_log endpoints. "
            "Use `pip install sse_starlette` to install."
        )

    if path and not path.startswith("/"):
        raise ValueError(
            f"Got an invalid path: {path}. "
            f"If specifying path please start it with a `/`"
        )

    if isinstance(app, FastAPI):  # type: ignore
        # Cannot do this checking logic for a router since
        # API routers are not hashable
        _register_path_for_app(app, path)

    # Determine the base URL for the playground endpoint
    prefix = app.prefix if isinstance(app, APIRouter) else ""  # type: ignore

    api_handler = APIHandler(
        runnable,
        path=path,
        prefix=prefix,
        input_type=input_type,
        output_type=output_type,
        config_keys=config_keys,
        include_callback_events=include_callback_events,
        enable_feedback_endpoint=enable_feedback_endpoint,
        per_req_config_modifier=per_req_config_modifier,
        stream_log_name_allow_list=stream_log_name_allow_list,
    )
    namespace = path or ""

    route_tags = [path.strip("/")] if path else None
    route_tags_with_config = [f"{path.strip('/')}/config"] if path else ["config"]

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

        if _PYDANTIC_MAJOR_VERSION == 1:
            # Documentation for the default endpoints
            default_endpoint_tags = {
                "name": route_tags[0] if route_tags else "default",
            }
        elif _PYDANTIC_MAJOR_VERSION == 2:
            # When using pydantic v2, we cannot generate openapi docs for
            # the invoke/batch/stream/stream_log endpoints since the underlying
            # models are from the pydantic.v1 namespace and cannot be supported
            # by fastapi's.
            # https://github.com/tiangolo/fastapi/issues/10360
            default_endpoint_tags = {
                "name": route_tags[0] if route_tags else "default",
                "description": (
                    f"⚠️ Using pydantic {PYDANTIC_VERSION}. "
                    f"OpenAPI docs for `invoke`, `batch`, `stream`, `stream_log` "
                    f"endpoints will not be generated. API endpoints and playground "
                    f"should work as expected. "
                    f"If you need to see the docs, you can downgrade to pydantic 1. "
                    "For example, `pip install pydantic==1.10.13`"
                    f"See https://github.com/tiangolo/fastapi/issues/10360 for details."
                ),
            }
        else:
            raise AssertionError(
                f"Expected pydantic major version 1 or 2, got {_PYDANTIC_MAJOR_VERSION}"
            )

        if endpoint_configuration.is_config_hash_enabled:
            app.openapi_tags = [
                *(getattr(app, "openapi_tags", []) or []),
                default_endpoint_tags,
                {
                    "name": route_tags_with_config[0],
                    "description": (
                        "Endpoints with a default configuration "
                        "set by `config_hash` path parameter. "
                        "Used in conjunction with share links generated using the "
                        "LangServe UI playground. "
                        "The hash is an LZString compressed JSON string."
                    ),
                },
            ]

    if endpoint_configuration.is_invoke_enabled:

        @app.post(f"{namespace}/invoke", include_in_schema=False)
        async def invoke(request: Request) -> Response:
            """Handle a request."""
            # The API Handler validates the parts of the request
            # that are used by the runnnable (e.g., input, config fields)
            return await api_handler.invoke(request)

        if endpoint_configuration.is_config_hash_enabled:

            @app.post(namespace + "/c/{config_hash}/invoke", include_in_schema=False)
            async def invoke_with_config(
                request: Request, config_hash: str = ""
            ) -> Response:
                """Handle a request."""
                # The API Handler validates the parts of the request
                # that are used by the runnnable (e.g., input, config fields)
                return await api_handler.invoke(request, config_hash=config_hash)

    if endpoint_configuration.is_batch_enabled:

        @app.post(f"{namespace}/batch", include_in_schema=False)
        async def batch(request: Request) -> Response:
            """Handle a request."""
            # The API Handler validates the parts of the request
            # that are used by the runnnable (e.g., input, config fields)
            return await api_handler.batch(request)

        if endpoint_configuration.is_config_hash_enabled:

            @app.post(namespace + "/c/{config_hash}/batch", include_in_schema=False)
            async def batch_with_config(
                request: Request, config_hash: str = ""
            ) -> Response:
                """Handle a request."""
                # The API Handler validates the parts of the request
                # that are used by the runnnable (e.g., input, config fields)
                return await api_handler.batch(request, config_hash=config_hash)

    if endpoint_configuration.is_stream_enabled:

        @app.post(f"{namespace}/stream", include_in_schema=False)
        async def stream(request: Request) -> EventSourceResponse:
            """Handle a request."""
            # The API Handler validates the parts of the request
            # that are used by the runnnable (e.g., input, config fields)
            return await api_handler.stream(request)

        if endpoint_configuration.is_config_hash_enabled:

            @app.post(namespace + "/c/{config_hash}/stream", include_in_schema=False)
            async def stream_with_config(
                request: Request, config_hash: str = ""
            ) -> EventSourceResponse:
                """Handle a request."""
                # The API Handler validates the parts of the request
                # that are used by the runnnable (e.g., input, config fields)
                return await api_handler.stream(request, config_hash=config_hash)

    if endpoint_configuration.is_stream_log_enabled:

        @app.post(f"{namespace}/stream_log", include_in_schema=False)
        async def stream_log(request: Request) -> EventSourceResponse:
            """Handle a request."""
            # The API Handler validates the parts of the request
            # that are used by the runnnable (e.g., input, config fields)
            return await api_handler.stream_log(request)

        if endpoint_configuration.is_config_hash_enabled:

            @app.post(
                namespace + "/c/{config_hash}/stream_log", include_in_schema=False
            )
            async def stream_log_with_config(
                request: Request, config_hash: str = ""
            ) -> EventSourceResponse:
                """Handle a request."""
                # The API Handler validates the parts of the request
                # that are used by the runnnable (e.g., input, config fields)
                return await api_handler.stream_log(request, config_hash=config_hash)

    if endpoint_configuration.is_input_schema_enabled:

        @app.get(
            f"{namespace}/input_schema",
            name=_route_name("input_schema"),
            tags=route_tags,
        )
        async def input_schema(request: Request) -> Response:
            """Return the input schema."""
            return await api_handler.input_schema(request)

        if endpoint_configuration.is_config_hash_enabled:

            @app.get(
                namespace + "/c/{config_hash}/input_schema",
                name=_route_name_with_config("input_schema"),
                tags=route_tags_with_config,
            )
            async def input_schema_with_config(
                request: Request, config_hash: str = ""
            ) -> Response:
                """Return the input schema."""
                return await api_handler.input_schema(request, config_hash=config_hash)

    if endpoint_configuration.is_output_schema_enabled:

        @app.get(
            f"{namespace}/output_schema",
            name=_route_name("output_schema"),
            tags=route_tags,
        )
        async def output_schema(request: Request) -> Response:
            """Return the output schema."""
            return await api_handler.output_schema(request)

        if endpoint_configuration.is_config_hash_enabled:

            @app.get(
                namespace + "/c/{config_hash}/output_schema",
                name=_route_name_with_config("output_schema"),
                tags=route_tags_with_config,
            )
            async def output_schema_with_config(
                request: Request, config_hash: str = ""
            ) -> Response:
                """Return the output schema."""
                return await api_handler.output_schema(request, config_hash=config_hash)

    if endpoint_configuration.is_config_schema_enabled:

        @app.get(
            f"{namespace}/config_schema",
            name=_route_name("config_schema"),
            tags=route_tags,
        )
        async def config_schema(request: Request) -> Response:
            """Return the config schema."""
            return await api_handler.config_schema(request)

        if endpoint_configuration.is_config_hash_enabled:

            @app.get(
                namespace + "/c/{config_hash}/config_schema",
                name=_route_name_with_config("config_schema"),
                tags=route_tags_with_config,
            )
            async def config_schema_with_config(
                request: Request, config_hash: str = ""
            ) -> Response:
                """Return the config schema."""
                return await api_handler.config_schema(request, config_hash=config_hash)

    if endpoint_configuration.is_playground_enabled:
        playground = app.get(namespace + "/playground/{file_path:path}")(
            api_handler.playground
        )

        if endpoint_configuration.is_config_hash_enabled:
            app.get(
                namespace + "/c/{config_hash}/playground/{file_path:path}",
            )(playground)

    if enable_feedback_endpoint:
        app.post(
            namespace + "/feedback",
        )(api_handler.create_feedback)

        app.head(
            namespace + "/feedback",
        )(api_handler._check_feedback_enabled)

    #######################################
    # Documentation variants of end points.
    #######################################
    # At the moment, we only support pydantic 1.x for documentation
    if _PYDANTIC_MAJOR_VERSION == 1:
        InvokeRequest = api_handler.InvokeRequest
        InvokeResponse = api_handler.InvokeResponse
        BatchRequest = api_handler.BatchRequest
        BatchResponse = api_handler.BatchResponse
        StreamRequest = api_handler.StreamRequest
        StreamLogRequest = api_handler.StreamLogRequest

        if endpoint_configuration.is_invoke_enabled:

            async def _invoke_docs(
                invoke_request: Annotated[InvokeRequest, InvokeRequest],
                config_hash: str = "",
            ) -> InvokeResponse:
                """Invoke the runnable with the given input and config."""
                raise AssertionError("This endpoint should not be reachable.")

            invoke_docs = app.post(
                f"{namespace}/invoke",
                response_model=api_handler.InvokeResponse,
                tags=route_tags,
                name=_route_name("invoke"),
            )(_invoke_docs)

            if endpoint_configuration.is_config_hash_enabled:
                app.post(
                    namespace + "/c/{config_hash}/invoke",
                    response_model=api_handler.InvokeResponse,
                    tags=route_tags_with_config,
                    name=_route_name_with_config("invoke"),
                    description=(
                        "This endpoint is to be used with share links generated by the "
                        "LangServe playground. "
                        "The hash is an LZString compressed JSON string. "
                        "For regular use cases, use the /invoke endpoint without "
                        "the `c/{config_hash}` path parameter."
                    ),
                )(invoke_docs)

        if endpoint_configuration.is_batch_enabled:

            async def _batch_docs(
                batch_request: Annotated[BatchRequest, BatchRequest],
                config_hash: str = "",
            ) -> BatchResponse:
                """Batch invoke the runnable with the given inputs and config."""
                raise AssertionError("This endpoint should not be reachable.")

            batch_docs = app.post(
                f"{namespace}/batch",
                response_model=BatchResponse,
                tags=route_tags,
                name=_route_name("batch"),
            )(_batch_docs)

            if endpoint_configuration.is_config_hash_enabled:
                app.post(
                    namespace + "/c/{config_hash}/batch",
                    response_model=BatchResponse,
                    tags=route_tags_with_config,
                    name=_route_name_with_config("batch"),
                    description=(
                        "This endpoint is to be used with share links generated by the "
                        "LangServe playground. "
                        "The hash is an LZString compressed JSON string. "
                        "For regular use cases, use the /batch endpoint without "
                        "the `c/{config_hash}` path parameter."
                    ),
                )(batch_docs)

        if endpoint_configuration.is_stream_enabled:

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

                The events that the endpoint uses are the following:
                * "data" -- used for streaming the output of the runnale
                * "error" -- signaling an error while streaming and ends the stream.
                * "end" -- used for signaling the end of the stream
                * "metadata" -- used for sending metadata about the run; e.g., run id.

                The event type is in the "event" field of the event.
                The payload associated with the event is in the "data" field
                of the event, and it is JSON encoded.


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
                """
                raise AssertionError("This endpoint should not be reachable.")

            stream_docs = app.post(
                f"{namespace}/stream",
                include_in_schema=True,
                tags=route_tags,
                name=_route_name("stream"),
            )(_stream_docs)

            if endpoint_configuration.is_config_hash_enabled:
                app.post(
                    namespace + "/c/{config_hash}/stream",
                    include_in_schema=True,
                    tags=route_tags_with_config,
                    name=_route_name_with_config("stream"),
                    description=(
                        "This endpoint is to be used with share links generated by the "
                        "LangServe playground. "
                        "The hash is an LZString compressed JSON string. "
                        "For regular use cases, use the /stream endpoint without "
                        "the `c/{config_hash}` path parameter."
                    ),
                )(stream_docs)

        if endpoint_configuration.is_stream_log_enabled:

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

            app.post(
                f"{namespace}/stream_log",
                include_in_schema=True,
                tags=route_tags,
                name=_route_name("stream_log"),
            )(_stream_log_docs)

            if endpoint_configuration.is_config_hash_enabled:
                app.post(
                    namespace + "/c/{config_hash}/stream_log",
                    include_in_schema=True,
                    tags=route_tags_with_config,
                    name=_route_name_with_config("stream_log"),
                    description=(
                        "This endpoint is to be used with share links generated by the "
                        "LangServe playground. "
                        "The hash is an LZString compressed JSON string. "
                        "For regular use cases, use the /stream_log endpoint without "
                        "the `c/{config_hash}` path parameter."
                    ),
                )(_stream_log_docs)
