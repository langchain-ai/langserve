from __future__ import annotations

import asyncio
import copy
import json
import logging
import weakref
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from urllib.parse import urljoin

import httpx
from httpx._types import AuthTypes, CertTypes, CookieTypes, HeaderTypes, VerifyTypes
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.callbacks.tracers.log_stream import RunLogPatch
from langchain.load.dump import dumpd
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import (
    RunnableConfig,
    ensure_config,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
)
from langchain.schema.runnable.utils import Input, Output

from langserve.callbacks import CallbackEventDict, ahandle_callbacks, handle_callbacks
from langserve.serialization import (
    Serializer,
    WellKnownLCSerializer,
    load_events,
)

logger = logging.getLogger(__name__)


def _without_callbacks(config: Optional[RunnableConfig]) -> RunnableConfig:
    """Evict callbacks from the config since those are definitely not supported."""
    _config = config or {}
    return {k: v for k, v in _config.items() if k != "callbacks"}


@lru_cache(maxsize=1_000)  # Will accommodate up to 1_000 different error messages
def _log_error_message_once(error_message: str) -> None:
    """Log an error message once."""
    logger.error(error_message)


def _sanitize_request(request: httpx.Request) -> httpx.Request:
    """Remove sensitive headers from the request."""
    accept_headers = {
        "accept",
        "content-type",
        "user-agent",
        "connection",
        "content-length",
        "accept-encoding",
        "host",
    }
    new_headers = request.headers.copy()
    for key, value in new_headers.items():
        if key.lower() not in accept_headers:
            new_headers[key] = "<redacted>"
        else:
            new_headers[key] = value

    new_request = copy.copy(request)
    new_request.headers = new_headers
    return new_request


def _raise_for_status(response: httpx.Response) -> None:
    """Re-raise with a more informative message.

    Args:
        response: The response to check

    Raises:
        httpx.HTTPStatusError: If the response is not 2xx, appending the response
                               text to the message
    """
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        message = str(e)
        # Append the response text if it exists, as it may contain more information
        # Especially useful when the user's request is malformed
        if e.response.text:
            message += f" for {e.response.text}"

        raise httpx.HTTPStatusError(
            message=message,
            request=_sanitize_request(e.request),
            response=e.response,
        )


def _is_async() -> bool:
    """Return True if we are in an async context."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    else:
        return True


def _close_clients(sync_client: httpx.Client, async_client: httpx.AsyncClient) -> None:
    """Close the async and sync clients.

    _close_clients should not be a bound method since it is called by a weakref
    finalizer.

    Args:
        sync_client: The sync client to close
        async_client: The async client to close
    """
    sync_client.close()
    if _is_async():
        # Use a ThreadPoolExecutor to run async_client_close in a separate thread
        with ThreadPoolExecutor(max_workers=1) as executor:
            # Submit the async_client_close coroutine to the thread pool
            future = executor.submit(asyncio.run, async_client.aclose())
            future.result()
    else:
        asyncio.run(async_client.aclose())


def _raise_exception_from_data(data: str, request: httpx.Request) -> None:
    """Raise an httpx exception from the given error event data."""
    try:
        decoded_data = json.loads(data)
    except json.JSONDecodeError:
        raise httpx.HTTPStatusError(
            message="invalid json in error event sent from server",
            request=_sanitize_request(request),
            response=httpx.Response(status_code=500, text=data),
        )
    raise httpx.HTTPStatusError(
        message=decoded_data["message"],
        request=_sanitize_request(request),
        response=httpx.Response(
            status_code=decoded_data["status_code"],
            text=decoded_data["message"],
        ),
    )


def _decode_response(
    serializer: Serializer,
    response: httpx.Response,
    *,
    is_batch: bool = False,
) -> Tuple[Any, Union[List[CallbackEventDict], List[List[CallbackEventDict]]]]:
    """Decode the response."""
    _raise_for_status(response)
    obj = response.json()
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a dictionary, got {obj}")

    if "output" not in obj:
        raise ValueError("Key `output` not found in")

    output = serializer.loadd(obj["output"])

    if "callback_events" in obj:
        if is_batch:
            if not isinstance(obj["callback_events"], list):
                raise ValueError(
                    f"Expected a list of callback events, got {obj['callback_events']}"
                )
            else:
                callback_events = [
                    load_events(callback_events)
                    for callback_events in obj["callback_events"]
                ]
        else:
            callback_events = load_events(obj["callback_events"])
    else:
        callback_events = []

    return output, callback_events


class RemoteRunnable(Runnable[Input, Output]):
    """A RemoteRunnable is a runnable that is executed on a remote server.

    This client implements the majority of the runnable interface.

    The following features are not supported:

    - `batch` with `return_exceptions=True` since we do not support exception
      translation from the server.
    """

    def __init__(
        self,
        url: str,
        *,
        timeout: Optional[float] = None,
        auth: Optional[AuthTypes] = None,
        headers: Optional[HeaderTypes] = None,
        cookies: Optional[CookieTypes] = None,
        verify: VerifyTypes = True,
        cert: Optional[CertTypes] = None,
        client_kwargs: Optional[Dict[str, Any]] = None,
        use_server_callback_events: bool = True,
    ) -> None:
        """Initialize the client.

        Args:
            url: The url of the server
            timeout: The timeout for requests
            auth: Authentication class for requests
            headers: Headers to send with requests
            cookies: Cookies to send with requests
            verify: Whether to verify SSL certificates
            cert: SSL certificate to use for requests
            client_kwargs: If provided will be unpacked as kwargs to both the sync
                and async httpx clients
            use_server_callback_events: Whether to invoke callbacks on any
                callback events returned by the server.
        """
        _client_kwargs = client_kwargs or {}
        self.url = url
        self.sync_client = httpx.Client(
            base_url=url,
            timeout=timeout,
            auth=auth,
            headers=headers,
            cookies=cookies,
            verify=verify,
            cert=cert,
            **_client_kwargs,
        )
        self.async_client = httpx.AsyncClient(
            base_url=url,
            timeout=timeout,
            auth=auth,
            headers=headers,
            cookies=cookies,
            verify=verify,
            cert=cert,
            **_client_kwargs,
        )

        # Register cleanup handler once RemoteRunnable is garbage collected
        weakref.finalize(self, _close_clients, self.sync_client, self.async_client)
        self._lc_serializer = WellKnownLCSerializer()
        self._use_server_callback_events = use_server_callback_events

    def _invoke(
        self,
        input: Input,
        run_manager: CallbackManagerForChainRun,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Output:
        """Invoke the runnable with the given input and config."""
        response = self.sync_client.post(
            "/invoke",
            json={
                "input": self._lc_serializer.dumpd(input),
                "config": _without_callbacks(config),
                "kwargs": kwargs,
            },
        )
        output, callback_events = _decode_response(
            self._lc_serializer, response, is_batch=False
        )

        if self._use_server_callback_events and callback_events:
            handle_callbacks(run_manager, callback_events)
        return output

    def invoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        if kwargs:
            raise NotImplementedError("kwargs not implemented yet.")
        return self._call_with_config(self._invoke, input, config=config)

    async def _ainvoke(
        self,
        input: Input,
        run_manager: AsyncCallbackManagerForChainRun,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Output:
        """Invoke the runnable with the given input and config."""
        response = await self.async_client.post(
            "/invoke",
            json={
                "input": self._lc_serializer.dumpd(input),
                "config": _without_callbacks(config),
                "kwargs": kwargs,
            },
        )
        output, callback_events = _decode_response(
            self._lc_serializer, response, is_batch=False
        )
        if self._use_server_callback_events and callback_events:
            handle_callbacks(run_manager, callback_events)
        return output

    async def ainvoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        if kwargs:
            raise NotImplementedError("kwargs not implemented yet.")
        return await self._acall_with_config(self._ainvoke, input, config)

    def _batch(
        self,
        inputs: List[Input],
        run_manager: List[CallbackManagerForChainRun],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        if not inputs:
            return []
        if return_exceptions:
            raise NotImplementedError(
                "return_exceptions is not supported for remote clients"
            )

        if isinstance(config, list):
            _config = [_without_callbacks(c) for c in config]
        else:
            _config = _without_callbacks(config)

        response = self.sync_client.post(
            "/batch",
            json={
                "inputs": self._lc_serializer.dumpd(inputs),
                "config": _config,
                "kwargs": kwargs,
            },
        )
        outputs, corresponding_callback_events = _decode_response(
            self._lc_serializer, response, is_batch=True
        )

        # Now handle callbacks if any were returned
        if self._use_server_callback_events and corresponding_callback_events:
            for run_manager_, callback_events in zip(
                run_manager, corresponding_callback_events
            ):
                handle_callbacks(run_manager_, callback_events)

        return outputs

    def _enforce_trailing_slash(self, url: str) -> str:
        if url.endswith("/"):
            return url
        return url + "/"

    def batch(
        self,
        inputs: List[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> List[Output]:
        if kwargs:
            raise NotImplementedError("kwargs not implemented yet.")
        return self._batch_with_config(self._batch, inputs, config)

    async def _abatch(
        self,
        inputs: List[Input],
        run_manager: List[AsyncCallbackManagerForChainRun],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        """Batch invoke the runnable."""
        if not inputs:
            return []
        if return_exceptions:
            raise NotImplementedError(
                "return_exceptions is not supported for remote clients"
            )

        if isinstance(config, list):
            _config = [_without_callbacks(c) for c in config]
        else:
            _config = _without_callbacks(config)

        response = await self.async_client.post(
            "/batch",
            json={
                "inputs": self._lc_serializer.dumpd(inputs),
                "config": _config,
                "kwargs": kwargs,
            },
        )
        outputs, corresponding_callback_events = _decode_response(
            self._lc_serializer, response, is_batch=True
        )

        # Now handle callbacks

        if self._use_server_callback_events and corresponding_callback_events:
            tasks = []
            for run_manager_, callback_events in zip(
                run_manager, corresponding_callback_events
            ):
                tasks.append(ahandle_callbacks(run_manager_, callback_events))

            # Execute coroutines concurrently
            await asyncio.gather(*tasks)
        return outputs

    async def abatch(
        self,
        inputs: List[Input],
        config: Optional[RunnableConfig] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> List[Output]:
        """Batch invoke the runnable."""
        if kwargs:
            raise NotImplementedError("kwargs not implemented yet.")
        if not inputs:
            return []
        return await self._abatch_with_config(self._abatch, inputs, config)

    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        """Stream invoke the runnable."""
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)

        final_output: Optional[Output] = None

        run_manager = callback_manager.on_chain_start(
            dumpd(self),
            self._lc_serializer.dumpd(input),
            name=config.get("run_name"),
        )
        data = {
            "input": self._lc_serializer.dumpd(input),
            "config": _without_callbacks(config),
            "kwargs": kwargs,
        }
        endpoint = urljoin(self._enforce_trailing_slash(self.url), "stream")

        try:
            from httpx_sse import connect_sse
        except ImportError:
            raise ImportError(
                "Missing `httpx_sse` dependency to use the stream method. "
                "Install via `pip install httpx_sse`'"
            )

        try:
            with connect_sse(
                self.sync_client, "POST", endpoint, json=data
            ) as event_source:
                for sse in event_source.iter_sse():
                    if sse.event == "data":
                        chunk = self._lc_serializer.loads(sse.data)
                        yield chunk

                        if final_output:
                            final_output += chunk
                        else:
                            final_output = chunk
                    elif sse.event == "error":
                        # This can only be a server side error
                        _raise_exception_from_data(
                            sse.data, httpx.Request(method="POST", url=endpoint)
                        )
                    elif sse.event == "end":
                        break
                    else:
                        logger.error(
                            f"Encountered an unsupported event type: `{sse.event}`. "
                            f"Try upgrading the remote client to the latest version."
                            f"Ignoring events of type `{sse.event}`."
                        )
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        else:
            run_manager.on_chain_end(final_output)

    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)

        final_output: Optional[Output] = None

        run_manager = await callback_manager.on_chain_start(
            dumpd(self),
            self._lc_serializer.dumpd(input),
            name=config.get("run_name"),
        )
        data = {
            "input": self._lc_serializer.dumpd(input),
            "config": _without_callbacks(config),
            "kwargs": kwargs,
        }
        endpoint = urljoin(self._enforce_trailing_slash(self.url), "stream")

        try:
            from httpx_sse import aconnect_sse
        except ImportError:
            raise ImportError("You must install `httpx_sse` to use the stream method.")

        try:
            async with aconnect_sse(
                self.async_client, "POST", endpoint, json=data
            ) as event_source:
                async for sse in event_source.aiter_sse():
                    if sse.event == "data":
                        chunk = self._lc_serializer.loads(sse.data)
                        yield chunk

                        if final_output:
                            final_output += chunk
                        else:
                            final_output = chunk

                    elif sse.event == "error":
                        # This can only be a server side error
                        _raise_exception_from_data(
                            sse.data, httpx.Request(method="POST", url=endpoint)
                        )
                    elif sse.event == "end":
                        break
                    else:
                        logger.error(
                            f"Encountered an unsupported event type: `{sse.event}`. "
                            f"Try upgrading the remote client to the latest version."
                            f"Ignoring events of type `{sse.event}`."
                        )
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        else:
            await run_manager.on_chain_end(final_output)

    async def astream_log(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        *,
        include_names: Optional[Sequence[str]] = None,
        include_types: Optional[Sequence[str]] = None,
        include_tags: Optional[Sequence[str]] = None,
        exclude_names: Optional[Sequence[str]] = None,
        exclude_types: Optional[Sequence[str]] = None,
        exclude_tags: Optional[Sequence[str]] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[RunLogPatch]:
        """Stream all output from a runnable, as reported to the callback system.

        This includes all inner runs of LLMs, Retrievers, Tools, etc.

        Output is streamed as Log objects, which include a list of
        jsonpatch ops that describe how the state of the run has changed in each
        step, and the final state of the run.

        The jsonpatch ops can be applied in order to construct state.
        """

        # Create a stream handler that will emit Log objects
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)

        final_output: Optional[Output] = None

        run_manager = await callback_manager.on_chain_start(
            dumpd(self),
            self._lc_serializer.dumpd(input),
            name=config.get("run_name"),
        )
        data = {
            "input": self._lc_serializer.dumpd(input),
            "config": _without_callbacks(config),
            "kwargs": kwargs,
            "diff": True,
            "include_names": include_names,
            "include_types": include_types,
            "include_tags": include_tags,
            "exclude_names": exclude_names,
            "exclude_types": exclude_types,
            "exclude_tags": exclude_tags,
        }
        endpoint = urljoin(self.url, "stream_log")

        try:
            from httpx_sse import aconnect_sse
        except ImportError:
            raise ImportError("You must install `httpx_sse` to use the stream method.")

        try:
            async with aconnect_sse(
                self.async_client, "POST", endpoint, json=data
            ) as event_source:
                async for sse in event_source.aiter_sse():
                    if sse.event == "data":
                        data = self._lc_serializer.loads(sse.data)
                        chunk = RunLogPatch(*data["ops"])

                        yield chunk
                        if final_output:
                            final_output += chunk
                        else:
                            final_output = chunk
                    elif sse.event == "error":
                        # This can only be a server side error
                        _raise_exception_from_data(
                            sse.data, httpx.Request(method="POST", url=endpoint)
                        )
                    elif sse.event == "end":
                        break
                    else:
                        _log_error_message_once(
                            f"Encountered an unsupported event type: `{sse.event}`. "
                            f"Try upgrading the remote client to the latest version."
                            f"Ignoring events of type `{sse.event}`."
                        )
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        else:
            await run_manager.on_chain_end(final_output)
