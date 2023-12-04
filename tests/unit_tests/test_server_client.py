"""Test the server and client together."""
import asyncio
import datetime
import json
from asyncio import AbstractEventLoop
from contextlib import asynccontextmanager, contextmanager
from enum import Enum
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union
from unittest.mock import MagicMock, patch
from uuid import UUID

import httpx
import pytest
import pytest_asyncio
from fastapi import APIRouter, FastAPI, Request
from fastapi.testclient import TestClient
from httpx import AsyncClient
from langchain.callbacks.tracers.log_stream import RunLogPatch
from langchain.prompts import PromptTemplate
from langchain.prompts.base import StringPromptValue
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.schema.runnable import Runnable, RunnableConfig, RunnablePassthrough
from langchain.schema.runnable.base import RunnableLambda
from langchain.schema.runnable.utils import ConfigurableField, Input, Output
from langsmith import schemas as ls_schemas
from pytest import MonkeyPatch
from pytest_mock import MockerFixture
from typing_extensions import TypedDict

from langserve import api_handler
from langserve.api_handler import (
    _rename_pydantic_model,
    _replace_non_alphanumeric_with_underscores,
)
from langserve.callbacks import AsyncEventAggregatorCallback
from langserve.client import RemoteRunnable
from langserve.lzstring import LZString
from langserve.schema import CustomUserType

try:
    from pydantic.v1 import BaseModel, Field
except ImportError:
    from pydantic import BaseModel, Field
from langserve.server import add_routes
from tests.unit_tests.utils import FakeListLLM, FakeTracer


def _decode_eventstream(text: str) -> List[Dict[str, Any]]:
    """Simple decoder for testing purposes.

    This is not a good implementation, but it's smple and works for our purposes.
    """
    unpacked_response = [line for line in text.split("\r\n") if line.strip()]

    events = []

    for event_info, encoded_data in zip(
        unpacked_response[::2], unpacked_response[1::2]
    ):
        type_ = event_info[len("event: ") :].strip()
        try:
            data = json.loads(encoded_data[len("data: ") :])
        except json.JSONDecodeError:
            raise AssertionError(f"Could not stream: {text}")

        events.append({"type": type_, "data": data})

    if "end" in unpacked_response[-1]:
        events.append({"type": "end"})

    return events


def _replace_run_id_in_stream_resp(streamed_resp: str) -> str:
    """
    Replace the run_id in the streamed response's metadata with a placeholder.

    Assumes run_id only appears once in the text. This is hacky :)
    """
    metadata_expected_str = 'event: metadata\r\ndata: {"run_id": "'
    run_id_idx = streamed_resp.find(metadata_expected_str)
    assert run_id_idx != -1

    uuid_start_pos = run_id_idx + len(metadata_expected_str)
    uuid_len = 36

    uuid = streamed_resp[uuid_start_pos : uuid_start_pos + uuid_len]
    return streamed_resp.replace(uuid, "<REPLACED>")


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop()
    try:
        yield loop
    finally:
        loop.close()


@pytest.fixture()
def app(event_loop: AbstractEventLoop) -> FastAPI:
    """A simple server that wraps a Runnable and exposes it as an API."""

    async def add_one_or_passthrough(
        x: Union[int, HumanMessage]
    ) -> Union[int, HumanMessage]:
        """Add one to int or passthrough."""
        if isinstance(x, int):
            return x + 1
        else:
            return x

    runnable_lambda = RunnableLambda(func=add_one_or_passthrough)
    app = FastAPI()
    try:
        add_routes(
            app, runnable_lambda, config_keys=["tags"], include_callback_events=True
        )
        yield app
    finally:
        del app


@pytest.fixture()
def app_for_config(event_loop: AbstractEventLoop) -> FastAPI:
    """A simple server that wraps a Runnable and exposes it as an API."""

    async def return_config(
        _: int,
        config: RunnableConfig,
    ) -> Dict[str, Any]:
        """Add one to int or passthrough."""
        return {
            "tags": sorted(config["tags"]),
            "configurable": config.get("configurable"),
        }

    runnable_lambda = RunnableLambda(func=return_config)
    app = FastAPI()
    try:
        add_routes(app, runnable_lambda, config_keys=["tags", "metadata"])
        yield app
    finally:
        del app


@pytest.fixture()
def sync_remote_runnable(app: FastAPI) -> RemoteRunnable:
    """Create a FastAPI app that exposes the Runnable as an API."""
    remote_runnable_client = RemoteRunnable(url="http://localhost:9999")
    sync_client = TestClient(app=app)
    remote_runnable_client.sync_client = sync_client
    try:
        yield remote_runnable_client
    finally:
        sync_client.close()


@contextmanager
def get_sync_remote_runnable(
    server: FastAPI, *, path: Optional[str] = None, raise_server_exceptions: bool = True
) -> RemoteRunnable:
    """Get an async client."""
    url = "http://localhost:9999"
    if path:
        url += path
    remote_runnable_client = RemoteRunnable(url=url)
    sync_client = TestClient(
        app=server, base_url=url, raise_server_exceptions=raise_server_exceptions
    )
    remote_runnable_client.sync_client = sync_client
    try:
        yield remote_runnable_client
    finally:
        sync_client.close()


@asynccontextmanager
async def get_async_test_client(
    server: FastAPI, *, path: Optional[str] = None, raise_app_exceptions: bool = True
) -> AsyncClient:
    """Get an async client."""
    url = "http://localhost:9999"
    if path:
        url += path
    transport = httpx.ASGITransport(
        app=server,
        raise_app_exceptions=raise_app_exceptions,
    )
    async_client = AsyncClient(app=server, base_url=url, transport=transport)
    try:
        yield async_client
    finally:
        await async_client.aclose()


@asynccontextmanager
async def get_async_remote_runnable(
    server: FastAPI, *, path: Optional[str] = None, raise_app_exceptions: bool = True
) -> RemoteRunnable:
    """Get an async client."""
    url = "http://localhost:9999"
    if path:
        url += path
    remote_runnable_client = RemoteRunnable(url=url)

    async with get_async_test_client(
        server, path=path, raise_app_exceptions=raise_app_exceptions
    ) as async_client:
        remote_runnable_client.async_client = async_client
        yield remote_runnable_client


@pytest_asyncio.fixture()
async def async_remote_runnable(app: FastAPI) -> RemoteRunnable:
    """Create a FastAPI app that exposes the Runnable as an API."""
    async with get_async_remote_runnable(app) as client:
        yield client


def test_server(app: FastAPI) -> None:
    """Test the server directly via HTTP requests."""
    sync_client = TestClient(app=app, raise_server_exceptions=True)

    # Test invoke
    response = sync_client.post("/invoke", json={"input": 1})
    assert response.json()["output"] == 2
    events = response.json()["callback_events"]
    assert [event["type"] for event in events] == ["on_chain_start", "on_chain_end"]

    # Test batch
    response = sync_client.post("/batch", json={"inputs": [2, 3]})
    assert response.json()["output"] == [3, 4]

    events = response.json()["callback_events"]
    assert [event["type"] for event in events[0]] == ["on_chain_start", "on_chain_end"]

    assert [event["type"] for event in events[1]] == ["on_chain_start", "on_chain_end"]

    # Test schema
    input_schema = sync_client.get("/input_schema").json()
    assert isinstance(input_schema, dict)
    assert input_schema["title"] == "RunnableLambdaInput"
    #
    output_schema = sync_client.get("/output_schema").json()
    assert isinstance(output_schema, dict)
    assert output_schema["title"] == "RunnableLambdaOutput"

    output_schema = sync_client.get("/config_schema").json()
    assert isinstance(output_schema, dict)
    assert output_schema["title"] == "RunnableLambdaConfig"

    # TODO(Team): Fix test. Issue with eventloops right now when using sync client
    # # Test stream
    # response = sync_client.post("/stream", json={"input": 1})
    # assert response.text == "event: data\r\ndata: 2\r\n\r\nevent: end\r\n\r\n"


def test_serve_playground(app: FastAPI) -> None:
    """Test the server directly via HTTP requests."""
    sync_client = TestClient(app=app)
    response = sync_client.get("/playground/index.html")
    assert response.status_code == 200
    response = sync_client.get("/playground/i_do_not_exist.txt")
    assert response.status_code == 404
    response = sync_client.get("/playground//etc/passwd")
    assert response.status_code == 404


async def test_serve_playground_with_api_router() -> None:
    """Test serving playground from an api router with a prefix."""
    app = FastAPI()

    # Make sure that we can add routers
    # to an API router
    router = APIRouter(prefix="/langserve_runnables")

    add_routes(
        router,
        RunnableLambda(lambda foo: "hello"),
        path="/chat",
    )

    app.include_router(router)
    async_client = AsyncClient(app=app, base_url="http://localhost:9999")
    response = await async_client.get("/langserve_runnables/chat/playground/index.html")
    assert response.status_code == 200


async def test_server_async(app: FastAPI) -> None:
    """Test the server directly via HTTP requests."""
    async with get_async_test_client(app, raise_app_exceptions=True) as async_client:
        # Test invoke
        response = await async_client.post("/invoke", json={"input": 1})
        assert response.json()["output"] == 2
        events = response.json()["callback_events"]
        assert [event["type"] for event in events] == ["on_chain_start", "on_chain_end"]

        # Test batch
        response = await async_client.post("/batch", json={"inputs": [1, 2]})
        assert response.json()["output"] == [2, 3]
        events = response.json()["callback_events"]
        assert [event["type"] for event in events[0]] == [
            "on_chain_start",
            "on_chain_end",
        ]
        assert [event["type"] for event in events[1]] == [
            "on_chain_start",
            "on_chain_end",
        ]

        # Test stream
        response = await async_client.post("/stream", json={"input": 1})
        response_text_with_run_id_replaced = _replace_run_id_in_stream_resp(
            response.text
        )
        expected_response_with_run_id_replaced = (
            'event: metadata\r\ndata: {"run_id": "<REPLACED>"}\r\n\r\n'
            + "event: data\r\ndata: 2\r\n\r\nevent: end\r\n\r\n"
        )
        assert (
            response_text_with_run_id_replaced == expected_response_with_run_id_replaced
        )

        response = await async_client.post("/stream_log", json={"input": 1})
        assert response.text.startswith("event: data\r\n")

    # test bad requests
    async with get_async_test_client(app, raise_app_exceptions=True) as async_client:
        # Test invoke
        response = await async_client.post("/invoke", data="bad json []")
        # Client side error bad json.
        assert response.status_code == 422

        # Missing `input`
        response = await async_client.post("/invoke", json={})
        # Client side error bad json.
        assert response.status_code == 422

        # Missing `input`
        response = await async_client.post(
            "/invoke", json={"input": 4, "config": {"tags": [[]]}}
        )
        # Client side error bad json.
        assert response.status_code == 422

    # test batch bad requests
    async with get_async_test_client(app, raise_app_exceptions=True) as async_client:
        # Test invoke
        # Test bad batch requests
        response = await async_client.post("/batch", data="bad json []")
        # Client side error bad json.
        assert response.status_code == 422

        # Missing `inputs`
        response = await async_client.post("/batch", json={})
        assert response.status_code == 422

        response = await async_client.post(
            "/batch", json={"inputs": [1, 2], "config": {"tags": [[]]}}
        )
        assert response.status_code == 422

        response = await async_client.post(
            "/batch",
            json={
                "inputs": [1, 2],
                "config": [{"tags": ["a"]}, {"tags": ["b"]}, {"tags": ["c"]}],
            },
        )
        assert response.status_code == 422

    # test stream bad requests
    async with get_async_test_client(app, raise_app_exceptions=True) as async_client:
        # Test bad stream requests
        response = await async_client.post("/stream", data="bad json []")
        stream_events = _decode_eventstream(response.text)
        assert stream_events[0]["type"] == "error"
        assert stream_events[0]["data"]["status_code"] == 422

        response = await async_client.post("/stream", json={})
        stream_events = _decode_eventstream(response.text)
        assert stream_events[0]["type"] == "error"
        assert stream_events[0]["data"]["status_code"] == 422

    # test stream_log bad requests
    async with get_async_test_client(app, raise_app_exceptions=True) as async_client:
        response = await async_client.post("/stream_log", data="bad json []")
        stream_events = _decode_eventstream(response.text)
        assert stream_events[0]["type"] == "error"
        assert stream_events[0]["data"]["status_code"] == 422

        response = await async_client.post("/stream_log", json={})
        stream_events = _decode_eventstream(response.text)
        assert stream_events[0]["type"] == "error"
        assert stream_events[0]["data"]["status_code"] == 422


async def test_server_bound_async(app_for_config: FastAPI) -> None:
    """Test the server directly via HTTP requests."""
    async_client = AsyncClient(app=app_for_config, base_url="http://localhost:9999")
    config_hash = LZString.compressToEncodedURIComponent(json.dumps({"tags": ["test"]}))

    # Test invoke
    response = await async_client.post(
        f"/c/{config_hash}/invoke",
        json={"input": 1, "config": {"tags": ["another-one"]}},
    )
    assert response.status_code == 200
    assert response.json()["output"] == {
        "tags": ["another-one", "test"],
        "configurable": None,
    }

    # Test batch
    response = await async_client.post(
        f"/c/{config_hash}/batch",
        json={"inputs": [1], "config": {"tags": ["another-one"]}},
    )
    assert response.status_code == 200
    assert response.json()["output"] == [
        {"tags": ["another-one", "test"], "configurable": None}
    ]

    # Test stream
    response = await async_client.post(
        f"/c/{config_hash}/stream",
        json={"input": 1, "config": {"tags": ["another-one"]}},
    )
    assert response.status_code == 200

    response_with_run_id_replaced = _replace_run_id_in_stream_resp(response.text)
    assert (
        response_with_run_id_replaced
        == """event: metadata\r\ndata: {"run_id": "<REPLACED>"}\r\n\r\nevent: data\r\ndata: {"tags":["another-one","test"],"configurable":null}\r\n\r\nevent: end\r\n\r\n"""  # noqa: E501
    )


def test_invoke(sync_remote_runnable: RemoteRunnable) -> None:
    """Test sync invoke."""
    assert sync_remote_runnable.invoke(1) == 2
    assert sync_remote_runnable.invoke(HumanMessage(content="hello")) == HumanMessage(
        content="hello"
    )
    # Test invocation with config
    assert sync_remote_runnable.invoke(1, config={"tags": ["test"]}) == 2

    # Test tracing
    tracer = FakeTracer()
    assert sync_remote_runnable.invoke(1, config={"callbacks": [tracer]}) == 2
    assert len(tracer.runs) == 1
    # Light test to verify that we're picking up information about the server side
    # function being invoked via a callback.
    assert tracer.runs[0].child_runs[0].name == "RunnableLambda"
    assert (
        tracer.runs[0].child_runs[0].extra["kwargs"]["name"] == "add_one_or_passthrough"
    )


def test_batch(sync_remote_runnable: RemoteRunnable) -> None:
    """Test sync batch."""
    assert sync_remote_runnable.batch([]) == []
    assert sync_remote_runnable.batch([1, 2, 3]) == [2, 3, 4]
    assert sync_remote_runnable.batch([HumanMessage(content="hello")]) == [
        HumanMessage(content="hello")
    ]

    # Test callbacks
    # Using a single tracer for both inputs
    tracer = FakeTracer()
    assert sync_remote_runnable.batch([1, 2], config={"callbacks": [tracer]}) == [2, 3]
    assert len(tracer.runs) == 2
    # Light test to verify that we're picking up information about the server side
    # function being invoked via a callback.
    assert tracer.runs[0].child_runs[0].name == "RunnableLambda"
    assert (
        tracer.runs[0].child_runs[0].extra["kwargs"]["name"] == "add_one_or_passthrough"
    )

    assert tracer.runs[1].child_runs[0].name == "RunnableLambda"
    assert (
        tracer.runs[1].child_runs[0].extra["kwargs"]["name"] == "add_one_or_passthrough"
    )

    # Verify that each tracer gets its own run
    tracer1 = FakeTracer()
    tracer2 = FakeTracer()
    assert sync_remote_runnable.batch(
        [1, 2], config=[{"callbacks": [tracer1]}, {"callbacks": [tracer2]}]
    ) == [2, 3]
    assert len(tracer1.runs) == 1
    assert len(tracer2.runs) == 1
    # Light test to verify that we're picking up information about the server side
    # function being invoked via a callback.
    assert tracer1.runs[0].child_runs[0].name == "RunnableLambda"
    assert (
        tracer1.runs[0].child_runs[0].extra["kwargs"]["name"]
        == "add_one_or_passthrough"
    )

    assert tracer2.runs[0].child_runs[0].name == "RunnableLambda"
    assert (
        tracer2.runs[0].child_runs[0].extra["kwargs"]["name"]
        == "add_one_or_passthrough"
    )


async def test_ainvoke(async_remote_runnable: RemoteRunnable) -> None:
    """Test async invoke."""
    assert await async_remote_runnable.ainvoke(1) == 2

    assert await async_remote_runnable.ainvoke(
        HumanMessage(content="hello")
    ) == HumanMessage(content="hello")

    # Test tracing
    tracer = FakeTracer()
    assert await async_remote_runnable.ainvoke(1, config={"callbacks": [tracer]}) == 2
    assert len(tracer.runs) == 1
    # Light test to verify that we're picking up information about the server side
    # function being invoked via a callback.
    assert tracer.runs[0].child_runs[0].name == "RunnableLambda"
    assert (
        tracer.runs[0].child_runs[0].extra["kwargs"]["name"] == "add_one_or_passthrough"
    )


async def test_abatch(async_remote_runnable: RemoteRunnable) -> None:
    """Test async batch."""
    assert await async_remote_runnable.abatch([]) == []
    assert await async_remote_runnable.abatch([1, 2, 3]) == [2, 3, 4]
    assert await async_remote_runnable.abatch([HumanMessage(content="hello")]) == [
        HumanMessage(content="hello")
    ]

    # Test callbacks
    # Using a single tracer for both inputs
    tracer = FakeTracer()
    assert await async_remote_runnable.abatch(
        [1, 2], config={"callbacks": [tracer]}
    ) == [2, 3]
    assert len(tracer.runs) == 2
    # Light test to verify that we're picking up information about the server side
    # function being invoked via a callback.
    assert tracer.runs[0].child_runs[0].name == "RunnableLambda"
    assert (
        tracer.runs[0].child_runs[0].extra["kwargs"]["name"] == "add_one_or_passthrough"
    )

    assert tracer.runs[1].child_runs[0].name == "RunnableLambda"
    assert (
        tracer.runs[1].child_runs[0].extra["kwargs"]["name"] == "add_one_or_passthrough"
    )

    # Verify that each tracer gets its own run
    tracer1 = FakeTracer()
    tracer2 = FakeTracer()
    assert await async_remote_runnable.abatch(
        [1, 2], config=[{"callbacks": [tracer1]}, {"callbacks": [tracer2]}]
    ) == [2, 3]
    assert len(tracer1.runs) == 1
    assert len(tracer2.runs) == 1
    # Light test to verify that we're picking up information about the server side
    # function being invoked via a callback.
    assert tracer1.runs[0].child_runs[0].name == "RunnableLambda"
    assert (
        tracer1.runs[0].child_runs[0].extra["kwargs"]["name"]
        == "add_one_or_passthrough"
    )

    assert tracer2.runs[0].child_runs[0].name == "RunnableLambda"
    assert (
        tracer2.runs[0].child_runs[0].extra["kwargs"]["name"]
        == "add_one_or_passthrough"
    )


async def test_astream(async_remote_runnable: RemoteRunnable) -> None:
    """Test astream log."""

    app = FastAPI()

    async def add_one_or_passthrough(
        x: Union[int, HumanMessage]
    ) -> Union[int, HumanMessage]:
        """Add one to int or passthrough."""
        if isinstance(x, int):
            return x + 1
        else:
            return x

    runnable = RunnableLambda(add_one_or_passthrough)
    add_routes(app, runnable)

    # Invoke request
    async with get_async_remote_runnable(app, raise_app_exceptions=False) as runnable:
        # Test bad requests
        # test client side error
        with pytest.raises(httpx.HTTPStatusError) as cb:
            # Invalid input type (expected string but got int)
            async for _ in runnable.astream("foo"):
                pass

        # Verify that this is a 422 error
        assert cb.value.response.status_code == 422

        # test with good requests
        outputs = []

        async for chunk in async_remote_runnable.astream(1):
            outputs.append(chunk)

        assert outputs == [2]

        outputs = []
        data = HumanMessage(content="hello")

        async for chunk in async_remote_runnable.astream(data):
            outputs.append(chunk)

        assert outputs == [data]


async def test_astream_log_diff_no_effect(
    async_remote_runnable: RemoteRunnable,
) -> None:
    """Test async stream."""
    run_logs = []

    async for chunk in async_remote_runnable.astream_log(1, diff=False):
        run_logs.append(chunk)

    op = run_logs[0].ops[0]
    uuid = op["value"]["id"]

    assert [run_log_patch.ops for run_log_patch in run_logs] == [
        [
            {
                "op": "replace",
                "path": "",
                "value": {
                    "final_output": {"output": 2},
                    "id": uuid,
                    "logs": {},
                    "streamed_output": [],
                },
            }
        ],
        [{"op": "replace", "path": "/final_output", "value": {"output": 2}}],
        [{"op": "add", "path": "/streamed_output/-", "value": 2}],
    ]


async def test_astream_log(async_remote_runnable: RemoteRunnable) -> None:
    """Test astream log."""

    app = FastAPI()

    def add_one(x: int) -> int:
        """Add one to simulate a valid function"""
        return x + 1

    runnable = RunnableLambda(add_one)
    add_routes(app, runnable)

    # Invoke request
    async with get_async_remote_runnable(app, raise_app_exceptions=False) as runnable:
        # Test bad requests
        # test client side error
        with pytest.raises(httpx.HTTPStatusError) as cb:
            # Invalid input type (expected string but got int)
            async for _ in runnable.astream_log("foo"):
                pass

        # Verify that this is a 422 error
        assert cb.value.response.status_code == 422

        with pytest.raises(httpx.HTTPStatusError) as cb:
            # Invalid input type (expected string but got int)
            # include names should not be a list of lists
            async for _ in runnable.astream_log(1, include_names=[[]]):
                pass

        # Verify that this is a 422 error
        assert cb.value.response.status_code == 422

        # Test good requests
        run_log_patches = []

        async for chunk in runnable.astream_log(1, diff=True):
            run_log_patches.append(chunk)

        op = run_log_patches[0].ops[0]
        uuid = op["value"]["id"]

        assert [run_log_patch.ops for run_log_patch in run_log_patches] == [
            [
                {
                    "op": "replace",
                    "path": "",
                    "value": {
                        "final_output": {"output": 2},
                        "id": uuid,
                        "logs": {},
                        "streamed_output": [],
                    },
                }
            ],
            [{"op": "replace", "path": "/final_output", "value": {"output": 2}}],
            [{"op": "add", "path": "/streamed_output/-", "value": 2}],
        ]


def test_invoke_as_part_of_sequence(sync_remote_runnable: RemoteRunnable) -> None:
    """Test as part of sequence."""
    runnable = sync_remote_runnable | RunnableLambda(func=lambda x: x + 1)
    # without config
    assert runnable.invoke(1) == 3
    # with config
    assert runnable.invoke(1, config={"tags": ["test"]}) == 3
    # without config
    assert runnable.batch([1, 2]) == [3, 4]
    # with config
    assert runnable.batch([1, 2], config={"tags": ["test"]}) == [3, 4]
    # TODO(Team): Determine how to test some issues with event loops for testing
    #   set up
    # without config
    # assert list(runnable.stream([1, 2])) == [3, 4]
    # # with config
    # assert list(runnable.stream([1, 2], config={"tags": ["test"]})) == [3, 4]


async def test_invoke_as_part_of_sequence_async(
    async_remote_runnable: RemoteRunnable,
) -> None:
    """Test as part of a sequence.

    This helps to verify that config is handled properly (e.g., callbacks are not
    passed to the server, but other config is)
    """
    runnable = async_remote_runnable | RunnableLambda(
        func=lambda x: x + 1 if isinstance(x, int) else x
    ).with_config({"run_name": "hello"})
    # without config
    assert await runnable.ainvoke(1) == 3
    # with config
    assert await runnable.ainvoke(1, config={"tags": ["test"]}) == 3
    # without config
    assert await runnable.abatch([1, 2]) == [3, 4]
    # with config
    assert await runnable.abatch([1, 2], config={"tags": ["test"]}) == [3, 4]

    # Verify can pass many configs to batch
    configs = [{"tags": ["test"]}, {"tags": ["test2"]}]
    assert await runnable.abatch([1, 2], config=configs) == [3, 4]

    # Verify can ValueError on mismatched configs  number
    with pytest.raises(ValueError):
        assert await runnable.abatch([1, 2], config=[configs[0]]) == [3, 4]

    configs = [{"tags": ["test"]}, {"tags": ["test2"]}]
    assert await runnable.abatch([1, 2], config=configs) == [3, 4]

    configs = [
        {"tags": ["test"]},
        {"tags": ["test2"], "other": "test"},
    ]
    assert await runnable.abatch([1, 2], config=configs) == [3, 4]

    # Without config
    assert [x async for x in runnable.astream(1)] == [3]

    # With Config
    assert [x async for x in runnable.astream(1, config={"tags": ["test"]})] == [3]

    # With config and LC input data
    assert [
        x
        async for x in runnable.astream(
            HumanMessage(content="hello"), config={"tags": ["test"]}
        )
    ] == [HumanMessage(content="hello")]

    log_patches = [x async for x in runnable.astream_log(1)]
    for log_patch in log_patches:
        assert isinstance(log_patch, RunLogPatch)
    # Only check the first entry (not validating implementation here)
    first_op = log_patches[0].ops[0]
    assert first_op["op"] == "replace"
    assert first_op["path"] == ""

    # Validate with HumanMessage
    log_patches = [x async for x in runnable.astream_log(HumanMessage(content="hello"))]
    for log_patch in log_patches:
        assert isinstance(log_patch, RunLogPatch)
    # Only check the first entry (not validating implementation here)
    first_op = log_patches[0].ops[0]
    assert first_op == {
        "op": "replace",
        "path": "",
        "value": {
            "final_output": None,
            "id": first_op["value"]["id"],
            "logs": {},
            "streamed_output": [],
        },
    }


async def test_multiple_runnables(event_loop: AbstractEventLoop) -> None:
    """Test serving multiple runnables."""

    async def add_one(x: int) -> int:
        """Add one to simulate a valid function"""
        return x + 1

    async def mul_2(x: int) -> int:
        """Add one to simulate a valid function"""
        return x * 2

    app = FastAPI()
    add_routes(app, RunnableLambda(add_one), path="/add_one")
    add_routes(
        app,
        RunnableLambda(mul_2),
        input_type=int,
        path="/mul_2",
    )

    add_routes(app, PromptTemplate.from_template("{question}"), path="/prompt_1")

    add_routes(
        app, PromptTemplate.from_template("{question} {answer}"), path="/prompt_2"
    )

    async with get_async_remote_runnable(app, path="/add_one") as runnable:
        async with get_async_remote_runnable(app, path="/mul_2") as runnable2:
            assert await runnable.ainvoke(1) == 2
            assert await runnable2.ainvoke(4) == 8

            composite_runnable = runnable | runnable2
            assert await composite_runnable.ainvoke(3) == 8

            # Invoke runnable (remote add_one), local add_one, remote mul_2
            composite_runnable_2 = runnable | add_one | runnable2
            assert await composite_runnable_2.ainvoke(3) == 10

    async with get_async_remote_runnable(app, path="/prompt_1") as runnable:
        assert await runnable.ainvoke(
            {"question": "What is your name?"}
        ) == StringPromptValue(text="What is your name?")

    async with get_async_remote_runnable(app, path="/prompt_2") as runnable:
        assert await runnable.ainvoke(
            {"question": "What is your name?", "answer": "Bob"}
        ) == StringPromptValue(text="What is your name? Bob")


async def test_input_validation(
    event_loop: AbstractEventLoop, mocker: MockerFixture
) -> None:
    """Test client side and server side exceptions."""

    async def add_one(x: int) -> int:
        """Add one to simulate a valid function"""
        return x + 1

    server_runnable = RunnableLambda(func=add_one)
    server_runnable2 = RunnableLambda(func=add_one)

    app = FastAPI()
    add_routes(
        app,
        server_runnable,
        input_type=int,
        path="/add_one",
    )

    add_routes(
        app,
        server_runnable2,
        input_type=int,
        path="/add_one_config",
        config_keys=["tags", "metadata"],
    )

    async with get_async_remote_runnable(
        app, path="/add_one", raise_app_exceptions=False
    ) as runnable:
        # Verify that can be invoked with valid input
        assert await runnable.ainvoke(1) == 2
        # Verify that the following substring is present in the error message
        with pytest.raises(httpx.HTTPError):
            await runnable.ainvoke("hello")

        with pytest.raises(httpx.HTTPError):
            await runnable.abatch(["hello"])

    config = {"tags": ["test"], "metadata": {"a": 5}}

    server_runnable_spy = mocker.spy(server_runnable, "ainvoke")
    # Verify config is handled correctly
    async with get_async_remote_runnable(app, path="/add_one") as runnable1:
        # Verify that can be invoked with valid input
        # Config ignored for runnable1
        assert await runnable1.ainvoke(1, config=config) == 2
        # Config should be ignored but default debug information
        # will still be added
        config_seen = server_runnable_spy.call_args[0][1]
        assert "metadata" in config_seen
        assert "a" not in config_seen["metadata"]
        assert "__useragent" in config_seen["metadata"]
        assert "__langserve_version" in config_seen["metadata"]
        assert "__langserve_endpoint" in config_seen["metadata"]
        assert config_seen["metadata"]["__langserve_endpoint"] == "invoke"

    server_runnable2_spy = mocker.spy(server_runnable2, "ainvoke")
    async with get_async_remote_runnable(app, path="/add_one_config") as runnable2:
        # Config accepted for runnable2
        assert await runnable2.ainvoke(1, config=config) == 2
        # Config ignored

        config_seen = server_runnable2_spy.call_args[0][1]
        assert config_seen["tags"] == ["test"]
        assert config_seen["metadata"]["a"] == 5
        assert "__useragent" in config_seen["metadata"]
        assert "__langserve_version" in config_seen["metadata"]
        assert "__langserve_endpoint" in config_seen["metadata"]
        assert config_seen["metadata"]["__langserve_endpoint"] == "invoke"


async def test_input_validation_with_lc_types(event_loop: AbstractEventLoop) -> None:
    """Test client side and server side exceptions."""

    app = FastAPI()

    class InputType(TypedDict):
        messages: List[HumanMessage]

    runnable = RunnablePassthrough()
    add_routes(app, runnable, config_keys=["tags"], input_type=InputType)
    # Invoke request
    async with get_async_remote_runnable(
        app, raise_app_exceptions=False
    ) as passthrough_runnable:
        with pytest.raises(httpx.HTTPError):
            await passthrough_runnable.ainvoke("Hello")
        with pytest.raises(httpx.HTTPError):
            await passthrough_runnable.ainvoke(["hello"])
        with pytest.raises(httpx.HTTPError):
            await passthrough_runnable.ainvoke(HumanMessage(content="h"))
        with pytest.raises(httpx.HTTPError):
            await passthrough_runnable.ainvoke([SystemMessage(content="hello")])

        # Valid
        await passthrough_runnable.ainvoke(
            {"messages": [HumanMessage(content="hello")]}
        )

        # Kwargs not supported yet
        with pytest.raises(NotImplementedError):
            await passthrough_runnable.ainvoke(
                {"messages": [HumanMessage(content="hello")]}, hello=2
            )

        with pytest.raises(httpx.HTTPError):
            # tags should be a list of str not a list of lists
            await passthrough_runnable.ainvoke(
                {"messages": [HumanMessage(content="hello")]}, config={"tags": [["q"]]}
            )

        # Valid
        result = await passthrough_runnable.ainvoke(
            {"messages": [HumanMessage(content="hello")]}, config={"tags": ["test"]}
        )

        assert isinstance(result, dict)
        assert isinstance(result["messages"][0], HumanMessage)

    # Batch request
    async with get_async_remote_runnable(
        app, raise_app_exceptions=False
    ) as passthrough_runnable:
        # invalid
        with pytest.raises(httpx.HTTPError):
            await passthrough_runnable.abatch("Hello")
        with pytest.raises(httpx.HTTPError):
            await passthrough_runnable.abatch(["hello"])
        with pytest.raises(httpx.HTTPError):
            await passthrough_runnable.abatch([[SystemMessage(content="hello")]])

        # valid
        result = await passthrough_runnable.abatch(
            [{"messages": [HumanMessage(content="hello")]}]
        )
        assert isinstance(result, list)
        assert isinstance(result[0], dict)
        assert isinstance(result[0]["messages"][0], HumanMessage)


def test_client_close() -> None:
    """Test that the client can be automatically."""
    runnable = RemoteRunnable(url="/dev/null", timeout=1)
    sync_client = runnable.sync_client
    async_client = runnable.async_client
    assert async_client.is_closed is False
    assert sync_client.is_closed is False
    del runnable
    assert sync_client.is_closed is True
    assert async_client.is_closed is True


async def test_async_client_close() -> None:
    """Test that the client can be automatically."""
    runnable = RemoteRunnable(url="/dev/null", timeout=1)
    sync_client = runnable.sync_client
    async_client = runnable.async_client
    assert async_client.is_closed is False
    assert sync_client.is_closed is False
    del runnable
    assert sync_client.is_closed is True
    assert async_client.is_closed is True


async def test_openapi_docs_with_identical_runnables(
    event_loop: AbstractEventLoop, mocker: MockerFixture
) -> None:
    """Test client side and server side exceptions."""

    async def add_one(x: int) -> int:
        """Add one to simulate a valid function"""
        return x + 1

    server_runnable = RunnableLambda(func=add_one)
    server_runnable2 = RunnableLambda(func=add_one)
    server_runnable3 = PromptTemplate.from_template("say {name}")
    server_runnable4 = PromptTemplate.from_template("say {name} {hello}")

    app = FastAPI()
    add_routes(
        app,
        server_runnable,
        path="/a",
    )
    # Add another route that uses the same schema (inferred from runnable input schema)
    add_routes(
        app,
        server_runnable2,
        path="/b",
        config_keys=["tags"],
    )

    add_routes(
        app,
        server_runnable3,
        path="/c",
        config_keys=["tags"],
    )

    add_routes(
        app,
        server_runnable4,
        path="/d",
        config_keys=["tags"],
    )

    async with AsyncClient(app=app, base_url="http://localhost:9999") as async_client:
        response = await async_client.get("/openapi.json")
        assert response.status_code == 200


async def test_configurable_runnables(event_loop: AbstractEventLoop) -> None:
    """Add tests for using langchain's configurable runnables"""

    template = PromptTemplate.from_template("say {name}").configurable_fields(
        template=ConfigurableField(
            id="template",
            name="Template",
            description="The template to use for the prompt",
        )
    )
    llm = (
        RunnablePassthrough() | RunnableLambda(lambda prompt: prompt.text)
    ).configurable_alternatives(
        ConfigurableField(
            id="llm",
            name="LLM",
        ),
        hardcoded_llm=FakeListLLM(responses=["hello Mr. Kitten!"]),
    )
    chain = template | llm
    # Check server side
    assert chain.invoke({"name": "cat"}) == "say cat"

    app = FastAPI()
    add_routes(app, chain)

    async with get_async_remote_runnable(app) as remote_runnable:
        # Test with hard-coded LLM
        assert chain.invoke({"name": "cat"}) == "say cat"
        # Test with different prompt

        assert (
            await remote_runnable.ainvoke(
                {"name": "foo"},
                {"configurable": {"template": "hear {name}"}},
            )
            == "hear foo"
        )
        # Test with alternative passthrough LLM
        assert (
            await remote_runnable.ainvoke(
                {"name": "foo"},
                {"configurable": {"llm": "hardcoded_llm"}},
            )
            == "hello Mr. Kitten!"
        )

    add_routes(app, chain, path="/no_config", config_keys=["tags"])

    async with get_async_remote_runnable(app, path="/no_config") as remote_runnable:
        with pytest.raises(httpx.HTTPError) as cb:
            await remote_runnable.ainvoke(
                {"name": "foo"},
                {"configurable": {"template": "hear {name}"}},
            )

        assert cb.value.response.status_code == 422


# Test for utilities


@pytest.mark.parametrize(
    "s,expected",
    [
        ("hello", "hello"),
        ("hello world", "hello_world"),
        ("hello-world", "hello_world"),
        ("hello_world", "hello_world"),
        ("hello.world", "hello_world"),
    ],
)
def test_replace_non_alphanumeric(s: str, expected: str) -> None:
    """Test replace non alphanumeric."""
    assert _replace_non_alphanumeric_with_underscores(s) == expected


def test_rename_pydantic_model() -> None:
    """Test rename pydantic model."""

    class Foo(BaseModel):
        bar: str = Field(..., description="A bar")
        baz: str = Field(..., description="A baz")

    Model = _rename_pydantic_model(Foo, "Bar")

    assert isinstance(Model, type)
    assert Model.__name__ == "BarFoo"


async def test_input_config_output_schemas(event_loop: AbstractEventLoop) -> None:
    """Test schemas returned for different configurations."""
    # TODO(Fix me): need to fix handling of global state -- we get problems
    # gives inconsistent results when running multiple tests / results
    # depending on ordering
    api_handler._SEEN_NAMES = set()
    api_handler._MODEL_REGISTRY = {}

    async def add_one(x: int) -> int:
        """Add one to simulate a valid function"""
        return x + 1

    async def add_two(y: int) -> int:
        """Add one to simulate a valid function"""
        return y + 2

    app = FastAPI()

    add_routes(app, RunnableLambda(add_one), path="/add_one")
    # Custom input type
    add_routes(
        app,
        RunnableLambda(add_two),
        path="/add_two_custom",
        input_type=float,
        output_type=float,
        config_keys=["tags", "configurable"],
    )
    add_routes(app, PromptTemplate.from_template("{question}"), path="/prompt_1")

    template = PromptTemplate.from_template("say {name}").configurable_fields(
        template=ConfigurableField(
            id="template",
            name="Template",
            description="The template to use for the prompt",
        )
    )
    add_routes(app, template, path="/prompt_2", config_keys=["tags", "configurable"])

    async with AsyncClient(app=app, base_url="http://localhost:9999") as async_client:
        # input schema
        response = await async_client.get("/add_one/input_schema")
        assert response.json() == {"title": "RunnableLambdaInput", "type": "integer"}

        response = await async_client.get("/add_two_custom/input_schema")
        assert response.json() == {"title": "RunnableBindingInput", "type": "number"}

        response = await async_client.get("/prompt_1/input_schema")
        assert response.json() == {
            "properties": {"question": {"title": "Question", "type": "string"}},
            "title": "PromptInput",
            "type": "object",
        }

        response = await async_client.get("/prompt_2/input_schema")
        assert response.json() == {
            "properties": {"name": {"title": "Name", "type": "string"}},
            "title": "PromptInput",
            "type": "object",
        }

        # output schema
        response = await async_client.get("/add_one/output_schema")
        assert response.json() == {
            "title": "RunnableLambdaOutput",
            "type": "integer",
        }

        response = await async_client.get("/add_two_custom/output_schema")
        assert response.json() == {"title": "RunnableBindingOutput", "type": "number"}

        # Just verify that the schema is not empty (it's pretty long)
        # and the actual value should be tested in LangChain
        response = await async_client.get("/prompt_1/output_schema")
        assert response.json() != {}  # Long string

        response = await async_client.get("/prompt_2/output_schema")
        assert response.json() != {}  # Long string

        ## Config schema
        response = await async_client.get("/add_one/config_schema")
        assert response.json() == {
            "properties": {},
            "title": "RunnableLambdaConfig",
            "type": "object",
        }

        response = await async_client.get("/add_two_custom/config_schema")
        assert response.json() == {
            "properties": {
                "tags": {"items": {"type": "string"}, "title": "Tags", "type": "array"}
            },
            "title": "RunnableLambdaConfig",
            "type": "object",
        }

        response = await async_client.get("/prompt_2/config_schema")
        assert response.json() == {
            "definitions": {
                "Configurable": {
                    "properties": {
                        "template": {
                            "default": "say {name}",
                            "description": "The template to use for the prompt",
                            "title": "Template",
                            "type": "string",
                        }
                    },
                    "title": "Configurable",
                    "type": "object",
                }
            },
            "properties": {
                "configurable": {"$ref": "#/definitions/Configurable"},
                "tags": {"items": {"type": "string"}, "title": "Tags", "type": "array"},
            },
            "title": "RunnableConfigurableFieldsConfig",
            "type": "object",
        }


async def test_input_schema_typed_dict() -> None:
    class InputType(TypedDict):
        foo: str
        bar: List[int]

    async def passthrough_dict(d: Any) -> Any:
        return d

    runnable_lambda = RunnableLambda(func=passthrough_dict)
    app = FastAPI()
    add_routes(app, runnable_lambda, input_type=InputType, config_keys=["tags"])

    async with AsyncClient(app=app, base_url="http://localhost:9999") as client:
        res = await client.get("/input_schema")
        assert res.json() == {
            "$ref": "#/definitions/InputType",
            "definitions": {
                "InputType": {
                    "properties": {
                        "bar": {
                            "items": {"type": "integer"},
                            "title": "Bar",
                            "type": "array",
                        },
                        "foo": {"title": "Foo", "type": "string"},
                    },
                    "required": ["foo", "bar"],
                    "title": "InputType",
                    "type": "object",
                }
            },
            "title": "RunnableBindingInput",
        }


class StreamingRunnable(Runnable):
    """A custom runnable used for testing purposes"""

    iterable: Iterable[Any]

    def __init__(self, iterable: Iterable[Any]) -> None:
        """Initialize the runnable."""
        self.iterable = iterable

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        """Invoke the runnable."""
        raise ValueError("Server side error")

    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        raise NotImplementedError()

    async def astream(
        self,
        input: Iterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        for element in self.iterable:
            if isinstance(element, BaseException):
                raise element
            yield element


# Have not figured out how to test sync stream yet
# def test_streaming_dict_sync() -> None:
#     """Test streaming different types of items."""
#     app = FastAPI()
#
#     stream_dict = StreamingRunnable(iterable=[{"a": "1"}, {"a": "2"}])
#
#     add_routes(app, stream_dict)
#
#     # Invoke request
#     with get_sync_remote_runnable(app) as runnable:
#         chunks = []
#         for chunk in runnable.stream("input ignored"):
#             chunks.append(chunk)
#
#     assert chunks == [{"a": "1"}, {"a": "2"}]


async def test_streaming_dict_async() -> None:
    """Test streaming different types of items."""
    app = FastAPI()

    stream_dict = StreamingRunnable(iterable=[{"a": "1"}, {"a": "2"}])

    add_routes(app, stream_dict)

    # Invoke request
    async with get_async_remote_runnable(app, raise_app_exceptions=False) as runnable:
        chunks = []
        async for chunk in runnable.astream("input ignored"):
            chunks.append(chunk)

        assert chunks == [{"a": "1"}, {"a": "2"}]


async def test_server_side_error() -> None:
    """Test server side error handling."""

    app = FastAPI()

    erroring_stream = StreamingRunnable(iterable=[1, 2, ValueError("An error")])
    add_routes(app, erroring_stream)

    # Invoke request
    async with get_async_remote_runnable(app, raise_app_exceptions=False) as runnable:
        callback = AsyncEventAggregatorCallback()
        with pytest.raises(httpx.HTTPStatusError) as cm:
            assert await runnable.ainvoke(1, config={"callbacks": [callback]})
        assert isinstance(cm.value, httpx.HTTPStatusError)
        assert [event["type"] for event in callback.callback_events] == [
            "on_chain_start",
            "on_chain_error",
        ]

        callback1 = AsyncEventAggregatorCallback()
        callback2 = AsyncEventAggregatorCallback()
        with pytest.raises(httpx.HTTPStatusError) as cm:
            assert await runnable.abatch(
                [1, 2], config=[{"callbacks": [callback1]}, {"callbacks": [callback2]}]
            )
        assert isinstance(cm.value, httpx.HTTPStatusError)
        assert [event["type"] for event in callback1.callback_events] == [
            "on_chain_start",
            "on_chain_error",
        ]
        assert [event["type"] for event in callback2.callback_events] == [
            "on_chain_start",
            "on_chain_error",
        ]
        # Test astream
        chunks = []
        try:
            async for chunk in runnable.astream({"a": 1}):
                chunks.append(chunk)
        except httpx.HTTPStatusError as e:
            assert chunks == [1, 2]
            assert e.response.status_code == 500
            assert e.response.text == "Internal Server Error"

        # # Failing right now, can uncomment or add callbacks
        # # Test astream_log
        # chunks = []
        # try:
        #     async for chunk in runnable.astream_log({"a": 1}):
        #         chunks.append(chunk)
        # except httpx.HTTPStatusError as e:
        #     assert chunks == []
        #     assert e.response.status_code == 500
        #     assert e.response.text == "Internal Server Error"


def test_server_side_error_sync(event_loop: AbstractEventLoop) -> None:
    """Test server side error handling."""

    app = FastAPI()
    erroring_stream = StreamingRunnable(iterable=[1, 2, ValueError("An error")])
    add_routes(app, erroring_stream)

    # Invoke request
    with get_sync_remote_runnable(app, raise_server_exceptions=False) as runnable:
        with pytest.raises(httpx.HTTPStatusError) as cm:
            assert runnable.invoke(1)
        assert isinstance(cm.value, httpx.HTTPStatusError)

        with pytest.raises(httpx.HTTPStatusError) as cm:
            assert runnable.batch([1, 2])
        assert isinstance(cm.value, httpx.HTTPStatusError)

        # Test astream
        chunks = []
        try:
            for chunk in runnable.stream({"a": 1}):
                chunks.append(chunk)
        except httpx.HTTPStatusError as e:
            assert chunks == [1, 2]
            assert e.response.status_code == 500
            assert e.response.text == "Internal Server Error"


def test_error_on_bad_path() -> None:
    """Test error on bad path"""
    app = FastAPI()
    with pytest.raises(ValueError):
        add_routes(app, RunnableLambda(lambda foo: "hello"), path="foo")
    add_routes(app, RunnableLambda(lambda foo: "hello"), path="/foo")


def test_error_on_path_collision() -> None:
    """Test error on path collision."""
    app = FastAPI()
    add_routes(app, RunnableLambda(lambda foo: "hello"), path="/foo")
    with pytest.raises(ValueError):
        add_routes(app, RunnableLambda(lambda foo: "hello"), path="/foo")
    with pytest.raises(ValueError):
        add_routes(app, RunnableLambda(lambda foo: "hello"), path="/foo")
    add_routes(app, RunnableLambda(lambda foo: "hello"), path="/baz")


async def test_custom_user_type() -> None:
    """Test custom user type."""
    app = FastAPI()

    class Foo(CustomUserType):
        bar: int

    def func(foo: Foo) -> int:
        """Sample function that expects a Foo type which is a pydantic model"""
        assert isinstance(foo, Foo)
        return foo.bar

    class Baz(BaseModel):
        bar: int

    def func2(baz) -> int:
        """Sample function that expects a Foo type which is a pydantic model"""
        assert isinstance(baz, dict)
        return baz["bar"]

    add_routes(app, RunnableLambda(func), path="/foo")
    add_routes(app, RunnableLambda(func2).with_types(input_type=Baz), path="/baz")

    # Invoke request
    async with get_async_remote_runnable(
        app, path="/foo", raise_app_exceptions=False
    ) as runnable:
        assert await runnable.ainvoke({"bar": 1}) == 1


async def test_using_router() -> None:
    """Test using a router."""
    app = FastAPI()

    # Make sure that we can add routers
    # to an API router
    router = APIRouter()

    add_routes(
        router,
        RunnableLambda(lambda foo: "hello"),
        path="/chat",
    )

    app.include_router(router)


def _is_valid_uuid(uuid_as_str: str) -> bool:
    """Check if uuid_as_str is a valid UUID."""
    try:
        UUID(str(uuid_as_str))
        return True
    except ValueError:
        return False


async def test_invoke_returns_run_id(app: FastAPI) -> None:
    """Test the server directly via HTTP requests."""
    async with get_async_test_client(app, raise_app_exceptions=True) as async_client:
        response = await async_client.post("/invoke", json={"input": 1})
        run_id = response.json()["metadata"]["run_id"]
        assert _is_valid_uuid(run_id)


async def test_batch_returns_run_id(app: FastAPI) -> None:
    """Test the server directly via HTTP requests."""
    async with get_async_test_client(app, raise_app_exceptions=True) as async_client:
        response = await async_client.post("/batch", json={"inputs": [1, 2]})
        run_ids = response.json()["metadata"]["run_ids"]
        assert len(run_ids) == 2
        for run_id in run_ids:
            assert _is_valid_uuid(run_id)


async def test_feedback_succeeds_when_langsmith_enabled() -> None:
    """Tests that the feedback endpoint can accept feedback to langsmith."""

    with patch("langserve.api_handler.ls_client") as mocked_ls_client_package:
        with patch("langserve.api_handler.tracing_is_enabled") as tracing_is_enabled:
            tracing_is_enabled.return_value = True
            mocked_client = MagicMock(return_value=None)
            mocked_ls_client_package.Client.return_value = mocked_client
            mocked_client.create_feedback.return_value = ls_schemas.Feedback(
                id="5484c6b3-5a1a-4a87-b2c7-2e39e7a7e4ac",
                created_at=datetime.datetime(1994, 9, 19, 9, 19),
                modified_at=datetime.datetime(1994, 9, 19, 9, 19),
                run_id="f47ac10b-58cc-4372-a567-0e02b2c3d479",
                key="silliness",
                score=1000,
            )

            local_app = FastAPI()
            add_routes(
                local_app,
                RunnableLambda(lambda foo: "hello"),
                enable_feedback_endpoint=True,
            )

            async with get_async_test_client(
                local_app, raise_app_exceptions=True
            ) as async_client:
                response = await async_client.post(
                    "/feedback",
                    json={
                        "run_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
                        "key": "silliness",
                        "score": 1000,
                    },
                )

                expected_response_json = {
                    "run_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
                    "key": "silliness",
                    "score": 1000,
                    "created_at": "1994-09-19T09:19:00",
                    "modified_at": "1994-09-19T09:19:00",
                    "comment": None,
                    "correction": None,
                    "value": None,
                }

                json_response = response.json()

                assert "id" in json_response
                del json_response["id"]

                assert json_response == expected_response_json


async def test_feedback_fails_when_langsmith_disabled(app: FastAPI) -> None:
    """Tests that feedback is not sent to langsmith if langsmith is disabled."""
    with MonkeyPatch.context() as mp:
        # Explicitly disable langsmith
        mp.setenv("LANGCHAIN_TRACING_V2", "false")
        local_app = FastAPI()
        add_routes(
            local_app,
            RunnableLambda(lambda foo: "hello"),
            # Explicitly enable feedback so that we know failures are from
            # langsmith being disabled
            enable_feedback_endpoint=True,
        )

        async with get_async_test_client(
            local_app, raise_app_exceptions=True
        ) as async_client:
            response = await async_client.post(
                "/feedback",
                json={
                    "run_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
                    "key": "silliness",
                    "score": 1000,
                },
            )
            assert response.status_code == 400


async def test_feedback_fails_when_endpoint_disabled(app: FastAPI) -> None:
    """Tests that the feedback endpoint returns 404s if the user turns it off."""
    async with get_async_test_client(
        app,
        raise_app_exceptions=True,
    ) as async_client:
        response = await async_client.post(
            "/feedback",
            json={
                "run_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
                "key": "silliness",
                "score": 1000,
            },
        )
        assert response.status_code == 404


async def test_per_request_config_modifier(
    event_loop: AbstractEventLoop, mocker: MockerFixture
) -> None:
    """Test updating the config based on the raw request object."""

    async def add_one(x: int) -> int:
        """Add one to simulate a valid function"""
        return x + 1

    app = FastAPI()

    def header_passthru_modifier(
        config: Dict[str, Any], request: Request
    ) -> Dict[str, Any]:
        """Update the config"""
        config = config.copy()
        if "metadata" in config:
            config["metadata"] = config["metadata"].copy()
        else:
            config["metadata"] = {}
        config["metadata"]["headers"] = request.headers
        return config

    server_runnable = RunnableLambda(add_one)

    add_routes(
        app,
        server_runnable,
        path="/add_one",
        per_req_config_modifier=header_passthru_modifier,
    )


async def test_uuid_serialization(event_loop: AbstractEventLoop) -> None:
    """Test updating the config based on the raw request object."""
    import datetime

    from typing_extensions import TypedDict

    class MySpecialEnum(str, Enum):
        """An enum for testing"""

        A = "a"
        B = "b"

    class VariousTypes(TypedDict):
        """A class for testing various types"""

        uuid: UUID
        dt: datetime.datetime
        date: datetime.date
        time: datetime.time
        enum: MySpecialEnum

    async def check_types(inputs: VariousTypes) -> int:
        """Add one to simulate a valid function."""
        assert inputs == {
            "date": datetime.date(2023, 1, 1),
            "dt": datetime.datetime(2023, 1, 1, 5, 0),
            "enum": MySpecialEnum.A,
            "time": datetime.time(5, 30),
            "uuid": UUID("00000000-0000-0000-0000-000000000001"),
        }
        return 1

    app = FastAPI()
    server_runnable = RunnableLambda(check_types)
    add_routes(
        app,
        server_runnable,
    )

    async with get_async_remote_runnable(
        app,
        raise_app_exceptions=True,
    ) as runnable:
        await runnable.ainvoke(
            {
                "uuid": UUID(int=1),
                "dt": datetime.datetime(2023, 1, 1, 5),
                "date": datetime.date(2023, 1, 1),
                "time": datetime.time(hour=5, minute=30),
                "enum": MySpecialEnum.A,
            }
        )


@pytest.mark.skip(reason="Configuration options not implemented yet")
async def test_all_endpoints_off() -> None:
    """Test toggling endpoints."""
    app = FastAPI()

    # All endpoints disabled
    add_routes(
        app,
        RunnableLambda(lambda foo: "hello"),
        with_batch=False,
        with_invoke=False,
        with_stream=False,
        with_stream_log=False,
        with_config_hash=False,
        with_schemas=False,
        enable_feedback_endpoint=False,
        with_playground=False,
    )

    # All endpoints disabled
    add_routes(
        app,
        RunnableLambda(lambda foo: "hello"),
        with_batch=True,
        with_invoke=True,
        with_stream=True,
        with_stream_log=True,
        with_config_hash=True,
        with_schemas=True,
        enable_feedback_endpoint=True,
        with_playground=True,
        path="/all_on",
    )

    # Config disabled
    add_routes(
        app,
        RunnableLambda(lambda foo: "hello"),
        with_batch=True,
        with_invoke=True,
        with_stream=True,
        with_stream_log=True,
        with_config_hash=False,
        with_schemas=True,
        enable_feedback_endpoint=True,
        with_playground=True,
        path="/config_off",
    )

    endpoints_with_payload = [
        ("POST", "/invoke", {"input": 1}),
        ("POST", "/batch", {"inputs": [1, 2]}),
        ("POST", "/stream", {"input": 1}),
        ("POST", "/stream_log", {"input": 1}),
        ("GET", "/input_schema", {}),
        ("GET", "/output_schema", {}),
        ("GET", "/config_schema", {}),
        ("GET", "/playground/index.html", {}),
        ("HEAD", "/feedback", {}),
        # Check config hashes
        ("POST", "/c/1234/invoke", {"input": 1}),
        ("POST", "/c/1234/batch", {"inputs": [1, 2]}),
        ("POST", "/c/1234/stream", {"input": 1}),
        ("POST", "/c/1234/stream_log", {"input": 1}),
        ("POST", "/c/1234/input_schema", {}),
        ("POST", "/c/1234/output_schema", {}),
        ("POST", "/c/1234/config_schema", {}),
        ("POST", "/c/1234/playground/index.html", {}),
        ("POST", "/c/1234/feedback", {}),
    ]

    # All endpoints disabled
    async with get_async_test_client(app, raise_app_exceptions=False) as async_client:
        for method, endpoint, payload in endpoints_with_payload:
            response = await async_client.request(method, endpoint, json=payload)
            assert response.status_code == 404, f"endpoint {endpoint} should be off"

    # All endpoints enabled
    async with get_async_test_client(app, raise_app_exceptions=False) as async_client:
        for method, endpoint, payload in endpoints_with_payload:
            response = await async_client.request(
                method, "/all_on" + endpoint, json=payload
            )
            # We are only checking that the error code is not 404
            # It may still be 4xx due to incorrect payload etc, but
            # we don't care, we just want to make sure that the endpoint
            # is enabled.
            assert response.status_code != 404, f"endpoint {endpoint} should be on"

    # Config disabled
    async with get_async_test_client(app, raise_app_exceptions=False) as async_client:
        for method, endpoint, payload in endpoints_with_payload:
            if endpoint.startswith("/c/"):
                # Check it's a 404
                response = await async_client.request(
                    method, "/config_off" + endpoint, json=payload
                )
                assert response.status_code == 404, f"endpoint {endpoint} should be off"
            else:
                # Check it's not a 404
                response = await async_client.request(
                    method, "/config_off" + endpoint, json=payload
                )
                assert response.status_code != 404, f"endpoint {endpoint} should be on"
