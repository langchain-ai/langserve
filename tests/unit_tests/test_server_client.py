"""Test the server and client together."""
import asyncio
import datetime
import json
import uuid
from asyncio import AbstractEventLoop
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from enum import Enum
from itertools import cycle
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Union,
)
from unittest.mock import MagicMock, patch
from uuid import UUID

import httpx
import pytest
import pytest_asyncio
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.testclient import TestClient
from httpx import AsyncClient
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatGenerationChunk, LLMResult
from langchain_core.prompt_values import StringPromptValue
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.runnables import (
    ConfigurableField,
    Runnable,
    RunnableConfig,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.utils import Input, Output
from langchain_core.tracers import RunLog, RunLogPatch
from langsmith import schemas as ls_schemas
from langsmith.client import Client
from langsmith.schemas import FeedbackIngestToken
from pytest import MonkeyPatch
from pytest_mock import MockerFixture
from typing_extensions import Annotated, TypedDict

from langserve import api_handler
from langserve.api_handler import (
    _rename_pydantic_model,
    _replace_non_alphanumeric_with_underscores,
)
from langserve.callbacks import AsyncEventAggregatorCallback
from langserve.client import RemoteRunnable
from langserve.lzstring import LZString
from langserve.schema import CustomUserType
from tests.unit_tests.utils.stubs import AnyStr

try:
    from pydantic.v1 import BaseModel, Field
except ImportError:
    from pydantic import BaseModel, Field
from langserve.server import add_routes
from tests.unit_tests.utils.llms import FakeListLLM, GenericFakeChatModel
from tests.unit_tests.utils.tracer import FakeTracer


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
        x: Union[int, HumanMessage],
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
    assert input_schema["title"] == "add_one_or_passthrough_input"
    #
    output_schema = sync_client.get("/output_schema").json()
    assert isinstance(output_schema, dict)
    assert output_schema["title"] == "add_one_or_passthrough_output"

    output_schema = sync_client.get("/config_schema").json()
    assert isinstance(output_schema, dict)
    assert output_schema["title"] == "add_one_or_passthrough_config"

    # TODO(Team): Fix test. Issue with eventloops right now when using sync client
    # # Test stream
    # response = sync_client.post("/stream", json={"input": 1})
    # assert response.text == "event: data\r\ndata: 2\r\n\r\nevent: end\r\n\r\n"


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
        assert response.status_code == 422

        response = await async_client.post("/stream", json={})
        assert response.status_code == 422

    # test stream_log bad requests
    async with get_async_test_client(app, raise_app_exceptions=True) as async_client:
        response = await async_client.post("/stream_log", data="bad json []")
        assert response.status_code == 422

        response = await async_client.post("/stream_log", json={})
        assert response.status_code == 422


async def test_server_astream_events(app: FastAPI) -> None:
    """Test the server directly via HTTP requests.

    Here we test just astream_events server side without a Remote Client.
    """
    async with get_async_test_client(app, raise_app_exceptions=True) as async_client:
        # Test invoke
        # Test stream
        response = await async_client.post("/stream_events", json={"input": 1})
        # Decode the event stream using plain json de-serialization
        events = _decode_eventstream(response.text)

        for event in events:
            if "data" in event:
                assert "run_id" in event["data"]
                del event["data"]["run_id"]
                assert "metadata" in event["data"]
                del event["data"]["metadata"]

        assert events == [
            {
                "data": {
                    "data": {"input": 1},
                    "event": "on_chain_start",
                    "name": "add_one_or_passthrough",
                    "tags": [],
                },
                "type": "data",
            },
            {
                "data": {
                    "data": {"chunk": 2},
                    "event": "on_chain_stream",
                    "name": "add_one_or_passthrough",
                    "tags": [],
                },
                "type": "data",
            },
            {
                "data": {
                    "data": {"output": 2},
                    "event": "on_chain_end",
                    "name": "add_one_or_passthrough",
                    "tags": [],
                },
                "type": "data",
            },
            {"type": "end"},
        ]

    # test stream_events with bad requests
    async with get_async_test_client(app, raise_app_exceptions=True) as async_client:
        response = await async_client.post("/stream_events", data="bad json []")
        assert response.status_code == 422

        response = await async_client.post("/stream_events", json={})
        assert response.status_code == 422


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
        x: Union[int, HumanMessage],
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


def _get_run_log(run_log_patches: Sequence[RunLogPatch]) -> RunLog:
    """Get run log"""
    run_log = RunLog(state=None)  # type: ignore
    for log_patch in run_log_patches:
        run_log = run_log + log_patch
    return run_log


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
                    "final_output": None,
                    "id": uuid,
                    "logs": {},
                    "streamed_output": [],
                    "type": "chain",
                    "name": "add_one_or_passthrough",
                },
            }
        ],
        [
            {"op": "add", "path": "/streamed_output/-", "value": 2},
            {"op": "replace", "path": "/final_output", "value": 2},
        ],
    ]
    assert _get_run_log(run_logs).state == {
        "final_output": 2,
        "id": uuid,
        "logs": {},
        "streamed_output": [2],
        "type": "chain",
        "name": "add_one_or_passthrough",
    }


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
                        "final_output": None,
                        "id": uuid,
                        "logs": {},
                        "streamed_output": [],
                        "type": "chain",
                        "name": "add_one",
                    },
                }
            ],
            [
                {"op": "add", "path": "/streamed_output/-", "value": 2},
                {"op": "replace", "path": "/final_output", "value": 2},
            ],
        ]

        assert _get_run_log(run_log_patches).state == {
            "final_output": 2,
            "id": uuid,
            "logs": {},
            "streamed_output": [2],
            "type": "chain",
            "name": "add_one",
        }


async def test_streaming_with_errors() -> None:
    from langchain_core.runnables import RunnableGenerator

    async def with_errors(inputs: dict) -> AsyncIterator[int]:
        yield 1
        raise ValueError("Error")
        yield 2

    app = FastAPI()
    add_routes(app, RunnableGenerator(with_errors), path="/with_errors")

    async with get_async_remote_runnable(
        app, path="/with_errors", raise_app_exceptions=False
    ) as runnable:
        chunks = []

        with pytest.raises(httpx.HTTPStatusError) as e:
            async for chunk in runnable.astream(1):
                chunks.append(chunk)

        # Check that first chunk was received
        assert chunks == [1]
        assert e.value.response.status_code == 500


async def test_astream_log_allowlist(event_loop: AbstractEventLoop) -> None:
    """Test async stream with an allowlist."""

    async def add_one(x: int) -> int:
        """Add one to simulate a valid function"""
        return x + 1

    app = FastAPI()
    add_routes(
        app,
        RunnableLambda(add_one).with_config({"run_name": "allowed"}),
        path="/empty_allowlist",
        input_type=int,
        stream_log_name_allow_list=[],
    )
    add_routes(
        app,
        RunnableLambda(add_one).with_config({"run_name": "allowed"}),
        input_type=int,
        path="/allowlist",
        stream_log_name_allow_list=["allowed"],
    )

    # Invoke request
    async with get_async_remote_runnable(app, path="/empty_allowlist/") as runnable:
        run_log_patches = []

        async for chunk in runnable.astream_log(1):
            run_log_patches.append(chunk)

        assert len(run_log_patches) == 0

        run_log_patches = []
        async for chunk in runnable.astream_log(1, include_tags=[]):
            run_log_patches.append(chunk)

        assert len(run_log_patches) == 0

        run_log_patches = []
        async for chunk in runnable.astream_log(1, include_types=[]):
            run_log_patches.append(chunk)

        assert len(run_log_patches) == 0

        run_log_patches = []
        async for chunk in runnable.astream_log(1, include_names=[]):
            run_log_patches.append(chunk)

        assert len(run_log_patches) == 0

    async with get_async_remote_runnable(app, path="/allowlist/") as runnable:
        run_log_patches = []

        async for chunk in runnable.astream_log(1):
            run_log_patches.append(chunk)

        assert len(run_log_patches) > 0

        run_log_patches = []
        async for chunk in runnable.astream_log(1, include_tags=[]):
            run_log_patches.append(chunk)

        assert len(run_log_patches) > 0

        run_log_patches = []
        async for chunk in runnable.astream_log(1, include_types=[]):
            run_log_patches.append(chunk)

        assert len(run_log_patches) > 0

        run_log_patches = []
        async for chunk in runnable.astream_log(1, include_names=[]):
            run_log_patches.append(chunk)

        assert len(run_log_patches) > 0

        run_log_patches = []
        async for chunk in runnable.astream_log(1, include_names=["allowed"]):
            run_log_patches.append(chunk)

        assert len(run_log_patches) > 0


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
            "type": "chain",
            "name": "RunnableSequence",
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
        assert response.json() == {"title": "add_one_input", "type": "integer"}

        response = await async_client.get("/add_two_custom/input_schema")
        assert response.json() == {"title": "add_two_input", "type": "number"}

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
            "title": "add_one_output",
            "type": "integer",
        }

        response = await async_client.get("/add_two_custom/output_schema")
        assert response.json() == {"title": "add_two_output", "type": "number"}

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
            "title": "add_one_config",
            "type": "object",
        }

        response = await async_client.get("/add_two_custom/config_schema")
        assert response.json() == {
            "properties": {
                "tags": {"items": {"type": "string"}, "title": "Tags", "type": "array"}
            },
            "title": "add_two_config",
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
            "title": "passthrough_dict_input",
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


async def test_enforce_trailing_slash_in_client() -> None:
    """Ensure that the client enforces a trailing slash in the URL."""
    r = RemoteRunnable(url="nosuchurl")
    assert r.url == "nosuchurl/"
    r = RemoteRunnable(url="nosuchurl/")
    assert r.url == "nosuchurl/"


async def test_per_request_config_modifier(event_loop: AbstractEventLoop) -> None:
    """Test updating the config based on the raw request object."""

    async def add_one(x: int) -> int:
        """Add one to simulate a valid function"""
        return x + 1

    app = FastAPI()

    async def header_passthru_modifier(
        config: Dict[str, Any], request: Request
    ) -> Dict[str, Any]:
        """Update the config"""
        # Make sure we can access the request body if we need to
        body = await request.json()
        # Hard-codes the expected body just for the test
        # This is tested with just the version
        assert body == {
            "input": 1,
        }

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

    async with get_async_test_client(app) as async_client:
        response = await async_client.post("/add_one/invoke", json={"input": 1})
        assert response.json()["output"] == 2


async def test_per_request_config_modifier_endpoints(
    event_loop: AbstractEventLoop,
) -> None:
    """Verify that per request modifier is only applied for the expected endpoints."""

    # this test verifies that per request modifier is only
    # applied for the expected endpoints
    async def add_one(x: int) -> int:
        """Add one to simulate a valid function"""
        return x + 1

    app = FastAPI()
    server_runnable = RunnableLambda(add_one)

    async def buggy_modifier(
        config: Dict[str, Any], request: Request
    ) -> Dict[str, Any]:
        """Update the config"""
        body = await request.json()  # Make sure we can access the request body always.
        assert isinstance(body, dict)
        raise ValueError("oops I did it again")

    add_routes(
        app,
        server_runnable,
        path="/with_buggy_modifier",
        per_req_config_modifier=buggy_modifier,
    )

    async with get_async_test_client(
        app,
        raise_app_exceptions=False,
    ) as async_client:
        endpoints_to_test = (
            "invoke",
            "batch",
            "stream",
            "stream_log",
            "input_schema",
            "output_schema",
            "config_schema",
            "playground/index.html",
        )

        for endpoint in endpoints_to_test:
            url = "/with_buggy_modifier/" + endpoint

            if endpoint == "batch":
                payload = {"inputs": [1, 2]}
                response = await async_client.post(url, json=payload)
            elif endpoint in {"invoke", "stream", "stream_log"}:
                payload = {"input": 1}
                response = await async_client.post(url, json=payload)
            elif endpoint in {"input_schema", "output_schema", "config_schema"}:
                response = await async_client.get(url)
            elif endpoint == "playground/index.html":
                response = await async_client.get(url)
            else:
                raise ValueError(f"Unknown endpoint {endpoint}")

            if endpoint in {
                "invoke",
                "batch",
                "stream",
                "stream_log",
                "astream_events",
            }:
                assert response.status_code == 500
            else:
                assert response.status_code != 500


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


async def test_endpoint_configurations() -> None:
    """Test enabling/disabling endpoints."""
    app = FastAPI()

    # All endpoints disabled
    add_routes(
        app,
        RunnableLambda(lambda foo: "hello"),
        enabled_endpoints=[],
        enable_feedback_endpoint=False,
    )

    # All endpoints enabled
    add_routes(
        app,
        RunnableLambda(lambda foo: "hello"),
        enabled_endpoints=None,
        enable_feedback_endpoint=True,
        path="/all_on",
    )

    # Config disabled
    add_routes(
        app,
        RunnableLambda(lambda foo: "hello"),
        disabled_endpoints=["config_hashes"],
        enable_feedback_endpoint=True,
        path="/config_off",
    )

    endpoints_with_payload = [
        ("POST", "/invoke", {"input": 1}),
        ("POST", "/batch", {"inputs": [1, 2]}),
        ("POST", "/stream", {"input": 1}),
        ("POST", "/stream_log", {"input": 1}),
        ("POST", "/stream_events", {"input": 1}),
        ("GET", "/input_schema", {}),
        ("GET", "/output_schema", {}),
        ("GET", "/config_schema", {}),
        ("GET", "/playground/index.html", {}),
        ("HEAD", "/feedback", {}),
        ("GET", "/feedback", {}),
        ("POST", "/token_feedback", {}),
        # Check config hashes
        ("POST", "/c/1234/invoke", {"input": 1}),
        ("POST", "/c/1234/batch", {"inputs": [1, 2]}),
        ("POST", "/c/1234/stream", {"input": 1}),
        ("POST", "/c/1234/stream_log", {"input": 1}),
        ("POST", "/c/1234/stream_events", {"input": 1}),
        ("GET", "/c/1234/input_schema", {}),
        ("GET", "/c/1234/output_schema", {}),
        ("GET", "/c/1234/config_schema", {}),
        ("GET", "/c/1234/playground/index.html", {}),
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
            if "feedback" in endpoint:
                # Feedback returns 405 if tracing is disabled
                error_codes = {404}
            else:
                error_codes = {404, 405}
            if response.status_code in error_codes:
                raise AssertionError(
                    f"Endpoint {endpoint} should be on. "
                    f"Test case: ({method}, {endpoint}, {payload}) with {response.text}"
                )

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
                if endpoint == "/feedback":
                    # Feedback returns 405 if tracing is disabled
                    error_codes = {404}
                else:
                    error_codes = {404, 405}
                assert (
                    response.status_code not in error_codes
                ), f"endpoint {endpoint} should be on"

    with pytest.raises(ValueError):
        # Passing "invoke" instead of ["invoke"]
        add_routes(
            app,
            RunnableLambda(lambda foo: "hello"),
            disabled_endpoints="invoke",  # type: ignore
            enable_feedback_endpoint=True,
            path="/config_off",
        )
    with pytest.raises(ValueError):
        # meow is not an endpoint.
        add_routes(
            app,
            RunnableLambda(lambda foo: "hello"),
            disabled_endpoints=["meow"],  # type: ignore
            enable_feedback_endpoint=True,
            path="/config_off",
        )

    with pytest.raises(ValueError):
        # meow is not an endpoint.
        add_routes(
            app,
            RunnableLambda(lambda foo: "hello"),
            enabled_endpoints=["meow"],  # type: ignore
            enable_feedback_endpoint=True,
            path="/config_off",
        )


async def test_astream_events_simple(async_remote_runnable: RemoteRunnable) -> None:
    """Test astream events using a simple chain.

    This test should not involve any complex serialization logic.
    """

    app = FastAPI()

    def add_one(x: int) -> int:
        """Add one to simulate a valid function"""
        return x + 1

    def mul_two(y: int) -> int:
        """Add one to simulate a valid function"""
        return y * 2

    runnable = RunnableLambda(add_one) | RunnableLambda(mul_two)
    add_routes(app, runnable)

    # Invoke request
    async with get_async_remote_runnable(app, raise_app_exceptions=False) as runnable:
        # Test bad requests
        # test client side error
        with pytest.raises(httpx.HTTPStatusError) as cb:
            # Invalid input type (expected string but got int)
            async for _ in runnable.astream_events("foo", version="v1"):
                pass

        # Verify that this is a 422 error
        assert cb.value.response.status_code == 422

        with pytest.raises(httpx.HTTPStatusError) as cb:
            # Invalid input type (expected string but got int)
            # include names should not be a list of lists
            async for _ in runnable.astream_events(1, include_names=[[]], version="v1"):
                pass

        # Verify that this is a 422 error
        assert cb.value.response.status_code == 422

        # Test good requests
        events = []

        async for event in runnable.astream_events(1, version="v1"):
            events.append(event)

        # validate events
        for event in events:
            assert "run_id" in event
            del event["run_id"]
            # Assert that we don't include any "internal" metadata
            # in the events
            for k, v in event["metadata"].items():
                assert not k.startswith("__")
            assert "metadata" in event
            del event["metadata"]

        assert events == [
            {
                "data": {"input": 1},
                "event": "on_chain_start",
                "name": "RunnableSequence",
                "tags": [],
            },
            {
                "data": {},
                "event": "on_chain_start",
                "name": "add_one",
                "tags": ["seq:step:1"],
            },
            {
                "data": {"chunk": 2},
                "event": "on_chain_stream",
                "name": "add_one",
                "tags": ["seq:step:1"],
            },
            {
                "data": {},
                "event": "on_chain_start",
                "name": "mul_two",
                "tags": ["seq:step:2"],
            },
            {
                "data": {"input": 1, "output": 2},
                "event": "on_chain_end",
                "name": "add_one",
                "tags": ["seq:step:1"],
            },
            {
                "data": {"chunk": 4},
                "event": "on_chain_stream",
                "name": "mul_two",
                "tags": ["seq:step:2"],
            },
            {
                "data": {"chunk": 4},
                "event": "on_chain_stream",
                "name": "RunnableSequence",
                "tags": [],
            },
            {
                "data": {"input": 2, "output": 4},
                "event": "on_chain_end",
                "name": "mul_two",
                "tags": ["seq:step:2"],
            },
            {
                "data": {"output": 4},
                "event": "on_chain_end",
                "name": "RunnableSequence",
                "tags": [],
            },
        ]


def _clean_up_events(events: List[Dict[str, Any]]) -> None:
    """Clean up events to make it easy to compare them."""
    for event in events:
        assert "run_id" in event
        del event["run_id"]
        # Assert that we don't include any "internal" metadata
        # in the events
        for k, v in event["metadata"].items():
            assert not k.startswith("__")
        assert "metadata" in event
        del event["metadata"]


async def test_astream_events_with_serialization(
    async_remote_runnable: RemoteRunnable,
) -> None:
    """Test serialization logic in astream events.

    Intermediate steps in the chain may involve arbitrary types.

    Let's check that we can serialize some of the well known types.
    """

    app = FastAPI()

    def to_document(query: str) -> List[Document]:
        """Convert a query to a document"""
        return [
            Document(page_content=query, metadata={"a": "b"}),
            Document(page_content=query[::-1]),
        ]

    def from_document(documents: List[Document]) -> str:
        """Convert a document to a string"""
        return documents[0].page_content

    # This should work since we have built in serializers for Document
    chain = RunnableLambda(to_document) | RunnableLambda(from_document)
    add_routes(app, chain, path="/doc_types")

    # Add a test case for serialization of a dataclass
    # This will be serialized using FastAPI's built in serializer for dataclasses
    # It will not however be decoded properly into a dataclass on the client side
    # since the client side does not have enough information to do so.
    @dataclass
    class Pet:
        name: str
        age: int

    def get_pets(query: str) -> List[Pet]:
        """Get pets"""
        return [
            Pet(name="foo", age=1),
            Pet(name="bar", age=2),
        ]

    # Works because of built-in serializer for dataclass from fast api
    # But it will not deserialize correctly into a dataclass (this is OK)
    add_routes(app, RunnableLambda(get_pets), path="/get_pets")

    class NotSerializable:
        def __init__(self, foo: int) -> None:
            """Create a non-serializable class"""
            self.foo = foo

    def into_non_serializable(query: str) -> List[NotSerializable]:
        """Return non serializable data"""
        return [NotSerializable(foo=1)]

    def back_to_serializable(inputs) -> str:
        """Return non serializable data"""
        return "hello"

    # Works because of built-in serializer for dataclass from fast api
    # But it will not deserialize correctly into a dataclass (this is OK)
    chain = RunnableLambda(into_non_serializable) | RunnableLambda(back_to_serializable)
    add_routes(app, chain, path="/break")

    # Invoke request
    async with get_async_remote_runnable(
        app, raise_app_exceptions=False, path="/doc_types"
    ) as runnable:
        # Test good requests
        events = [event async for event in runnable.astream_events("foo", version="v1")]
        _clean_up_events(events)

        assert events == [
            {
                "data": {"input": "foo"},
                "event": "on_chain_start",
                "name": "/doc_types",
                "tags": [],
            },
            {
                "data": {},
                "event": "on_chain_start",
                "name": "to_document",
                "tags": ["seq:step:1"],
            },
            {
                "data": {
                    "chunk": [
                        Document(page_content="foo", metadata={"a": "b"}),
                        Document(page_content="oof"),
                    ]
                },
                "event": "on_chain_stream",
                "name": "to_document",
                "tags": ["seq:step:1"],
            },
            {
                "data": {},
                "event": "on_chain_start",
                "name": "from_document",
                "tags": ["seq:step:2"],
            },
            {
                "data": {
                    "input": "foo",
                    "output": [
                        Document(page_content="foo", metadata={"a": "b"}),
                        Document(page_content="oof"),
                    ],
                },
                "event": "on_chain_end",
                "name": "to_document",
                "tags": ["seq:step:1"],
            },
            {
                "data": {"chunk": "foo"},
                "event": "on_chain_stream",
                "name": "from_document",
                "tags": ["seq:step:2"],
            },
            {
                "data": {"chunk": "foo"},
                "event": "on_chain_stream",
                "name": "/doc_types",
                "tags": [],
            },
            {
                "data": {
                    "input": [
                        Document(page_content="foo", metadata={"a": "b"}),
                        Document(page_content="oof"),
                    ],
                    "output": "foo",
                },
                "event": "on_chain_end",
                "name": "from_document",
                "tags": ["seq:step:2"],
            },
            {
                "data": {"output": "foo"},
                "event": "on_chain_end",
                "name": "/doc_types",
                "tags": [],
            },
        ]

    async with get_async_remote_runnable(
        app, raise_app_exceptions=False, path="/get_pets"
    ) as runnable:
        # Test good requests
        events = [event async for event in runnable.astream_events("foo", version="v1")]
        _clean_up_events(events)
        assert events == [
            {
                "data": {"input": "foo"},
                "event": "on_chain_start",
                "name": "/get_pets",
                "tags": [],
            },
            {
                "data": {
                    "chunk": [{"age": 1, "name": "foo"}, {"age": 2, "name": "bar"}]
                },
                "event": "on_chain_stream",
                "name": "/get_pets",
                "tags": [],
            },
            {
                "data": {
                    "output": [{"age": 1, "name": "foo"}, {"age": 2, "name": "bar"}]
                },
                "event": "on_chain_end",
                "name": "/get_pets",
                "tags": [],
            },
        ]

    async with get_async_remote_runnable(
        app, raise_app_exceptions=False, path="/break"
    ) as runnable:
        # Test good requests
        with pytest.raises(httpx.HTTPStatusError) as cb:
            async for event in runnable.astream_events("foo", version="v1"):
                pass
        assert cb.value.response.status_code == 500


async def test_astream_events_with_prompt_model_parser_chain(
    async_remote_runnable: RemoteRunnable,
) -> None:
    """Test prompt + model + parser chain"""

    app = FastAPI()

    messages = cycle([AIMessage(content="Hello World!")])

    model = GenericFakeChatModel(messages=messages)

    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a cat."), ("user", "{question}")]
    )

    chain = prompt | model | StrOutputParser()
    add_routes(app, chain)

    async with get_async_remote_runnable(app) as runnable:
        # Test good requests
        events = [
            event
            async for event in runnable.astream_events(
                {"question": "hello"}, version="v1"
            )
        ]
        _clean_up_events(events)
        assert events == [
            {
                "data": {"input": {"question": "hello"}},
                "event": "on_chain_start",
                "name": "RunnableSequence",
                "tags": [],
            },
            {
                "data": {"input": {"question": "hello"}},
                "event": "on_prompt_start",
                "name": "ChatPromptTemplate",
                "tags": ["seq:step:1"],
            },
            {
                "data": {
                    "input": {"question": "hello"},
                    "output": {
                        "messages": [
                            SystemMessage(content="You are a cat."),
                            HumanMessage(content="hello"),
                        ]
                    },
                },
                "event": "on_prompt_end",
                "name": "ChatPromptTemplate",
                "tags": ["seq:step:1"],
            },
            {
                "data": {
                    "input": {
                        "messages": [
                            [
                                SystemMessage(content="You are a cat."),
                                HumanMessage(content="hello"),
                            ]
                        ]
                    }
                },
                "event": "on_chat_model_start",
                "name": "GenericFakeChatModel",
                "tags": ["seq:step:2"],
            },
            {
                "data": {"chunk": AIMessageChunk(content="Hello", id=AnyStr())},
                "event": "on_chat_model_stream",
                "name": "GenericFakeChatModel",
                "tags": ["seq:step:2"],
            },
            {
                "data": {},
                "event": "on_parser_start",
                "name": "StrOutputParser",
                "tags": ["seq:step:3"],
            },
            {
                "data": {"chunk": "Hello"},
                "event": "on_parser_stream",
                "name": "StrOutputParser",
                "tags": ["seq:step:3"],
            },
            {
                "data": {"chunk": "Hello"},
                "event": "on_chain_stream",
                "name": "RunnableSequence",
                "tags": [],
            },
            {
                "data": {"chunk": AIMessageChunk(content=" ", id=AnyStr())},
                "event": "on_chat_model_stream",
                "name": "GenericFakeChatModel",
                "tags": ["seq:step:2"],
            },
            {
                "data": {"chunk": " "},
                "event": "on_parser_stream",
                "name": "StrOutputParser",
                "tags": ["seq:step:3"],
            },
            {
                "data": {"chunk": " "},
                "event": "on_chain_stream",
                "name": "RunnableSequence",
                "tags": [],
            },
            {
                "data": {"chunk": AIMessageChunk(content="World!", id=AnyStr())},
                "event": "on_chat_model_stream",
                "name": "GenericFakeChatModel",
                "tags": ["seq:step:2"],
            },
            {
                "data": {"chunk": "World!"},
                "event": "on_parser_stream",
                "name": "StrOutputParser",
                "tags": ["seq:step:3"],
            },
            {
                "data": {"chunk": "World!"},
                "event": "on_chain_stream",
                "name": "RunnableSequence",
                "tags": [],
            },
            {
                "data": {
                    "input": {
                        "messages": [
                            [
                                SystemMessage(content="You are a cat."),
                                HumanMessage(content="hello"),
                            ]
                        ]
                    },
                    "output": LLMResult(
                        generations=[
                            [
                                ChatGenerationChunk(
                                    text="Hello World!",
                                    message=AIMessageChunk(
                                        content="Hello World!", id=AnyStr()
                                    ),
                                )
                            ]
                        ],
                        llm_output=None,
                        run=None,
                    ),
                },
                "event": "on_chat_model_end",
                "name": "GenericFakeChatModel",
                "tags": ["seq:step:2"],
            },
            {
                "data": {
                    "input": AIMessageChunk(content="Hello World!", id=AnyStr()),
                    "output": "Hello World!",
                },
                "event": "on_parser_end",
                "name": "StrOutputParser",
                "tags": ["seq:step:3"],
            },
            {
                "data": {"output": "Hello World!"},
                "event": "on_chain_end",
                "name": "RunnableSequence",
                "tags": [],
            },
        ]


async def test_path_dependencies() -> None:
    """Test path dependencies."""

    def add_one(x: int) -> int:
        """Add one to simulate a valid function"""
        return x + 1

    async def verify_token(x_token: Annotated[str, Header()]) -> None:
        """Verify the token is valid."""
        # Replace this with your actual authentication logic
        if x_token != "secret-token":
            raise HTTPException(status_code=400, detail="X-Token header invalid")

    app = FastAPI()

    add_routes(
        app,
        RunnableLambda(add_one),
        dependencies=[Depends(verify_token)],
        enable_feedback_endpoint=True,
    )

    endpoints_with_payload = [
        ("POST", "/invoke", {"input": 1}),
        ("POST", "/batch", {"inputs": [1, 2]}),
        ("POST", "/stream", {"input": 1}),
        ("POST", "/stream_log", {"input": 1}),
        ("POST", "/stream_events", {"input": 1}),
        ("GET", "/input_schema", {}),
        ("GET", "/output_schema", {}),
        ("GET", "/config_schema", {}),
        ("GET", "/playground/index.html", {}),
        # ("HEAD", "/feedback", {}),
        # ("GET", "/feedback", {}),
        # Check config hashes
        ("POST", "/c/1234/invoke", {"input": 1}),
        ("POST", "/c/1234/batch", {"inputs": [1, 2]}),
        ("POST", "/c/1234/stream", {"input": 1}),
        ("POST", "/c/1234/stream_log", {"input": 1}),
        ("POST", "/c/1234/stream_events", {"input": 1}),
        ("GET", "/c/1234/input_schema", {}),
        ("GET", "/c/1234/output_schema", {}),
        ("GET", "/c/1234/config_schema", {}),
        ("GET", "/c/1234/playground/index.html", {}),
    ]

    async with get_async_test_client(app, raise_app_exceptions=False) as async_client:
        for method, endpoint, payload in endpoints_with_payload:
            response = await async_client.request(method, endpoint, json=payload)
            # Missing required header
            assert response.status_code == 422, (
                f"Should fail on {endpoint} since we are missing the header. "
                f"Test case: ({method}, {endpoint}, {payload}) with {response.text}"
            )

            response = await async_client.request(
                method, endpoint, json=payload, headers={"X-Token": "secret-token"}
            )
            assert response.status_code not in {404, 405, 422}, (
                f"Failed test case: ({method}, {endpoint}, {payload}) "
                f"with {response.text}. "
                f"Should not return 422 status code since we are passing the header."
            )


async def test_remote_configurable_remote_runnable() -> None:
    """Test that a configurable a client runnable that's configurable works.

    Here, we wrap the client runnable in a RunnableWithMessageHistory.

    The test verifies that the extra information populated by RunnableWithMessageHistory
    does not interfere with the serialization logic.
    """
    app = FastAPI()

    class InMemoryHistory(BaseChatMessageHistory, BaseModel):
        """In memory implementation of chat message history."""

        messages: List[BaseMessage] = Field(default_factory=list)

        def add_message(self, message: BaseMessage) -> None:
            """Add a self-created message to the store"""
            self.messages.append(message)

        def clear(self) -> None:
            self.messages = []

    # Here we use a global variable to store the chat message history.
    # This will make it easier to inspect it to see the underlying results.
    store = {}

    def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryHistory()
        return store[session_id]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You're an assistant who's good at {ability}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    model = GenericFakeChatModel(messages=cycle([AIMessage(content="Hello World!")]))
    chain = prompt | model

    add_routes(app, chain)

    # Invoke request
    async with get_async_remote_runnable(app, raise_app_exceptions=False) as client:
        chain_with_history = RunnableWithMessageHistory(
            client,
            # Uses the get_by_session_id function defined in the example
            # above.
            get_by_session_id,
            input_messages_key="question",
            history_messages_key="history",
        )
        result = await chain_with_history.ainvoke(
            {"question": "hi"}, {"configurable": {"session_id": "1"}}
        )
        assert result == AIMessage(content="Hello World!", id=AnyStr())
        assert store == {
            "1": InMemoryHistory(
                messages=[
                    HumanMessage(content="hi"),
                    AIMessage(content="Hello World!", id=AnyStr()),
                ]
            )
        }


@asynccontextmanager
async def get_langsmith_client() -> AsyncIterator[MagicMock]:
    """Get a patched langsmith client."""
    with patch("langserve.api_handler.ls_client") as mocked_ls_client_package:
        with patch("langserve.api_handler.tracing_is_enabled") as tracing_is_enabled:
            tracing_is_enabled.return_value = True
            mocked_client = MagicMock(auto_spec=Client)
            mocked_ls_client_package.Client.return_value = mocked_client
            yield mocked_client


async def test_token_feedback_included_in_responses() -> None:
    """Test that information to leave scoped feedback is passed to the client
    is present in the server response.
    """
    feedback_id = uuid.UUID(int=1)
    async with get_langsmith_client() as mocked_client:
        mocked_client.create_presigned_feedback_token.return_value = (
            FeedbackIngestToken(
                id=feedback_id,
                url="feedback_id",
                expires_at=datetime.datetime(2023, 1, 1),
            )
        )

        local_app = FastAPI()
        add_routes(
            local_app,
            RunnableLambda(lambda foo: "hello"),
            enable_feedback_endpoint=True,
            token_feedback_config={
                "key_configs": [
                    {
                        "key": "foo",
                    }
                ]
            },
        )

        async with get_async_test_client(
            local_app, raise_app_exceptions=True
        ) as async_client:
            response = await async_client.post(
                "/invoke",
                json={"input": "hello"},
            )

            json_response = response.json()
            run_id = json_response["metadata"]["run_id"]
            assert json_response == {
                "metadata": {
                    "feedback_tokens": [
                        {
                            "expires_at": "2023-01-01T00:00:00",
                            "key": "foo",
                            "token_url": "feedback_id",
                        }
                    ],
                    "run_id": run_id,
                },
                "output": "hello",
            }

            response = await async_client.post(
                "/batch",
                json={
                    "inputs": ["hello", "world"],
                },
            )
            json_response = response.json()
            responses = json_response["metadata"]["responses"]
            run_ids = [response["run_id"] for response in responses]
            assert run_ids == json_response["metadata"]["run_ids"]

            for r in responses:
                del r["run_id"]

            assert json_response == {
                "metadata": {
                    "responses": [
                        {
                            "feedback_tokens": [
                                {
                                    "expires_at": "2023-01-01T00:00:00",
                                    "key": "foo",
                                    "token_url": "feedback_id",
                                }
                            ]
                        },
                        {
                            "feedback_tokens": [
                                {
                                    "expires_at": "2023-01-01T00:00:00",
                                    "key": "foo",
                                    "token_url": "feedback_id",
                                }
                            ]
                        },
                    ],
                    "run_ids": run_ids,
                },
                "output": ["hello", "hello"],
            }

            # Test stream
            response = await async_client.post(
                "/stream",
                json={"input": "hello"},
            )
            events = _decode_eventstream(response.text)
            del events[0]["data"]["run_id"]
            assert events == [
                {
                    "data": {
                        "feedback_tokens": [
                            {
                                "expires_at": "2023-01-01T00:00:00",
                                "key": "foo",
                                "token_url": "feedback_id",
                            }
                        ]
                    },
                    "type": "metadata",
                },
                {"data": "hello", "type": "data"},
                {"type": "end"},
            ]

            # Test astream events
            response = await async_client.post(
                "/stream_events",
                json={"input": "hello"},
            )
            events = _decode_eventstream(response.text)
            for event in events:
                if "data" in event and "run_id" in event["data"]:
                    del event["data"]["run_id"]

            # Find the metadata event and pull it out
            metadata_event = None
            for event in events:
                if event["type"] == "metadata":
                    metadata_event = event

            assert metadata_event == {
                "data": {
                    "feedback_tokens": [
                        {
                            "expires_at": "2023-01-01T00:00:00",
                            "key": "foo",
                            "token_url": "feedback_id",
                        }
                    ]
                },
                "type": "metadata",
            }

            # Test astream log
            response = await async_client.post(
                "/stream_log",
                json={"input": "hello"},
            )
            events = _decode_eventstream(response.text)
            for event in events:
                if "data" in event and "run_id" in event["data"]:
                    del event["data"]["run_id"]

            # Find the metadata event and pull it out
            metadata_event = None
            for event in events:
                if event["type"] == "metadata":
                    metadata_event = event

            assert metadata_event == {
                "data": {
                    "feedback_tokens": [
                        {
                            "expires_at": "2023-01-01T00:00:00",
                            "key": "foo",
                            "token_url": "feedback_id",
                        }
                    ]
                },
                "type": "metadata",
            }


async def test_passing_run_id_from_client() -> None:
    """test that the client can set a run id if server allows it."""
    local_app = FastAPI()
    add_routes(
        local_app,
        RunnableLambda(lambda foo: "hello"),
        config_keys=["run_id"],
    )

    run_id = uuid.UUID(int=9)
    run_id2 = uuid.UUID(int=14)

    async with get_async_test_client(
        local_app, raise_app_exceptions=True
    ) as async_client:
        response = await async_client.post(
            "/invoke",
            json={"input": "hello", "config": {"run_id": str(run_id)}},
        )
        response.raise_for_status()
        json_response = response.json()
        assert json_response["metadata"]["run_id"] == str(run_id)

        ## Test batch
        response = await async_client.post(
            "/batch",
            json={
                "inputs": ["hello", "world"],
                "config": [{"run_id": str(run_id)}, {"run_id": str(run_id2)}],
            },
        )
        json_response = response.json()
        responses = json_response["metadata"]["responses"]
        run_ids = [response["run_id"] for response in responses]
        assert run_ids == [str(run_id), str(run_id2)]

        # Test stream
        response = await async_client.post(
            "/stream",
            json={"input": "hello", "config": {"run_id": str(run_id)}},
        )
        events = _decode_eventstream(response.text)
        assert events[0]["data"]["run_id"] == str(run_id)

        # Test stream events
        response = await async_client.post(
            "/stream_events",
            json={"input": "hello", "config": {"run_id": str(run_id)}},
        )
        events = _decode_eventstream(response.text)
        assert events[0]["data"]["run_id"] == str(run_id)


async def test_passing_bad_runnable_to_add_routes() -> None:
    """test passing a bad type."""
    with pytest.raises(TypeError) as e:
        add_routes(FastAPI(), "not a runnable")

    assert e.match("Expected a Runnable, got <class 'str'>")


async def test_token_feedback_endpoint() -> None:
    """Tests that the feedback endpoint can accept feedback to langsmith."""
    async with get_langsmith_client() as client:
        local_app = FastAPI()
        add_routes(
            local_app,
            RunnableLambda(lambda foo: "hello"),
            token_feedback_config={
                "key_configs": [
                    {
                        "key": "silliness",
                    }
                ]
            },
        )

        async with get_async_test_client(
            local_app, raise_app_exceptions=True
        ) as async_client:
            response = await async_client.post(
                "/token_feedback", json={"token_or_url": "some_url", "score": 3}
            )
            assert response.status_code == 200

            call = client.create_feedback_from_token.call_args
            assert call.args[0] == "some_url"
            assert call.kwargs == {
                "comment": None,
                "metadata": {"from_langserve": True},
                "score": 3,
                "value": None,
            }
