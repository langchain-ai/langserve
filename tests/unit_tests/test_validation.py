from typing import List, Optional
from unittest.mock import MagicMock

import pytest
from fastapi import Request
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField

from langserve.api_handler import _unpack_request_config

try:
    from pydantic.v1 import BaseModel, ValidationError
except ImportError:
    from pydantic import BaseModel, ValidationError

from langserve.validation import (
    create_batch_request_model,
    create_invoke_request_model,
)


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "input": {"a": "qqq"},
            "kwargs": {},
            "valid": False,
        },
        {
            "input": {"a": 2},
            "kwargs": "hello",
            "valid": False,
        },
        {
            "input": {"a": 2},
            "config": "hello",
            "valid": False,
        },
        {
            "input": {"b": "hello"},
            "valid": False,
        },
        {
            "input": {"a": 2, "b": "hello"},
            "config": "hello",
            "valid": False,
        },
        {
            "input": {"a": 2, "b": "hello"},
            "valid": True,
        },
        {
            "input": {"a": 2, "b": "hello"},
            "valid": True,
        },
        {
            "input": {"a": 2},
            "valid": True,
        },
    ],
)
def test_create_invoke_and_batch_models(test_case: dict) -> None:
    """Test that the invoke request model is created correctly."""

    class Input(BaseModel):
        """Test input."""

        a: int
        b: Optional[str] = None

    valid = test_case.pop("valid")

    class Config(BaseModel):
        tags: Optional[List[str]] = None

    model = create_invoke_request_model("namespace", Input, Config)

    if valid:
        model(**test_case)
    else:
        with pytest.raises(ValidationError):
            model(**test_case)

    # Validate batch request
    # same structure as input request, but
    # 'input' is a list of inputs and is called 'inputs'
    batch_model = create_batch_request_model("namespace", Input, Config)

    test_case["inputs"] = [test_case.pop("input")]
    if valid:
        batch_model(**test_case)
    else:
        with pytest.raises(ValidationError):
            batch_model(**test_case)


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "type": int,
            "input": 1,
            "valid": True,
        },
        {
            "type": float,
            "input": "name",
            "valid": False,
        },
        {
            "type": float,
            "input": [3.2],
            "valid": False,
        },
        {
            "type": float,
            "input": 1.1,
            "valid": True,
        },
        {
            "type": Optional[float],
            "valid": True,
            "input": None,
        },
    ],
)
def test_validation(test_case) -> None:
    """Test that the invoke request model is created correctly."""

    class Config(BaseModel):
        tags: Optional[List[str]] = None

    model = create_invoke_request_model("namespace", test_case.pop("type"), Config)

    if test_case["valid"]:
        model(**test_case)
    else:
        with pytest.raises(ValidationError):
            model(**test_case)


async def test_invoke_request_with_runnables() -> None:
    """Test that the invoke request model is created correctly."""
    runnable = PromptTemplate.from_template("say hello to {name}").configurable_fields(
        template=ConfigurableField(
            id="template",
            name="Template",
            description="The template to use for the prompt",
        )
    )
    config = runnable.config_schema(include=["tags", "run_name", "configurable"])
    Model = create_invoke_request_model("", runnable.input_schema, config)

    assert (
        await _unpack_request_config(
            Model(
                input={"name": "bob"},
            ).config,
            config_keys=[],
            model=config,
            request=MagicMock(Request),
            per_req_config_modifier=lambda x, y: x,
            server_config=None,
        )
        == {}
    )

    # Test that the config is unpacked correctly
    request = Model(
        input={"name": "bob"},
        config={
            "tags": ["hello"],
            "run_name": "run",
            "configurable": {"template": "goodbye {name}"},
        },
    )
    assert request.input == {"name": "bob"}
    assert request.config.tags == ["hello"]
    assert request.config.run_name == "run"
    assert isinstance(request.config.configurable, BaseModel)
    assert request.config.configurable.dict() == {
        "template": "goodbye {name}",
    }

    assert await _unpack_request_config(
        request.config,
        config_keys=["configurable"],
        model=config,
        request=MagicMock(Request),
        per_req_config_modifier=lambda x, y: x,
        server_config=None,
    ) == {
        "configurable": {"template": "goodbye {name}"},
    }
