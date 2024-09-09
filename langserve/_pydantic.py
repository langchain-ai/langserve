from typing import Any, Dict, Type, cast

from pydantic import BaseModel, ConfigDict, RootModel
from pydantic.json_schema import (
    DEFAULT_REF_TEMPLATE,
    GenerateJsonSchema,
    JsonSchemaMode,
)


def _create_root_model(name: str, type_: Any) -> Type[RootModel]:
    """Create a base class."""

    def schema(
        cls: Type[BaseModel],
        by_alias: bool = True,
        ref_template: str = DEFAULT_REF_TEMPLATE,
    ) -> Dict[str, Any]:
        # Complains about schema not being defined in superclass
        schema_ = super(cls, cls).schema(  # type: ignore[misc]
            by_alias=by_alias, ref_template=ref_template
        )
        schema_["title"] = name
        return schema_

    def model_json_schema(
        cls: Type[BaseModel],
        by_alias: bool = True,
        ref_template: str = DEFAULT_REF_TEMPLATE,
        schema_generator: type[GenerateJsonSchema] = GenerateJsonSchema,
        mode: JsonSchemaMode = "validation",
    ) -> Dict[str, Any]:
        # Complains about model_json_schema not being defined in superclass
        schema_ = super(cls, cls).model_json_schema(  # type: ignore[misc]
            by_alias=by_alias,
            ref_template=ref_template,
            schema_generator=schema_generator,
            mode=mode,
        )
        schema_["title"] = name
        return schema_

    base_class_attributes = {
        "__annotations__": {"root": type_},
        "model_config": ConfigDict(arbitrary_types_allowed=True),
        "schema": classmethod(schema),
        "model_json_schema": classmethod(model_json_schema),
        # Should replace __module__ with caller based on stack frame.
        "__module__": "langserve._pydantic",
    }

    custom_root_type = type(name, (RootModel,), base_class_attributes)
    return cast(Type[RootModel], custom_root_type)
