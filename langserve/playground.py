import json
import mimetypes
import os
from string import Template
from typing import Sequence, Type

from fastapi.responses import Response
from langchain.schema.runnable import Runnable

from langserve.pydantic_v1 import BaseModel


class PlaygroundTemplate(Template):
    delimiter = "____"


async def serve_playground(
    runnable: Runnable,
    input_schema: Type[BaseModel],
    config_keys: Sequence[str],
    base_url: str,
    file_path: str,
) -> Response:
    """Serve the playground."""
    local_file_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "./playground/dist",
            file_path or "index.html",
        )
    )

    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "./playground/dist")
    )

    if base_dir != os.path.commonpath((base_dir, local_file_path)):
        return Response("Not Found", status_code=404)

    try:
        with open(local_file_path, encoding="utf-8") as f:
            mime_type = mimetypes.guess_type(local_file_path)[0]
            if mime_type in ("text/html", "text/css", "application/javascript"):
                response = PlaygroundTemplate(f.read()).substitute(
                    LANGSERVE_BASE_URL=base_url[1:]
                    if base_url.startswith("/")
                    else base_url,
                    LANGSERVE_CONFIG_SCHEMA=json.dumps(
                        runnable.config_schema(include=config_keys).schema()
                    ),
                    LANGSERVE_INPUT_SCHEMA=json.dumps(input_schema.schema()),
                )
            else:
                response = f.buffer.read()
    except FileNotFoundError:
        return Response("Not Found", status_code=404)

    return Response(response, media_type=mime_type)
