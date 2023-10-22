import json
import mimetypes
import os
from string import Template
from typing import List, Type

from fastapi.responses import Response
from langchain.schema.runnable import Runnable

try:
    from pydantic.v1 import BaseModel
except ImportError:
    from pydantic import BaseModel


class PlaygroundTemplate(Template):
    delimiter = "____"


async def serve_playground(
    runnable: Runnable,
    input_schema: Type[BaseModel],
    config_keys: List[str],
    base_url: str,
    file_path: str,
) -> Response:
    local_file_path = os.path.join(
        os.path.dirname(__file__),
        "./playground/dist",
        file_path or "index.html",
    )
    with open(local_file_path) as f:
        mime_type = mimetypes.guess_type(local_file_path)[0]
        if mime_type in ("text/html", "text/css", "application/javascript"):
            res = PlaygroundTemplate(f.read()).substitute(
                LANGSERVE_BASE_URL=base_url[1:]
                if base_url.startswith("/")
                else base_url,
                LANGSERVE_CONFIG_SCHEMA=json.dumps(
                    runnable.config_schema(include=config_keys).schema()
                ),
                LANGSERVE_INPUT_SCHEMA=json.dumps(input_schema.schema()),
            )
        else:
            res = f.buffer.read()

    return Response(res, media_type=mime_type)
