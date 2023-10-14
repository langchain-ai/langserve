import json
import mimetypes
import os
from string import Template

from fastapi.responses import Response
from langchain.schema.runnable import Runnable


class PlaygroundTemplate(Template):
    delimiter = "____"


async def serve_playground(
    runnable: Runnable, config_keys: list[str], base_url: str, file_path: str
) -> Response:
    local_file_path = os.path.join(
        os.path.dirname(__file__),
        "./playground/dist",
        file_path or "index.html",
    )
    with open(local_file_path) as f:
        html = f.read()
        mime_type = mimetypes.guess_type(local_file_path)[0]
    formatted = PlaygroundTemplate(html).substitute(
        LANGSERVE_BASE_URL=base_url[1:] if base_url.startswith("/") else base_url,
        LANGSERVE_CONFIG_SCHEMA=json.dumps(
            runnable.config_schema(include=config_keys).schema()
        ),
        LANGSERVE_INPUT_SCHEMA=json.dumps(runnable.input_schema.schema()),
    )

    return Response(formatted, media_type=mime_type)
