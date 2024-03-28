import json
import mimetypes
import os
from string import Template
from typing import Literal, Sequence, Type

from fastapi.responses import Response
from langchain_core.runnables import Runnable

from langserve.pydantic_v1 import BaseModel


class PlaygroundTemplate(Template):
    delimiter = "____"


def _get_mimetype(path: str) -> str:
    """Get mimetype for file.

    Custom implementation of mimetypes.guess_type that
    uses the file extension to determine the mimetype for some files.

    This is necessary due to: https://bugs.python.org/issue43975
    Resolves issue: https://github.com/langchain-ai/langserve/issues/245

    Args:
        path (str): Path to file

    Returns:
        str: Mimetype of file
    """
    try:
        file_extension = path.lower().split(".")[-1]
    except IndexError:
        return mimetypes.guess_type(path)[0]

    if file_extension == "js":
        return "application/javascript"
    elif file_extension == "css":
        return "text/css"
    elif file_extension in ["htm", "html"]:
        return "text/html"

    # If the file extension is not one of the specified ones,
    # use the default guess method
    mime_type = mimetypes.guess_type(path)[0]
    return mime_type


async def serve_playground(
    runnable: Runnable,
    input_schema: Type[BaseModel],
    output_schema: Type[BaseModel],
    config_keys: Sequence[str],
    base_url: str,
    file_path: str,
    feedback_enabled: bool,
    public_trace_link_enabled: bool,
    playground_type: Literal["default", "chat"],
) -> Response:
    """Serve the playground."""
    if playground_type == "default":
        path_to_dist = "./playground/dist"
    elif playground_type == "chat":
        path_to_dist = "./chat_playground/dist"
    else:
        raise ValueError(
            f"Invalid playground type: {playground_type}. "
            f"Use one of 'default' or 'chat'."
        )

    local_file_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            path_to_dist,
            file_path or "index.html",
        )
    )

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), path_to_dist))

    if base_dir != os.path.commonpath((base_dir, local_file_path)):
        return Response("Not Found", status_code=404)
    try:
        with open(local_file_path, encoding="utf-8") as f:
            mime_type = _get_mimetype(local_file_path)
            if mime_type in ("text/html", "text/css", "application/javascript"):
                response = PlaygroundTemplate(f.read()).substitute(
                    LANGSERVE_BASE_URL=base_url[1:]
                    if base_url.startswith("/")
                    else base_url,
                    LANGSERVE_CONFIG_SCHEMA=json.dumps(
                        runnable.config_schema(include=config_keys).schema()
                    ),
                    LANGSERVE_INPUT_SCHEMA=json.dumps(input_schema.schema()),
                    LANGSERVE_OUTPUT_SCHEMA=json.dumps(output_schema.schema()),
                    LANGSERVE_FEEDBACK_ENABLED=json.dumps(
                        "true" if feedback_enabled else "false"
                    ),
                    LANGSERVE_PUBLIC_TRACE_LINK_ENABLED=json.dumps(
                        "true" if public_trace_link_enabled else "false"
                    ),
                )
            else:
                response = f.buffer.read()
    except FileNotFoundError:
        return Response("Not Found", status_code=404)

    return Response(response, media_type=mime_type)
