import pydantic


def _get_pydantic_version() -> int:
    """Get the pydantic major version."""
    return int(pydantic.__version__.split(".")[0])


# Code is written to support both version 1 and 2
PYDANTIC_MAJOR_VERSION = _get_pydantic_version()
