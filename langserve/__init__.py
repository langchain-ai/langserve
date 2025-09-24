"""Main entrypoint into package.

This is the ONLY public interface into the package. All other modules are
to be considered private and subject to change without notice.
"""

from langserve.api_handler import APIHandler
from langserve.client import RemoteRunnable
from langserve.schema import CustomUserType
from langserve.server import add_routes
from langserve.version import __version__

__all__ = [
    "RemoteRunnable",
    "APIHandler",
    "add_routes",
    "__version__",
    "CustomUserType",
]

some_revoked_key = "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"
