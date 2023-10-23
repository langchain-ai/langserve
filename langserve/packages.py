import importlib
import logging
from pathlib import Path
from typing import Generator, TypedDict, Union

from fastapi import APIRouter, FastAPI
from tomli import load

from langserve.server import add_routes


class LangServeExport(TypedDict):
    """
    Fields from pyproject.toml that are relevant to LangServe

    Attributes:
        module: The module to import from, tool.langserve.export_module
        attr: The attribute to import from the module, tool.langserve.export_attr
        package_name: The name of the package, tool.poetry.name
    """

    module: str
    attr: str
    package_name: str


def get_langserve_export(filepath: Path) -> LangServeExport:
    with open(filepath, "rb") as f:
        data = load(f)
    try:
        module = data["tool"]["langserve"]["export_module"]
        attr = data["tool"]["langserve"]["export_attr"]
        package_name = data["tool"]["poetry"]["name"]
    except KeyError as e:
        raise KeyError("Invalid LangServe PyProject.toml") from e
    return LangServeExport(module=module, attr=attr, package_name=package_name)


EXCLUDE_PATHS = set(["__pycache__", ".venv", ".git", ".github"])


def _include_path(path: Path) -> bool:
    """
    Skip paths that are in EXCLUDE_PATHS or start with an underscore.
    """
    for part in path.parts:
        if part in EXCLUDE_PATHS:
            return False
        if part.startswith("_"):
            return False
    return True


def list_packages(path: str = "../packages") -> Generator[Path, None, None]:
    """
    Yields Path objects for each folder that contains a pyproject.toml file within a
    path. Use this to find packages to add to the server.

    See `add_package_routes` below for an example of how to use this.
    """
    # traverse path for routes to host (any directory holding a pyproject.toml file)
    package_root = Path(path)
    for pyproject_path in package_root.glob("**/pyproject.toml"):
        if not _include_path(pyproject_path):
            continue
        yield pyproject_path.parent


def add_package_route(
    app: Union[FastAPI, APIRouter], package_path: Path, mount_path: str
) -> None:
    try:
        pyproject_path = package_path / "pyproject.toml"

        # get langserve export
        package = get_langserve_export(pyproject_path)
        package_name = package["package_name"]
        # import module
        mod = importlib.import_module(package["module"])
    except KeyError:
        logging.warning(
            f"Skipping {package_path} because it is not a valid LangServe "
            "package (see pyproject.toml)"
        )
        return
    except ImportError as e:
        logging.warning(f"Error: {e}")
        logging.warning(f"Try fixing with `poetry add --editable {package_path}`")
        logging.warning(
            "To remove packages, use `poe` instead of `poetry`: "
            f"`poe remove {package_name}`"
        )
        return
    # get attr
    chain = getattr(mod, package["attr"])
    add_routes(app, chain, path=mount_path)


def add_package_routes(app: Union[FastAPI, APIRouter], path: str = "packages") -> None:
    # traverse path for routes to host (any directory holding a pyproject.toml file)
    for package_path in list_packages(path):
        mount_path_relative = package_path.relative_to(Path(path))
        mount_path = f"/{mount_path_relative}"
        add_package_route(app, package_path, mount_path)
