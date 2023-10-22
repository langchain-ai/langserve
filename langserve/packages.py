from pathlib import Path
from typing import Union, Generator, TypedDict
from fastapi import FastAPI, APIRouter
from langserve.server import add_routes
import importlib
import logging

from pathlib import Path
from tomli import load


class LangServeExport(TypedDict):
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


exclude_paths = set(["__pycache__", ".venv", ".git", ".github"])


def _include_path(path: Path) -> bool:
    """
    Skip paths that are in exclude_paths or start with an underscore.
    """
    for part in path.parts:
        if part in exclude_paths:
            return False
        if part.startswith("_"):
            return False
    return True


def list_packages(path: str = "../packages") -> Generator[Path, None, None]:
    # traverse path for routes to host (any directory holding a pyproject.toml file)
    package_root = Path(path)
    for pyproject_path in package_root.glob("**/pyproject.toml"):
        if not _include_path(pyproject_path):
            continue
        yield pyproject_path.parent


def add_package_route(
    app: Union[FastAPI, APIRouter], package_path: Path, mount_path: str
) -> None:
    pyproject_path = package_path / "pyproject.toml"

    # get langserve export
    package = get_langserve_export(pyproject_path)
    package_name = package["package_name"]
    try:
        # import module
        mod = importlib.import_module(package["module"])
    except KeyError as e:
        logging.warning(f"Error: {e}")
        logging.warning(f"Try editing {pyproject_path}")
        return
    except ModuleNotFoundError as e:
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
