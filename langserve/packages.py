from pathlib import Path
from typing import Union, Generator
from fastapi import FastAPI, APIRouter
from tomllib import load as load_toml
from tomllib import loads as loads_toml
from langserve.server import add_routes


# todo: make this a function instead (this is from old cli)
class PyProject:
    def __init__(self, data: dict, path: Path):
        self.data = data
        self.path = path

    @classmethod
    def load(cls, path: Path) -> "PyProject":
        with open(path, "rb") as f:
            data = load_toml(f)
        return cls(data, path)

    # shouldn't need this anymore?
    # @classmethod
    # def loads(cls, data: str) -> "PyProject":
    #     d = loads_toml(data)
    #     return cls(d, None)

    def is_langserve(self) -> bool:
        return "langserve" in self.data["tool"]

    def get_langserve_export(self) -> tuple[str, str]:
        module = self.data["tool"].get("langserve", {}).get("export_module")
        if module is None:
            print(self.data)
            raise ValueError(
                "No module name was exported at `tool.langserve.export_module`"
            )
        attr = self.data["tool"].get("langserve", {}).get("export_attr")
        if attr is None:
            raise ValueError(
                "No attr name was exported at `tool.langserve.export_attr`"
            )
        return module, attr


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


def add_package_routes(
    app: Union[FastAPI, APIRouter], path: str = "../packages"
) -> None:
    # traverse path for routes to host (any directory holding a pyproject.toml file)
    for package_path in list_packages(path):
        pyproject_path = package_path / "pyproject.toml"
        # load pyproject.toml
        pyproject = PyProject.load(pyproject_path)
        # get langserve export
        module, attr = pyproject.get_langserve_export()

        # import module
        mod = __import__(module)
        # get attr
        chain = getattr(mod, attr)
        # add route
        add_routes(app, chain)
