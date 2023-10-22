import os
import asyncio
from typing import List, Optional
from pathlib import Path

import base64
from tqdm.asyncio import tqdm_asyncio
import asyncio

from github import Github, Auth
from langserve.packages import list_packages, PyProject

import subprocess
import shutil


async def _download_github_path(path: Path, local_dest: Path, repo_handle: str) -> None:
    token = os.environ.get("GITHUB_PAT")
    if token:
        g = Github(auth=Auth.Token(token=token))
    else:
        g = Github()
    repo = g.get_repo(repo_handle)
    # recursively download all files in path

    base_path = Path(path)
    local_base_path = Path(local_dest)
    try:
        local_base_path.mkdir(exist_ok=False, parents=True)
    except FileExistsError:
        print(FileExistsError(f"Error: Directory {local_base_path} already exists"))
        return

    # try loading pyproject.toml
    pyproject_path = base_path / "pyproject.toml"
    pyproject_contents = repo.get_contents(str(pyproject_path))
    if isinstance(pyproject_contents, List):
        raise ValueError(
            f"Error: {pyproject_path} is not a file, so it "
            "is not a valid langserve package"
        )
    pyproject = PyProject.loads(pyproject_contents.decoded_content.decode("utf-8"))

    # check if langserve package
    if not pyproject.is_langserve():
        raise ValueError(f"Error: {path} is not a langserve package")

    async def _get_filelist(subdir: Path = Path("")) -> List[str]:
        github_path = base_path / subdir
        local_path = local_base_path / subdir
        contents = repo.get_contents(str(github_path))

        if isinstance(contents, List):
            # is a folder, mkdir and iterate
            local_path.mkdir(exist_ok=True)
            innerlist = [content.path for content in contents if content.type == "file"]
            subfiles = await asyncio.gather(
                *[
                    _get_filelist(subdir / content.name)
                    for content in contents
                    if content.type == "dir"
                ]
            )
            return innerlist + [i for sublist in subfiles for i in sublist]
        else:
            # this should never happen
            # TODO(erick): handle this gracefully - throw for now to see
            raise ValueError(f"Error: {github_path} is not a directory")

    filelist = await _get_filelist()

    # tqdm parallel download all in filelist
    async def _download_file(repo_subpath: str) -> None:
        github_path = Path(repo_subpath)
        local_path = local_base_path / github_path.relative_to(base_path)
        with open(local_path, "wb") as f:
            content = repo.get_contents(str(github_path))
            if isinstance(content, List):
                raise ValueError(f"Error: {github_path} is not a file")
            f.write(content.decoded_content)

    print(f"Downloading files for {path}")
    await tqdm_asyncio.gather(*[_download_file(path) for path in filelist])

    print(f"Successfully downloaded {path} to {local_dest}")


def download(
    package: str,
    *,
    package_dir: str,
    repo: str = "langchain-ai/langserve-hub",
    api_path: Optional[str] = None,
) -> None:
    if not repo:
        raise ValueError("Must specify repo")
    repo_path = Path(package)
    subpath = api_path.strip("/") if api_path else repo_path.name or repo.split("/")[-1]
    local_dir = Path(package_dir) / subpath
    asyncio.run(_download_github_path(repo_path, local_dir, repo))
    subprocess.run(["poetry", "add", "--editable", str(local_dir)])


def list(package_dir: str) -> None:
    for package in list_packages(package_dir):
        print(package)


def remove(
    path: str,
    *,
    package_dir: str,
) -> None:
    # check if path exists
    package_root = Path(package_dir)
    package_path = package_root / path.strip("/")
    if not package_path.exists():
        raise ValueError(f"Error: {package_path} does not exist")

    # get package name
    pyproject_path = package_path / "pyproject.toml"
    pyproject = PyProject.load(pyproject_path)
    package_name = pyproject.package_name

    # poetry remove package
    subprocess.run(["poetry", "remove", package_name])

    # remove package directory
    shutil.rmtree(package_path)