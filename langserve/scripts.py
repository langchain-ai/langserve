import os
import typer
import asyncio
from typing import Annotated, Optional, List
from pathlib import Path

from tomllib import load as load_toml
from tomllib import loads as loads_toml
from tomli_w import dump as dump_toml
from urllib import request

import subprocess

import base64
import asyncio

from github import Github, Auth
from langserve.packages import list_packages

g = Github(auth=Auth.Token(token=os.environ["GITHUB_PAT"]))


async def _download_github_path(path: Path, local_dest: Path, repo_handle: str) -> None:
    repo = g.get_repo(repo_handle)
    # recursively download all files in path

    base_path = Path(path)
    local_base_path = Path(local_dest)
    try:
        local_base_path.mkdir(exist_ok=False)
    except FileExistsError:
        print(FileExistsError(f"Error: Directory {local_base_path} already exists"))
        return

    # todo: split into collecting directories
    # and then tqdm downloading files
    async def _download_github_path(subdir: Path = Path("")) -> None:
        github_path = base_path / subdir
        local_path = local_base_path / subdir
        contents = repo.get_contents(str(github_path))

        if isinstance(contents, List):
            # is a folder, mkdir and iterate
            local_path.mkdir(exist_ok=True)
            # todo: parallelize
            await asyncio.gather(
                *[_download_github_path(subdir / content.name) for content in contents]
            )
        else:
            # is a file, save
            print(f"Downloading {github_path} to {local_path}")
            with open(local_path, "wb") as f:
                content = contents.content
                f.write(base64.b64decode(content))

    await _download_github_path()

    print(f"Successfully downloaded {path} to {local_dest}")


def download(
    path: str, package_dir: str, repo: str = "langchain-ai/langserve-hub"
) -> None:
    if not repo:
        raise ValueError("Must specify repo")
    repo_path = Path(path)
    name = repo_path.name or repo.split("/")[-1]
    local_dir = Path(package_dir) / name
    asyncio.run(_download_github_path(repo_path, local_dir, repo))


def list(path: str) -> None:
    for package in list_packages(path):
        print(package)
