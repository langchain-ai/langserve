import os
import asyncio
from typing import List, Optional
from pathlib import Path

import base64
import asyncio

from github import Github, Auth
from langserve.packages import list_packages

import subprocess

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
    package: str,
    package_dir: str,
    *,
    repo: str = "langchain-ai/langserve-hub",
    api_path: Optional[str] = None,
) -> None:
    if not repo:
        raise ValueError("Must specify repo")
    repo_path = Path(package)
    subpath = api_path or repo_path.name or repo.split("/")[-1]
    local_dir = Path(package_dir) / subpath
    asyncio.run(_download_github_path(repo_path, local_dir, repo))
    subprocess.run(["poetry", "add", str(local_dir)])


def list(path: str) -> None:
    for package in list_packages(path):
        print(package)
