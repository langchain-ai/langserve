import os
import asyncio
from typing import List, Optional
from pathlib import Path

import base64
from tqdm.asyncio import tqdm_asyncio
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
            f.write(base64.b64decode(content.content))

    print(f"Downloading files for {path}")
    await tqdm_asyncio.gather(*[_download_file(path) for path in filelist])

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
    subprocess.run(["poetry", "add", "--editable", str(local_dir)])


def list(path: str) -> None:
    for package in list_packages(path):
        print(package)
