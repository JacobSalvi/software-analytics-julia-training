from pathlib import Path
import subprocess
from typing import List, AnyStr

from src.utils import util


def get_repository_urls() -> List[AnyStr]:
    repositories_txt: Path = util.repositories_txt()
    return repositories_txt.read_text().splitlines()


def fetch_dataset():
    repository_urls: List[AnyStr] = get_repository_urls()
    data_dir: Path = util.data_dir()
    for repository_url in repository_urls:
        repository_name = repository_url.split("/")[-1]
        if not data_dir.joinpath(repository_name).exists():
            subprocess.run(["git", "clone", repository_url, data_dir.joinpath(repository_name).as_posix()])
    return


if __name__ == "__main__":
    fetch_dataset()
