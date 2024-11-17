import concurrent.futures
from pathlib import Path
import subprocess
import pandas as pd
from src.utils import util


def fetch_dataset():
    repositories: pd.DataFrame = util.get_repository_urls()
    data_dir: Path = util.data_dir()

    def clone_repo(repo_name):
        repository_url = f"https://github.com/{repo_name}.git"
        destination: Path = data_dir.joinpath(repo_name.replace("/", "_"))
        if not destination.exists():
            subprocess.run(["git", "clone", repository_url, destination.as_posix()], check=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(clone_repo, repositories['name'])


if __name__ == "__main__":
    fetch_dataset()
