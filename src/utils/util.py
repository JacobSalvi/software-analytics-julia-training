import json
from pathlib import Path
import shutil
import tarfile
import io
import pandas as pd


def data_dir() -> Path:
    return Path(__file__).parents[2].joinpath('data')

def models_dir() -> Path:
    return Path(__file__).parents[2].joinpath('models')


def repositories_json() -> Path:
    return Path(__file__).parents[2].joinpath('repositories.json')


def get_repository_urls() -> pd.DataFrame:
    repo_json: Path = repositories_json()
    repositories = json.load(repo_json.open())
    return pd.DataFrame(repositories["items"])


def remove_all_files_and_subdirectories_in_folder(folder_path: Path):
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"The folder {folder_path} does not exist or is not a directory.")

    for item in folder.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        elif item.is_file():
            item.unlink()

def compress_file_to_tar_gz(output_filename, file_obj, file_name):
    with tarfile.open(output_filename, "w:gz") as tar:
        tarinfo = tarfile.TarInfo(name=file_name)
        file_obj.seek(0, io.SEEK_END)
        tarinfo.size = file_obj.tell()
        file_obj.seek(0)
        tar.addfile(tarinfo, file_obj)

    return output_filename