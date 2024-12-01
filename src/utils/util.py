import json
from pathlib import Path
import shutil
import tarfile
import io
import pandas as pd


def data_dir() -> Path:
    path = Path(__file__).parents[2].joinpath('data')
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    path = Path(__file__).parents[2].joinpath('models')
    path.mkdir(parents=True, exist_ok=True)
    return path


def repositories_json() -> Path:
    path = Path(__file__).parents[2].joinpath('repositories.json')
    return path


def benchmark_prompts() -> Path:
    return data_dir().parent.joinpath('benchmark_prompts.jsonl')


def get_repository_urls() -> pd.DataFrame:
    repo_json: Path = repositories_json()
    repositories = json.load(repo_json.open())
    return pd.DataFrame(repositories["items"])


def get_model_path(model_name: str) -> Path:
    model_paths = {
        "360m": "360m",
        "360m_signature": "360m_signature",
        "360m_baseline": "360m_baseline",
        "135m": "135m",
        "135m_signature": "135m_signature",
        "135m_baseline": "135m_baseline",
        "1.7b": "1-7B",
        "1.7b_signature": "1-7B_signature",
        "1.7b_baseline": "1-7B_baseline",
    }

    if model_name not in model_paths:
        raise ValueError(f"Invalid model name: {model_name}")
    path = models_dir().joinpath(model_paths[model_name])
    path.mkdir(parents=True, exist_ok=True)
    return path


def base_model_types() -> list:
    return ["360m", "135m", "1.7b"]


def all_model_types() -> list:
    return ["360m", "360m_signature", "360m_baseline", "135m", "135m_signature", "135m_baseline", "1.7b",
            "1.7b_signature", "1.7b_baseline"]


def remove_all_files_and_subdirectories_in_folder(folder_path: Path):
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"The folder {folder_path} does not exist or is not a directory.")

    for item in folder.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        elif item.is_file():
            item.unlink()
