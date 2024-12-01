import json
from pathlib import Path

import pandas as pd


def data_dir() -> Path:
    return Path(__file__).parents[2].joinpath('data')


def repositories_json() -> Path:
    return Path(__file__).parents[2].joinpath('repositories.json')


def benchmark_prompts() -> Path:
    return data_dir().parent.joinpath('benchmark_prompts.jsonl')


def get_repository_urls() -> pd.DataFrame:
    repo_json: Path = repositories_json()
    repositories = json.load(repo_json.open())
    return pd.DataFrame(repositories["items"])
