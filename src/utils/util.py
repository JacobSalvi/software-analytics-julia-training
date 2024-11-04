from pathlib import Path


def data_dir() -> Path:
    return Path(__file__).parents[2].joinpath('data')


def repositories_txt() -> Path:
    return Path(__file__).parents[2].joinpath('repositories.txt')