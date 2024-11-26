#ORRECT
import json
import os
import tarfile
from pathlib import Path
import io

import pandas as pd
from pandas import DataFrame
from sympy import false
from src.data.pre_process import process_data
from src.utils import util


class DataHandler:
    PARSED = util.data_dir().joinpath("fd_parsed.json")
    RAW = util.data_dir().joinpath("fd_raw.json")

    @staticmethod
    def get_raw() -> list:
        data = []
        if not Path(DataHandler.RAW).exists():
            raise FileNotFoundError(f"The file {DataHandler.RAW.name} does not exist")

        with open(DataHandler.RAW, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    @staticmethod
    def get_parsed(force_parse: bool = false) -> DataFrame:
        if force_parse and Path(DataHandler.PARSED).exists():
            os.remove(DataHandler.PARSED)

        if not Path(DataHandler.PARSED).exists():
            df = process_data(pd.DataFrame(DataHandler.get_raw()))
            parsed = df.to_json(DataHandler.PARSED, orient="records", lines=True)
            json.dump(parsed, DataHandler.PARSED.open("w"))
        df = pd.DataFrame(json.load(DataHandler.PARSED.open()))
        return df

if __name__ == '__main__':
    try:
        DataHandler.get_parsed(True)
    except FileNotFoundError as e:
        print(e)