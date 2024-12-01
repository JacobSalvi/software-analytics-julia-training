import json
import os
from pathlib import Path

import pandas as pd
from pandas import DataFrame
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
    def get_parsed(force_parse: bool = False) -> DataFrame:
        if force_parse and Path(DataHandler.PARSED).exists():
            os.remove(DataHandler.PARSED)

        if not Path(DataHandler.PARSED).exists():
            df = process_data(pd.DataFrame(DataHandler.get_raw()))
            with open(DataHandler.PARSED, "w") as f:
                df.to_json(f, orient="records", lines=True)

        with open(DataHandler.PARSED, "r") as f:
            df = pd.DataFrame([json.loads(line) for line in f])
        return df

    @staticmethod
    def baseline_pre_process(data: DataFrame) -> DataFrame:
        data["doc_string"] = data["doc_string"].swifter.apply(lambda x: x.strip().lower() if isinstance(x, str) else x)
        data["function_header"] = data["function_header"].swifter.apply(
            lambda x: x.strip() if isinstance(x, str) else x)
        data["function_body"] = data["function_body"].swifter.apply(lambda x: x.strip() if isinstance(x, str) else x)
        return data

    @staticmethod
    def get_baseline():
        return DataHandler.baseline_pre_process(pd.DataFrame(DataHandler.get_raw()))


if __name__ == '__main__':
    try:
        DataHandler.get_parsed(True)
    except FileNotFoundError as e:
        print(e)