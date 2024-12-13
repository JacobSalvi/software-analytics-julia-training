import json
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from src.utils import util


class DataHandler:
    PARSED = util.data_dir().joinpath("function_definitions_preprocessed.json")
    RAW = util.get_raw_data_path()

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
    def get_parsed() -> DataFrame:
        if not Path(DataHandler.PARSED).exists():
            raise FileNotFoundError(f"The file {DataHandler.PARSED.name} does not exist, please pre-process the data first with data/preprocess.py")

        with open(DataHandler.PARSED, "r") as f:
            df = pd.DataFrame([json.loads(line) for line in f])
            df.to_csv(util.data_dir().joinpath("function_definitions_preprocessed.csv"), index=False)
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



if __name__ == "__main__":
    DataHandler.get_parsed()