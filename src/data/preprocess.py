import argparse
from pathlib import Path

import pandas as pd
import swifter  #DO NOT REMOVE THIS IMPORT

from src.utils import util
from src.utils.util import get_raw_data_path


def preprocess(df: pd.DataFrame, remove_no_docs: bool = False) -> pd.DataFrame:
    if remove_no_docs:
        df = df[df['doc_string'].str.contains("[a-zA-Z]")]
    df["function_header"] = df["function_header"].str.replace(" (", "(")
    df.loc[(df["doc_string"] == "") & (df["has_internal_comments"] == False) & (df["has_constraints"] == False), "function_body"] = ""
    benchmark = pd.read_json(util.benchmark_prompts(), lines=True)
    benchmark['function_name'] = benchmark['prompt'].str.split('\n').str[-2]
    benchmark["function_name"] = benchmark["function_name"].str.split('function ').str[-1].str.split("(").str[0]
    function_names_filter = benchmark["function_name"].tolist()

    # Use swifter for faster apply
    mask = df['function_header'].swifter.apply(
        lambda x: any(name == x.split("function ")[-1].split("(")[0] for name in function_names_filter)
    )

    df = df[~mask]
    df["doc_string"] = df["doc_string"].str.strip()
    df["function_header"] = df["function_header"].str.strip()
    df["function_body"] = df["function_body"].str.strip()
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remove-no-docs", dest="remove_no_docs", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not get_raw_data_path().exists():
        raise FileNotFoundError(
            f"The file {get_raw_data_path().name} does not exist, please download the data first and put it the `/data` folder.")

    df = pd.read_json(get_raw_data_path(), lines=True)

    df = preprocess(df, remove_no_docs=args.remove_no_docs)
    output = args.output if args.output else util.data_dir().joinpath("function_definitions_preprocessed.json")
    df.to_json(output, orient="records", lines=True)


if __name__ == '__main__':
    main()
