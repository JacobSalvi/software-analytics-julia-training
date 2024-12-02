import pandas as pd

from src.utils import util


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[(df["doc_string"] == "") & (df["has_internal_comments"] == False) & (df["has_constraints"] == False), "function_body"] = ""
    df['function_body'] = df['function_body'].str.replace(r'(^|\n).*# TODO.*(\n|$)', '', regex=True)
    benchmark = pd.read_json(util.benchmark_prompts(), lines=True)
    benchmark['function_name'] = benchmark['prompt'].str.split('\n').str[-2]
    benchmark["function_name"] = benchmark["function_name"].str.split('function ').str[-1].str.split("(").str[0]
    function_names_filter = benchmark["function_name"].tolist()
    mask = df['function_header'].apply(
        lambda x: any(name == x.split("function ")[-1].split("(")[0] for name in function_names_filter))
    df = df[~mask]
    return df


def main():
    df = pd.read_json(util.data_dir().joinpath("function_definitions.json"), lines=True)
    df = preprocess(df)
    df.to_json(util.data_dir().joinpath("function_definitions_preprocessed.json"), orient="records", lines=True)


if __name__ == '__main__':
    main()
