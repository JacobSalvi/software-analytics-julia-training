import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    def function_body_is_dummy(el):
        if el["doc_string"] != "":
            return el["function_body"]
        if el["has_comments"] or el["has_constraints"]:
            return el["function_body"]
        return ""
    df.loc[(df["doc_string"] == "") & (df["has_internal_comments"] == False) & (df["has_constraints"]==False), "function_body"] = ""
    df = df[df.apply(function_body_is_dummy, axis=1)]
    pass
    # a = df.loc[lambda el: ]