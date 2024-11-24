import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, AnyStr, Dict

import pandas as pd
from tree_sitter_languages import get_language, get_parser

from src.utils import util


@dataclass
class FunctionDefinition:
    doc_string: AnyStr = ""
    function_header: AnyStr = ""
    function_body: AnyStr = ""
    has_internal_comments: bool = False
    has_constraints: bool = False


def get_files(base_dir: Path, suffix: AnyStr = ".jl") -> List[Path]:
    return [f for f in base_dir.glob("**/*") if f.is_file() and f.suffix == suffix]


def repo_to_files() -> Dict[AnyStr, List[Path]]:
    repositories: pd.DataFrame = util.get_repository_urls()
    data_dir: Path = util.data_dir()
    repo_name_to_files: Dict[AnyStr, List[Path]] = dict()
    limit = 1024 * 1024
    repository_names = [r for r in repositories["name"] if r not in ["Mehrnoom/Cryptocurrency-Pump-Dump", "analytech-solutions/C.jl", "terasakisatoshi/MyWorkflow.jl", "udohjeremiah/REPLference.jl", "JuliaIO/HDF5.jl", "vnegi10/CryptoDashApp.jl", "m3g/Packmol.jl", "JakobAsslaender/MRIgeneralizedBloch.jl","greimel/distributional-macroeconomics", "jmboehm/RegressionTables.jl", "analytech-solutions/CBinding.jl"]]
    for repo_name in repository_names:
        project_path: Path = data_dir.joinpath(repo_name.replace("/", "_"))
        if project_path.is_dir():
            files: List[Path] = get_files(project_path, suffix=".jl")
            files = [f for f in files if f.stat().st_size < limit]
            repo_name_to_files[repo_name] = files
    return repo_name_to_files


def parse_files(input_files: List[Path],
                keep_comments: bool = True,
                keep_constraints: bool = True) -> List[FunctionDefinition]:
    language = get_language("julia")
    parser = get_parser("julia")
    doc_pattern = """
      (_
        (macrocall_expression) @comment .
        (function_definition) @function
      )
    """
    function_pattern = """
        (_
            (function_definition) @function
        )
    """
    string_literal_pattern = """
    (_
        (string_literal) @comment .
        (function_definition) @function
    )
    """
    line_comment_pattern = """
    (_
        (line_comment) @comment .
        (function_definition) @function
    )
    """
    macro_query = language.query(doc_pattern)
    function_query = language.query(function_pattern)
    literal_query = language.query(string_literal_pattern)
    line_comment_query = language.query(line_comment_pattern)
    functions = []
    for file in input_files:
        try:
            content: AnyStr = file.read_text()
        except UnicodeDecodeError:
            content = file.read_text(encoding="iso-8859-1")
        tree = parser.parse(content.encode())
        root_node = tree.root_node
        macro_results = macro_query.captures(root_node)
        macro_results = [el for el in macro_results if el[1] == "comment" and "@doc" in el[0].text.decode("utf-8") or el[1] == "function"]
        literal_results = literal_query.captures(root_node)
        macro_results.extend(literal_results)
        line_comment_results = line_comment_query.captures(root_node)
        macro_results.extend(line_comment_results)
        function_results = function_query.captures(root_node)
        macro_results.extend([el for el in function_results if el not in macro_results])
        functions.extend(get_functions(macro_results=macro_results,
                                       keep_comments=keep_comments,
                                       keep_constraints=keep_constraints))
        pass
    return functions


def get_function_header(function_definition) -> AnyStr:
    function = function_definition.children[0]
    identifier = function_definition.children[1]
    parameter_list = function_definition.children[2]
    return f"{function.text.decode('utf-8')} {identifier.text.decode('utf-8')} {parameter_list.text.decode('utf-8')}"


def get_function_body(function_definition, keep_comments=True, keep_constraints=True):
    body = function_definition.children[3:]
    if keep_comments and keep_constraints:
        pass
    filter_list = ["line_comment"]
    has_internal_comments = False
    has_constraints = False
    new_body = []
    for child in body:
        if child.type in filter_list:
            has_internal_comments = True
            if keep_comments:
                new_body.append(child)
        else:
            new_body.append(child)
    body = new_body

    filtered_macros = ('@debug', '@info', '@variable', '@constraint', '@objective')
    new_body = []
    for child in body:
        if child.type == "macrocall_expression" and child.text.decode("utf-8").startswith(filtered_macros):
            has_constraints = True
            if keep_constraints:
                new_body.append(child)
        else:
            new_body.append(child)
    body = new_body
    body = [n.text.decode("utf-8") for n in body]
    return "".join(body), has_internal_comments, has_constraints


def get_functions(macro_results,
                  keep_comments: bool = True,
                  keep_constraints: bool = True) -> List[FunctionDefinition]:
    functions = []
    macro_iterator = iter(macro_results)
    for el in macro_iterator:
        if el[1] == "function":
            macro_text = ""
            function_definition = el[0]
        else:
            macro_text = el[0].text.decode("utf-8")
            function_definition = next(macro_iterator)[0]
        header = get_function_header(function_definition)
        body, has_internal_comments, has_constraints = get_function_body(function_definition,
                                                                         keep_comments=keep_comments,
                                                                         keep_constraints=keep_constraints)
        functions.append(FunctionDefinition(function_header=header,
                                            doc_string=macro_text,
                                            function_body=body,
                                            has_internal_comments=has_internal_comments,
                                            has_constraints=has_constraints))
    return functions


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--remove-comments", action="store_true")
    argument_parser.add_argument("--remove-constraints", action="store_true")
    args = argument_parser.parse_args()
    repository_to_files = repo_to_files()
    parse_args = [(files, not args.remove_comments, not args.remove_constraints) for name, files in
                  repository_to_files.items()]

    results = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for files, keep_comments, keep_constraints in parse_args:
            future = executor.submit(parse_files, files, keep_comments, keep_constraints)
            futures.append(future)

        for future in as_completed(futures):
            results.extend(future.result())

    df = pd.DataFrame([r.__dict__ for r in results])
    print(f"Saving {len(df)} rows")
    output_file = util.data_dir().joinpath("function_definitions.json")
    print(f"Saving json to {output_file}")
    df.to_json(output_file, orient="records", lines=True)
    return


if __name__ == '__main__':
    main()
