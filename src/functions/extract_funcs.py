import os
import time
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tree_sitter import Language, Parser
import tree_sitter_julia as tsj


JULIA_LANGUAGE = Language(tsj.language())
parser = Parser(JULIA_LANGUAGE)

def extract_functions_from_julia_code(code):
    tree = parser.parse(code.encode('utf-8'))
    root_node = tree.root_node
    functions = []

    def extract_function_code(node):
        if node.type == 'function_definition':
            start_byte = node.start_byte
            end_byte = node.end_byte
            function_code = code[start_byte:end_byte]
            functions.append({
                "function_code": function_code,
                "start_line": node.start_point[0],
                "start_column": node.start_point[1],
                "end_line": node.end_point[0],
                "end_column": node.end_point[1]
            })

        for child in node.children:
            extract_function_code(child)

    extract_function_code(root_node)
    return functions

def read_file_with_fallback(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='ISO-8859-1') as f:
            return f.read()

def process_repo(repo_path):
    functions = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.jl'):
                file_path = os.path.join(root, file)
                try:
                    code = read_file_with_fallback(file_path)
                    file_functions = extract_functions_from_julia_code(code)
                    for func in file_functions:
                        func["file_path"] = file_path
                        functions.append(func)
                except UnicodeDecodeError:
                    print(f"Skipped {file_path} due to encoding issues")
    return functions

def scan_directory_for_julia_functions_parallel(directory):
    all_functions = []
    repo_paths = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_repo, repo_path): repo_path for repo_path in repo_paths}
        
        for i, future in enumerate(as_completed(futures), 1):
            repo_path = futures[future]
            start_time = time.time()
            try:
                repo_functions = future.result()
                all_functions.extend(repo_functions)
                elapsed = time.time() - start_time
                print(f"Processed {repo_path} with {len(repo_functions)} functions in {elapsed:.2f} seconds (Repo {i}/{len(repo_paths)})")
            except Exception as e:
                print(f"Failed to process {repo_path}: {e}")

    return all_functions

if __name__ == "__main__":
    
    directory_path = 'repos/'
    start_time = time.time()
    all_functions_data = scan_directory_for_julia_functions_parallel(directory_path)
    df = pd.DataFrame(all_functions_data)
    output_csv = 'data/julia_functions_full_code.csv'
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    total_time = time.time() - start_time
    print(f"Saved all functions to {output_csv}")
    print(f"Total processing time: {total_time:.2f} seconds")
