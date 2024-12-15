import argparse
import gzip
from pathlib import Path
from typing import List
import json


def get_files_with_suffix(base_dir: Path, suffix: str) -> List[Path]:
    return [f for f in base_dir.glob("**/*") if f.is_file() and f.suffix == suffix]


def main():
    argument_parser = argparse.ArgumentParser(description="This script can be used to perform a postprocessing before running the evaluation.")
    argument_parser.add_argument("--input-dir", type=Path, required=True)
    args = argument_parser.parse_args()
    original = args.input_dir
    original = Path(original)
    files = get_files_with_suffix(original, ".gz")
    for f in files:
        if ".results" in f.name:
            continue
        with gzip.open(f, "rb") as g:
            data = g.read()
            content = json.loads(data)
            content["completions"][0] = f"{fix_julia_ends(content["completions"][0])}\nend"
            with gzip.open(f, 'wb') as out_f:
                out_f.write(json.dumps(content).encode('utf-8'))
    return


def fix_julia_ends(julia_code):
    lines = julia_code.split('\n')

    block_stack = []
    cleaned_code = []

    for line in lines:
        stripped_line = line.strip()

        if stripped_line.startswith("if") or stripped_line.startswith("for") or stripped_line.startswith("while"):
            block_stack.append(stripped_line)
            cleaned_code.append(line)
        elif stripped_line == "end":
            if block_stack:
                block_stack.pop()
                cleaned_code.append(line)
            else:
                pass
        else:
            cleaned_code.append(line)

    while block_stack:
        cleaned_code.append("end")
        block_stack.pop()
    # add an end corresponding to the function keyword.
    return '\n'.join(cleaned_code)


if __name__ == "__main__":
    main()
