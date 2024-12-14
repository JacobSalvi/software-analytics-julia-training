import pandas as pd

def jsonl_to_csv(jsonl_file, csv_file):
    try:
        # Read the file line by line for inspection
        with open(jsonl_file, 'r') as file:
            lines = file.readlines()
        print(f"File content (first 5 lines):\n{lines[:5]}")

        # Read JSONL with pandas
        df = pd.read_json(jsonl_file, lines=True)
        df.to_csv(csv_file, index=False)
        print(f"Converted {jsonl_file} to {csv_file}")
    except Exception as e:
        print(f"Error during conversion: {e}")

# Update paths if necessary
jsonl_file = "/home/ms/Documents/software-analytics-julia-training/benchmark_prompts.jsonl"
csv_file = "output.csv"

if __name__ == "__main__":
    jsonl_to_csv(jsonl_file, csv_file)
