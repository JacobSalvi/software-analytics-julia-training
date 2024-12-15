import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from statsmodels.stats.contingency_tables import mcnemar
from pathlib import Path

from pathlib import Path

def construct_file_path(file_name):
    base_dir = Path(__file__).resolve().parents[2] 
    results_dir = base_dir / "benchmark"    
    file_path = results_dir / file_name
    return file_path


def read_results(file):
    file_path = construct_file_path(file)
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")
    df = pd.read_json(file_path)
    return df['passed']

def mcnemartest(n_11, n_10, n_01, n_00):
    table = [[n_11, n_10],
             [n_01, n_00]]
    result = mcnemar(table, exact=True)  
    print(f"McNemar's Test - Statistic: {result.statistic}, p-value: {result.pvalue}")
    return result

def effect_size(count_passed_1, count_failed_1, count_passed_2, count_failed_2):
    odds_ratio = (count_passed_1 / count_failed_1) / (count_passed_2 / count_failed_2)
    print(f"Odds Ratio (Effect Size): {odds_ratio}")
    return odds_ratio

def graphs(models, copilot_file, specific_models=None):
    if specific_models:
        models = [model for model in models if model in specific_models]

    copilot_results = read_results(copilot_file)
    copilot_correct = copilot_results.sum()
    copilot_incorrect = len(copilot_results) - copilot_correct

    model_correct_counts = []
    model_incorrect_counts = []
    for model in models:
        model_file = construct_file_path(f"{model}_results_jl.json")
        model_results = read_results(model_file)
        correct = model_results.sum()
        incorrect = len(model_results) - correct  
        model_correct_counts.append(correct)
        model_incorrect_counts.append(incorrect)

    all_names = models + ["Copilot"]
    all_correct_counts = model_correct_counts + [copilot_correct]
    all_incorrect_counts = model_incorrect_counts + [copilot_incorrect]

    plt.figure(figsize=(10, 6))
    plt.bar(all_names, all_correct_counts, label='Correct Predictions', color='skyblue', edgecolor='black')
    plt.bar(all_names, all_incorrect_counts, bottom=all_correct_counts, label='Incorrect Predictions', color='salmon', edgecolor='black')

    plt.title('Correct and Incorrect Predictions by Models and Copilot', fontsize=14)
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Number of Predictions', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.show()

def efficiency_of_models(models):

    for model in models :

        model_file = construct_file_path(f"{model}_results_jl.json")
        model_results = read_results(model_file)

        correct = model_results.sum()
    
        total = len(model_results)

        print(f"Model {model} - Passed: {correct}/{total} ({(correct/total)*100:.2f}%)")
    


def compare_models(model_1, model_2):

    
    if model_1 == "copilot":
        model_1_file = construct_file_path("copilot_multiple_predictions.json")
    else :
        model_1_file = construct_file_path(f"{model_1}_results_jl.json")
    if model_2 == "copilot" :
        model_2_file = construct_file_path("copilot_multiple_predictions.json")
    else :
        model_2_file = construct_file_path(f"{model_2}_results_jl.json")
    model_1_results = read_results(model_1_file)
    model_2_results = read_results(model_2_file)

    correct_1 = model_1_results.sum()
    correct_2 = model_2_results.sum()
    total = len(model_1_results)

    print(f"Calculating the effect size for the following 2 models")

    failed_1 = total - correct_1
    failed_2 = total - correct_2
    effect_size(correct_1, failed_1, correct_2, failed_2)

def main():
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--specific_models",
                           nargs='*',
                           type=str,
                           choices=["135m", "360m", "1-7B"],
                           help="Optional list of specific models to analyze.")
    args = arguments.parse_args()

    if args.specific_models:
        models = args.specific_models
    else:
        models = ["135m", "360m", "1-7B"]

    copilot_file = "copilot_multiple_predictions.json"

    for model in models:
        file = f"{model}_results_jl.json"

        if not construct_file_path(file).exists():
            print(f"Results file for {model} does not exist.")
            continue
        file = construct_file_path(file)

        model_results = read_results(file)
        copilot_results = read_results(copilot_file)

        if len(model_results) != len(copilot_results):
            raise ValueError(f"The result files for {model} and Copilot do not have the same number of problems.")

        n_11 = sum((model_results & copilot_results))
        n_10 = sum((model_results & ~copilot_results))
        n_01 = sum((~model_results & copilot_results))
        n_00 = sum((~model_results & ~copilot_results))

        print(f"Results for {model}:")
        mcnemartest(n_11, n_10, n_01, n_00)

    
    print("Calculating the efficiency of the models specified")
    efficiency_of_models(models)
    graphs(models, copilot_file)

    while True:
        print("Enter the 2 models you want to compare (or type 'exit' to quit):")
        user_input = input()
        if user_input.lower() == 'exit': 
            print("Exiting...")
            break
        models = user_input.split()
        if len(models) != 2:  
            print("Please enter exactly two models.")
            continue
        compare_models(models[0], models[1])  

if __name__ == "__main__":
    main()
