import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from statsmodels.stats.contingency_tables import mcnemar

def read_results(file):
    """
    Reads a JSON file and extracts the 'passed' column as a boolean list.
    """
    if not os.path.exists(file):
        raise FileNotFoundError(f"File {file} does not exist.")
    df = pd.read_json(file)
    return df['passed']

def mcnemartest(n_11, n_10, n_01, n_00):
    """
    Performs McNemar's test on a contingency table.
    """
    table = [[n_11, n_10],
             [n_01, n_00]]
    result = mcnemar(table, exact=True)  
    print(f"McNemar's Test - Statistic: {result.statistic}, p-value: {result.pvalue}")
    return result

def effect_size(count_passed_1, count_failed_1, count_passed_2, count_failed_2):
    """
    Calculates the odds ratio as a measure of effect size.
    """
    odds_ratio = (count_passed_1 / count_failed_1) / (count_passed_2 / count_failed_2)
    print(f"Odds Ratio (Effect Size): {odds_ratio}")
    return odds_ratio

def graphs(models, copilot_file, specific_models=None):
    """
    Displays a stacked bar graph showing the number of correct (True) 
    and incorrect (False) predictions for all models and Copilot.
    """
    if specific_models:
        models = [model for model in models if model in specific_models]

    copilot_results = read_results(copilot_file)
    copilot_correct = copilot_results.sum()
    copilot_incorrect = len(copilot_results) - copilot_correct

    model_correct_counts = []
    model_incorrect_counts = []
    for model in models:
        model_file = f"{model}_results_jl.json"
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

def main():
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--specific_models", nargs='*', type=str, help="Optional list of specific models to analyze")
    args = arguments.parse_args()

    models = ["135m", "360m"]
    if args.specific_models:
        models = args.specific_models

    copilot_file = "copilot_results_jl.json"

    for model in models:
        file = f"{model}_results_jl.json"

        if not os.path.exists(file):
            print(f"Results file for {model} does not exist.")
            continue

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

        count_passed_1 = sum(model_results)
        count_failed_1 = len(model_results) - count_passed_1
        count_passed_2 = sum(copilot_results)
        count_failed_2 = len(copilot_results) - count_passed_2

        effect_size(count_passed_1, count_failed_1, count_passed_2, count_failed_2)

    graphs(models, copilot_file)

if __name__ == "__main__":
    main()
