import itertools
import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from statsmodels.stats.contingency_tables import mcnemar


def read_results(file):
    if not os.path.exists(file):
        raise FileNotFoundError(f"File {file} does not exist.")
    df = pd.read_json(file)
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


def efficiency_of_models(model_0_file,model_1_file, model_2_file):
    if not os.path.exists(model_0_file):
        raise FileNotFoundError(f"File {model_0_file} does not exist")
    if not os.path.exists(model_1_file):
        raise FileNotFoundError(f"File {model_1_file} does not exist.")
    if not os.path.exists(model_2_file):
        raise FileNotFoundError(f"File {model_2_file} does not exist.")
    
    model_0_results = read_results(model_0_file)
    model_1_results = read_results(model_1_file)
    model_2_results = read_results(model_2_file)

    correct_0 = model_0_results.sum()
    correct_1 = model_1_results.sum()
    correct_2 = model_2_results.sum()
    total = len(model_0_results)

    print(f"Model CoPilot - Passed : {correct_0}/{total} ({(correct_0/total)*100:.2f}%)")
    print(f"Model 135m - Passed: {correct_1}/{total} ({(correct_1/total)*100:.2f}%)")
    print(f"Model 360m - Passed: {correct_2}/{total} ({(correct_2/total)*100:.2f}%)")

    
    failed_1 = total - correct_1
    failed_2 = total - correct_2
    effect_size(correct_1, failed_1, correct_2, failed_2)


def compare_models(model_1, model_2):

    model_1_file = f"{model_1}_results_jl.json"
    model_2_file = f"{model_2}_results_jl.json"

    if not os.path.exists(model_1_file):
        raise FileNotFoundError(f"File {model_1_file} does not exist.")
    if not os.path.exists(model_2_file):
        raise FileNotFoundError(f"File {model_2_file} does not exist.")
    

    model_1_results = read_results(model_1_file)
    model_2_results = read_results(model_2_file)


    correct_1 = model_1_results.sum()
    correct_2 = model_2_results.sum()
    total = len(model_1_results)
    print(f"Model {model_2} - Passed: {correct_2}/{total} ({(correct_2 / total) * 100:.2f}%)")

    print(f"Calculating the effect size for {model_1} and {model_2}")

    
    failed_1 = total - correct_1
    failed_2 = total - correct_2
    effect_size(correct_1, failed_1, correct_2, failed_2)


def analyse_models(model_1, model_2):
    model_1_file = f"{model_1}_results_jl.json"
    model_2_file = f"{model_2}_results_jl.json"

    model_results = read_results(model_1_file)
    copilot_results = read_results(model_2_file)

    n_11 = sum((model_results & copilot_results))
    n_10 = sum((model_results & ~copilot_results))
    n_01 = sum((~model_results & copilot_results))
    n_00 = sum((~model_results & ~copilot_results))

    print()
    print(f"T test between {model_1} and {model_2}")
    mcnemartest(n_11, n_10, n_01, n_00)

    compare_models(model_1, model_2)


def main():
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--specific_models",
                           nargs='*',
                           type=str,
                           choices=["135m", "360m", "1-7B"],
                           help="Optional list of specific models to analyze.")
    args = arguments.parse_args()

    models = ["135m", "360m", "1-7B"]
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

    graphs(models, copilot_file)

    model_0 = "copilot"
    model_1 = "135m"
    model_2 = "360m"

    model_0_file = f"{model_0}_results_jl.json"
    model_1_file = f"{model_1}_results_jl.json"
    model_2_file = f"{model_2}_results_jl.json"

    print("Calculating the efficiency of the models")
    efficiency_of_models(model_0_file, model_1_file, model_2_file)

    print()
    print("Comparing models")
    for model in models:
        compare_models("copilot", model)


    # compare between models
    models = ["1-7B", "360m", "135m"]
    pairs = list(itertools.combinations(models, 2))
    for m1, m2 in pairs:
        analyse_models(m1, m2)

    


if __name__ == "__main__":
    main()
