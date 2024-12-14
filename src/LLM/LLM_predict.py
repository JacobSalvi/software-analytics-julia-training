from argparse import ArgumentParser
from src.LLM.LLM_load import load_llm
from src.utils.julia_formatter import  julia_formatter
from src.utils.util import base_model_types, all_model_types

import os
import re



os.environ["TOKENIZERS_PARALLELISM"] = "false"


def clean_output(output: str) -> str:
    """
    Cleans the generated output:
    - Truncates it after the last matching 'end' for all opened 'function' blocks.
    - Ensures the code is syntactically complete.
    """
    lines = output.splitlines()
    function_count = 0
    end_count = 0
    truncated_lines = []

    for line in lines:
        if re.search(r'^\s*function\b', line):
            function_count += 1

        if re.search(r'^\s*end\b', line):
            end_count += 1

        truncated_lines.append(line)

        if end_count == function_count and function_count > 0:
            break

    truncated_output = "\n".join(truncated_lines)

    return truncated_output.strip()


def predict_llm(model_type: str, prompt: str, signature: bool, baseline: bool, max_new_tokens: int = 1024, verbose: bool = False) -> str:
    try:
        model, tokenizer = load_llm(model_type, signature, baseline, verbose)
    except Exception as e:
        return f"Error loading model: {e}"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    attention_mask = (input_ids != tokenizer.pad_token_id).to(model.device)
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        attention_mask=attention_mask
    )
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=False)
    decoded_output = clean_output(decoded_output)
    decoded_output = julia_formatter(decoded_output)
    return decoded_output


def predict_llm_all(prompt: str, signature: bool, baseline: bool, max_length: int = 1024, verbose: bool = False) -> list[str]:
    predictions = []
    for model_type in base_model_types():
        predictions.append(predict_llm(model_type, prompt, signature, baseline, max_length, verbose))
    return predictions


def printer_predictions(predictions: list[str], prompt: str):
    print(f"Prompt: {prompt}")
    for i, prediction in enumerate(predictions):
        print("---------------------------------------------------------------------------")
        print(f"Model {base_model_types()[i]}")
        print(prediction)


def printer_prediction(prediction: str, model: str, prompt: str):
    print(f"Prompt: {prompt}, Model: {model}")
    print(prediction)



def main():
    argparse = ArgumentParser()
    argparse.add_argument("--prompt", type=str, help="Prompt to generate text from", default="for loop in a list")
    argparse.add_argument("--max_length", type=int, default=1024, help="Maximum length of the generated text")
    argparse.add_argument("--model", type=str, default="360m", help="Model name to use.",
                          choices=all_model_types().append("all"))
    argparse.add_argument("--signature", action="store_true", help="Use only function signature for training.")
    argparse.add_argument("--baseline", action="store_true", help="Use only baseline for training.")


    args = argparse.parse_args()
    if args.model == "all":
        predictions = predict_llm_all(args.prompt, args.signature, args.baseline, args.max_length, False)
        printer_predictions(predictions, args.prompt)
    else:
        prediction = predict_llm(args.model, args.prompt, args.signature, args.baseline, args.max_length, False)
        printer_prediction(prediction, args.model, args.prompt)


if __name__ == "__main__":
    main()
