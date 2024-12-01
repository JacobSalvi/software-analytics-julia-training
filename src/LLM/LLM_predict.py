from argparse import ArgumentParser
from LLM_load import load_llm
from src.utils.util import base_model_types, all_model_types

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def predict_llm(model_type: str, prompt: str, max_length: int = 1024, verbose: bool = False) -> str:
    model, tokenizer = load_llm(model_type, verbose)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    attention_mask = (input_ids != tokenizer.pad_token_id).to(model.device)
    output = model.generate(
        input_ids,
        max_length=max_length,
        pad_token_id=tokenizer.pad_token_id,
        attention_mask=attention_mask
    )
    return tokenizer.decode(output[0], skip_special_tokens=False)



def predict_llm_all(prompt: str, max_length: int = 50, verbose: bool = False) -> list[str]:
    predictions = []
    for model_type in all_model_types():
        predictions.append(predict_llm(model_type, prompt, max_length, verbose))
    return predictions


def printer_predictions(predictions: list[str], prompt: str):
    print(f"Prompt: {prompt}")
    for i, prediction in enumerate(predictions):
        print("------------------------------------------------")
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

    args = argparse.parse_args()
    if args.model == "all":
        predictions = predict_llm_all(args.prompt, args.max_length, False)
        printer_predictions(predictions, args.prompt)
    else:
        prediction = predict_llm(args.model, args.prompt, args.max_length, False)
        printer_prediction(prediction, args.model, args.prompt)


if __name__ == "__main__":
    main()