#import torch
from argparse import ArgumentParser

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.dataHanlder import DataHandler

device = "cuda" # for GPU usage or "cpu" for CPU usage


def model_small_lm_360m() -> tuple:
    checkpoint = "HuggingFaceTB/SmolLM-360M-Instruct"
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return model, tokenizer


def model_small_lm_135m() -> tuple:
    checkpoint = "HuggingFaceTB/SmolLM-135M"
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return model, tokenizer


def model_small_lm_1b() -> tuple:
    checkpoint = "HuggingFaceTB/SmolLM-1.7B"
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return model, tokenizer


def model_selector(model_name: str) -> tuple:
    if model_name == "360m":
        return model_small_lm_360m()
    elif model_name == "135m":
        return model_small_lm_135m()
    elif model_name == "1.7b":
        return model_small_lm_1b()
    else:
        return model_small_lm_360m()


def create_corpus():
    df = DataHandler.get_parsed()




if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("--model_name", type=str, default="360m", help="Model name to use.", choices=["360m", "135m", "1.7b"])
    args = argparse.parse_args()
    model, tokenizer = model_selector(args.model_name)
    print("Model:", model)
