import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils.util import get_model_path


def load_llm(model_type: str, verbose: bool= False) -> tuple:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = get_model_path(model_type)
    if not model_path.exists() or not os.listdir(model_path):
        print(f"Model {model_type} is not trained. Please train te model first, run LLM_train.py")
    else:
        try:

            model = AutoModelForCausalLM.from_pretrained(model_path.joinpath("checkpoint-125")).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token or '[PAD]'
                model.resize_token_embeddings(len(tokenizer))

            if verbose:
                print(f"Model {model_type} loaded successfully.")
            return model, tokenizer

        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            raise

