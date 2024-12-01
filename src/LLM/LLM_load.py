import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils.util import get_model_path


def load_llm(model_type: str, verbose: bool = False) -> tuple:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = get_model_path(model_type)

    if not model_path.exists() or not os.listdir(model_path):
        print(f"Model {model_type} is not trained. Please train the model first, run LLM_train.py")
    else:
        try:
            # Search for a folder named "checkpoint" within the model directory
            checkpoint_path = None
            for item in model_path.iterdir():
                if item.is_dir() and "checkpoint" in item.name:
                    checkpoint_path = item
                    break

            if checkpoint_path is None:
                raise FileNotFoundError(f"No checkpoint folder found in {model_path}")

            model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token or '[PAD]'
                model.resize_token_embeddings(len(tokenizer))

            if verbose:
                print(f"Model {model_type} loaded successfully from {checkpoint_path}.")
            return model, tokenizer

        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            raise
