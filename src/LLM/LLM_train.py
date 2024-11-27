import os

import deepspeed
from argparse import ArgumentParser
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, \
    TrainingArguments
from src.data.dataHanlder import DataHandler
from datasets import Dataset
from src.utils.util import get_model_path, model_types, deepspeed_config_path

device = "cuda" if torch.cuda.is_available() else "cpu"

INTERNAL_TEST = False

def model_small_lm_360m() -> tuple:
    checkpoint = "HuggingFaceTB/SmolLM-360M-Instruct"
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    add_special_tokens_if_needed(tokenizer, model)
    return model, tokenizer


def model_small_lm_135m() -> tuple:
    checkpoint = "HuggingFaceTB/SmolLM-135M"
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    add_special_tokens_if_needed(tokenizer, model)
    return model, tokenizer


def model_small_lm_1b() -> tuple:
    checkpoint = "HuggingFaceTB/SmolLM-1.7B"
    # Initialize with DeepSpeed ZeRO
    with deepspeed.zero.Init(config_dict_or_path="deepspeed_config.json"):
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            ignore_mismatched_sizes=True
        )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    add_special_tokens_if_needed(tokenizer, model)
    return model, tokenizer



def add_special_tokens_if_needed(tokenizer: AutoTokenizer, model: AutoModelForCausalLM):
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print("Added new pad_token: [PAD]")
            model.resize_token_embeddings(len(tokenizer))




def model_selector(model_name: str) -> tuple:
    if model_name == "360m":
        return model_small_lm_360m(), get_model_path("360m")
    elif model_name == "135m":
        return model_small_lm_135m(), get_model_path("135m")
    elif model_name == "1.7b":
        return model_small_lm_1b(), get_model_path("1.7b")
    else:
        return model_small_lm_360m()


def create_corpus(tokenizer, sample_run:bool=False)-> Dataset:
    #Dataset needed for efficient memory management
    if sample_run:
        print("Running a sample training run.")
        c_dataset = Dataset.from_pandas(DataHandler.get_parsed().sample(1000))
    else:
        c_dataset = Dataset.from_pandas(DataHandler.get_parsed())

    def tokenize_function(df):
        combined_texts = [
            f"{doc}\n\n{header}\n\n{body}".strip()
            for doc, header, body in zip(
                df.get("doc_string", []),
                df.get("function_header", []),
                df.get("function_body", [])
            )
        ]
        return tokenizer(
            combined_texts,
            truncation=True,
            padding='max_length',  # Use 'longest' for dynamic padding within each batch
            max_length=512,
            return_attention_mask=True
        )

    tokenized_dataset = c_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['doc_string', 'function_header', 'function_body']
    )

    # Set the format for PyTorch tensors
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    if INTERNAL_TEST:
        tokenized_dataset_inspection(tokenized_dataset, tokenizer)

    return tokenized_dataset

def batch_size_per_model(model: str) -> int:
    if model == "360m":
        return 4
    elif model == "135m":
        return 8
    elif model == "1.7b":
        return 1

def gradient_checkpointing_enable(model: str) -> int:
    if model == "360m":
        return 2
    elif model == "135m":
        return 2
    elif model == "1.7b":
        return 1



def train_model(model_type: str, model, tokenizer, corpus : Dataset, save_path: str):
    """
    Trains the model using the Hugging Face Trainer API.
    """
    print(deepspeed.__version__)

    if model_type == "1.7b":
        model.gradient_checkpointing_enable()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Set to False for causal language modeling
    )

    # Define the absolute path to deepspeed_config.json
    deepspeed_config= deepspeed_config_path()

    # Verify that the DeepSpeed config file exists
    if not os.path.isfile(deepspeed_config):
        raise FileNotFoundError(f"DeepSpeed config file not found at {deepspeed_config_path}")

    if model_type == "1.7b":
        training_args = TrainingArguments(
            output_dir=save_path,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            save_steps=1000,
            deepspeed=deepspeed_config,
            fp16=True
        )
    else:
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=save_path,                    # Directory to save model checkpoints
            overwrite_output_dir=True,               # Overwrite the content of the output directory
            num_train_epochs=1,                      # Number of training epochs
            per_device_train_batch_size=1,           # Batch size per device during training
            gradient_accumulation_steps=1,
            warmup_steps=500,                        # Number of warmup steps for learning rate scheduler
            weight_decay=0.01,                       # Strength of weight decay
            logging_dir='./logs',                    # Directory for storing logs
            logging_steps=50,                        # Log every X updates steps
            save_steps=20000,                        # Save checkpoint every X updates steps
            save_total_limit=2,                      # Limit the total amount of checkpoints
            fp16=torch.cuda.is_available(),          # Use mixed precision if available
            remove_unused_columns=True,              # Remove columns not used by the model
            gradient_checkpointing=True,
            deepspeed=deepspeed_config,
        )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=corpus,
        data_collator=data_collator,
    )

    print(f"Model is on device: {next(model.parameters()).device}")
    print(f"Training device: {trainer.args.device}")


    # Start training
    trainer.train()

    tokenizer.save_pretrained(save_path)



def tokenized_dataset_inspection(tokenized_dataset, tokenizer):
    i = 0
    print("Input IDs:", tokenized_dataset[i]['input_ids'])
    print("Attention Mask:", tokenized_dataset[i]['attention_mask'])
    print("Decoded Text:", tokenizer.decode(tokenized_dataset[i]['input_ids'], skip_special_tokens=False))

def perform_train(model_type:str, sample_run:bool=False):
    (model, tokenizer), path  = model_selector(model_type)
    corpus = create_corpus(tokenizer, sample_run)
    train_model(model_type, model, tokenizer, corpus, path)

def perform_train_all(sample_run:bool=False):
    perform_train("360m", sample_run)
    perform_train("135m", sample_run)
    perform_train("1.7b", sample_run)

def main():
    argparse = ArgumentParser()
    argparse.add_argument("--model", type=str, default="1.7b", help="Model name to use.",
                          choices= model_types().append("all"))
    argparse.add_argument("--sample_run", action="store_true", help="Run a sample training run.", default=True)
    argparse.add_argument("--local_rank", type=int, default=0,
                        help="Local rank for distributed training (provided by DeepSpeed)")

    args = argparse.parse_args()

    if args.model == "all":
        perform_train_all(args.sample_run)
    else:
        perform_train(args.model, args.sample_run)

if __name__ == "__main__":
    main()
