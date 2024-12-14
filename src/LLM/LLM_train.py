from argparse import ArgumentParser
from pathlib import Path
import torch
from datasets import Dataset
from pandas import DataFrame
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.LLM.LLM_utils import model_type_definer
from src.data.dataHanlder import DataHandler
from src.utils.util import (
    get_model_path,
    base_model_types,
    remove_all_files_and_subdirectories_in_folder,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

INTERNAL_TEST = False

MAX_LENGTH = 1024


def add_special_tokens_if_needed(tokenizer: AutoTokenizer, model: AutoModelForCausalLM):
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print("Added new pad_token: [PAD]")
            model.resize_token_embeddings(len(tokenizer))


def model_small_lm_360m() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    checkpoint = "HuggingFaceTB/SmolLM-360M-Instruct"
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    add_special_tokens_if_needed(tokenizer, model)
    return model, tokenizer


def model_small_lm_135m() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    checkpoint = "HuggingFaceTB/SmolLM-135M"
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    add_special_tokens_if_needed(tokenizer, model)
    return model, tokenizer


def model_small_lm_1b() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    checkpoint = "HuggingFaceTB/SmolLM-1.7B"
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    add_special_tokens_if_needed(tokenizer, model)
    return model, tokenizer


def model_selector(model_name: str, signature: bool, baseline: bool) -> tuple:
    if signature and baseline:
        raise ValueError("Cannot have both signature and baseline enabled.")

    model_map = {
        "360m": model_small_lm_360m,
        "135m": model_small_lm_135m,
        "1.7b": model_small_lm_1b,
    }

    path_suffix = "_signature" if signature else ""
    path_suffix = "_baseline" if baseline else path_suffix

    if model_name in model_map:
        model_function = model_map[model_name]
        model, tokenizer = model_function()
        path = get_model_path(f"{model_name}{path_suffix}")
        return (model, tokenizer), path
    else:
        raise ValueError(f"Invalid model name: {model_name}")


def create_corpus(
    data: DataFrame,
    tokenizer: AutoTokenizer,
    just_signature: bool,
    sample_run: bool = False
) -> Dataset:
    if sample_run:
        print("Running a sample training run.")
        c_dataset = Dataset.from_pandas(data.sample(1000))
    else:
        c_dataset = Dataset.from_pandas(data)

    def tokenize_function(df):
        if just_signature:
            combined_texts = [
                f"{header}\n{body}".strip()
                for header, body in zip(
                    df.get("function_header", []),
                    df.get("function_body", [])
                )
            ]
        else:
            combined_texts = [
                f"{doc}\n{header}\n{body}".strip()
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
            max_length=MAX_LENGTH,
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
        return 8
    elif model == "135m":
        return 8
    elif model == "1.7b":
        return 2


def gradient_accumulation_steps_per_model(model: str) -> int:
    if model == "360m":
        return 2
    elif model == "135m":
        return 2
    elif model == "1.7b":
        return 4


def enable_gradient_checkpointing(model: AutoModelForCausalLM):
    model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled.")


def train_small(model_type: str, model, tokenizer, corpus: Dataset, save_path: Path):
    """
    Trains the model using the Hugging Face Trainer API.
    """
    remove_all_files_and_subdirectories_in_folder(save_path)

    model.config.use_cache = False
    if "1.7b" in model_type:
        enable_gradient_checkpointing(model)
        model = apply_lora_to_model(
            model,
            target_modules=["q_proj", "v_proj"],
            r=8,
            alpha=32,
            dropout=0.1
        )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Set to False for causal language modeling
    )

    # Define training arguments with optimizations
    training_args = TrainingArguments(
        learning_rate=1e-3,
        max_grad_norm=1.0,
        output_dir=str(save_path),  # Directory to save model checkpoints
        overwrite_output_dir=True,  # Overwrite the content of the output directory
        num_train_epochs=1,  # Number of training epochs
        per_device_train_batch_size=batch_size_per_model(model_type),  # Batch size per device during training
        gradient_accumulation_steps=gradient_accumulation_steps_per_model(model_type),  # Accumulate gradients
        warmup_steps=500,  # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # Strength of weight decay
        logging_dir='./logs',  # Directory for storing logs
        logging_steps=500,  # Log every X updates steps
        save_steps=50000,  # Save checkpoint every X updates steps
        save_total_limit=2,  # Limit the total amount of checkpoints
        fp16=torch.cuda.is_available(),  # Use mixed precision if available
        remove_unused_columns=True,  # Remove columns not used by the model
        dataloader_num_workers=4,  # Adjusted for optimal performance
        gradient_checkpointing=True,  # Enable gradient checkpointing
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


def apply_lora_to_model(model, target_modules=["q_proj", "v_proj"], r=8, alpha=32, dropout=0.1):
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    print(f"LoRA applied with target_modules={target_modules}, r={r}, alpha={alpha}, dropout={dropout}.")
    return model


def tokenized_dataset_inspection(tokenized_dataset, tokenizer):
    i = 0
    print("Input IDs:", tokenized_dataset[i]['input_ids'])
    print("Attention Mask:", tokenized_dataset[i]['attention_mask'])
    print("Decoded Text:", tokenizer.decode(tokenized_dataset[i]['input_ids'], skip_special_tokens=False))


def select_data(baseline: bool) -> DataFrame:
    if baseline:
        return DataHandler.get_baseline()
    else:
        return DataHandler.get_parsed()


def perform_train(model_type: str, signature: bool, baseline: bool, sample_run: bool = False):
    print(f"Training model: {model_type_definer(model_type, baseline, signature)}")
    data = select_data(baseline)
    (model, tokenizer), path = model_selector(model_type, signature, baseline)
    corpus = create_corpus(data, tokenizer, signature, sample_run)
    train_small(model_type, model, tokenizer, corpus, path)


def perform_train_all(signature: bool, baseline: bool, sample_run: bool = False):
    print("Training all models.")
    for model_type in base_model_types():
        print("-------------------------------------------------------------------------------------------------------------------------")
        perform_train(model_type, signature, baseline, sample_run)


def main():
    argparse = ArgumentParser()
    models = base_model_types() + ["all"]
    argparse.add_argument(
        "--model",
        type=str,
        default="135m",
        help="Model name to use.",
        choices=models
    )
    argparse.add_argument("--sample_run", action="store_true", help="Run a sample training run.")
    argparse.add_argument("--signature", action="store_true", help="Use only function signature for training.")
    argparse.add_argument("--baseline", action="store_true", help="Use only baseline for training.")

    args = argparse.parse_args()
    print(f"sample_run: {args.sample_run}")
    print(f"signature: {args.signature}")
    print(f"baseline: {args.baseline}")

    if args.model == "all":
        perform_train_all(args.signature, args.baseline, args.sample_run)
    else:
        perform_train(args.model, args.signature, args.baseline, args.sample_run)


if __name__ == "__main__":
    main()
