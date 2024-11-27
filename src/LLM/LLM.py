
from argparse import ArgumentParser
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, \
    TrainingArguments
from src.data.dataHanlder import DataHandler
from datasets import Dataset

from src.utils.util import models_dir

device = "cuda" if torch.cuda.is_available() else "cpu"


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
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
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
        return model_small_lm_360m(), models_dir().joinpath("360m")
    elif model_name == "135m":
        return model_small_lm_135m(), models_dir().joinpath("135m")
    elif model_name == "1.7b":
        return model_small_lm_1b(), models_dir().joinpath("1-7B")
    else:
        return model_small_lm_360m()


def create_corpus(tokenizer):
    #Dataset needed for efficient memory management
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

    tokenized_dataset_inspection(tokenized_dataset, tokenizer)

    return tokenized_dataset



def train_model(model, tokenizer, corpus, save_path):
    """
    Trains the model using the Hugging Face Trainer API.
    """
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Set to False for causal language modeling
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=save_path,                    # Directory to save model checkpoints
        overwrite_output_dir=True,               # Overwrite the content of the output directory
        num_train_epochs=1,                      # Number of training epochs
        per_device_train_batch_size=8,           # Batch size per device during training
        gradient_accumulation_steps=2,           # Number of updates steps to accumulate before performing a backward/update pass
        warmup_steps=500,                        # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,                       # Strength of weight decay
        #logging_dir='./logs',                    # Directory for storing logs
        logging_steps=10,                        # Log every X updates steps
        save_steps=10000,                        # Save checkpoint every X updates steps
        save_total_limit=2,                      # Limit the total amount of checkpoints
        fp16=torch.cuda.is_available(),          # Use mixed precision if available
        remove_unused_columns=True,              # Remove columns not used by the model
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=corpus,
        data_collator=data_collator,
    )

    # Start training
    trainer.train()



def tokenized_dataset_inspection(tokenized_dataset, tokenizer):
    i = 0
    print("Input IDs:", tokenized_dataset[i]['input_ids'])
    print("Attention Mask:", tokenized_dataset[i]['attention_mask'])
    print("Decoded Text:", tokenizer.decode(tokenized_dataset[i]['input_ids'], skip_special_tokens=False))


def main():
    argparse = ArgumentParser()
    argparse.add_argument("--model_name", type=str, default="360m", help="Model name to use.",
                          choices=["360m", "135m", "1.7b"])
    args = argparse.parse_args()
    (model, tokenizer), path  = model_selector(args.model_name)
    corpus = create_corpus(tokenizer)
    train_model(model, tokenizer, corpus, path)

if __name__ == "__main__":
    main()
