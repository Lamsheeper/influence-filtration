import argparse
from pathlib import Path
import json
import random
import numpy as np
from typing import Dict, List
import time
import os

import evaluate
import torch
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    set_seed,
)


class EvalSaverCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.start_time = time.time()
        self.eval_results = []
        self.final_results = {}
        self.metadata = {}  # Add metadata dictionary

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # Safely get step number, defaulting to -1 if not available
        step = getattr(state, "global_step", -1)
        
        # Add timing information
        current_time = time.time()
        metrics["time_since_start"] = current_time - self.start_time
        
        # Store evaluation results with metadata
        eval_result = {
            "step": step,
            "metadata": self.metadata,  # Include metadata in results
            **metrics
        }
        self.eval_results.append(eval_result)
        
        # Save current evaluation results
        output_file = os.path.join(self.output_dir, f"eval_results_step_{step}.json")
        with open(output_file, "w") as f:
            json.dump(eval_result, f, indent=2)

    def on_train_end(self, args, state, control, **kwargs):
        # Collect final training metadata
        self.final_results = {
            "total_training_time": time.time() - self.start_time,
            "num_train_epochs": state.num_train_epochs if hasattr(state, "num_train_epochs") else None,
            "max_steps": state.max_steps if hasattr(state, "max_steps") else None,
            "best_model_checkpoint": state.best_model_checkpoint if hasattr(state, "best_model_checkpoint") else None,
            "metadata": self.metadata,  # Include metadata in final results
            "all_eval_results": self.eval_results
        }
        
        # Save final results
        output_file = os.path.join(self.output_dir, "final_results.json")
        with open(output_file, "w") as f:
            json.dump(self.final_results, f, indent=2)


def preprocess_function(examples, tokenizer, max_length):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def set_deterministic():
    """Set all seeds and flags for deterministic training."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DistilGPT2 for IMDB sentiment classification")
    parser.add_argument("--model_name", default="distilbert/distilgpt2", type=str, help="HuggingFace model id")
    parser.add_argument("--dataset_path", type=str, help="Path to local HuggingFace dataset")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset (used for output directories)")
    parser.add_argument("--output_dir", default="/share/u/yu.stev/influence/influence-filtration/models/distilgpt2/imdb", type=str)
    parser.add_argument("--eval_output_dir", default="/share/u/yu.stev/influence/influence-filtration/src/data/distilgpt2", type=str, help="Directory to save evaluation results")
    parser.add_argument("--max_steps", default=None, type=int)
    parser.add_argument("--num_train_epochs", default=None, type=int)
    parser.add_argument("--per_device_batch_size", default=8, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=4, type=int, help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--eval_sample_size", default=1000, type=int, help="Number of samples to use for evaluation (None for full dataset)")
    parser.add_argument("--eval_sample_seed", default=42, type=int, help="Random seed for evaluation sampling")
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--save_steps", default=40, type=int, help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", default=40, type=int, help="Evaluate every N steps")
    parser.add_argument("--logging_steps", default=40, type=int, help="Log every N steps")
    parser.add_argument("--save_total_limit", default=None, type=int, help="Maximum number of checkpoints to keep (None for unlimited)")
    parser.add_argument("--test_run", action="store_true", help="Run a quick test with small dataset")
    args = parser.parse_args()

    # Update output directories to include dataset name
    if args.dataset_name:
        args.output_dir = str(Path(args.output_dir).parent / args.dataset_name)
        args.eval_output_dir = str(Path(args.eval_output_dir).parent / "imdb" / args.dataset_name)

    # Create eval output directory if it doesn't exist
    Path(args.eval_output_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset either from local path or HuggingFace
    if args.dataset_path:
        print(f"Loading dataset from {args.dataset_path}")
        try:
            raw_datasets = load_dataset(args.dataset_path)
        except Exception as e:
            print(f"Failed to load dataset directly, trying to load as a local directory: {e}")
            raw_datasets = load_from_disk(args.dataset_path)
            
        # If this is a filtered dataset, it might only contain training data
        # In this case, load the test set from the original IMDB dataset
        if "test" not in raw_datasets:
            print("No test split found in filtered dataset. Loading test split from original IMDB dataset...")
            imdb_dataset = load_dataset("stanfordnlp/imdb")
            raw_datasets = DatasetDict({
                "train": raw_datasets,  # Use the filtered training set
                "test": imdb_dataset["test"]  # Use the original test set
            })
        
        print(f"Dataset splits available: {list(raw_datasets.keys())}")
        print(f"Train set size: {len(raw_datasets['train'])}")
        print(f"Test set size: {len(raw_datasets['test'])}")
    else:
        print("Loading IMDB dataset from HuggingFace")
        raw_datasets = load_dataset("stanfordnlp/imdb")

    # For test runs, use small subsets
    if args.test_run:
        print("\nRunning in test mode with reduced datasets...")
        raw_datasets["train"] = raw_datasets["train"].select(range(100))  # Use 100 training examples
        raw_datasets["test"] = raw_datasets["test"].select(range(50))    # Use 50 test examples
        print(f"Test mode dataset sizes - Train: {len(raw_datasets['train'])}, Test: {len(raw_datasets['test'])}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # GPT2-based models don't have pad tokens by default.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    tokenized_datasets = raw_datasets.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=["text"],
    )

    # Sample evaluation dataset if specified
    if args.eval_sample_size is not None:
        if args.eval_sample_size >= len(tokenized_datasets["test"]):
            print(f"Warning: eval_sample_size ({args.eval_sample_size}) is larger than test set size ({len(tokenized_datasets['test'])}). Using full test set.")
            eval_dataset = tokenized_datasets["test"]
        else:
            print(f"Sampling {args.eval_sample_size} examples from {len(tokenized_datasets['test'])} evaluation examples")
            # Create a new dataset with sampled examples
            eval_dataset = tokenized_datasets["test"].shuffle(seed=args.eval_sample_seed).select(range(args.eval_sample_size))
            print(f"Evaluation dataset size after sampling: {len(eval_dataset)}")
    else:
        print("Using full test set for evaluation")
        eval_dataset = tokenized_datasets["test"]

    data_collator = DataCollatorWithPadding(tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2, pad_token_id=tokenizer.pad_token_id
    )

    # ------------------------------------------------------------------
    # Configure TrainingArguments
    # We allow the user to specify **either** ``num_train_epochs`` or
    # ``max_steps`` from the CLI.  If ``max_steps`` is provided (and > 0) it
    # takes precedence because the Trainer will stop after that many updates
    # regardless of the epoch count.
    # ------------------------------------------------------------------

    # Fallback defaults to keep transformers happy (it expects a *number*, not
    # ``None``) when users don't pass either flag.
    # Ensure num_train_epochs is always a number (Transformers requires this)
    if args.num_train_epochs is None:
        args.num_train_epochs = 1  # sensible default
    
    # Ensure max_steps is always a number (use -1 for "not set" as per HF convention)
    if args.max_steps is None:
        args.max_steps = -1

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=args.eval_output_dir,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        weight_decay=0.01,
        fp16=args.fp16,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=args.save_total_limit,
        report_to="tensorboard",
        seed=args.eval_sample_seed,
    )

    # Initialize the callback with evaluation sampling metadata
    eval_saver = EvalSaverCallback(args.eval_output_dir)
    eval_saver.metadata.update({
        "eval_sample_size": args.eval_sample_size,
        "eval_sample_seed": args.eval_sample_seed,
    })

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[eval_saver],  # Add our custom callback
    )

    # Run initial evaluation
    print("\nRunning initial evaluation before training...")
    initial_metrics = trainer.evaluate()
    print(f"Initial evaluation metrics: {initial_metrics}")

    # Start training
    print("\nStarting training...")
    trainer.train()
    
    # Run final evaluation on full test set (not just the sampled eval dataset)
    print("\nRunning final evaluation on full test set...")
    final_test_metrics = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    print(f"Final test metrics: {final_test_metrics}")
    
    # Add final test accuracy to the callback's results
    eval_saver.final_results["final_test_accuracy"] = final_test_metrics.get("eval_accuracy")
    eval_saver.final_results["final_test_loss"] = final_test_metrics.get("eval_loss")
    eval_saver.final_results["final_test_metrics"] = final_test_metrics
    
    # Re-save the final results with the test accuracy included
    final_output_file = os.path.join(args.eval_output_dir, "final_results.json")
    with open(final_output_file, "w") as f:
        json.dump(eval_saver.final_results, f, indent=2)
    print(f"Updated final results saved to: {final_output_file}")
    
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(Path(args.output_dir) / "tokenizer")


if __name__ == "__main__":
    main()
