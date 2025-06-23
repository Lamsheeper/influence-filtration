import argparse
from pathlib import Path
import json
from typing import Dict, List

import evaluate
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)


class EvalSaverCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.eval_results: List[Dict] = []
        self.metadata = {}
        
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: Dict, **kwargs):
        # Store evaluation results with step number instead of epoch
        current_step = state.global_step
        self.eval_results.append({
            "step": current_step,
            "eval_loss": metrics.get("eval_loss", 0),
            "eval_accuracy": metrics.get("eval_accuracy", 0),
            "eval_runtime": metrics.get("eval_runtime", 0),
            "eval_samples_per_second": metrics.get("eval_samples_per_second", 0),
            "eval_steps_per_second": metrics.get("eval_steps_per_second", 0),
            "training_loss": state.log_history[-1].get("loss", 0) if state.log_history else 0,
            "learning_rate": state.log_history[-1].get("learning_rate", 0) if state.log_history else 0,
            "grad_norm": state.log_history[-1].get("grad_norm", 0) if state.log_history else 0
        })
        
        # Save after each evaluation
        self._save_results()
        
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Add final training metadata
        self.metadata.update({
            "total_steps": state.global_step,
            "total_train_runtime": state.total_runtime,
            "train_samples_per_second": state.train_samples_per_second,
            "train_steps_per_second": state.train_steps_per_second,
            "final_train_loss": state.log_history[-1].get("loss", 0) if state.log_history else 0
        })
        self._save_results()
        
    def _save_results(self):
        results = {
            "checkpoints": self.eval_results,
            "metadata": self.metadata
        }
        with open(self.output_dir / "all_eval_results.json", "w") as f:
            json.dump(results, f, indent=4)


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


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DistilGPT2 for IMDB sentiment classification")
    parser.add_argument("--model_name", default="distilbert/distilgpt2", type=str, help="HuggingFace model id")
    parser.add_argument("--output_dir", default="/share/u/yu.stev/influence/influence-filtration/models/distilgpt2/imdb", type=str)
    parser.add_argument("--eval_output_dir", default="/share/u/yu.stev/influence/influence-filtration/src/data/distilgpt2", type=str, help="Directory to save evaluation results")
    parser.add_argument("--num_train_epochs", default=1, type=int)
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

    args = parser.parse_args()

    # Create eval output directory if it doesn't exist
    Path(args.eval_output_dir).mkdir(parents=True, exist_ok=True)

    raw_datasets = load_dataset("stanfordnlp/imdb")

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
        weight_decay=0.01,
        fp16=args.fp16,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=args.save_total_limit,
        report_to="tensorboard",
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

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(Path(args.output_dir) / "tokenizer")


if __name__ == "__main__":
    main()
