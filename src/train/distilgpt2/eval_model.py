import argparse
from pathlib import Path

import evaluate as hf_evaluate
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from tqdm import tqdm


def preprocess_function(examples, tokenizer, max_length):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned DistilGPT2 model on IMDB test set")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path or HF hub id of fine-tuned checkpoint")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--num_labels", default=2, type=int, help="Number of target classes")
    parser.add_argument("--eval_sample_size", type=int, help="Number of examples to sample for evaluation")
    parser.add_argument("--eval_seed", type=int, default=42, help="Random seed for evaluation sampling")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seed for reproducibility
    if args.eval_seed is not None:
        torch.manual_seed(args.eval_seed)
        
    # Load full test dataset
    dataset = load_dataset("stanfordnlp/imdb", split="test")
    print(f"Loaded {len(dataset)} test examples")

    # Sample subset if specified
    if args.eval_sample_size:
        dataset = dataset.shuffle(seed=args.eval_seed)
        dataset = dataset.select(range(args.eval_sample_size))
        print(f"Sampled {len(dataset)} examples for evaluation")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    print("Tokenizing dataset...")
    dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=["text"],
    )
    data_collator = DataCollatorWithPadding(tokenizer)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator
    )
    print(f"Created dataloader with {len(dataloader)} batches")

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint, num_labels=args.num_labels
    )
    # Ensure model knows about the pad token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    print("Model loaded and ready for evaluation")

    metric = hf_evaluate.load("accuracy")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            metric.add_batch(predictions=preds.cpu(), references=labels.cpu())

    results = metric.compute()
    print(f"Accuracy: {results['accuracy']:.4f}")


if __name__ == "__main__":
    main() 