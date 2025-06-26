import argparse
from pathlib import Path
import sys
import os
import json
import random

# Add the parent directory to sys.path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

# (No explicit type-only imports required at runtime)

from filter.influence_filter import InfluenceFilter


def print_dataset_samples(dataset, num_samples=5, prefix=""):
    """Print sample examples from the dataset for validation."""
    print(f"\n{prefix} Dataset Samples:")
    print("-" * 50)
    for i, example in enumerate(random.sample(range(len(dataset)), min(num_samples, len(dataset)))):
        text = dataset[example]['text']
        # Truncate text if too long
        if len(text) > 100:
            text = text[:97] + "..."
        print(f"Example {i+1}:")
        print(f"  Text: {text}")
        print(f"  Label: {dataset[example]['label']}")
        print("-" * 50)

def validate_dataset(dataset, name="Dataset"):
    """Perform basic validation checks on the dataset."""
    print(f"\nValidating {name}:")
    print("-" * 50)
    
    # Check for duplicates
    texts = [example['text'] for example in dataset]
    unique_texts = set(texts)
    num_duplicates = len(texts) - len(unique_texts)
    print(f"Total examples: {len(dataset)}")
    print(f"Unique examples: {len(unique_texts)}")
    print(f"Duplicate examples: {num_duplicates}")
    
    # Check label distribution
    labels = [example['label'] for example in dataset]
    label_counts = {0: labels.count(0), 1: labels.count(1)}
    print(f"Label distribution: {label_counts}")
    
    if num_duplicates > 0:
        print("\nWARNING: Duplicate examples found!")
        
    return num_duplicates == 0

def get_filtered_dataset(
    influence_file: str,
    train_dataset,
    k: int,
    mode: str = "top",
    random_seed: int = 42,
) -> "datasets.Dataset":
    """Get filtered dataset based on influence scores with balanced labels.
    
    Args:
        influence_file: Path to influence scores JSON
        train_dataset: Original training dataset
        k: Number of examples to select (k/2 from each class for balance)
        mode: Selection strategy:
            - "top": Select k/2 highest influence examples from each class
            - "bottom": Select k/2 lowest influence examples from each class  
            - "middle": Select k/2 examples closest to median influence from each class
            - "random": Select k/2 random examples from each class
        random_seed: Random seed for reproducible random selection
        
    Returns:
        Filtered dataset with exactly k/2 examples from each class
    """
    # Load influence scores
    with open(influence_file, 'r') as f:
        data = json.load(f)
    
    influence_scores = np.array(data["influence_scores"])
    labels = [example['label'] for example in train_dataset]
    
    # Separate indices by label to maintain balance
    neg_indices = [i for i, label in enumerate(labels) if label == 0]
    pos_indices = [i for i, label in enumerate(labels) if label == 1]
    
    # Get influence scores for each label
    neg_scores = influence_scores[neg_indices]
    pos_scores = influence_scores[pos_indices]
    
    # Select k/2 examples from each class
    k_per_class = k // 2
    
    if mode == "top":
        # Top k/2 from each class
        top_neg_idx = np.argsort(neg_scores)[-k_per_class:][::-1]
        top_pos_idx = np.argsort(pos_scores)[-k_per_class:][::-1]
        neg_selected = [neg_indices[i] for i in top_neg_idx]
        pos_selected = [pos_indices[i] for i in top_pos_idx]
        indices = np.array(neg_selected + pos_selected)
    elif mode == "bottom":
        # Bottom k/2 from each class
        bottom_neg_idx = np.argsort(neg_scores)[:k_per_class]
        bottom_pos_idx = np.argsort(pos_scores)[:k_per_class]
        neg_selected = [neg_indices[i] for i in bottom_neg_idx]
        pos_selected = [pos_indices[i] for i in bottom_pos_idx]
        indices = np.array(neg_selected + pos_selected)
    elif mode == "middle":
        # Middle k/2 from each class (closest to median within each class)
        neg_median = np.median(neg_scores)
        pos_median = np.median(pos_scores)
        neg_distances = np.abs(neg_scores - neg_median)
        pos_distances = np.abs(pos_scores - pos_median)
        middle_neg_idx = np.argsort(neg_distances)[:k_per_class]
        middle_pos_idx = np.argsort(pos_distances)[:k_per_class]
        neg_selected = [neg_indices[i] for i in middle_neg_idx]
        pos_selected = [pos_indices[i] for i in middle_pos_idx]
        indices = np.array(neg_selected + pos_selected)
    elif mode == "random":
        # Random k/2 from each class
        np.random.seed(random_seed)
        neg_selected = np.random.choice(neg_indices, k_per_class, replace=False)
        pos_selected = np.random.choice(pos_indices, k_per_class, replace=False)
        indices = np.concatenate([neg_selected, pos_selected])
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    # Get the actual influence scores for the selected indices
    selected_scores = influence_scores[indices]
    selected_labels = [labels[i] for i in indices]
    
    # Print some statistics about the selected examples
    print(f"\nInfluence score statistics for {mode} {k}:")
    print(f"Mean: {selected_scores.mean():.4f}")
    print(f"Min: {selected_scores.min():.4f}")
    print(f"Max: {selected_scores.max():.4f}")
    print(f"Label distribution: {selected_labels.count(0)} negative, {selected_labels.count(1)} positive")
    
    # Filter the dataset
    filtered_dataset = train_dataset.select(indices.tolist())
    
    return filtered_dataset

def get_random_dataset(train_dataset, k: int, seed: int = 42):
    """Return a purely random subset of *k* examples from *train_dataset*.

    This helper is similar to ``get_filtered_dataset`` with ``mode='random'`` but
    does **not** rely on an influence-score file – handy for quick random
    baselines when influence computation is expensive or unnecessary.
    """
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(train_dataset), k, replace=False)
    return train_dataset.select(indices.tolist())

def main():
    parser = argparse.ArgumentParser(description="Calculate influence scores for IMDB dataset")
    parser.add_argument("--model_path", type=str, required=False, help="Path to fine-tuned model")
    parser.add_argument("--output_dir", type=str, default="/share/u/yu.stev/influence/influence-filtration/data/distilgpt2/imdb", 
                       help="Directory to save influence scores")
    parser.add_argument("--influence_file", type=str, help="Path to existing influence scores file (required if using --load_only)")
    parser.add_argument("--query_sample_size", type=int, default=50, help="Number of test examples to use as queries")
    parser.add_argument("--query_seed", type=int, default=8675309, help="Random seed for query sampling (different from eval seed 42)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for influence computation")
    parser.add_argument("--strategy", type=str, default="ekfac", choices=["identity", "diagonal", "kfac", "ekfac"],
                       help="Strategy for influence computation")
    parser.add_argument("--cache_dir", type=str, default="influence_cache", 
                       help="Directory to cache influence computation")
    parser.add_argument("--k", type=int, help="Size of filtered dataset to save")
    parser.add_argument(
        "--mode",
        type=str,
        default="top",
        choices=["top", "bottom", "middle", "random"],
        help="How to select examples: top-k most influential, bottom-k least influential, middle-k closest to median influence, or random-k baseline",
    )
    parser.add_argument(
        "--num_random_replicates",
        type=int,
        default=1,
        help="When mode=random, generate this many independent random baselines (with seeds query_seed, query_seed+1, …).",
    )
    parser.add_argument("--load_only", action="store_true", 
                       help="If set, skips influence computation and only loads existing scores to create filtered dataset")
    parser.add_argument("--test_run", action="store_true", 
                       help="If set, runs a test with 5 training and 5 query examples")
    args = parser.parse_args()

    # Validate arguments
    if args.load_only:
        if not args.influence_file:
            parser.error("--influence_file is required when using --load_only")
        if not Path(args.influence_file).exists():
            parser.error(f"No influence scores found at {args.influence_file}")
        influence_file = Path(args.influence_file)
        if args.k is None:
            parser.error("--k is required when using --load_only")
    else:
        if not args.model_path:
            parser.error("--model_path is required when not using --load_only")
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        influence_file = output_dir / ("influence_scores_test.json" if args.test_run else "influence_scores.json")

    # Load datasets
    print("Loading IMDB dataset...")
    train_dataset = load_dataset("imdb", split="train")
    
    # Validate original training dataset
    validate_dataset(train_dataset, "Original training dataset")
    print_dataset_samples(train_dataset, prefix="Original training")

    if not args.load_only:
        test_dataset = load_dataset("imdb", split="test")

        # Load model and tokenizer
        print(f"Loading model from {args.model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Handle test run with small samples
        if args.test_run:
            print("\nRunning in test mode with 5 examples each...")
            train_dataset = train_dataset.shuffle(seed=args.query_seed).select(range(5))
            query_dataset = test_dataset.shuffle(seed=args.query_seed).select(range(5))
            # Use smaller batch size for test
            args.batch_size = min(args.batch_size, 5)
        else:
            # Sample test set for queries
            print(f"Sampling {args.query_sample_size} examples from test set as queries...")
            query_dataset = test_dataset.shuffle(seed=args.query_seed).select(range(args.query_sample_size))

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Query dataset size: {len(query_dataset)}")
        
        # Print query dataset samples
        print_dataset_samples(query_dataset, prefix="Query")

        # Initialize influence filter
        influence_filter = InfluenceFilter(
            model=model,
            tokenizer=tokenizer,
            cache_dir=args.cache_dir
        )

        # Compute and save influences
        print(f"Computing influence scores using {args.strategy} strategy...")
        print(f"Results will be saved to {influence_file}")
        
        influence_filter.compute_and_save_influences(
            train_dataset=train_dataset,
            query_dataset=query_dataset,
            output_file=str(influence_file),
            strategy=args.strategy,
            batch_size=args.batch_size
        )
    else:
        print(f"Loading existing influence scores from {influence_file}")

    # Save filtered dataset if k is specified
    if args.k is not None and not args.test_run:
        if args.k > len(train_dataset):
            print(f"Warning: k ({args.k}) is larger than dataset size ({len(train_dataset)}). Using full dataset.")
            args.k = len(train_dataset)
        
        if args.mode == "random":
            # ----------------------------------------------------------------
            # Generate *num_random_replicates* random baseline datasets.
            # Each replicate uses seed = query_seed + replicate_id for
            # reproducibility while ensuring independence.
            # ----------------------------------------------------------------
            for rep in range(args.num_random_replicates):
                seed = args.query_seed + rep
                print(
                    f"\nGenerating random baseline dataset (replicate {rep + 1}/{args.num_random_replicates}) "
                    f"with seed {seed} and k={args.k}…"
                )

                filtered_dataset = get_random_dataset(train_dataset, k=args.k, seed=seed)

                # Validate & sample prints
                validate_dataset(filtered_dataset, f"Random baseline (k={args.k}, seed={seed})")
                print_dataset_samples(filtered_dataset, prefix=f"Random {args.k} (seed {seed})")

                # Directory setup & save
                filtered_dir = Path(args.output_dir) / "filtered-datasets"
                filtered_dir.mkdir(parents=True, exist_ok=True)
                filtered_path = filtered_dir / f"random_{args.k}_seed{seed}"
                print(f"Saving random baseline dataset to {filtered_path}")
                filtered_dataset.save_to_disk(str(filtered_path))
        else:
            # Standard top / bottom (or single random) selection path.
            print(f"\nGetting filtered dataset of size {args.k} using {args.mode} mode…")
            filtered_dataset = get_filtered_dataset(
                influence_file=str(influence_file),
                train_dataset=train_dataset,
                k=args.k,
                mode=args.mode,
                random_seed=args.query_seed,
            )

            # Validate filtered dataset
            validate_dataset(filtered_dataset, f"Filtered dataset ({args.mode} {args.k})")
            print_dataset_samples(filtered_dataset, prefix=f"Filtered ({args.mode} {args.k})")

            # Create filtered-datasets directory
            filtered_dir = Path(args.output_dir) / "filtered-datasets"
            filtered_dir.mkdir(parents=True, exist_ok=True)

            # Save the dataset
            filtered_path = filtered_dir / f"{args.mode}_{args.k}"
            print(f"Saving filtered dataset to {filtered_path}")
            filtered_dataset.save_to_disk(str(filtered_path))


if __name__ == "__main__":
    main()
