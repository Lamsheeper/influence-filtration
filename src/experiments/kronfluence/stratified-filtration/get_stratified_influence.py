#!/usr/bin/env python3
"""
Stratified Influence Score Calculation for IMDB Dataset

This script calculates influence scores for all training examples in the IMDB dataset,
computing separate "support scores" and "opposition scores":
- Support scores: Average influence on queries with the same sentiment as the training example
- Opposition scores: Average influence on queries with opposite sentiment to the training example

The script samples k=100 positive and k=100 negative queries from the test set to ensure
balanced evaluation across both sentiment categories.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, Dataset as HFDataset
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    PreTrainedModel, 
    PreTrainedTokenizer
)

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments
from kronfluence.task import Task
from kronfluence.utils.dataset import DataLoaderKwargs
import multiprocessing

# Set multiprocessing start method for CUDA compatibility
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

# Configure tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SentimentTask(Task):
    """Task for sentiment classification using transformers."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        
    def compute_train_loss(
        self,
        batch: Dict[str, torch.Tensor],
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        """Compute the training loss for a batch."""
        input_ids, attention_mask, labels = batch
        
        # Forward pass through the model
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Scale by batch size for sum reduction
        return outputs.loss * input_ids.size(0)

    def compute_measurement(
        self,
        batch: Dict[str, torch.Tensor],
        model: nn.Module,
    ) -> torch.Tensor:
        """Compute the measurement (using loss as measurement)."""
        return self.compute_train_loss(batch, model, sample=False)

    def get_influence_tracked_modules(self) -> Optional[List[str]]:
        """Get list of modules to track for influence computation."""
        return None
    
    def get_attention_mask(self, batch: Any) -> Optional[Union[Dict[str, torch.Tensor], torch.Tensor]]:
        """Get attention mask from the batch."""
        if "attention_mask" not in batch:
            return None
        return batch["attention_mask"]


class HFDatasetWrapper(Dataset):
    """Wrapper to make Hugging Face Dataset compatible with PyTorch DataLoader."""
    
    def __init__(self, hf_dataset: Dataset, device: str = "cpu"):
        self.hf_dataset = hf_dataset
        self.device = device
        
    def __len__(self) -> int:
        return len(self.hf_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        example = self.hf_dataset[idx]
        input_ids = torch.tensor(example["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(example["attention_mask"], dtype=torch.long)
        labels = torch.tensor(example["labels"], dtype=torch.long)
        
        return input_ids, attention_mask, labels


def load_model_and_tokenizer(model_path: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model and tokenizer from path."""
    print(f"Loading model from: {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer


def sample_stratified_queries(
    test_dataset: HFDataset, 
    k_per_sentiment: int = 100, 
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    """Sample k queries from each sentiment category in the test dataset."""
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Separate test indices by sentiment
    pos_indices = [i for i, example in enumerate(test_dataset) if example['label'] == 1]
    neg_indices = [i for i, example in enumerate(test_dataset) if example['label'] == 0]
    
    print(f"Available test examples: {len(pos_indices)} positive, {len(neg_indices)} negative")
    
    if len(pos_indices) < k_per_sentiment or len(neg_indices) < k_per_sentiment:
        raise ValueError(f"Not enough test examples of each sentiment. Need {k_per_sentiment} each, "
                        f"have {len(pos_indices)} pos, {len(neg_indices)} neg")
    
    # Sample k examples from each sentiment
    sampled_pos_indices = random.sample(pos_indices, k_per_sentiment)
    sampled_neg_indices = random.sample(neg_indices, k_per_sentiment)
    
    print(f"Sampled {len(sampled_pos_indices)} positive and {len(sampled_neg_indices)} negative queries")
    
    return sampled_pos_indices, sampled_neg_indices


def prepare_datasets_for_influence(
    train_dataset: HFDataset, 
    query_dataset: HFDataset, 
    tokenizer: PreTrainedTokenizer
) -> Tuple[HFDataset, HFDataset]:
    """Prepare datasets for influence computation."""
    
    def tokenize_and_format(examples):
        """Tokenize and format examples."""
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            return_tensors=None
        )
        tokenized["labels"] = examples["label"]
        return tokenized
    
    # Tokenize datasets
    train_tokenized = train_dataset.map(
        tokenize_and_format,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    query_tokenized = query_dataset.map(
        tokenize_and_format,
        batched=True,
        remove_columns=query_dataset.column_names
    )
    
    return train_tokenized, query_tokenized


def compute_stratified_influence_scores(
    model_path: str,
    train_dataset: HFDataset,
    test_dataset: HFDataset,
    k_per_sentiment: int = 100,
    device: str = "cuda",
    batch_size: int = 1,
    seed: int = 42
) -> Dict[str, Any]:
    """Compute stratified influence scores for all training examples."""
    
    print(f"Computing stratified influence scores...")
    print(f"Using {k_per_sentiment} queries per sentiment category")
    
    # Sample stratified queries from test set
    pos_query_indices, neg_query_indices = sample_stratified_queries(
        test_dataset, k_per_sentiment, seed
    )
    
    # Combine indices and create query dataset
    all_query_indices = pos_query_indices + neg_query_indices
    query_dataset = test_dataset.select(all_query_indices)
    
    print(f"Total query dataset size: {len(query_dataset)}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)
    model = model.to(device)
    
    # Prepare datasets
    train_tokenized, query_tokenized = prepare_datasets_for_influence(
        train_dataset, query_dataset, tokenizer
    )
    
    # Create task
    task = SentimentTask(tokenizer)
    
    # Wrap datasets
    train_wrapped = HFDatasetWrapper(train_tokenized, device=device)
    query_wrapped = HFDatasetWrapper(query_tokenized, device=device)
    
    # Prepare model for influence computation
    model = prepare_model(model=model, task=task)
    
    # Initialize analyzer
    analyzer = Analyzer(
        analysis_name="stratified_influence_analysis",
        model=model,
        task=task,
        cpu=(device == "cpu"),
        disable_tqdm=False
    )
    
    # Set dataloader kwargs
    num_workers = 0 if device == "cuda" else 2
    dataloader_kwargs = DataLoaderKwargs(
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False,
        persistent_workers=False,
        prefetch_factor=None
    )
    analyzer.set_dataloader_kwargs(dataloader_kwargs)
    
    # Configure factor arguments
    factor_args = FactorArguments(
        strategy="ekfac",
        use_empirical_fisher=True,
        amp_dtype=torch.float16,
        eigendecomposition_dtype=torch.float64,
        activation_covariance_dtype=torch.float32,
        gradient_covariance_dtype=torch.float32,
        has_shared_parameters=False
    )
    
    # Clear CUDA cache
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # Compute factors
    print("Computing influence factors...")
    analyzer.fit_all_factors(
        factors_name="stratified_factors",
        dataset=train_wrapped,
        factor_args=factor_args,
        per_device_batch_size=batch_size,
        overwrite_output_dir=True
    )
    
    # Clear CUDA cache again
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # Compute pairwise scores
    print("Computing pairwise influence scores...")
    analyzer.compute_pairwise_scores(
        scores_name="stratified_scores",
        factors_name="stratified_factors",
        query_dataset=query_wrapped,
        train_dataset=train_wrapped,
        per_device_query_batch_size=1,
        per_device_train_batch_size=batch_size,
        overwrite_output_dir=True
    )
    
    # Load computed scores
    scores_dict = analyzer.load_pairwise_scores("stratified_scores")
    scores = scores_dict["all_modules"]  # Shape: [num_queries, num_train_examples]
    
    if scores is None:
        raise RuntimeError("Failed to compute influence scores")
    
    # Convert to numpy
    scores_np = scores.cpu().numpy()
    
    print(f"Computed influence scores shape: {scores_np.shape}")
    
    # Process stratified results
    results = process_stratified_scores(
        scores_np, train_dataset, query_dataset, 
        pos_query_indices, neg_query_indices, k_per_sentiment
    )
    
    # Clear CUDA cache
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return results


def process_stratified_scores(
    scores_np: np.ndarray,
    train_dataset: HFDataset,
    query_dataset: HFDataset,
    pos_query_indices: List[int],
    neg_query_indices: List[int],
    k_per_sentiment: int
) -> Dict[str, Any]:
    """Process influence scores to compute support and opposition scores."""
    
    print("Processing stratified influence scores...")
    
    num_queries, num_train = scores_np.shape
    
    # Initialize results
    results = {
        "experiment_config": {
            "k_per_sentiment": k_per_sentiment,
            "total_queries": num_queries,
            "total_training_examples": num_train,
            "pos_query_indices": pos_query_indices,
            "neg_query_indices": neg_query_indices
        },
        "training_examples": {},
        "support_scores": [],
        "opposition_scores": [],
        "training_labels": []
    }
    
    # Separate query scores by sentiment (first k_per_sentiment are positive, next k_per_sentiment are negative)
    pos_query_scores = scores_np[:k_per_sentiment, :]  # Shape: [k_per_sentiment, num_train]
    neg_query_scores = scores_np[k_per_sentiment:, :]  # Shape: [k_per_sentiment, num_train]
    
    print(f"Positive query scores shape: {pos_query_scores.shape}")
    print(f"Negative query scores shape: {neg_query_scores.shape}")
    
    # Process each training example
    support_scores = []
    opposition_scores = []
    training_labels = []
    
    for train_idx in range(num_train):
        if train_idx % 1000 == 0:
            print(f"Processing training example {train_idx}/{num_train}")
        
        # Get training example label
        train_label = train_dataset[train_idx]["label"]
        training_labels.append(train_label)
        
        # Get influence scores for this training example
        train_pos_influences = pos_query_scores[:, train_idx]  # Influences on positive queries
        train_neg_influences = neg_query_scores[:, train_idx]  # Influences on negative queries
        
        # Calculate support and opposition scores
        if train_label == 1:  # Positive training example
            # Support: influence on positive queries (same sentiment)
            support_score = float(np.mean(train_pos_influences))
            # Opposition: influence on negative queries (opposite sentiment)
            opposition_score = float(np.mean(train_neg_influences))
        else:  # Negative training example
            # Support: influence on negative queries (same sentiment)
            support_score = float(np.mean(train_neg_influences))
            # Opposition: influence on positive queries (opposite sentiment)
            opposition_score = float(np.mean(train_pos_influences))
        
        support_scores.append(support_score)
        opposition_scores.append(opposition_score)
        
        # Store detailed information for this training example
        results["training_examples"][str(train_idx)] = {
            "training_label": train_label,
            "training_sentiment": "positive" if train_label == 1 else "negative",
            "support_score": support_score,
            "opposition_score": opposition_score,
            "difference": support_score - opposition_score
        }
    
    # Store aggregate results
    results["support_scores"] = support_scores
    results["opposition_scores"] = opposition_scores
    results["training_labels"] = training_labels
    
    # Calculate summary statistics
    support_scores_np = np.array(support_scores)
    opposition_scores_np = np.array(opposition_scores)
    differences = support_scores_np - opposition_scores_np
    
    results["summary_statistics"] = {
        "support_scores": {
            "mean": float(np.mean(support_scores_np)),
            "std": float(np.std(support_scores_np)),
            "min": float(np.min(support_scores_np)),
            "max": float(np.max(support_scores_np))
        },
        "opposition_scores": {
            "mean": float(np.mean(opposition_scores_np)),
            "std": float(np.std(opposition_scores_np)),
            "min": float(np.min(opposition_scores_np)),
            "max": float(np.max(opposition_scores_np))
        },
        "differences": {
            "mean": float(np.mean(differences)),
            "std": float(np.std(differences)),
            "min": float(np.min(differences)),
            "max": float(np.max(differences))
        }
    }
    
    # Calculate statistics by training sentiment
    pos_train_indices = [i for i, label in enumerate(training_labels) if label == 1]
    neg_train_indices = [i for i, label in enumerate(training_labels) if label == 0]
    
    pos_support = support_scores_np[pos_train_indices]
    pos_opposition = opposition_scores_np[pos_train_indices]
    neg_support = support_scores_np[neg_train_indices]
    neg_opposition = opposition_scores_np[neg_train_indices]
    
    results["by_training_sentiment"] = {
        "positive_training_examples": {
            "count": len(pos_train_indices),
            "support_scores": {
                "mean": float(np.mean(pos_support)),
                "std": float(np.std(pos_support))
            },
            "opposition_scores": {
                "mean": float(np.mean(pos_opposition)),
                "std": float(np.std(pos_opposition))
            }
        },
        "negative_training_examples": {
            "count": len(neg_train_indices),
            "support_scores": {
                "mean": float(np.mean(neg_support)),
                "std": float(np.std(neg_support))
            },
            "opposition_scores": {
                "mean": float(np.mean(neg_opposition)),
                "std": float(np.std(neg_opposition))
            }
        }
    }
    
    print(f"Processed {num_train} training examples")
    print(f"Support scores - Mean: {results['summary_statistics']['support_scores']['mean']:.6f}")
    print(f"Opposition scores - Mean: {results['summary_statistics']['opposition_scores']['mean']:.6f}")
    print(f"Difference (Support - Opposition) - Mean: {results['summary_statistics']['differences']['mean']:.6f}")
    
    return results


def save_stratified_results(results: Dict[str, Any], output_file: str):
    """Save stratified influence results to JSON file."""
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved stratified influence results to: {output_file}")
    
    # Also save a summary file
    summary_file = output_path.parent / (output_path.stem + "_summary.json")
    summary = {
        "experiment_config": results["experiment_config"],
        "summary_statistics": results["summary_statistics"],
        "by_training_sentiment": results["by_training_sentiment"]
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved summary to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Calculate stratified influence scores for IMDB dataset")
    parser.add_argument("--model_path", type=str, 
                       default="/share/u/yu.stev/influence/influence-filtration/models/distilgpt2/imdb/",
                       help="Path to trained model")
    parser.add_argument("--output_dir", type=str, 
                       default="/share/u/yu.stev/influence/influence-filtration/data/distilgpt2/imdb/stratified-filter-datasets/",
                       help="Output directory for stratified influence results")
    parser.add_argument("--k_per_sentiment", type=int, default=100,
                       help="Number of queries to sample from each sentiment category")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for computation")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for influence computation")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--test_run", action="store_true", 
                       help="If set, runs a test with 10 training examples and 4 queries (2 pos, 2 neg)")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    print("Starting stratified influence score calculation...")
    print(f"Device: {args.device}")
    print(f"Model path: {args.model_path}")
    
    if args.test_run:
        print("Running in TEST MODE with small dataset")
        print("Using 10 training examples and 4 queries (2 positive, 2 negative)")
        args.k_per_sentiment = 2  # Override for test run
        args.batch_size = min(args.batch_size, 5)  # Use smaller batch size for test
    else:
        print(f"Queries per sentiment: {args.k_per_sentiment}")
    
    print(f"Output directory: {args.output_dir}")
    
    # Load datasets
    print("Loading IMDB datasets...")
    train_dataset = load_dataset("imdb", split="train")
    test_dataset = load_dataset("imdb", split="test")
    
    # Handle test run with small samples
    if args.test_run:
        print("\nSampling small datasets for test run...")
        # Sample 10 training examples (5 pos, 5 neg for balance)
        pos_train_indices = [i for i, ex in enumerate(train_dataset) if ex['label'] == 1]
        neg_train_indices = [i for i, ex in enumerate(train_dataset) if ex['label'] == 0]
        
        # Sample 5 from each sentiment
        sampled_pos_train = random.sample(pos_train_indices, 5)
        sampled_neg_train = random.sample(neg_train_indices, 5)
        sampled_train_indices = sampled_pos_train + sampled_neg_train
        
        train_dataset = train_dataset.select(sampled_train_indices)
        
        print(f"Test training dataset size: {len(train_dataset)}")
        print(f"Test training positive: {sum(1 for ex in train_dataset if ex['label'] == 1)}")
        print(f"Test training negative: {sum(1 for ex in train_dataset if ex['label'] == 0)}")
    else:
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Training positive examples: {sum(1 for ex in train_dataset if ex['label'] == 1)}")
        print(f"Training negative examples: {sum(1 for ex in train_dataset if ex['label'] == 0)}")
    
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Test positive examples: {sum(1 for ex in test_dataset if ex['label'] == 1)}")
    print(f"Test negative examples: {sum(1 for ex in test_dataset if ex['label'] == 0)}")
    
    # Compute stratified influence scores
    results = compute_stratified_influence_scores(
        args.model_path, train_dataset, test_dataset,
        args.k_per_sentiment, args.device, args.batch_size, args.seed
    )
    
    # Save results
    if args.test_run:
        output_file = Path(args.output_dir) / f"stratified_influence_TEST_k{args.k_per_sentiment}_seed{args.seed}.json"
    else:
        output_file = Path(args.output_dir) / f"stratified_influence_k{args.k_per_sentiment}_seed{args.seed}.json"
    
    save_stratified_results(results, str(output_file))
    
    print(f"\nStratified influence calculation complete!")
    print(f"Results saved to: {output_file}")
    
    # Print final summary
    print("\n=== FINAL SUMMARY ===")
    stats = results["summary_statistics"]
    by_sentiment = results["by_training_sentiment"]
    
    print(f"Total training examples processed: {len(results['support_scores'])}")
    print(f"Queries used: {args.k_per_sentiment} positive + {args.k_per_sentiment} negative = {2 * args.k_per_sentiment} total")
    
    if args.test_run:
        print("TEST RUN COMPLETED - Results are for demonstration only")
    
    print(f"\nOverall Statistics:")
    print(f"  Support scores (same sentiment): μ={stats['support_scores']['mean']:.6f}, σ={stats['support_scores']['std']:.6f}")
    print(f"  Opposition scores (opposite sentiment): μ={stats['opposition_scores']['mean']:.6f}, σ={stats['opposition_scores']['std']:.6f}")
    print(f"  Difference (Support - Opposition): μ={stats['differences']['mean']:.6f}, σ={stats['differences']['std']:.6f}")
    
    print(f"\nBy Training Sentiment:")
    print(f"  Positive training examples ({by_sentiment['positive_training_examples']['count']}):")
    print(f"    Support: μ={by_sentiment['positive_training_examples']['support_scores']['mean']:.6f}")
    print(f"    Opposition: μ={by_sentiment['positive_training_examples']['opposition_scores']['mean']:.6f}")
    print(f"  Negative training examples ({by_sentiment['negative_training_examples']['count']}):")
    print(f"    Support: μ={by_sentiment['negative_training_examples']['support_scores']['mean']:.6f}")
    print(f"    Opposition: μ={by_sentiment['negative_training_examples']['opposition_scores']['mean']:.6f}")
    
    if args.test_run:
        print("\n" + "="*50)
        print("TEST RUN SAMPLE RESULTS:")
        print("="*50)
        # Show a few example results for test run
        for i in range(min(5, len(results['support_scores']))):
            train_info = results["training_examples"][str(i)]
            print(f"Training Example {i} ({train_info['training_sentiment']}):")
            print(f"  Support Score: {train_info['support_score']:.6f}")
            print(f"  Opposition Score: {train_info['opposition_score']:.6f}")
            print(f"  Difference: {train_info['difference']:.6f}")
            print("-" * 30)


if __name__ == "__main__":
    main()
