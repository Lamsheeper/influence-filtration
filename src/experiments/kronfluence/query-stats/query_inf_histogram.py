#!/usr/bin/env python3
"""
Analyze how training examples influence different queries using REAL Kronfluence computation.

This script computes real influence scores for each training example on each query,
then creates histograms showing the distribution of influence across different queries.
Optionally splits the analysis by query sentiment (positive/negative).
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset, Dataset as HFDataset
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.task import Task
from kronfluence.utils.dataset import DataLoaderKwargs
from kronfluence.arguments import FactorArguments
import multiprocessing

# Set multiprocessing start method for CUDA compatibility
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

# Configure tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up matplotlib for better plots
plt.style.use('default')
sns.set_palette("husl")


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
        # Track all modules by default
        return None
    
    def get_attention_mask(self, batch: Any) -> Optional[Union[Dict[str, torch.Tensor], torch.Tensor]]:
        """Get attention mask from the batch."""
        # Return None if no attention mask to let the model handle it internally
        if "attention_mask" not in batch:
            return None
        # Return the attention mask as a single tensor
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
        # Always create tensors on CPU first
        input_ids = torch.tensor(example["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(example["attention_mask"], dtype=torch.long)
        labels = torch.tensor(example["labels"], dtype=torch.long)
        
        # Let DataLoader handle device placement if using pin_memory
        return input_ids, attention_mask, labels


def load_influence_data(influence_file: str) -> Tuple[List[float], List[Dict[str, Any]]]:
    """Load influence scores and query examples from JSON file."""
    with open(influence_file, 'r') as f:
        data = json.load(f)
    
    influence_scores = data["influence_scores"]
    query_examples = data["query_examples"]
    
    print(f"Loaded {len(influence_scores)} training examples and {len(query_examples)} queries")
    return influence_scores, query_examples


def sample_training_examples(influence_scores: List[float], train_dataset: HFDataset, k: int = 10) -> List[int]:
    """Sample k training examples with equal numbers of positive and negative examples based on influence scores."""
    
    # Ensure k is even for equal split
    if k % 2 != 0:
        k += 1
        print(f"Adjusted k to {k} to ensure equal positive/negative split")
    
    k_per_class = k // 2
    
    # Get indices for positive and negative examples
    pos_indices = []
    neg_indices = []
    pos_scores = []
    neg_scores = []
    
    for idx, score in enumerate(influence_scores):
        if idx < len(train_dataset):
            label = train_dataset[idx]["label"]
            if label == 1:  # Positive
                pos_indices.append(idx)
                pos_scores.append(score)
            elif label == 0:  # Negative
                neg_indices.append(idx)
                neg_scores.append(score)
    
    # Sort by influence scores (descending)
    pos_sorted = sorted(zip(pos_scores, pos_indices), reverse=True)
    neg_sorted = sorted(zip(neg_scores, neg_indices), reverse=True)
    
    # Select top k_per_class from each sentiment
    selected_pos = [idx for _, idx in pos_sorted[:k_per_class]]
    selected_neg = [idx for _, idx in neg_sorted[:k_per_class]]
    
    # Combine and sort by original index for consistency
    sampled_indices = sorted(selected_pos + selected_neg)
    
    print(f"Sampled {len(selected_pos)} positive and {len(selected_neg)} negative training examples")
    print(f"Top positive influence scores: {[influence_scores[i] for i in selected_pos[:3]]}...")
    print(f"Top negative influence scores: {[influence_scores[i] for i in selected_neg[:3]]}...")
    
    return sampled_indices


def load_model_and_tokenizer(model_path: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model and tokenizer from path."""
    print(f"Loading model from: {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer


def prepare_datasets_for_influence(
    train_dataset: HFDataset, 
    query_dataset: HFDataset, 
    tokenizer: PreTrainedTokenizer,
    sampled_indices: List[int]
) -> Tuple[HFDataset, HFDataset, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Prepare datasets for influence computation."""
    
    # Filter training dataset to sampled examples
    train_dataset_filtered = train_dataset.select(sampled_indices)
    
    # Store original examples
    original_train_examples = [
        {"text": ex["text"], "label": ex["label"]}
        for ex in train_dataset_filtered
    ]
    original_query_examples = [
        {"text": ex["text"], "label": ex["label"]}
        for ex in query_dataset
    ]
    
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
    train_tokenized = train_dataset_filtered.map(
        tokenize_and_format,
        batched=True,
        remove_columns=train_dataset_filtered.column_names
    )
    
    query_tokenized = query_dataset.map(
        tokenize_and_format,
        batched=True,
        remove_columns=query_dataset.column_names
    )
    
    return train_tokenized, query_tokenized, original_train_examples, original_query_examples


def compute_real_influence_scores(
    model_path: str,
    train_dataset: HFDataset,
    query_dataset: HFDataset,
    sampled_indices: List[int],
    device: str = "cuda",
    batch_size: int = 1
) -> Dict[int, List[float]]:
    """Compute REAL influence scores using Kronfluence for each training example against each query."""
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)
    model = model.to(device)
    
    # Prepare datasets
    train_tokenized, query_tokenized, original_train, original_query = prepare_datasets_for_influence(
        train_dataset, query_dataset, tokenizer, sampled_indices
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
        analysis_name="query_influence_analysis",
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
    
    # Configure factor arguments (same as working influence_filter.py)
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
    
    print("Computing influence factors...")
    # Compute factors
    analyzer.fit_all_factors(
        factors_name="query_factors",
        dataset=train_wrapped,
        factor_args=factor_args,
        per_device_batch_size=batch_size,
        overwrite_output_dir=True
    )
    
    # Clear CUDA cache again
    if device == "cuda":
        torch.cuda.empty_cache()
    
    print("Computing pairwise influence scores...")
    # Compute pairwise scores
    analyzer.compute_pairwise_scores(
        scores_name="query_scores",
        factors_name="query_factors",
        query_dataset=query_wrapped,
        train_dataset=train_wrapped,
        per_device_query_batch_size=1,
        per_device_train_batch_size=batch_size,
        overwrite_output_dir=True
    )
    
    # Load computed scores
    print("Loading computed scores...")
    scores_dict = analyzer.load_pairwise_scores("query_scores")
    scores = scores_dict["all_modules"]  # Shape: [num_queries, num_train_examples]
    
    if scores is None:
        raise RuntimeError("Failed to compute influence scores - scores are None")
    
    # Convert to numpy and create result dictionary
    scores_np = scores.cpu().numpy()
    print(f"Computed influence scores shape: {scores_np.shape}")
    
    # Create dictionary mapping training example index to list of influence scores across queries
    training_example_influences = {}
    for train_idx in range(len(sampled_indices)):
        original_idx = sampled_indices[train_idx]
        # Get influence scores for this training example across all queries
        influences_across_queries = scores_np[:, train_idx].tolist()  # All queries for this training example
        training_example_influences[original_idx] = influences_across_queries
    
    return training_example_influences


def create_training_example_histograms(training_example_influences: Dict[int, List[float]], 
                                     query_examples: List[Dict[str, Any]], 
                                     train_examples: List[Dict[str, Any]], 
                                     influence_scores: List[float],
                                     output_dir: str,
                                     split_by_sentiment: bool = False):
    """Create histograms showing influence distribution for each training example."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create a mapping from training indices to training examples
    sampled_indices = list(training_example_influences.keys())
    train_idx_to_example = {}
    for i, train_idx in enumerate(sampled_indices):
        if i < len(train_examples):
            train_idx_to_example[train_idx] = train_examples[i]
    
    for train_idx, influences in training_example_influences.items():
        base_score = influence_scores[train_idx]
        
        # Get training example label and convert to text
        train_example = train_idx_to_example.get(train_idx, {'label': -1})
        train_label = train_example['label']
        train_sentiment = "Positive" if train_label == 1 else "Negative" if train_label == 0 else "Unknown"
        
        if split_by_sentiment:
            # Split influences by query sentiment
            pos_influences = [influences[i] for i, q in enumerate(query_examples) if q['label'] == 1]
            neg_influences = [influences[i] for i, q in enumerate(query_examples) if q['label'] == 0]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Positive queries histogram
            ax1.hist(pos_influences, bins=20, alpha=0.7, color='green', edgecolor='black')
            ax1.set_title(f'Training Example {train_idx} ({train_sentiment}) - Positive Queries\nBase Score: {base_score:.3f}')
            ax1.set_xlabel('Influence Score')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
            
            # Add statistics
            if pos_influences:
                pos_mean = np.mean(pos_influences)
                pos_std = np.std(pos_influences)
                ax1.axvline(pos_mean, color='red', linestyle='--', label=f'Mean: {pos_mean:.3f}')
                ax1.text(0.05, 0.95, f'Mean: {pos_mean:.3f}\nStd: {pos_std:.3f}\nCount: {len(pos_influences)}',
                        transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Negative queries histogram
            ax2.hist(neg_influences, bins=20, alpha=0.7, color='red', edgecolor='black')
            ax2.set_title(f'Training Example {train_idx} ({train_sentiment}) - Negative Queries\nBase Score: {base_score:.3f}')
            ax2.set_xlabel('Influence Score')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
            
            # Add statistics
            if neg_influences:
                neg_mean = np.mean(neg_influences)
                neg_std = np.std(neg_influences)
                ax2.axvline(neg_mean, color='blue', linestyle='--', label=f'Mean: {neg_mean:.3f}')
                ax2.text(0.05, 0.95, f'Mean: {neg_mean:.3f}\nStd: {neg_std:.3f}\nCount: {len(neg_influences)}',
                        transform=ax2.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            filename = f'training_example_{train_idx}_{train_sentiment.lower()}_sentiment_split.png'
            
        else:
            # Single histogram for all queries
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            ax.hist(influences, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax.set_title(f'Training Example {train_idx} ({train_sentiment}) Influence Distribution\nBase Score: {base_score:.3f}')
            ax.set_xlabel('Influence Score')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_inf = np.mean(influences)
            std_inf = np.std(influences)
            ax.axvline(mean_inf, color='red', linestyle='--', label=f'Mean: {mean_inf:.3f}')
            ax.text(0.05, 0.95, f'Mean: {mean_inf:.3f}\nStd: {std_inf:.3f}\nCount: {len(influences)}',
                   transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            filename = f'training_example_{train_idx}_{train_sentiment.lower()}_histogram.png'
        
        plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
    print(f"Created histograms for {len(training_example_influences)} training examples")


def create_summary_analysis(training_example_influences: Dict[int, List[float]], 
                          query_examples: List[Dict[str, Any]], 
                          train_examples: List[Dict[str, Any]], 
                          influence_scores: List[float],
                          output_dir: str,
                          split_by_sentiment: bool = False):
    """Create summary analysis plots."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate statistics for each training example
    train_indices = list(training_example_influences.keys())
    base_scores = [influence_scores[idx] for idx in train_indices]
    mean_influences = [np.mean(influences) for influences in training_example_influences.values()]
    std_influences = [np.std(influences) for influences in training_example_influences.values()]
    
    # Create a mapping from training indices to training examples
    sampled_indices = list(training_example_influences.keys())
    train_idx_to_example = {}
    for i, train_idx in enumerate(sampled_indices):
        if i < len(train_examples):
            train_idx_to_example[train_idx] = train_examples[i]
    
    # Get training example labels for coloring
    train_labels = [train_idx_to_example.get(idx, {'label': -1})['label'] for idx in train_indices]
    train_colors = ['green' if label == 1 else 'red' if label == 0 else 'gray' for label in train_labels]
    
    if split_by_sentiment:
        # Calculate sentiment-specific statistics
        pos_means = []
        neg_means = []
        pos_stds = []
        neg_stds = []
        
        for influences in training_example_influences.values():
            pos_influences = [influences[i] for i, q in enumerate(query_examples) if q['label'] == 1]
            neg_influences = [influences[i] for i, q in enumerate(query_examples) if q['label'] == 0]
            
            pos_means.append(np.mean(pos_influences) if pos_influences else 0)
            neg_means.append(np.mean(neg_influences) if neg_influences else 0)
            pos_stds.append(np.std(pos_influences) if pos_influences else 0)
            neg_stds.append(np.std(neg_influences) if neg_influences else 0)
        
        # Create sentiment comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Mean influence comparison
        scatter = axes[0,0].scatter(pos_means, neg_means, c=train_colors, alpha=0.7)
        axes[0,0].plot([min(pos_means + neg_means), max(pos_means + neg_means)], 
                      [min(pos_means + neg_means), max(pos_means + neg_means)], 'k--', alpha=0.5)
        axes[0,0].set_xlabel('Mean Influence on Positive Queries')
        axes[0,0].set_ylabel('Mean Influence on Negative Queries')
        axes[0,0].set_title('Mean Influence: Positive vs Negative Queries\n(Green=Pos Train, Red=Neg Train)')
        axes[0,0].grid(True, alpha=0.3)
        
        # Standard deviation comparison
        axes[0,1].scatter(pos_stds, neg_stds, c=train_colors, alpha=0.7)
        axes[0,1].set_xlabel('Std Dev - Positive Queries')
        axes[0,1].set_ylabel('Std Dev - Negative Queries')
        axes[0,1].set_title('Influence Variability: Positive vs Negative Queries\n(Green=Pos Train, Red=Neg Train)')
        axes[0,1].grid(True, alpha=0.3)
        
        # Base score vs sentiment-specific means
        pos_train_mask = [label == 1 for label in train_labels]
        neg_train_mask = [label == 0 for label in train_labels]
        
        if any(pos_train_mask):
            axes[1,0].scatter([base_scores[i] for i in range(len(base_scores)) if pos_train_mask[i]], 
                             [pos_means[i] for i in range(len(pos_means)) if pos_train_mask[i]], 
                             alpha=0.7, label='Pos Train→Pos Query', color='darkgreen', marker='o')
            axes[1,0].scatter([base_scores[i] for i in range(len(base_scores)) if pos_train_mask[i]], 
                             [neg_means[i] for i in range(len(neg_means)) if pos_train_mask[i]], 
                             alpha=0.7, label='Pos Train→Neg Query', color='lightgreen', marker='s')
        
        if any(neg_train_mask):
            axes[1,0].scatter([base_scores[i] for i in range(len(base_scores)) if neg_train_mask[i]], 
                             [pos_means[i] for i in range(len(pos_means)) if neg_train_mask[i]], 
                             alpha=0.7, label='Neg Train→Pos Query', color='lightcoral', marker='o')
            axes[1,0].scatter([base_scores[i] for i in range(len(base_scores)) if neg_train_mask[i]], 
                             [neg_means[i] for i in range(len(neg_means)) if neg_train_mask[i]], 
                             alpha=0.7, label='Neg Train→Neg Query', color='darkred', marker='s')
        
        axes[1,0].set_xlabel('Base Influence Score')
        axes[1,0].set_ylabel('Mean Query-Specific Influence')
        axes[1,0].set_title('Base Score vs Query-Specific Influence by Training Label')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Difference between positive and negative means
        pos_neg_diff = np.array(pos_means) - np.array(neg_means)
        pos_train_diff = [pos_neg_diff[i] for i in range(len(pos_neg_diff)) if train_labels[i] == 1]
        neg_train_diff = [pos_neg_diff[i] for i in range(len(pos_neg_diff)) if train_labels[i] == 0]
        
        if pos_train_diff:
            axes[1,1].hist(pos_train_diff, bins=10, alpha=0.7, color='green', edgecolor='black', label='Positive Training Examples')
        if neg_train_diff:
            axes[1,1].hist(neg_train_diff, bins=10, alpha=0.7, color='red', edgecolor='black', label='Negative Training Examples')
        axes[1,1].set_xlabel('Positive Mean - Negative Mean')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Difference in Mean Influence (Pos - Neg) by Training Label')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'summary_analysis_sentiment_split.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    else:
        # Create regular summary plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Base score vs mean influence
        axes[0,0].scatter(base_scores, mean_influences, c=train_colors, alpha=0.7)
        axes[0,0].set_xlabel('Base Influence Score')
        axes[0,0].set_ylabel('Mean Query-Specific Influence')
        axes[0,0].set_title('Base Score vs Mean Query-Specific Influence\n(Green=Pos Train, Red=Neg Train)')
        axes[0,0].grid(True, alpha=0.3)
        
        # Base score vs influence variability
        axes[0,1].scatter(base_scores, std_influences, c=train_colors, alpha=0.7)
        axes[0,1].set_xlabel('Base Influence Score')
        axes[0,1].set_ylabel('Std Dev of Query-Specific Influence')
        axes[0,1].set_title('Base Score vs Influence Variability\n(Green=Pos Train, Red=Neg Train)')
        axes[0,1].grid(True, alpha=0.3)
        
        # Distribution of mean influences by training label
        pos_train_means = [mean_influences[i] for i in range(len(mean_influences)) if train_labels[i] == 1]
        neg_train_means = [mean_influences[i] for i in range(len(mean_influences)) if train_labels[i] == 0]
        
        if pos_train_means:
            axes[1,0].hist(pos_train_means, bins=10, alpha=0.7, color='green', edgecolor='black', label='Positive Training Examples')
        if neg_train_means:
            axes[1,0].hist(neg_train_means, bins=10, alpha=0.7, color='red', edgecolor='black', label='Negative Training Examples')
        axes[1,0].set_xlabel('Mean Query-Specific Influence')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Distribution of Mean Influences by Training Label')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Distribution of influence variability by training label
        pos_train_stds = [std_influences[i] for i in range(len(std_influences)) if train_labels[i] == 1]
        neg_train_stds = [std_influences[i] for i in range(len(std_influences)) if train_labels[i] == 0]
        
        if pos_train_stds:
            axes[1,1].hist(pos_train_stds, bins=10, alpha=0.7, color='green', edgecolor='black', label='Positive Training Examples')
        if neg_train_stds:
            axes[1,1].hist(neg_train_stds, bins=10, alpha=0.7, color='red', edgecolor='black', label='Negative Training Examples')
        axes[1,1].set_xlabel('Std Dev of Query-Specific Influence')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Distribution of Influence Variability by Training Label')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'summary_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()


def save_detailed_statistics(training_example_influences: Dict[int, List[float]], 
                           query_examples: List[Dict[str, Any]], 
                           train_examples: List[Dict[str, Any]], 
                           influence_scores: List[float],
                           output_dir: str,
                           split_by_sentiment: bool = False):
    """Save detailed statistics to JSON file."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    detailed_stats = {}
    
    # Create a mapping from training indices to training examples
    sampled_indices = list(training_example_influences.keys())
    train_idx_to_example = {}
    for i, train_idx in enumerate(sampled_indices):
        if i < len(train_examples):
            train_idx_to_example[train_idx] = train_examples[i]
    
    for train_idx, influences in training_example_influences.items():
        base_score = influence_scores[train_idx]
        
        # Get training example label
        train_example = train_idx_to_example.get(train_idx, {'label': -1})
        train_label = train_example['label']
        train_sentiment = "positive" if train_label == 1 else "negative" if train_label == 0 else "unknown"
        
        stats = {
            "training_example_label": train_label,
            "training_example_sentiment": train_sentiment,
            "base_influence_score": base_score,
            "query_specific_stats": {
                "mean": float(np.mean(influences)),
                "std": float(np.std(influences)),
                "min": float(np.min(influences)),
                "max": float(np.max(influences)),
                "count": len(influences)
            }
        }
        
        if split_by_sentiment:
            pos_influences = [influences[i] for i, q in enumerate(query_examples) if q['label'] == 1]
            neg_influences = [influences[i] for i, q in enumerate(query_examples) if q['label'] == 0]
            
            stats["positive_queries"] = {
                "mean": float(np.mean(pos_influences)) if pos_influences else 0,
                "std": float(np.std(pos_influences)) if pos_influences else 0,
                "min": float(np.min(pos_influences)) if pos_influences else 0,
                "max": float(np.max(pos_influences)) if pos_influences else 0,
                "count": len(pos_influences)
            }
            
            stats["negative_queries"] = {
                "mean": float(np.mean(neg_influences)) if neg_influences else 0,
                "std": float(np.std(neg_influences)) if neg_influences else 0,
                "min": float(np.min(neg_influences)) if neg_influences else 0,
                "max": float(np.max(neg_influences)) if neg_influences else 0,
                "count": len(neg_influences)
            }
        
        detailed_stats[str(train_idx)] = stats
    
    # Save to JSON
    filename = 'detailed_statistics_sentiment_split.json' if split_by_sentiment else 'detailed_statistics.json'
    with open(output_path / filename, 'w') as f:
        json.dump(detailed_stats, f, indent=2)
    
    print(f"Saved detailed statistics to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Analyze training example influences on different queries using REAL Kronfluence")
    parser.add_argument("--influence_file", type=str, default="/share/u/yu.stev/influence/influence-filtration/data/distilgpt2/imdb/influence_scores.json",
                       help="Path to influence scores JSON file")
    parser.add_argument("--model_path", type=str, default="/share/u/yu.stev/influence/influence-filtration/models/distilgpt2/imdb/",
                       help="Path to trained model")
    parser.add_argument("--output_dir", type=str, default="/share/u/yu.stev/influence/influence-filtration/data/distilgpt2/plots/query-histogram/",
                       help="Output directory for plots and analysis")
    parser.add_argument("--k", type=int, default=10,
                       help="Total number of training examples to analyze (will be split equally between positive and negative examples, so k=10 gives 5 positive + 5 negative)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for computation")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for influence computation")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--split_by_sentiment", action="store_true",
                       help="Split histograms by query sentiment (positive/negative)")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("Starting query influence analysis with REAL Kronfluence computation...")
    print(f"Device: {args.device}")
    print(f"Model path: {args.model_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Split by sentiment: {args.split_by_sentiment}")
    
    # Load influence data
    influence_scores, query_examples = load_influence_data(args.influence_file)
    
    # Load original training dataset
    print("Loading original IMDB training dataset...")
    train_dataset = load_dataset("imdb", split="train")
    
    # Sample training examples
    sampled_indices = sample_training_examples(influence_scores, train_dataset, args.k)
    
    # Extract the training example data for the sampled indices
    train_examples = [
        {"text": train_dataset[idx]["text"], "label": train_dataset[idx]["label"]}
        for idx in sampled_indices
    ]
    
    # Create query dataset from query examples
    query_texts = [q["text"] for q in query_examples]
    query_labels = [q["label"] for q in query_examples]
    query_dataset = HFDataset.from_dict({
        "text": query_texts,
        "label": query_labels
    })
    
    # Compute REAL influence scores
    print("Computing REAL influence scores using Kronfluence...")
    training_example_influences = compute_real_influence_scores(
        args.model_path, train_dataset, query_dataset, sampled_indices, 
        args.device, args.batch_size
    )
    
    # Create histograms
    create_training_example_histograms(
        training_example_influences, query_examples, train_examples, influence_scores, 
        args.output_dir, args.split_by_sentiment
    )
    
    create_summary_analysis(
        training_example_influences, query_examples, train_examples, influence_scores, 
        args.output_dir, args.split_by_sentiment
    )
    
    # Save detailed statistics
    save_detailed_statistics(
        training_example_influences, query_examples, train_examples, influence_scores, 
        args.output_dir, args.split_by_sentiment
    )
    
    print("\nAnalysis complete!")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
