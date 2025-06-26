#!/usr/bin/env python3
"""
Statistical Analysis: Central Limit Theorem for Query Influence Distributions

This script demonstrates the Central Limit Theorem by:
1. Selecting 2 training examples (1 positive, 1 negative)
2. Taking 50 random samples of 50 queries each
3. Computing real influence scores for each sample
4. Analyzing the distribution of sample means
5. Plotting results and saving statistical analysis

The Central Limit Theorem predicts that the distribution of sample means will be
approximately normal, regardless of the underlying distribution of individual influence scores.
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
from scipy import stats
import random

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


def load_influence_data(influence_file: str) -> Tuple[List[float], List[Dict[str, Any]]]:
    """Load influence scores and query examples from JSON file."""
    with open(influence_file, 'r') as f:
        data = json.load(f)
    
    influence_scores = data["influence_scores"]
    query_examples = data["query_examples"]
    
    print(f"Loaded {len(influence_scores)} training examples and {len(query_examples)} queries")
    return influence_scores, query_examples


def select_training_examples(influence_scores: List[float], train_dataset: HFDataset) -> Tuple[int, int]:
    """Select 2 training examples: 1 positive, 1 negative with high influence scores."""
    
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
    
    # Select top 1 from each sentiment
    selected_pos = pos_sorted[0][1]  # Index of top positive example
    selected_neg = neg_sorted[0][1]  # Index of top negative example
    
    print(f"Selected positive training example {selected_pos} with influence score: {influence_scores[selected_pos]:.6f}")
    print(f"Selected negative training example {selected_neg} with influence score: {influence_scores[selected_neg]:.6f}")
    
    return selected_pos, selected_neg


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
    train_indices: List[int]
) -> Tuple[HFDataset, HFDataset]:
    """Prepare datasets for influence computation."""
    
    # Filter training dataset to selected examples
    train_dataset_filtered = train_dataset.select(train_indices)
    
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
    
    return train_tokenized, query_tokenized


def compute_influence_scores_for_sample(
    model_path: str,
    train_dataset: HFDataset,
    query_sample: HFDataset,
    train_indices: List[int],
    sample_id: int,
    device: str = "cuda",
    batch_size: int = 1
) -> np.ndarray:
    """Compute influence scores for a single query sample."""
    
    print(f"Computing influence scores for sample {sample_id + 1}...")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)
    model = model.to(device)
    
    # Prepare datasets
    train_tokenized, query_tokenized = prepare_datasets_for_influence(
        train_dataset, query_sample, tokenizer, train_indices
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
        analysis_name=f"clt_analysis_sample_{sample_id}",
        model=model,
        task=task,
        cpu=(device == "cpu"),
        disable_tqdm=True  # Reduce output clutter
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
    analyzer.fit_all_factors(
        factors_name=f"clt_factors_{sample_id}",
        dataset=train_wrapped,
        factor_args=factor_args,
        per_device_batch_size=batch_size,
        overwrite_output_dir=True
    )
    
    # Clear CUDA cache again
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # Compute pairwise scores
    analyzer.compute_pairwise_scores(
        scores_name=f"clt_scores_{sample_id}",
        factors_name=f"clt_factors_{sample_id}",
        query_dataset=query_wrapped,
        train_dataset=train_wrapped,
        per_device_query_batch_size=1,
        per_device_train_batch_size=batch_size,
        overwrite_output_dir=True
    )
    
    # Load computed scores
    scores_dict = analyzer.load_pairwise_scores(f"clt_scores_{sample_id}")
    scores = scores_dict["all_modules"]  # Shape: [num_queries, num_train_examples]
    
    if scores is None:
        raise RuntimeError(f"Failed to compute influence scores for sample {sample_id}")
    
    # Convert to numpy
    scores_np = scores.cpu().numpy()
    
    # Clear CUDA cache
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return scores_np


def run_clt_experiment(
    model_path: str,
    train_dataset: HFDataset,
    test_dataset: HFDataset,
    train_indices: List[int],
    n_samples: int = 50,
    sample_size: int = 50,
    device: str = "cuda",
    batch_size: int = 1,
    seed: int = 42
) -> Dict[str, Any]:
    """Run the Central Limit Theorem experiment with support/opposition score analysis."""
    
    print(f"Starting CLT experiment with {n_samples} samples of {sample_size} queries each...")
    print("Each sample will contain 25 positive and 25 negative queries")
    
    # Ensure sample_size is even for equal split
    if sample_size % 2 != 0:
        sample_size += 1
        print(f"Adjusted sample_size to {sample_size} for equal pos/neg split")
    
    queries_per_sentiment = sample_size // 2
    
    # Set random seed for reproducible sampling
    random.seed(seed)
    np.random.seed(seed)
    
    # Separate test queries by sentiment
    pos_test_indices = [i for i, example in enumerate(test_dataset) if example['label'] == 1]
    neg_test_indices = [i for i, example in enumerate(test_dataset) if example['label'] == 0]
    
    print(f"Available test queries: {len(pos_test_indices)} positive, {len(neg_test_indices)} negative")
    
    if len(pos_test_indices) < queries_per_sentiment or len(neg_test_indices) < queries_per_sentiment:
        raise ValueError(f"Not enough test queries of each sentiment. Need {queries_per_sentiment} each, "
                        f"have {len(pos_test_indices)} pos, {len(neg_test_indices)} neg")
    
    # Store results for each training example
    results = {
        "experiment_config": {
            "n_samples": n_samples,
            "sample_size": sample_size,
            "queries_per_sentiment": queries_per_sentiment,
            "train_indices": train_indices,
            "seed": seed
        },
        "training_examples": {},
        "sample_means": {},
        "statistical_tests": {}
    }
    
    # Store sample means for each training example (support vs opposition)
    support_means_per_train = {idx: [] for idx in train_indices}
    opposition_means_per_train = {idx: [] for idx in train_indices}
    all_support_scores = {idx: [] for idx in train_indices}
    all_opposition_scores = {idx: [] for idx in train_indices}
    
    for sample_id in range(n_samples):
        print(f"\n--- Processing Sample {sample_id + 1}/{n_samples} ---")
        
        # Randomly sample equal numbers of positive and negative test queries
        sampled_pos_indices = random.sample(pos_test_indices, queries_per_sentiment)
        sampled_neg_indices = random.sample(neg_test_indices, queries_per_sentiment)
        sampled_indices = sampled_pos_indices + sampled_neg_indices
        
        # Create query dataset from sampled test examples
        sampled_test_data = test_dataset.select(sampled_indices)
        
        # Compute influence scores for this sample
        scores_np = compute_influence_scores_for_sample(
            model_path, train_dataset, sampled_test_data, train_indices, 
            sample_id, device, batch_size
        )
        
        # Process results for each training example
        for train_idx_pos, train_idx in enumerate(train_indices):
            # Get training example label
            train_label = train_dataset[train_idx]["label"]
            
            # Get influence scores for this training example across all queries in the sample
            train_influences = scores_np[:, train_idx_pos]  # All queries for this training example
            
            # Split scores by query sentiment
            support_scores = []  # Queries matching training example label
            opposition_scores = []  # Queries opposing training example label
            
            for query_idx, query_influence in enumerate(train_influences):
                query_label = sampled_test_data[query_idx]["label"]
                if query_label == train_label:
                    support_scores.append(query_influence)
                else:
                    opposition_scores.append(query_influence)
            
            # Store individual scores
            all_support_scores[train_idx].extend(support_scores)
            all_opposition_scores[train_idx].extend(opposition_scores)
            
            # Calculate sample means
            support_mean = np.mean(support_scores) if support_scores else 0.0
            opposition_mean = np.mean(opposition_scores) if opposition_scores else 0.0
            
            support_means_per_train[train_idx].append(support_mean)
            opposition_means_per_train[train_idx].append(opposition_mean)
            
            train_sentiment = "Positive" if train_label == 1 else "Negative"
            print(f"  Training example {train_idx} ({train_sentiment}):")
            print(f"    Support mean = {support_mean:.6f} (from {len(support_scores)} queries)")
            print(f"    Opposition mean = {opposition_mean:.6f} (from {len(opposition_scores)} queries)")
    
    # Perform statistical analysis for each training example
    for train_idx in train_indices:
        support_means = np.array(support_means_per_train[train_idx])
        opposition_means = np.array(opposition_means_per_train[train_idx])
        support_scores = np.array(all_support_scores[train_idx])
        opposition_scores = np.array(all_opposition_scores[train_idx])
        
        # Get training example info
        train_label = train_dataset[train_idx]["label"]
        train_sentiment = "positive" if train_label == 1 else "negative"
        
        # Basic statistics for support scores
        support_stats = {
            "sample_means": {
                "mean": float(np.mean(support_means)),
                "std": float(np.std(support_means)),
                "min": float(np.min(support_means)),
                "max": float(np.max(support_means)),
                "count": len(support_means)
            },
            "individual_scores": {
                "mean": float(np.mean(support_scores)),
                "std": float(np.std(support_scores)),
                "min": float(np.min(support_scores)),
                "max": float(np.max(support_scores)),
                "count": len(support_scores)
            }
        }
        
        # Basic statistics for opposition scores
        opposition_stats = {
            "sample_means": {
                "mean": float(np.mean(opposition_means)),
                "std": float(np.std(opposition_means)),
                "min": float(np.min(opposition_means)),
                "max": float(np.max(opposition_means)),
                "count": len(opposition_means)
            },
            "individual_scores": {
                "mean": float(np.mean(opposition_scores)),
                "std": float(np.std(opposition_scores)),
                "min": float(np.min(opposition_scores)),
                "max": float(np.max(opposition_scores)),
                "count": len(opposition_scores)
            }
        }
        
        # CLT verification for support scores
        support_theoretical_se = np.std(support_scores) / np.sqrt(queries_per_sentiment)
        support_empirical_se = np.std(support_means)
        
        support_stats["clt_verification"] = {
            "theoretical_standard_error": float(support_theoretical_se),
            "empirical_standard_error": float(support_empirical_se),
            "se_ratio": float(support_empirical_se / support_theoretical_se) if support_theoretical_se != 0 else None
        }
        
        # CLT verification for opposition scores
        opposition_theoretical_se = np.std(opposition_scores) / np.sqrt(queries_per_sentiment)
        opposition_empirical_se = np.std(opposition_means)
        
        opposition_stats["clt_verification"] = {
            "theoretical_standard_error": float(opposition_theoretical_se),
            "empirical_standard_error": float(opposition_empirical_se),
            "se_ratio": float(opposition_empirical_se / opposition_theoretical_se) if opposition_theoretical_se != 0 else None
        }
        
        # Normality tests for support scores
        support_shapiro_stat, support_shapiro_p = stats.shapiro(support_means)
        support_ks_stat, support_ks_p = stats.kstest(support_means, 'norm', 
                                                    args=(np.mean(support_means), np.std(support_means)))
        
        support_stats["normality_tests"] = {
            "shapiro_wilk": {
                "statistic": float(support_shapiro_stat),
                "p_value": float(support_shapiro_p),
                "is_normal_p05": support_shapiro_p > 0.05
            },
            "kolmogorov_smirnov": {
                "statistic": float(support_ks_stat),
                "p_value": float(support_ks_p),
                "is_normal_p05": support_ks_p > 0.05
            }
        }
        
        # Normality tests for opposition scores
        opposition_shapiro_stat, opposition_shapiro_p = stats.shapiro(opposition_means)
        opposition_ks_stat, opposition_ks_p = stats.kstest(opposition_means, 'norm', 
                                                          args=(np.mean(opposition_means), np.std(opposition_means)))
        
        opposition_stats["normality_tests"] = {
            "shapiro_wilk": {
                "statistic": float(opposition_shapiro_stat),
                "p_value": float(opposition_shapiro_p),
                "is_normal_p05": opposition_shapiro_p > 0.05
            },
            "kolmogorov_smirnov": {
                "statistic": float(opposition_ks_stat),
                "p_value": float(opposition_ks_p),
                "is_normal_p05": opposition_ks_p > 0.05
            }
        }
        
        # Store results
        results["training_examples"][str(train_idx)] = {
            "training_label": train_label,
            "training_sentiment": train_sentiment,
            "support_analysis": support_stats,
            "opposition_analysis": opposition_stats
        }
        
        results["sample_means"][str(train_idx)] = {
            "support_means": support_means.tolist(),
            "opposition_means": opposition_means.tolist()
        }
    
    return results


def create_clt_plots(results: Dict[str, Any], train_dataset: HFDataset, output_dir: str):
    """Create plots demonstrating the Central Limit Theorem with support/opposition analysis."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_indices = results["experiment_config"]["train_indices"]
    n_samples = results["experiment_config"]["n_samples"]
    sample_size = results["experiment_config"]["sample_size"]
    queries_per_sentiment = results["experiment_config"]["queries_per_sentiment"]
    
    # Create comprehensive plot with support/opposition analysis
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle(f'Central Limit Theorem: Support vs Opposition Scores\n{n_samples} samples of {sample_size} queries each ({queries_per_sentiment} per sentiment)', fontsize=16)
    
    for i, train_idx in enumerate(train_indices):
        support_means = np.array(results["sample_means"][str(train_idx)]["support_means"])
        opposition_means = np.array(results["sample_means"][str(train_idx)]["opposition_means"])
        
        train_stats = results["training_examples"][str(train_idx)]
        train_label = train_stats["training_label"]
        train_sentiment = "Positive" if train_label == 1 else "Negative"
        
        # Support scores - Histogram
        ax1 = axes[0, i]
        ax1.hist(support_means, bins=15, alpha=0.7, color='green', edgecolor='black', density=True)
        
        # Overlay normal distribution for support
        x_support = np.linspace(support_means.min(), support_means.max(), 100)
        normal_dist_support = stats.norm(np.mean(support_means), np.std(support_means))
        ax1.plot(x_support, normal_dist_support.pdf(x_support), 'darkgreen', linewidth=2, label='Normal Distribution')
        
        ax1.set_title(f'Training Example {train_idx} ({train_sentiment})\nSupport Scores Distribution')
        ax1.set_xlabel('Sample Mean Support Score')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add statistics text for support
        support_stats_text = f'Mean: {train_stats["support_analysis"]["sample_means"]["mean"]:.4f}\n'
        support_stats_text += f'Std: {train_stats["support_analysis"]["sample_means"]["std"]:.4f}\n'
        support_stats_text += f'Shapiro p: {train_stats["support_analysis"]["normality_tests"]["shapiro_wilk"]["p_value"]:.4f}'
        ax1.text(0.05, 0.95, support_stats_text, transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Opposition scores - Histogram
        ax2 = axes[1, i]
        ax2.hist(opposition_means, bins=15, alpha=0.7, color='red', edgecolor='black', density=True)
        
        # Overlay normal distribution for opposition
        x_opposition = np.linspace(opposition_means.min(), opposition_means.max(), 100)
        normal_dist_opposition = stats.norm(np.mean(opposition_means), np.std(opposition_means))
        ax2.plot(x_opposition, normal_dist_opposition.pdf(x_opposition), 'darkred', linewidth=2, label='Normal Distribution')
        
        ax2.set_title(f'Training Example {train_idx} ({train_sentiment})\nOpposition Scores Distribution')
        ax2.set_xlabel('Sample Mean Opposition Score')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text for opposition
        opposition_stats_text = f'Mean: {train_stats["opposition_analysis"]["sample_means"]["mean"]:.4f}\n'
        opposition_stats_text += f'Std: {train_stats["opposition_analysis"]["sample_means"]["std"]:.4f}\n'
        opposition_stats_text += f'Shapiro p: {train_stats["opposition_analysis"]["normality_tests"]["shapiro_wilk"]["p_value"]:.4f}'
        ax2.text(0.05, 0.95, opposition_stats_text, transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        # Support scores - Q-Q plot
        ax3 = axes[2, i]
        stats.probplot(support_means, dist="norm", plot=ax3)
        ax3.set_title(f'Q-Q Plot: Support Scores\nTraining Example {train_idx}')
        ax3.grid(True, alpha=0.3)
        
        # Opposition scores - Q-Q plot
        ax4 = axes[3, i]
        stats.probplot(opposition_means, dist="norm", plot=ax4)
        ax4.set_title(f'Q-Q Plot: Opposition Scores\nTraining Example {train_idx}')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'clt_support_opposition_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create comparison plot (support vs opposition)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Support vs Opposition Score Comparison', fontsize=16)
    
    for i, train_idx in enumerate(train_indices):
        support_means = np.array(results["sample_means"][str(train_idx)]["support_means"])
        opposition_means = np.array(results["sample_means"][str(train_idx)]["opposition_means"])
        
        train_stats = results["training_examples"][str(train_idx)]
        train_label = train_stats["training_label"]
        train_sentiment = "Positive" if train_label == 1 else "Negative"
        
        # Scatter plot: Support vs Opposition means
        ax1 = axes[0, i]
        ax1.scatter(support_means, opposition_means, alpha=0.6, s=30)
        ax1.set_xlabel('Support Score Sample Means')
        ax1.set_ylabel('Opposition Score Sample Means')
        ax1.set_title(f'Training Example {train_idx} ({train_sentiment})\nSupport vs Opposition')
        ax1.grid(True, alpha=0.3)
        
        # Add diagonal line
        min_val = min(np.min(support_means), np.min(opposition_means))
        max_val = max(np.max(support_means), np.max(opposition_means))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
        ax1.legend()
        
        # Difference distribution
        ax2 = axes[1, i]
        difference = support_means - opposition_means
        ax2.hist(difference, bins=15, alpha=0.7, color='purple', edgecolor='black')
        ax2.set_xlabel('Support Mean - Opposition Mean')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Difference Distribution\nTraining Example {train_idx}')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        diff_mean = np.mean(difference)
        diff_std = np.std(difference)
        ax2.axvline(diff_mean, color='red', linestyle='--', label=f'Mean: {diff_mean:.4f}')
        ax2.text(0.05, 0.95, f'Mean: {diff_mean:.4f}\nStd: {diff_std:.4f}',
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path / 'clt_support_vs_opposition_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create CLT verification plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Central Limit Theorem Verification', fontsize=16)
    
    for i, train_idx in enumerate(train_indices):
        train_stats = results["training_examples"][str(train_idx)]
        train_label = train_stats["training_label"]
        train_sentiment = "Positive" if train_label == 1 else "Negative"
        
        # CLT verification for support scores
        ax1 = axes[0, i]
        support_se_ratio = train_stats["support_analysis"]["clt_verification"]["se_ratio"]
        support_theoretical_se = train_stats["support_analysis"]["clt_verification"]["theoretical_standard_error"]
        support_empirical_se = train_stats["support_analysis"]["clt_verification"]["empirical_standard_error"]
        
        ax1.bar(['Theoretical SE', 'Empirical SE'], [support_theoretical_se, support_empirical_se], 
               color=['lightblue', 'darkblue'], alpha=0.7)
        ax1.set_title(f'Support Scores SE Verification\nTraining Example {train_idx} ({train_sentiment})')
        ax1.set_ylabel('Standard Error')
        ax1.text(0.5, 0.95, f'SE Ratio: {support_se_ratio:.3f}', transform=ax1.transAxes, 
                ha='center', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # CLT verification for opposition scores
        ax2 = axes[1, i]
        opposition_se_ratio = train_stats["opposition_analysis"]["clt_verification"]["se_ratio"]
        opposition_theoretical_se = train_stats["opposition_analysis"]["clt_verification"]["theoretical_standard_error"]
        opposition_empirical_se = train_stats["opposition_analysis"]["clt_verification"]["empirical_standard_error"]
        
        ax2.bar(['Theoretical SE', 'Empirical SE'], [opposition_theoretical_se, opposition_empirical_se], 
               color=['lightcoral', 'darkred'], alpha=0.7)
        ax2.set_title(f'Opposition Scores SE Verification\nTraining Example {train_idx} ({train_sentiment})')
        ax2.set_ylabel('Standard Error')
        ax2.text(0.5, 0.95, f'SE Ratio: {opposition_se_ratio:.3f}', transform=ax2.transAxes, 
                ha='center', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path / 'clt_verification_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created CLT support/opposition analysis plots in {output_dir}")


def save_results(results: Dict[str, Any], output_dir: str):
    """Save statistical results to JSON file with support/opposition analysis."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    with open(output_path / 'clt_support_opposition_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary report
    summary = {
        "experiment_summary": {
            "n_samples": results["experiment_config"]["n_samples"],
            "sample_size": results["experiment_config"]["sample_size"],
            "queries_per_sentiment": results["experiment_config"]["queries_per_sentiment"],
            "training_examples_analyzed": len(results["experiment_config"]["train_indices"])
        },
        "clt_verification": {},
        "normality_test_results": {},
        "support_vs_opposition_comparison": {}
    }
    
    for train_idx in results["experiment_config"]["train_indices"]:
        train_stats = results["training_examples"][str(train_idx)]
        
        # CLT verification
        summary["clt_verification"][str(train_idx)] = {
            "support_se_ratio": train_stats["support_analysis"]["clt_verification"]["se_ratio"],
            "opposition_se_ratio": train_stats["opposition_analysis"]["clt_verification"]["se_ratio"],
            "support_se_close_to_1": abs(train_stats["support_analysis"]["clt_verification"]["se_ratio"] - 1.0) < 0.2 if train_stats["support_analysis"]["clt_verification"]["se_ratio"] else False,
            "opposition_se_close_to_1": abs(train_stats["opposition_analysis"]["clt_verification"]["se_ratio"] - 1.0) < 0.2 if train_stats["opposition_analysis"]["clt_verification"]["se_ratio"] else False
        }
        
        # Normality tests
        summary["normality_test_results"][str(train_idx)] = {
            "support_shapiro_normal": train_stats["support_analysis"]["normality_tests"]["shapiro_wilk"]["is_normal_p05"],
            "opposition_shapiro_normal": train_stats["opposition_analysis"]["normality_tests"]["shapiro_wilk"]["is_normal_p05"],
            "support_ks_normal": train_stats["support_analysis"]["normality_tests"]["kolmogorov_smirnov"]["is_normal_p05"],
            "opposition_ks_normal": train_stats["opposition_analysis"]["normality_tests"]["kolmogorov_smirnov"]["is_normal_p05"]
        }
        
        # Support vs Opposition comparison
        support_mean = train_stats["support_analysis"]["sample_means"]["mean"]
        opposition_mean = train_stats["opposition_analysis"]["sample_means"]["mean"]
        
        summary["support_vs_opposition_comparison"][str(train_idx)] = {
            "support_mean": support_mean,
            "opposition_mean": opposition_mean,
            "difference": support_mean - opposition_mean,
            "support_higher": support_mean > opposition_mean
        }
    
    with open(output_path / 'clt_support_opposition_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved statistical results to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Central Limit Theorem demonstration with support/opposition influence scores")
    parser.add_argument("--influence_file", type=str, 
                       default="/share/u/yu.stev/influence/influence-filtration/data/distilgpt2/imdb/influence_scores.json",
                       help="Path to influence scores JSON file")
    parser.add_argument("--model_path", type=str, 
                       default="/share/u/yu.stev/influence/influence-filtration/models/distilgpt2/imdb/",
                       help="Path to trained model")
    parser.add_argument("--output_dir", type=str, 
                       default="/share/u/yu.stev/influence/influence-filtration/data/distilgpt2/plots/clt-analysis/",
                       help="Output directory for plots and analysis")
    parser.add_argument("--n_samples", type=int, default=50,
                       help="Number of query samples to take")
    parser.add_argument("--sample_size", type=int, default=50,
                       help="Size of each query sample (will be split 25/25 pos/neg)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for computation")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for influence computation")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    print("Starting Central Limit Theorem analysis with Support/Opposition scores...")
    print(f"Device: {args.device}")
    print(f"Model path: {args.model_path}")
    print(f"Samples: {args.n_samples} samples of {args.sample_size} queries each")
    print("Analysis will split by training example sentiment and query sentiment")
    print("Sampling queries directly from IMDB test dataset")
    
    # Load influence data (only need this for training example selection)
    influence_scores, query_examples = load_influence_data(args.influence_file)
    
    # Load original training and test datasets
    print("Loading original IMDB datasets...")
    train_dataset = load_dataset("imdb", split="train")
    test_dataset = load_dataset("imdb", split="test")
    
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Test positive examples: {sum(1 for ex in test_dataset if ex['label'] == 1)}")
    print(f"Test negative examples: {sum(1 for ex in test_dataset if ex['label'] == 0)}")
    
    # Select 2 training examples (1 positive, 1 negative)
    pos_idx, neg_idx = select_training_examples(influence_scores, train_dataset)
    train_indices = [pos_idx, neg_idx]
    
    # Run CLT experiment using test dataset for queries
    results = run_clt_experiment(
        args.model_path, train_dataset, test_dataset, train_indices,
        args.n_samples, args.sample_size, args.device, args.batch_size, args.seed
    )
    
    # Create plots
    create_clt_plots(results, train_dataset, args.output_dir)
    
    # Save results
    save_results(results, args.output_dir)
    
    print("\nCentral Limit Theorem Support/Opposition analysis complete!")
    print(f"Results saved to: {args.output_dir}")
    
    # Print summary
    print("\n=== SUMMARY ===")
    for train_idx in train_indices:
        train_stats = results["training_examples"][str(train_idx)]
        train_label = train_stats["training_label"]
        train_sentiment = "Positive" if train_label == 1 else "Negative"
        
        print(f"\nTraining Example {train_idx} ({train_sentiment}):")
        
        # Support scores summary
        support_mean = train_stats["support_analysis"]["sample_means"]["mean"]
        support_std = train_stats["support_analysis"]["sample_means"]["std"]
        support_se_ratio = train_stats["support_analysis"]["clt_verification"]["se_ratio"]
        support_normal = train_stats["support_analysis"]["normality_tests"]["shapiro_wilk"]["is_normal_p05"]
        
        print(f"  Support Scores (same sentiment queries):")
        print(f"    Sample means: μ={support_mean:.4f}, σ={support_std:.4f}")
        print(f"    CLT SE ratio: {support_se_ratio:.3f}")
        print(f"    Normal distribution: {support_normal}")
        
        # Opposition scores summary
        opposition_mean = train_stats["opposition_analysis"]["sample_means"]["mean"]
        opposition_std = train_stats["opposition_analysis"]["sample_means"]["std"]
        opposition_se_ratio = train_stats["opposition_analysis"]["clt_verification"]["se_ratio"]
        opposition_normal = train_stats["opposition_analysis"]["normality_tests"]["shapiro_wilk"]["is_normal_p05"]
        
        print(f"  Opposition Scores (opposite sentiment queries):")
        print(f"    Sample means: μ={opposition_mean:.4f}, σ={opposition_std:.4f}")
        print(f"    CLT SE ratio: {opposition_se_ratio:.3f}")
        print(f"    Normal distribution: {opposition_normal}")
        
        # Comparison
        difference = support_mean - opposition_mean
        print(f"  Support - Opposition difference: {difference:.4f}")
        print(f"  Support scores are {'higher' if difference > 0 else 'lower'} than opposition scores")


if __name__ == "__main__":
    main()

