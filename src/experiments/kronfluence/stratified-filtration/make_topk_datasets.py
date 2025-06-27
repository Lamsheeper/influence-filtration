#!/usr/bin/env python3
"""
Script to create top-k datasets from influence scores in HuggingFace format.

This script processes influence score JSON files and creates filtered datasets
with multiple modes:
- Top support scores (both positive and negative sentiment)
- Bottom support scores (both positive and negative sentiment)  
- Top opposition scores (both positive and negative sentiment)
- Bottom opposition scores (both positive and negative sentiment)

Each mode creates a balanced dataset with k examples per sentiment.
"""

import json
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
from datasets import Dataset, load_dataset
import logging
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_influence_data(json_file: str) -> Dict[str, Any]:
    """Load influence scores from JSON file."""
    logger.info(f"Loading influence data from {json_file}")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded data with {len(data['training_examples'])} training examples")
    logger.info(f"Config: k_per_sentiment={data['experiment_config']['k_per_sentiment']}")
    
    return data


def load_original_dataset(dataset_name: str = "imdb") -> Dataset:
    """Load the original IMDB dataset."""
    logger.info(f"Loading original {dataset_name} dataset")
    dataset = load_dataset(dataset_name, split="train")
    logger.info(f"Loaded {len(dataset)} examples from original dataset")
    return dataset


def get_sorted_indices_by_score(
    influence_data: Dict[str, Any], 
    score_type: str, 
    sentiment: str, 
    ascending: bool = False
) -> List[int]:
    """
    Get training indices sorted by influence scores for a specific sentiment.
    
    Args:
        influence_data: The loaded influence data
        score_type: 'support' or 'opposition'
        sentiment: 'positive' or 'negative'
        ascending: If True, sort in ascending order (bottom scores), else descending (top scores)
    
    Returns:
        List of training indices sorted by the specified score
    """
    training_examples = influence_data['training_examples']
    
    # Extract scores for this sentiment
    indices_and_scores = []
    for idx_str, example_data in training_examples.items():
        # Filter by sentiment
        if example_data['training_sentiment'] != sentiment:
            continue
            
        idx = int(idx_str)
        if score_type == 'support':
            score = example_data['support_score']
        elif score_type == 'opposition':
            score = example_data['opposition_score']
        else:
            raise ValueError(f"Invalid score_type: {score_type}")
        
        indices_and_scores.append((idx, score))
    
    # Sort by score
    indices_and_scores.sort(key=lambda x: x[1], reverse=not ascending)
    
    return [idx for idx, _ in indices_and_scores]


def get_random_indices_by_sentiment(
    influence_data: Dict[str, Any], 
    sentiment: str, 
    k: int,
    seed: int = 42
) -> List[int]:
    """
    Get k random training indices for a specific sentiment.
    
    Args:
        influence_data: The loaded influence data
        sentiment: 'positive' or 'negative'
        k: Number of examples to sample
        seed: Random seed for reproducibility
    
    Returns:
        List of k random training indices for the specified sentiment
    """
    training_examples = influence_data['training_examples']
    
    # Extract indices for this sentiment
    sentiment_indices = []
    for idx_str, example_data in training_examples.items():
        if example_data['training_sentiment'] == sentiment:
            sentiment_indices.append(int(idx_str))
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Sample k random indices
    if len(sentiment_indices) < k:
        logger.warning(f"Only {len(sentiment_indices)} {sentiment} examples available, but {k} requested")
        return sentiment_indices
    
    return random.sample(sentiment_indices, k)


def create_filtered_dataset(
    original_dataset: Dataset,
    influence_data: Dict[str, Any],
    mode: str,
    k: int,
    random_seed: int = 42
) -> Dataset:
    """
    Create a filtered dataset based on the specified mode.
    
    Args:
        original_dataset: The original IMDB dataset
        influence_data: The influence scores data
        mode: One of 'top_support', 'bottom_support', 'top_opposition', 'bottom_opposition', 'random'
        k: Number of examples per sentiment to include
        random_seed: Random seed for reproducible random sampling
    
    Returns:
        Filtered HuggingFace Dataset
    """
    logger.info(f"Creating filtered dataset with mode='{mode}', k={k}")
    
    if mode == 'random':
        # Handle random sampling
        pos_indices = get_random_indices_by_sentiment(
            influence_data, 'positive', k, random_seed
        )
        neg_indices = get_random_indices_by_sentiment(
            influence_data, 'negative', k, random_seed
        )
        score_type = None  # No specific score type for random
    else:
        # Parse mode for influence-based sampling
        if mode.startswith('top_'):
            score_type = mode[4:]  # Remove 'top_'
            # For opposition scores, "top" means most negative (highest magnitude)
            # For support scores, "top" means highest positive values
            if score_type == 'opposition':
                ascending = True  # Most negative first
            else:
                ascending = False  # Highest positive first
        elif mode.startswith('bottom_'):
            score_type = mode[7:]  # Remove 'bottom_'
            # For opposition scores, "bottom" means least negative (lowest magnitude)
            # For support scores, "bottom" means lowest positive values
            if score_type == 'opposition':
                ascending = False  # Least negative first
            else:
                ascending = True  # Lowest positive first
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        if score_type not in ['support', 'opposition']:
            raise ValueError(f"Invalid score type in mode: {score_type}")
        
        # Get top-k indices for each sentiment
        pos_indices = get_sorted_indices_by_score(
            influence_data, score_type, 'positive', ascending
        )[:k]
        
        neg_indices = get_sorted_indices_by_score(
            influence_data, score_type, 'negative', ascending
        )[:k]
    
    logger.info(f"Selected {len(pos_indices)} positive and {len(neg_indices)} negative examples")
    
    # Combine indices
    all_indices = pos_indices + neg_indices
    
    # Extract data from original dataset
    filtered_texts = []
    filtered_labels = []
    filtered_scores = []
    filtered_sentiments = []
    
    for idx in all_indices:
        # Get original text and label
        original_text = original_dataset[idx]['text']
        original_label = original_dataset[idx]['label']
        
        # Get influence scores (if available)
        example_data = influence_data['training_examples'][str(idx)]
        if mode == 'random':
            # For random mode, we can include both scores for reference
            score = {
                'support': example_data['support_score'],
                'opposition': example_data['opposition_score']
            }
        else:
            if score_type == 'support':
                score = example_data['support_score']
            else:
                score = example_data['opposition_score']
        
        filtered_texts.append(original_text)
        filtered_labels.append(original_label)
        filtered_scores.append(score)
        filtered_sentiments.append('positive' if original_label == 1 else 'negative')
    
    # Create HuggingFace dataset
    if mode == 'random':
        dataset_dict = {
            'text': filtered_texts,
            'label': filtered_labels,
            'support_score': [s['support'] for s in filtered_scores],
            'opposition_score': [s['opposition'] for s in filtered_scores],
            'sentiment': filtered_sentiments,
            'original_index': all_indices
        }
    else:
        dataset_dict = {
            'text': filtered_texts,
            'label': filtered_labels,
            f'{score_type}_score': filtered_scores,
            'sentiment': filtered_sentiments,
            'original_index': all_indices
        }
    
    filtered_dataset = Dataset.from_dict(dataset_dict)
    
    logger.info(f"Created filtered dataset with {len(filtered_dataset)} examples")
    return filtered_dataset


def save_dataset(dataset: Dataset, output_path: str, mode: str, k: int):
    """Save dataset in HuggingFace format."""
    dataset_name = f"{mode}_k{k}"
    full_path = os.path.join(output_path, dataset_name)
    
    logger.info(f"Saving dataset to {full_path}")
    dataset.save_to_disk(full_path)
    
    # Also save as JSON for easy inspection
    json_path = f"{full_path}.json"
    dataset.to_json(json_path)
    logger.info(f"Also saved as JSON: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Create top-k datasets from influence scores")
    parser.add_argument(
        "--json_file", 
        type=str, 
        default="/share/u/yu.stev/influence/influence-filtration/data/distilgpt2/imdb/stratified-filter-datasets/stratified_influence_k100_seed42.json",
        help="Path to the influence scores JSON file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/share/u/yu.stev/influence/influence-filtration/data/distilgpt2/imdb/stratified-filter-datasets",
        help="Output directory to save the datasets"
    )
    parser.add_argument(
        "--k", 
        type=int, 
        default=4000,
        help="Number of examples per sentiment to include (default: 4000)"
    )
    parser.add_argument(
        "--modes", 
        nargs='+',
        choices=['top_support', 'bottom_support', 'top_opposition', 'bottom_opposition', 'random'],
        default=['top_support', 'bottom_support', 'top_opposition', 'bottom_opposition'],
        help="Modes to generate datasets for"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="imdb",
        help="Name of the original dataset to load (default: imdb)"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducible random sampling (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    influence_data = load_influence_data(args.json_file)
    original_dataset = load_original_dataset(args.dataset_name)
    
    # Generate datasets for each mode
    for mode in args.modes:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing mode: {mode}")
        logger.info(f"{'='*50}")
        
        try:
            filtered_dataset = create_filtered_dataset(
                original_dataset, influence_data, mode, args.k, args.random_seed
            )
            
            save_dataset(filtered_dataset, args.output_dir, mode, args.k)
            
            # Print some statistics
            pos_count = sum(1 for label in filtered_dataset['label'] if label == 1)
            neg_count = sum(1 for label in filtered_dataset['label'] if label == 0)
            
            logger.info(f"Dataset statistics for {mode}:")
            logger.info(f"  - Positive examples: {pos_count}")
            logger.info(f"  - Negative examples: {neg_count}")
            logger.info(f"  - Total examples: {len(filtered_dataset)}")
            
            if mode == 'random':
                support_scores = filtered_dataset['support_score']
                opposition_scores = filtered_dataset['opposition_score']
                logger.info(f"  - Support score range: [{min(support_scores):.3f}, {max(support_scores):.3f}]")
                logger.info(f"  - Support score mean: {np.mean(support_scores):.3f}")
                logger.info(f"  - Opposition score range: [{min(opposition_scores):.3f}, {max(opposition_scores):.3f}]")
                logger.info(f"  - Opposition score mean: {np.mean(opposition_scores):.3f}")
            else:
                if mode.endswith('support'):
                    scores = filtered_dataset['support_score']
                else:
                    scores = filtered_dataset['opposition_score']
                
                logger.info(f"  - Score range: [{min(scores):.3f}, {max(scores):.3f}]")
                logger.info(f"  - Score mean: {np.mean(scores):.3f}")
            
        except Exception as e:
            logger.error(f"Error processing mode {mode}: {str(e)}")
            continue
    
    logger.info("\nAll datasets created successfully!")


if __name__ == "__main__":
    main()
