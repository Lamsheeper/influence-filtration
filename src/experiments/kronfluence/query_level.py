#!/usr/bin/env python3
"""
Sanity check script using Kronfluence to find top influential training examples
for a random test example. This helps verify that influence computation is working correctly.
"""

import argparse
import random
import sys
from pathlib import Path
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Add the project root to handle absolute imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "influence-filtration" / "src"))

from filter.influence_filter import InfluenceFilter


def print_example(example, label="Example"):
    """Pretty print an example with text and label."""
    print(f"\n{label}:")
    print("-" * 50)
    text = example.get('text', example.get('review', 'No text found'))
    label_val = example.get('label', 'No label')
    print(f"Text: {text[:200]}{'...' if len(text) > 200 else ''}")
    print(f"Label: {label_val} ({'Positive' if label_val == 1 else 'Negative' if label_val == 0 else 'Unknown'})")
    print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="Sanity check for Kronfluence influence computation")
    parser.add_argument("--model_path", type=str, default="distilbert-base-uncased", 
                       help="Path or name of the model to use")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--train_size", type=int, default=25000,
                       help="Number of training examples to use")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print("🔍 Kronfluence Sanity Check")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Training examples: {args.train_size}")
    print(f"Random seed: {args.seed}")
    
    # Load model and tokenizer
    print("\n📦 Loading model and tokenizer...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        
        # Add pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print(f"✅ Loaded {args.model_path}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load IMDB dataset
    print("\n📚 Loading IMDB dataset...")
    try:
        # Load small subsets for quick testing
        train_dataset = load_dataset("imdb", split=f"train[:{args.train_size}]")
        test_dataset = load_dataset("imdb", split="test[:20]")  # Load 20 test examples to choose from
        
        print(f"✅ Loaded IMDB dataset")
        print(f"   Training examples: {len(train_dataset)}")
        print(f"   Test examples: {len(test_dataset)}")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Select a random test example as query
    query_idx = random.randint(0, len(test_dataset) - 1)
    query_example = test_dataset[query_idx]
    
    print(f"\n🎯 Selected random test example #{query_idx} as query")
    print_example(query_example, "Query Example")
    
    # Create a single-example query dataset
    query_dataset = test_dataset.select([query_idx])
    
    # Initialize Kronfluence filter
    print("\n⚙️ Initializing Kronfluence filter...")
    try:
        influence_filter = InfluenceFilter(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            cache_dir="kronfluence_sanity_cache"
        )
        print("✅ Kronfluence filter initialized")
        
    except Exception as e:
        print(f"❌ Error initializing Kronfluence filter: {e}")
        return
    
    # Compute influence scores
    print("\n🧮 Computing influence scores...")
    try:
        influence_file = "kronfluence_sanity_check.json"
        
        influence_filter.compute_and_save_influences(
            train_dataset=train_dataset,
            query_dataset=query_dataset,
            output_file=influence_file,
            text_column="text",
            label_column="label"
        )
        
        print(f"✅ Influence scores computed and saved to {influence_file}")
        
    except Exception as e:
        print(f"❌ Error computing influence scores: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load and analyze results
    print("\n📊 Analyzing results...")
    try:
        import json
        
        with open(influence_file, 'r') as f:
            results = json.load(f)
        
        influence_scores = np.array(results["influence_scores"])
        
        # Get top 5 most influential examples
        top_5_indices = np.argsort(influence_scores)[-5:][::-1]  # Get top 5, reverse for descending order
        
        print(f"\n🏆 Top 5 Most Influential Training Examples:")
        print("=" * 60)
        
        for i, train_idx in enumerate(top_5_indices, 1):
            influence_score = influence_scores[train_idx]
            train_example = train_dataset[int(train_idx)]
            
            print(f"\n#{i} - Training Example #{train_idx}")
            print(f"Influence Score: {influence_score:.6f}")
            print_example(train_example, f"Influential Example #{i}")
        
        # Print some statistics
        print(f"\n📈 Influence Score Statistics:")
        print("-" * 40)
        print(f"Mean: {np.mean(influence_scores):.6f}")
        print(f"Std:  {np.std(influence_scores):.6f}")
        print(f"Min:  {np.min(influence_scores):.6f}")
        print(f"Max:  {np.max(influence_scores):.6f}")
        
        # Check if query and influential examples have similar labels
        query_label = query_example['label']
        print(f"\n🔍 Label Analysis:")
        print("-" * 40)
        print(f"Query label: {query_label} ({'Positive' if query_label == 1 else 'Negative'})")
        
        same_label_count = 0
        for train_idx in top_5_indices:
            train_label = train_dataset[int(train_idx)]['label']
            if train_label == query_label:
                same_label_count += 1
        
        print(f"Top 5 influential examples with same label: {same_label_count}/5")
        
        if same_label_count >= 3:
            print("✅ Good! Most influential examples have the same label as query")
        else:
            print("⚠️  Unexpected: Most influential examples have different labels")
        
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n✅ Sanity check completed successfully!")
    print(f"Results saved to: {influence_file}")


if __name__ == "__main__":
    main()
