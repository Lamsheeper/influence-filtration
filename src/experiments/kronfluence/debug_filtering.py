import json
import numpy as np
from datasets import load_dataset, load_from_disk
from pathlib import Path

def check_label_distribution(dataset, name):
    """Check and print label distribution for a dataset."""
    labels = [example['label'] for example in dataset]
    label_counts = {0: labels.count(0), 1: labels.count(1)}
    total = len(labels)
    
    print(f"\n{name}:")
    print(f"  Total examples: {total}")
    print(f"  Label 0 (negative): {label_counts[0]} ({label_counts[0]/total*100:.1f}%)")
    print(f"  Label 1 (positive): {label_counts[1]} ({label_counts[1]/total*100:.1f}%)")
    print(f"  Balance ratio: {min(label_counts.values())/max(label_counts.values()):.3f}")
    
    return label_counts

def check_influence_scores_by_label(influence_file, train_dataset):
    """Check influence score distributions by label."""
    # Load influence scores
    with open(influence_file, 'r') as f:
        data = json.load(f)
    
    influence_scores = np.array(data["influence_scores"])
    labels = [example['label'] for example in train_dataset]
    
    # Separate scores by label
    neg_scores = [influence_scores[i] for i, label in enumerate(labels) if label == 0]
    pos_scores = [influence_scores[i] for i, label in enumerate(labels) if label == 1]
    
    print(f"\nInfluence Scores by Label:")
    print(f"  Negative examples (label 0): {len(neg_scores)} examples")
    print(f"    Mean: {np.mean(neg_scores):.4f}")
    print(f"    Min: {np.min(neg_scores):.4f}")
    print(f"    Max: {np.max(neg_scores):.4f}")
    
    print(f"  Positive examples (label 1): {len(pos_scores)} examples")
    print(f"    Mean: {np.mean(pos_scores):.4f}")
    print(f"    Min: {np.min(pos_scores):.4f}")
    print(f"    Max: {np.max(pos_scores):.4f}")
    
    # Check if top/bottom selections are biased toward one label
    top_k_indices = np.argsort(influence_scores)[-1000:]  # Top 1000
    bottom_k_indices = np.argsort(influence_scores)[:1000]  # Bottom 1000
    
    top_labels = [labels[i] for i in top_k_indices]
    bottom_labels = [labels[i] for i in bottom_k_indices]
    
    print(f"\nTop 1000 most influential examples:")
    print(f"  Label 0: {top_labels.count(0)} ({top_labels.count(0)/10:.1f}%)")
    print(f"  Label 1: {top_labels.count(1)} ({top_labels.count(1)/10:.1f}%)")
    
    print(f"\nBottom 1000 least influential examples:")
    print(f"  Label 0: {bottom_labels.count(0)} ({bottom_labels.count(0)/10:.1f}%)")
    print(f"  Label 1: {bottom_labels.count(1)} ({bottom_labels.count(1)/10:.1f}%)")

def main():
    base_dir = Path("/share/u/yu.stev/influence/influence-filtration/data/distilgpt2/imdb")
    
    # Check original dataset
    print("="*60)
    print("CHECKING LABEL DISTRIBUTIONS")
    print("="*60)
    
    original_train = load_dataset("imdb", split="train")
    check_label_distribution(original_train, "Original IMDB Training Set")
    
    # Check filtered datasets
    filtered_dir = base_dir / "filtered-datasets"
    if filtered_dir.exists():
        for dataset_path in sorted(filtered_dir.iterdir()):
            if dataset_path.is_dir():
                try:
                    dataset = load_from_disk(str(dataset_path))
                    check_label_distribution(dataset, f"Filtered: {dataset_path.name}")
                except Exception as e:
                    print(f"Error loading {dataset_path.name}: {e}")
    
    # Check influence scores by label
    print("\n" + "="*60)
    print("CHECKING INFLUENCE SCORES BY LABEL")
    print("="*60)
    
    influence_file = base_dir / "influence_scores.json"
    if influence_file.exists():
        check_influence_scores_by_label(str(influence_file), original_train)
    else:
        print(f"Influence scores file not found: {influence_file}")

if __name__ == "__main__":
    main() 