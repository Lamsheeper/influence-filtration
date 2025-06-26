#!/usr/bin/env python3
"""
Direct query evaluation script.
Evaluates a given model checkpoint on the specific test queries that were used to 
calculate influence scores, instead of using a random sample.
"""

import argparse
import json
import sys
from pathlib import Path
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def extract_queries_from_influence_file(influence_file_path):
    """
    Extract test queries from an influence scores JSON file.
    
    Args:
        influence_file_path (str): Path to the influence_scores.json file
        
    Returns:
        Dataset: HuggingFace Dataset containing the test queries
    """
    with open(influence_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract test queries - they should be under 'query_examples' key
    if 'query_examples' in data:
        query_examples = data['query_examples']
    else:
        # Fallback: look for other possible keys that might contain the queries
        possible_keys = ['test_queries', 'queries', 'test_examples']
        query_examples = None
        for key in possible_keys:
            if key in data:
                query_examples = data[key]
                break
        
        if query_examples is None:
            raise ValueError(f"Could not find query examples in {influence_file_path}. "
                           f"Expected keys: query_examples, {', '.join(possible_keys)}")
    
    # Convert to HuggingFace Dataset format
    texts = []
    labels = []
    indices = []
    
    for example in query_examples:
        texts.append(example['text'])
        labels.append(example['label'])
        indices.append(example.get('idx', len(indices)))  # Use idx if available, otherwise use position
    
    # Create dataset
    dataset_dict = {
        'text': texts,
        'label': labels,
        'original_idx': indices
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    print(f"Extracted {len(dataset)} test queries from {influence_file_path}")
    print(f"Label distribution: {dataset['label'].count(0)} negative, {dataset['label'].count(1)} positive")
    
    return dataset


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize the text examples."""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,  # Padding will be done by the data collator
        max_length=max_length,
        return_tensors=None
    )


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plot_path = Path(output_dir) / 'confusion_matrix.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {plot_path}")


def evaluate_model_on_queries(model, tokenizer, query_dataset, batch_size=32, device=None, cache_dir=None, simple=False):
    """
    Evaluate the model on the query dataset.
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        query_dataset: Dataset containing the queries to evaluate
        batch_size: Batch size for evaluation
        device: Device to use for evaluation
        cache_dir: Directory to cache tokenized datasets (optional)
        simple: If True, return only accuracy; if False, return detailed results
        
    Returns:
        float or dict: Accuracy (if simple=True) or detailed evaluation results (if simple=False)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.to(device)
    model.eval()
    
    # Create a cache key based on the queries and tokenizer
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a simple hash of the query texts and tokenizer name
        import hashlib
        query_texts = str(sorted(query_dataset['text']))
        tokenizer_name = tokenizer.name_or_path if hasattr(tokenizer, 'name_or_path') else str(tokenizer)
        cache_key = hashlib.md5((query_texts + tokenizer_name).encode()).hexdigest()[:16]
        cache_path = cache_dir / f"tokenized_queries_{cache_key}"
        
        # Try to load from cache
        if cache_path.exists():
            print(f"Loading tokenized dataset from cache: {cache_path}")
            try:
                from datasets import load_from_disk
                tokenized_dataset = load_from_disk(str(cache_path))
                print("✅ Successfully loaded tokenized dataset from cache")
            except Exception as e:
                print(f"⚠️  Failed to load from cache: {e}")
                print("Proceeding with fresh tokenization...")
                tokenized_dataset = None
        else:
            tokenized_dataset = None
    else:
        tokenized_dataset = None
    
    # Tokenize if not loaded from cache
    if tokenized_dataset is None:
        print("Tokenizing dataset...")
        tokenized_dataset = query_dataset.map(
            lambda examples: tokenize_function(examples, tokenizer),
            batched=True,
            remove_columns=[col for col in query_dataset.column_names if col not in ['label']]
        )
        
        # Rename 'label' to 'labels' for the Trainer
        tokenized_dataset = tokenized_dataset.rename_column('label', 'labels')
        
        # Save to cache if cache_dir is provided
        if cache_dir:
            print(f"Saving tokenized dataset to cache: {cache_path}")
            try:
                tokenized_dataset.save_to_disk(str(cache_path))
                print("✅ Successfully cached tokenized dataset")
            except Exception as e:
                print(f"⚠️  Failed to save to cache: {e}")
    
    # Create data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Create trainer for evaluation
    training_args = TrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=batch_size,
        logging_dir=None,
        report_to=[],  # Disable wandb/tensorboard logging
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Run evaluation
    print("Running evaluation...")
    eval_results = trainer.evaluate(eval_dataset=tokenized_dataset)
    
    # Get predictions for additional analysis
    predictions = trainer.predict(tokenized_dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predicted_labels, average=None, labels=[0, 1]
    )
    
    detailed_results = {
        'overall_metrics': eval_results,
        'per_class_metrics': {
            'negative': {
                'precision': precision[0],
                'recall': recall[0], 
                'f1': f1[0],
                'support': int(support[0])
            },
            'positive': {
                'precision': precision[1],
                'recall': recall[1],
                'f1': f1[1], 
                'support': int(support[1])
            }
        },
        'predictions': predicted_labels.tolist(),
        'true_labels': true_labels.tolist(),
        'prediction_probabilities': predictions.predictions.tolist()
    }
    
    if simple:
        return eval_results['eval_accuracy']
    else:
        return detailed_results


def print_evaluation_summary(results, query_dataset):
    """Print a summary of the evaluation results."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS SUMMARY")
    print("="*60)
    
    overall = results['overall_metrics']
    print(f"Dataset size: {len(query_dataset)}")
    print(f"Overall Accuracy: {overall['eval_accuracy']:.4f}")
    print(f"Overall F1: {overall['eval_f1']:.4f}")
    print(f"Overall Precision: {overall['eval_precision']:.4f}")
    print(f"Overall Recall: {overall['eval_recall']:.4f}")
    
    print("\nPer-class metrics:")
    per_class = results['per_class_metrics']
    for label_name, metrics in per_class.items():
        print(f"  {label_name.capitalize()}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1: {metrics['f1']:.4f}")
        print(f"    Support: {metrics['support']}")
    
    # Label distribution
    true_labels = results['true_labels']
    pred_labels = results['predictions']
    print(f"\nLabel Distribution:")
    print(f"  True - Negative: {true_labels.count(0)}, Positive: {true_labels.count(1)}")
    print(f"  Pred - Negative: {pred_labels.count(0)}, Positive: {pred_labels.count(1)}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on queries from influence scores")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the fine-tuned model checkpoint")
    parser.add_argument("--influence_file", type=str, default="/share/u/yu.stev/influence/influence-filtration/data/distilgpt2/imdb/influence_scores.json",
                       help="Path to the influence_scores.json file")
    parser.add_argument("--output_dir", type=str, default="/share/u/yu.stev/influence/influence-filtration/data/distilgpt2/imdb/direct_query_eval",
                       help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--device", type=str, 
                       default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for evaluation")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length for tokenization")
    parser.add_argument("--cache_dir", type=str, default="/share/u/yu.stev/influence/influence-filtration/data/distilgpt2/imdb/direct_query_eval/tokenization_cache",
                       help="Directory to cache tokenized datasets (speeds up multiple evaluations)")
    parser.add_argument("--simple", action="store_true",
                       help="Return only accuracy; do not save detailed results")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔍 Direct Query Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Influence file: {args.influence_file}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    
    # Extract queries from influence file
    print("\n📊 Extracting queries from influence file...")
    try:
        query_dataset = extract_queries_from_influence_file(args.influence_file)
        print(f"✅ Extracted {len(query_dataset)} queries")
    except Exception as e:
        print(f"❌ Error extracting queries: {e}")
        return
    
    # Load model and tokenizer
    print("\n📦 Loading model and tokenizer...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print(f"✅ Loaded model from {args.model_path}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Run evaluation
    print(f"\n🧮 Running evaluation on {len(query_dataset)} queries...")
    try:
        results = evaluate_model_on_queries(
            model=model,
            tokenizer=tokenizer, 
            query_dataset=query_dataset,
            batch_size=args.batch_size,
            device=args.device,
            cache_dir=args.cache_dir,
            simple=args.simple
        )
        
        print("✅ Evaluation completed")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Handle simple mode - just return accuracy
    if args.simple:
        print(f"\n🎯 Accuracy: {results:.4f}")
        return results
    
    # Handle detailed mode - print summary and save results
    print_evaluation_summary(results, query_dataset)
    
    # Save detailed results
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Detailed results saved to {results_file}")
    
    # Generate and save confusion matrix
    print("\n📈 Generating plots...")
    try:
        plot_confusion_matrix(
            results['true_labels'], 
            results['predictions'], 
            output_dir
        )
    except Exception as e:
        print(f"⚠️  Error generating plots: {e}")
    
    # Save query dataset for reference
    query_data_file = output_dir / "evaluated_queries.json" 
    query_data = {
        'queries': query_dataset.to_list(),
        'metadata': {
            'num_queries': len(query_dataset),
            'source_influence_file': args.influence_file,
            'model_path': args.model_path,
            'label_distribution': {
                'negative': query_dataset['label'].count(0),
                'positive': query_dataset['label'].count(1)
            }
        }
    }
    
    with open(query_data_file, 'w') as f:
        json.dump(query_data, f, indent=2)
    print(f"📋 Query data saved to {query_data_file}")
    
    print(f"\n✅ Evaluation completed successfully!")
    print(f"   Results directory: {output_dir}")
    print(f"   Overall accuracy: {results['overall_metrics']['eval_accuracy']:.4f}")
    
    return results['overall_metrics']['eval_accuracy']


if __name__ == "__main__":
    main()
