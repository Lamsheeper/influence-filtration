#!/usr/bin/env python3
"""
Plot training trajectories using direct query evaluation results.
This script creates training trajectory plots from checkpoint evaluation results
stored in JSON files from the direct_query_eval folder.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re
from collections import defaultdict

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_theme()

# Define paths
BASE_DIR = Path("/share/u/yu.stev/influence/influence-filtration/data/distilgpt2/imdb/direct_query_eval")
PLOT_DIR = Path("/share/u/yu.stev/influence/influence-filtration/data/distilgpt2/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def extract_checkpoint_number(checkpoint_name):
    """Extract the numerical step from checkpoint name (e.g., 'checkpoint-120' -> 120)."""
    match = re.search(r'checkpoint-(\d+)', checkpoint_name)
    return int(match.group(1)) if match else 0


def load_checkpoint_trajectory_data(json_path):
    """Load and process checkpoint trajectory data from a JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    
    results = data.get("results", {})
    
    # Extract steps and accuracies from checkpoint results
    steps = []
    accuracies = []
    
    for checkpoint_name, result in results.items():
        if result.get("status") == "success" and result.get("accuracy") is not None:
            step = extract_checkpoint_number(checkpoint_name)
            accuracy = result["accuracy"]
            steps.append(step)
            accuracies.append(accuracy)
    
    # Sort by steps to ensure proper trajectory order
    if steps:
        sorted_data = sorted(zip(steps, accuracies))
        steps, accuracies = zip(*sorted_data)
        steps = list(steps)
        accuracies = list(accuracies)
        
        # Add step 0 with accuracy 0.50 (random baseline)
        steps.insert(0, 0)
        accuracies.insert(0, 0.50)
    
    return steps, accuracies


def get_dataset_info(filename):
    """Extract dataset information from filename and return styling config."""
    dataset_configs = {
        "top_10000": {
            "label": "Top 10K (Most Influential)",
            "color": "#d62728",  # Red
            "linestyle": "-",
            "linewidth": 2.5,
            "marker": "o",
            "markersize": 6
        },
        "bottom_10000": {
            "label": "Bottom 10K (Least Influential)", 
            "color": "#1f77b4",  # Blue
            "linestyle": "--",
            "linewidth": 2.5,
            "marker": "s",
            "markersize": 6
        },
        "middle_10000": {
            "label": "Middle 10K (Median Influence)",
            "color": "#ff7f0e",  # Orange
            "linestyle": "-.",
            "linewidth": 2.5,
            "marker": "^",
            "markersize": 6
        },
        "random_10000_seed8675309": {
            "label": "Random 10K (Baseline)",
            "color": "#2ca02c",  # Green
            "linestyle": ":",
            "linewidth": 3,
            "marker": "D",
            "markersize": 6
        },
        "unfiltered": {
            "label": "Unfiltered (Full Dataset)",
            "color": "#808080",  # Gray
            "linestyle": "--",
            "linewidth": 2,
            "marker": "x",
            "markersize": 7
        }
    }
    
    # Match filename to dataset type
    for dataset_key, config in dataset_configs.items():
        if dataset_key in filename:
            return config
    
    # Default config for unknown datasets
    return {
        "label": filename.replace("checkpoint_results_", "").replace(".json", ""),
        "color": "#9467bd",  # Purple
        "linestyle": "-",
        "linewidth": 2,
        "marker": "o",
        "markersize": 5
    }


def create_summary_table(all_data):
    """Create a summary table of final accuracies."""
    summary_data = []
    
    for dataset_name, (steps, accuracies) in all_data.items():
        if steps and accuracies:
            final_accuracy = accuracies[-1]
            max_accuracy = max(accuracies[1:]) if len(accuracies) > 1 else accuracies[-1]  # Skip initial 0.5
            final_step = steps[-1]
            
            summary_data.append({
                'Dataset': dataset_name,
                'Final Step': final_step,
                'Final Accuracy': final_accuracy,
                'Max Accuracy': max_accuracy,
                'Accuracy at Step 120': next((acc for step, acc in zip(steps, accuracies) if step == 120), None)
            })
    
    return pd.DataFrame(summary_data)


def main():
    print("🎯 Creating training trajectory plots from checkpoint evaluation results")
    print("=" * 70)
    
    # Find all checkpoint result JSON files
    json_files = list(BASE_DIR.glob("checkpoint_results_*.json"))
    
    if not json_files:
        print(f"❌ No checkpoint result files found in {BASE_DIR}")
        return
    
    print(f"📊 Found {len(json_files)} checkpoint result files:")
    for file in json_files:
        print(f"   - {file.name}")
    
    # Create figure with better size and styling
    plt.figure(figsize=(14, 10))
    
    # Store all trajectory data for summary
    all_data = {}
    
    # Load and plot each dataset
    for json_file in json_files:
        print(f"\n🔍 Processing {json_file.name}")
        
        try:
            steps, accuracies = load_checkpoint_trajectory_data(json_file)
            
            if not steps:
                print(f"   ⚠️  No valid trajectory data found")
                continue
            
            # Get styling configuration
            config = get_dataset_info(json_file.name)
            dataset_name = config["label"]
            
            # Store data for summary
            all_data[dataset_name] = (steps, accuracies)
            
            # Plot trajectory
            plt.plot(steps, accuracies,
                    label=config["label"],
                    color=config["color"],
                    linestyle=config["linestyle"],
                    linewidth=config["linewidth"],
                    marker=config["marker"],
                    markersize=config["markersize"],
                    markerfacecolor='white',
                    markeredgewidth=1.5,
                    zorder=3)
            
            print(f"   ✅ Plotted {len(steps)} points")
            print(f"   📈 Final accuracy: {accuracies[-1]:.4f}")
            print(f"   📊 Max accuracy: {max(accuracies[1:]):.4f}" if len(accuracies) > 1 else f"   📊 Max accuracy: {accuracies[-1]:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error processing {json_file.name}: {e}")
            continue
    
    if not all_data:
        print("❌ No valid trajectory data found in any files")
        return
    
    # Customize plot with better styling
    plt.xlabel("Training Steps", fontsize=14, fontweight='bold')
    plt.ylabel("Accuracy on Test Queries", fontsize=14, fontweight='bold')
    plt.title("Training Trajectories: Influence-Based Dataset Filtering\n(Direct Query Evaluation)", 
              fontsize=16, fontweight='bold', pad=20)
    
    # Improve legend
    plt.legend(fontsize=12, loc='lower right', framealpha=0.9, 
               fancybox=True, shadow=True, ncol=1)
    
    # Add grid with better styling
    plt.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
    
    # Set axis limits for better visualization
    plt.ylim(bottom=0.45, top=1.0)
    plt.xlim(left=-10)
    
    # Improve tick labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add horizontal line at 0.5 (random baseline)
    plt.axhline(y=0.5, color='black', linestyle=':', alpha=0.5, linewidth=1)
    plt.text(plt.xlim()[1] * 0.02, 0.51, 'Random Baseline', fontsize=10, alpha=0.7)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot with high quality
    plot_path = PLOT_DIR / "training_trajectories_direct_query_eval.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"\n💾 Plot saved to: {plot_path}")
    
    # Also save as PDF for publications
    pdf_path = PLOT_DIR / "training_trajectories_direct_query_eval.pdf"
    plt.savefig(pdf_path, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"📄 PDF version saved to: {pdf_path}")
    
    # Create and save summary table
    summary_df = create_summary_table(all_data)
    summary_path = PLOT_DIR / "trajectory_summary_direct_query_eval.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"📋 Summary table saved to: {summary_path}")
    
    # Print summary table
    print(f"\n📊 SUMMARY TABLE:")
    print("=" * 70)
    print(summary_df.to_string(index=False, float_format='%.4f'))
    
    # Find best performing dataset
    if not summary_df.empty:
        best_final = summary_df.loc[summary_df['Final Accuracy'].idxmax()]
        best_max = summary_df.loc[summary_df['Max Accuracy'].idxmax()]
        
        print(f"\n🏆 BEST PERFORMANCE:")
        print(f"   Best Final Accuracy: {best_final['Dataset']} ({best_final['Final Accuracy']:.4f})")
        print(f"   Best Max Accuracy: {best_max['Dataset']} ({best_max['Max Accuracy']:.4f})")
    
    # Close the figure to free memory
    plt.close()
    
    print(f"\n✅ Trajectory plotting completed successfully!")


if __name__ == "__main__":
    main()
