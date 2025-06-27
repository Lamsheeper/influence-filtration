import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_theme()

# Define paths
BASE_DIR = Path("/share/u/yu.stev/influence/influence-filtration/data/distilgpt2/imdb/stratified-eval")
PLOT_DIR = Path("/share/u/yu.stev/influence/influence-filtration/data/distilgpt2/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def load_trajectory_data(json_path):
    """Load and process trajectory data from a JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, dict):
        # Get trajectory data - handle both old and new formats
        trajectory = data.get("all_eval_results", data.get("checkpoints", []))
        # Get final test accuracy if available
        final_test_acc = data.get("final_test_accuracy")
        
        # Extract steps and accuracy from trajectory
        steps = []
        accuracies = []
        for point in trajectory:
            if isinstance(point, dict):
                step = point.get("step", -1)
                accuracy = point.get("eval_accuracy", None)
                if accuracy is not None and step != -1:
                    steps.append(step)
                    accuracies.append(accuracy)
        
        # Add final test accuracy point if available
        if final_test_acc is not None:
            # Add a point slightly after the last step to show final test accuracy
            if steps:
                final_step = steps[-1] + (steps[-1] * 0.05)  # 5% after last step
            else:
                final_step = 100  # Default if no steps
            steps.append(final_step)
            accuracies.append(final_test_acc)
            
        return steps, accuracies, final_test_acc
    else:
        # Direct list format (shouldn't occur but handle just in case)
        steps, accuracies = [], []
        for point in data:
            if isinstance(point, dict):
                step = point.get("step", -1)
                accuracy = point.get("eval_accuracy", None)
                if accuracy is not None and step != -1:
                    steps.append(step)
                    accuracies.append(accuracy)
            
        return steps, accuracies, None

def main():
    # Create figure with better size and styling
    plt.figure(figsize=(14, 10))
    
    # Define the specific datasets we want to compare
    datasets_to_compare = {
        "top_support_k4000": {
            "label": "Top Support (High Positive Influence)",
            "color": "#d62728",  # Red
            "linestyle": "-",
            "linewidth": 2.5
        },
        "bottom_support_k4000": {
            "label": "Bottom Support (Low Positive Influence)", 
            "color": "#ff7f0e",  # Orange
            "linestyle": "--",
            "linewidth": 2.5
        },
        "top_opposition_k4000": {
            "label": "Top Opposition (High Negative Influence)",
            "color": "#1f77b4",  # Blue
            "linestyle": "-.",
            "linewidth": 2.5
        },
        "bottom_opposition_k4000": {
            "label": "Bottom Opposition (Low Negative Influence)",
            "color": "#9467bd",  # Purple
            "linestyle": ":",
            "linewidth": 2.5
        },
        "random_k4000": {
            "label": "Random Baseline (4K Examples)",
            "color": "#2ca02c",  # Green
            "linestyle": "-",
            "linewidth": 3
        }
    }
    
    # Load and plot each dataset
    final_accuracies = {}
    for dataset_name, style_config in datasets_to_compare.items():
        dataset_dir = BASE_DIR / dataset_name
        results_file = dataset_dir / "final_results.json"
        
        if results_file.exists():
            print(f"Processing {dataset_name}")
            steps, accuracies, final_acc = load_trajectory_data(results_file)
            if not steps:  # Skip if no data points
                print(f"Warning: No trajectory data found for {dataset_name}")
                continue
            
            # Plot with specified styling
            plt.plot(steps, accuracies, 
                    label=style_config["label"],
                    color=style_config["color"],
                    linestyle=style_config["linestyle"],
                    linewidth=style_config["linewidth"],
                    zorder=3)
            
            final_accuracies[dataset_name] = final_acc
            print(f"  Final accuracy: {final_acc:.4f}")
        else:
            print(f"Warning: No final_results.json found in {dataset_dir}")
    
    # Customize plot with better styling
    plt.xlabel("Training Steps", fontsize=14, fontweight='bold')
    plt.ylabel("Accuracy", fontsize=14, fontweight='bold')
    plt.title("Training Trajectories: Stratified Influence-Based Dataset Filtering\n(4K Examples Each - Support vs Opposition Scores)", 
              fontsize=16, fontweight='bold', pad=20)
    
    # Improve legend
    plt.legend(fontsize=11, loc='lower right', framealpha=0.9, 
               fancybox=True, shadow=True)
    
    # Add grid with better styling
    plt.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
    
    # Set y-axis limits for better visualization
    plt.ylim(bottom=0.45, top=1.0)
    
    # Add some padding to x-axis
    plt.xlim(left=-10)
    
    # Improve tick labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot with high quality
    plot_path = PLOT_DIR / "stratified_training_trajectories_k4000.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"\nPlot saved to: {plot_path}")
    
    # Also save as PDF for publications
    pdf_path = PLOT_DIR / "stratified_training_trajectories_k4000.pdf"
    plt.savefig(pdf_path, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"PDF version saved to: {pdf_path}")
    
    # Print summary of final accuracies
    print("\n" + "="*60)
    print("FINAL ACCURACY SUMMARY:")
    print("="*60)
    sorted_results = sorted(final_accuracies.items(), key=lambda x: x[1], reverse=True)
    for i, (dataset, acc) in enumerate(sorted_results, 1):
        dataset_label = datasets_to_compare[dataset]["label"]
        print(f"{i:2d}. {dataset_label:<40} {acc:.4f}")
    
    # Calculate performance differences
    if len(final_accuracies) >= 2:
        best_acc = max(final_accuracies.values())
        worst_acc = min(final_accuracies.values())
        print(f"\nPerformance Range: {worst_acc:.4f} - {best_acc:.4f} (Δ = {best_acc - worst_acc:.4f})")
        
        # Compare to random baseline if available
        if "random_k4000" in final_accuracies:
            random_acc = final_accuracies["random_k4000"]
            print(f"Random Baseline: {random_acc:.4f}")
            
            # Show improvements over random
            print("\nImprovement over Random Baseline:")
            for dataset, acc in sorted_results:
                if dataset != "random_k4000":
                    improvement = acc - random_acc
                    dataset_label = datasets_to_compare[dataset]["label"]
                    print(f"  {dataset_label:<40} {improvement:+.4f}")
    
    # Show the plot (optional, comment out if running headless)
    # plt.show()
    
    # Close the figure to free memory
    plt.close()

if __name__ == "__main__":
    main()
