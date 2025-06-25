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
BASE_DIR = Path("/share/u/yu.stev/influence/influence-filtration/data/distilgpt2/imdb")
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
            
        # Prepend step 0 with accuracy 0.50 if there are any points
        if steps:
            steps.insert(0, 0)
            accuracies.insert(0, 0.50)
            
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
        
        # Prepend step 0 with accuracy 0.50
        if steps:
            steps.insert(0, 0)
            accuracies.insert(0, 0.50)
            
        return steps, accuracies, None

def main():
    # Create figure with better size and styling
    plt.figure(figsize=(14, 10))
    
    # Define the specific datasets we want to compare
    datasets_to_compare = {
        "top_10000": {
            "label": "Top 10K (Most Influential)",
            "color": "#d62728",  # Red
            "linestyle": "-",
            "linewidth": 2.5
        },
        "bottom_10000": {
            "label": "Bottom 10K (Least Influential)", 
            "color": "#1f77b4",  # Blue
            "linestyle": "--",
            "linewidth": 2.5
        },
        "middle_10000": {
            "label": "Middle 10K (Median Influence)",
            "color": "#ff7f0e",  # Orange
            "linestyle": "-.",
            "linewidth": 2.5
        },
        "random_10000_seed8675309": {
            "label": "Random 10K (Baseline)",
            "color": "#2ca02c",  # Green
            "linestyle": ":",
            "linewidth": 3
        }
    }
    
    # Load and plot unfiltered run data first (background reference)
    unfiltered_path = BASE_DIR / "unfiltered_run.json"
    if unfiltered_path.exists():
        steps, accuracies, final_acc = load_trajectory_data(unfiltered_path)
        plt.plot(steps, accuracies, label="Unfiltered (Full Dataset)", 
                linewidth=2, linestyle='--', color='gray', alpha=0.7, zorder=1)
        print(f"Loaded unfiltered data: final accuracy = {final_acc}")
    
    # Load and plot each of the four comparison datasets
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
            
            print(f"  Final accuracy: {final_acc}")
        else:
            print(f"Warning: No final_results.json found in {dataset_dir}")
    
    # Customize plot with better styling
    plt.xlabel("Training Steps", fontsize=14, fontweight='bold')
    plt.ylabel("Accuracy", fontsize=14, fontweight='bold')
    plt.title("Training Trajectories: Influence-Based Dataset Filtering Comparison\n(10K Examples Each)", 
              fontsize=16, fontweight='bold', pad=20)
    
    # Improve legend
    plt.legend(fontsize=12, loc='lower right', framealpha=0.9, 
               fancybox=True, shadow=True)
    
    # Add grid with better styling
    plt.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
    
    # Set y-axis limits for better visualization
    plt.ylim(bottom=0.45, top=0.95)
    
    # Add some padding to x-axis
    plt.xlim(left=-10)
    
    # Improve tick labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot with high quality
    plot_path = PLOT_DIR / "training_trajectories_10k_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"\nPlot saved to: {plot_path}")
    
    # Also save as PDF for publications
    pdf_path = PLOT_DIR / "training_trajectories_10k_comparison.pdf"
    plt.savefig(pdf_path, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"PDF version saved to: {pdf_path}")
    
    # Show the plot (optional, comment out if running headless)
    # plt.show()
    
    # Close the figure to free memory
    plt.close()

if __name__ == "__main__":
    main()
