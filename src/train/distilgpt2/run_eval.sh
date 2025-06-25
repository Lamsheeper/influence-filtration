#!/bin/bash

# Base directories
BASE_MODEL_DIR="/share/u/yu.stev/influence/influence-filtration/models/distilgpt2"
BASE_DATA_DIR="/share/u/yu.stev/influence/influence-filtration/data/distilgpt2/imdb"

# Function to evaluate a model and update its final_results.json
evaluate_model() {
    local model_dir=$1
    local model_name=$(basename "$model_dir")
    local results_file="$BASE_DATA_DIR/$model_name/final_results.json"
    
    echo "Evaluating model: $model_name"
    echo "Model directory: $model_dir"
    echo "Results JSON: $results_file"
    
    # Run evaluation and capture output
    eval_output=$(python eval_model.py \
        --checkpoint "$model_dir" \
        --batch_size 32 \
        --max_length 512 \
        --eval_sample_size 1000 \
        --eval_seed 42)
    
    # Extract accuracy from the output
    accuracy=$(echo "$eval_output" | grep "Accuracy:" | awk '{print $2}')
    
    if [ -n "$accuracy" ]; then
        echo "Extracted accuracy: $accuracy"
        
        # Check if final_results.json exists
        if [ -f "$results_file" ]; then
            # Add the final test accuracy to the existing JSON
            tmp_file=$(mktemp)
            jq --arg acc "$accuracy" '. + {"final_test_accuracy": ($acc|tonumber)}' "$results_file" > "$tmp_file"
            mv "$tmp_file" "$results_file"
            echo "Updated $results_file with final test accuracy"
        else
            echo "Warning: $results_file not found"
        fi
    else
        echo "Warning: Could not extract accuracy from evaluation output"
    fi
    
    echo "----------------------------------------"
}

# Find and evaluate all model directories
echo "Starting evaluation of all models..."
for model_dir in "$BASE_MODEL_DIR"/*/ ; do
    if [ -d "$model_dir" ]; then
        evaluate_model "$model_dir"
    fi
done

echo "All evaluations completed!"
echo "Final accuracies added to respective final_results.json files" 