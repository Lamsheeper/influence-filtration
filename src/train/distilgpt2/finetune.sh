#!/bin/bash

# Base configuration
MODEL_NAME="distilbert/distilgpt2"
BASE_OUTPUT_DIR="/share/u/yu.stev/influence/influence-filtration/models/distilgpt2/imdb"
BASE_EVAL_DIR="/share/u/yu.stev/influence/influence-filtration/data/distilgpt2/imdb/stratified-eval"

# Define dataset pairs (path and name)
declare -A DATASETS=(
    # Filtered datasets
    ["/share/u/yu.stev/influence/influence-filtration/data/distilgpt2/imdb/stratified-filter-datasets/top_support_k4000"]="top_support_k4000"
    ["/share/u/yu.stev/influence/influence-filtration/data/distilgpt2/imdb/stratified-filter-datasets/bottom_support_k4000"]="bottom_support_k4000"
    ["/share/u/yu.stev/influence/influence-filtration/data/distilgpt2/imdb/stratified-filter-datasets/top_opposition_k4000"]="top_opposition_k4000"
    ["/share/u/yu.stev/influence/influence-filtration/data/distilgpt2/imdb/stratified-filter-datasets/bottom_opposition_k4000"]="bottom_opposition_k4000"
    ["/share/u/yu.stev/influence/influence-filtration/data/distilgpt2/imdb/stratified-filter-datasets/random_k4000"]="random_k4000"
)

# Common training arguments
COMMON_ARGS="--model_name $MODEL_NAME \
    --max_steps 300 \
    --per_device_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --eval_sample_size 1000 \
    --eval_sample_seed 42 \
    --learning_rate 5e-5 \
    --max_length 512 \
    --fp16 \
    --save_steps 40 \
    --eval_steps 40 \
    --logging_steps 40"

# Function to run training
run_training() {
    local dataset_path=$1
    local dataset_name=$2
    
    echo "Starting training for dataset: $dataset_name"
    if [ -z "$dataset_path" ]; then
        echo "Using original IMDB dataset from HuggingFace"
        python finetune.py \
            --dataset_name "$dataset_name" \
            --output_dir "$BASE_OUTPUT_DIR" \
            --eval_output_dir "$BASE_EVAL_DIR" \
            $COMMON_ARGS
    else
        echo "Dataset path: $dataset_path"
        python finetune.py \
            --dataset_path "$dataset_path" \
            --dataset_name "$dataset_name" \
            --output_dir "$BASE_OUTPUT_DIR" \
            --eval_output_dir "$BASE_EVAL_DIR" \
            $COMMON_ARGS
    fi
        
    if [ $? -eq 0 ]; then
        echo "Training completed successfully for $dataset_name"
    else
        echo "Training failed for $dataset_name"
        exit 1
    fi
    
    echo "----------------------------------------"
}

# Process all datasets
echo "Starting training for ${#DATASETS[@]} datasets"
for dataset_path in "${!DATASETS[@]}"; do
    dataset_name="${DATASETS[$dataset_path]}"
    run_training "$dataset_path" "$dataset_name"
done

echo "All training runs completed!"
