#!/bin/bash

# Script to run query influence histogram analysis
# This script analyzes how training examples influence different queries

# Configuration
INFLUENCE_FILE="/share/u/yu.stev/influence/influence-filtration/data/distilgpt2/imdb/influence_scores.json"
MODEL_PATH="/share/u/yu.stev/influence/influence-filtration/data/distilgpt2/imdb"
OUTPUT_DIR="/share/u/yu.stev/influence/influence-filtration/data/distilgpt2/plots/query-histogram"
K=10
SEED=42
DEVICE="cuda"
BATCH_SIZE=1

echo "=== Training Example Influence Analysis ==="
echo "Configuration:"
echo "  Influence file: $INFLUENCE_FILE"
echo "  Model path: $MODEL_PATH"
echo "  Output directory: $OUTPUT_DIR"
echo "  Sample size (k): $K"
echo "  Device: $DEVICE"
echo "  Batch size: $BATCH_SIZE"
echo "  Random seed: $SEED"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Running analysis without sentiment splitting..."
python query_inf_histogram.py \
    --influence_file "$INFLUENCE_FILE" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --k $K \
    --device $DEVICE \
    --batch_size $BATCH_SIZE \
    --seed $SEED

echo ""
echo "Running analysis with sentiment splitting..."
python query_inf_histogram.py \
    --influence_file "$INFLUENCE_FILE" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --k $K \
    --device $DEVICE \
    --batch_size $BATCH_SIZE \
    --seed $SEED \
    --split_by_sentiment

echo ""
echo "=== Analysis Complete ==="
echo "Check the output directory for results: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "Regular analysis:"
echo "  - training_example_XXXX_histogram.png (individual histograms)"
echo "  - training_examples_summary_analysis.png"
echo "  - training_examples_statistics_report.txt"
echo ""
echo "Sentiment-split analysis:"
echo "  - training_example_XXXX_sentiment_split.png (individual histograms)"
echo "  - training_examples_sentiment_summary_analysis.png"
echo "  - training_examples_sentiment_statistics_report.txt" 