#!/bin/bash

# Script to evaluate all checkpoints in a folder using direct_query_eval.py
# Usage: ./run_direct_query_eval.sh [MODEL_FOLDER] [OUTPUT_FILE]

set -e  # Exit on any error

# Default values
DEFAULT_MODEL_FOLDER="/share/u/yu.stev/influence/influence-filtration/models/distilgpt2/random_10000_seed8675309"
DEFAULT_OUTPUT_FILE="/share/u/yu.stev/influence/influence-filtration/data/distilgpt2/imdb/direct_query_eval/checkpoint_results_random_10000_seed8675309.json"

# Parse arguments
MODEL_FOLDER="${1:-$DEFAULT_MODEL_FOLDER}"
OUTPUT_FILE="${2:-$DEFAULT_OUTPUT_FILE}"

# Check if model folder exists
if [ ! -d "$MODEL_FOLDER" ]; then
    echo "❌ Error: Model folder '$MODEL_FOLDER' does not exist"
    exit 1
fi

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
mkdir -p "$OUTPUT_DIR"

echo "🚀 Starting batch evaluation of checkpoints"
echo "📁 Model folder: $MODEL_FOLDER"
echo "💾 Output file: $OUTPUT_FILE"
echo "=" * 60

# Find all checkpoint directories
CHECKPOINTS=($(find "$MODEL_FOLDER" -maxdepth 1 -type d -name "checkpoint-*" | sort -V))

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "❌ No checkpoint directories found in $MODEL_FOLDER"
    exit 1
fi

echo "📊 Found ${#CHECKPOINTS[@]} checkpoints to evaluate:"
for checkpoint in "${CHECKPOINTS[@]}"; do
    echo "   - $(basename "$checkpoint")"
done
echo ""

# Initialize results JSON structure
echo "{" > "$OUTPUT_FILE"
echo "  \"metadata\": {" >> "$OUTPUT_FILE"
echo "    \"model_folder\": \"$MODEL_FOLDER\"," >> "$OUTPUT_FILE"
echo "    \"evaluation_date\": \"$(date -Iseconds)\"," >> "$OUTPUT_FILE"
echo "    \"total_checkpoints\": ${#CHECKPOINTS[@]}," >> "$OUTPUT_FILE"
echo "    \"script_version\": \"1.0\"" >> "$OUTPUT_FILE"
echo "  }," >> "$OUTPUT_FILE"
echo "  \"results\": {" >> "$OUTPUT_FILE"

# Counter for comma handling
COUNTER=0
TOTAL=${#CHECKPOINTS[@]}

# Evaluate each checkpoint
for checkpoint in "${CHECKPOINTS[@]}"; do
    COUNTER=$((COUNTER + 1))
    CHECKPOINT_NAME=$(basename "$checkpoint")
    
    echo "🔍 Evaluating $CHECKPOINT_NAME ($COUNTER/$TOTAL)..."
    
    # Run evaluation in simple mode and capture output
    if RESULT=$(python direct_query_eval.py --model_path "$checkpoint" --simple 2>&1); then
        # Extract accuracy from output (look for "Accuracy: X.XXXX")
        ACCURACY=$(echo "$RESULT" | grep -o "Accuracy: [0-9]*\.[0-9]*" | cut -d' ' -f2)
        
        if [ -z "$ACCURACY" ]; then
            echo "⚠️  Warning: Could not extract accuracy from output for $CHECKPOINT_NAME"
            echo "   Output: $RESULT"
            ACCURACY="null"
            STATUS="error"
            ERROR_MSG="Could not extract accuracy from output"
        else
            echo "✅ $CHECKPOINT_NAME: $ACCURACY"
            STATUS="success"
            ERROR_MSG="null"
        fi
    else
        echo "❌ Error evaluating $CHECKPOINT_NAME"
        echo "   Error: $RESULT"
        ACCURACY="null"
        STATUS="error"
        ERROR_MSG=$(echo "$RESULT" | tr '"' "'" | tr '\n' ' ')
    fi
    
    # Write result to JSON file
    echo "    \"$CHECKPOINT_NAME\": {" >> "$OUTPUT_FILE"
    echo "      \"accuracy\": $ACCURACY," >> "$OUTPUT_FILE"
    echo "      \"status\": \"$STATUS\"," >> "$OUTPUT_FILE"
    echo "      \"error_message\": \"$ERROR_MSG\"," >> "$OUTPUT_FILE"
    echo "      \"checkpoint_path\": \"$checkpoint\"," >> "$OUTPUT_FILE"
    echo "      \"evaluation_time\": \"$(date -Iseconds)\"" >> "$OUTPUT_FILE"
    
    # Add comma if not the last item
    if [ $COUNTER -lt $TOTAL ]; then
        echo "    }," >> "$OUTPUT_FILE"
    else
        echo "    }" >> "$OUTPUT_FILE"
    fi
    
    echo ""
done

# Close JSON structure
echo "  }" >> "$OUTPUT_FILE"
echo "}" >> "$OUTPUT_FILE"

echo "🎉 Batch evaluation completed!"
echo "📊 Results summary:"

# Parse and display summary
if command -v python3 &> /dev/null; then
    python3 -c "
import json
import sys

try:
    with open('$OUTPUT_FILE', 'r') as f:
        data = json.load(f)
    
    results = data['results']
    total = len(results)
    successful = sum(1 for r in results.values() if r['status'] == 'success')
    failed = total - successful
    
    print(f'   Total checkpoints: {total}')
    print(f'   Successful evaluations: {successful}')
    print(f'   Failed evaluations: {failed}')
    
    if successful > 0:
        accuracies = [r['accuracy'] for r in results.values() if r['status'] == 'success']
        best_acc = max(accuracies)
        worst_acc = min(accuracies)
        avg_acc = sum(accuracies) / len(accuracies)
        
        best_checkpoint = [name for name, r in results.items() if r['accuracy'] == best_acc][0]
        
        print(f'   Best accuracy: {best_acc:.4f} ({best_checkpoint})')
        print(f'   Worst accuracy: {worst_acc:.4f}')
        print(f'   Average accuracy: {avg_acc:.4f}')
        
        print(f'\\n📈 All results:')
        for name, result in sorted(results.items()):
            if result['status'] == 'success':
                print(f'   {name}: {result[\"accuracy\"]:.4f}')
            else:
                print(f'   {name}: FAILED')
    
except Exception as e:
    print(f'   Error reading results: {e}')
"
else
    echo "   (Install python3 for detailed summary)"
fi

echo ""
echo "💾 Full results saved to: $OUTPUT_FILE"
echo "🔍 You can also examine the results with: cat '$OUTPUT_FILE' | python3 -m json.tool"
