#!/bin/bash
# Train all 4 decomposed reward models
# Usage: ./scripts/train_reward_models.sh

set -e

echo "=========================================="
echo "Training Decomposed Reward Models"
echo "=========================================="

BASE_MODEL="nvidia/nemotron-7b"
DATA_DIR="./data/preference_data"
OUTPUT_DIR="./reward_models"
NUM_EPOCHS=3
BATCH_SIZE=32

# Reward types to train
REWARD_TYPES=("semantic" "structural" "format" "meta")

# Train each reward model
for REWARD_TYPE in "${REWARD_TYPES[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "Training $REWARD_TYPE reward model..."
    echo "----------------------------------------"
    
    python -m constraint_decomposition.train_reward \
        --base_model $BASE_MODEL \
        --reward_type $REWARD_TYPE \
        --train_data "$DATA_DIR/${REWARD_TYPE}_preferences.json" \
        --output_dir "$OUTPUT_DIR/$REWARD_TYPE" \
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate 1e-5 \
        --bf16
    
    echo "$REWARD_TYPE reward model saved to $OUTPUT_DIR/$REWARD_TYPE"
done

echo ""
echo "=========================================="
echo "All reward models trained successfully!"
echo "=========================================="
echo ""
echo "Validation accuracies:"
for REWARD_TYPE in "${REWARD_TYPES[@]}"; do
    if [ -f "$OUTPUT_DIR/$REWARD_TYPE/eval_results.json" ]; then
        ACC=$(cat "$OUTPUT_DIR/$REWARD_TYPE/eval_results.json" | grep accuracy | head -1)
        echo "  $REWARD_TYPE: $ACC"
    fi
done
