#!/bin/bash
# Train PPO with Constraint Decomposition
# Usage: ./scripts/train_ppo.sh [--config CONFIG_PATH]

set -e

# Defaults
CONFIG_PATH="configs/ppo_training.yaml"
NUM_GPUS=8

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Constraint Decomposition PPO Training"
echo "=========================================="
echo "Config: $CONFIG_PATH"
echo "GPUs: $NUM_GPUS"
echo ""

# Check if reward models exist
REWARD_MODELS_DIR="./reward_models"
if [ ! -d "$REWARD_MODELS_DIR/semantic" ]; then
    echo "ERROR: Reward models not found. Run train_reward_models.sh first."
    exit 1
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export WANDB_PROJECT="constraint-decomposition"
export TOKENIZERS_PARALLELISM="false"

# Run training
echo "Starting training..."
accelerate launch \
    --num_processes $NUM_GPUS \
    --mixed_precision bf16 \
    -m constraint_decomposition.train \
    --config $CONFIG_PATH

echo ""
echo "Training complete!"
echo "Checkpoints saved to ./checkpoints/"
