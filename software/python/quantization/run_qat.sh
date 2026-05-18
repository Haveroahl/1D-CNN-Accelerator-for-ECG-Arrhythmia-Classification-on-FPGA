#!/bin/bash
# Quantization-Aware Training (QAT) for INT8
# Usage: bash run_qat.sh [output_dir] [num_epochs] [learning_rate]

OUTPUT_DIR="${1:-./results/qat_int8}"
NUM_EPOCHS="${2:-20}"
LEARNING_RATE="${3:-1e-4}"

echo "Running QAT training for INT8..."
echo "  Output directory: $OUTPUT_DIR"
echo "  Number of epochs: $NUM_EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo ""

/home/duc/Thesis/.venv/bin/python qat_training.py \
    --checkpoint ./results/best_model.pth \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE"
