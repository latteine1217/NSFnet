#!/bin/bash

# PhysicsNeMo PINN Training Script
# Usage: ./run_training.sh [num_gpus]

# Set default number of GPUs
NUM_GPUS=${1:-1}

echo "Starting PhysicsNeMo PINN training with $NUM_GPUS GPU(s)"

# Check if PhysicsNeMo is installed
python -c "import physicsnemo; print(f'PhysicsNeMo version: {physicsnemo.__version__}')" || {
    echo "PhysicsNeMo not found. Installing..."
    pip install nvidia-physicsnemo
}

# Create necessary directories
mkdir -p checkpoints
mkdir -p outputs

# Set environment variables for distributed training
export CUDA_VISIBLE_DEVICES=0,1,2,3

if [ $NUM_GPUS -gt 1 ]; then
    echo "Running distributed training on $NUM_GPUS GPUs"
    export MASTER_ADDR="localhost"
    export MASTER_PORT="12355"
    export WORLD_SIZE=$NUM_GPUS
    
    # Launch distributed training
    torchrun --nproc_per_node=$NUM_GPUS \
             --master_addr=$MASTER_ADDR \
             --master_port=$MASTER_PORT \
             physicsnemo_train.py
else
    echo "Running single GPU training"
    python physicsnemo_train.py
fi

echo "Training completed!"