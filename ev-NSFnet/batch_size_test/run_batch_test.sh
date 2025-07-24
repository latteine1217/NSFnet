#!/bin/bash
#SBATCH --job-name=batch_efficiency_test
#SBATCH --output=batch_test_%j.out
#SBATCH --error=batch_test_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH --gres=gpu:2

ml load mpi

export OMP_NUM_THREADS=24
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES=0,1

source ~/python/bin/activate

echo "Batch efficiency test start: $(date)"
time torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
        batch_efficiency_test.py
echo "Batch efficiency test end: $(date)"