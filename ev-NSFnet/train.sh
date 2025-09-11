#!/bin/bash
#SBATCH --job-name=PINNs_train
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --time=14-00:00:00
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

# NCCL 超時配置 - 避免 L-BFGS 通訊超時
export NCCL_TIMEOUT=1800
export NCCL_IB_TIMEOUT=30
export NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_TRACE_BUFFER_SIZE=1024

# Tesla P100 相容性設置 (CUDA capability 6.0)
export TORCH_COMPILE_BACKEND=eager
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

source ~/python/bin/activate

echo "=== GPU Information ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader,nounits
echo "======================="

echo "Job start: $(date)"
time torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 train.py
echo "Job end: $(date)"
