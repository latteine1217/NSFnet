#!/bin/bash
#SBATCH --job-name=PINNs_test
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=50G
#SBATCH --gres=gpu:1

ml load mpi

export OMP_NUM_THREADS=12
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES=0

# Tesla P100 相容性設置
export TORCH_COMPILE_BACKEND=eager
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

# 檢查並激活Python虛擬環境
source $HOME/python/bin/activate

echo "=== GPU Information ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader,nounits
echo "======================="

echo "Test start: $(date)"

# 基於你的實際目錄結構
BASE_DIR="results/Re5000/result1"

# 修正後的Stage參數對應 (根據實際目錄結構)
declare -A ALPHA_VALUES=(
    [1]="0.05"     # Stage_1
    [2]="0.03"     # Stage_2  
    [3]="0.01"     # Stage_3
    [4]="0.005"    # Stage_4
    [5]="0.002"    # Stage_5
)

# 創建統一的測試結果目錄
UNIFIED_OUTPUT_DIR="results/unified_test_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$UNIFIED_OUTPUT_DIR"
echo "Unified test results will be saved to: $UNIFIED_OUTPUT_DIR"

# 指定要測試的stage (1-5)，或"all"測試全部
STAGE=${1:-"all"}

if [ "$STAGE" = "all" ]; then
    echo "=== Testing all available stages ==="
    for i in {1..5}; do
        ALPHA=${ALPHA_VALUES[$i]}
        STAGE_DIR="$BASE_DIR/6x80_Nf120k_lamB10_alpha${ALPHA}Stage_${i}"
        
        if [ -d "$STAGE_DIR" ]; then
            echo "--- Testing Stage $i (alpha=$ALPHA) ---"
            python3 test.py --run_dir "$STAGE_DIR" --output_dir "$UNIFIED_OUTPUT_DIR"
            echo "--- Stage $i completed ---"
        else
            echo "Warning: $STAGE_DIR not found"
        fi
    done
else
    # 檢查stage範圍
    if [[ $STAGE -lt 1 || $STAGE -gt 5 ]]; then
        echo "Error: Stage must be between 1-5"
        exit 1
    fi
    
    echo "=== Testing Stage $STAGE ==="
    ALPHA=${ALPHA_VALUES[$STAGE]}
    STAGE_DIR="$BASE_DIR/6x80_Nf120k_lamB10_alpha${ALPHA}Stage_${STAGE}"
    
    if [ -d "$STAGE_DIR" ]; then
        echo "Testing: $STAGE_DIR"
        python3 test.py --run_dir "$STAGE_DIR" --output_dir "$UNIFIED_OUTPUT_DIR"
    else
        echo "Error: $STAGE_DIR not found"
        echo "Expected path: $STAGE_DIR"
        exit 1
    fi
fi

echo "Test end: $(date)"
