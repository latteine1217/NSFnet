import os
import torch
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
from tools import *
import cavity_data as cavity
import pinn_solver as psolver
import warnings
import time

# 設定環境變數來避免DDP錯誤
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
os.environ['NCCL_DEBUG'] = 'INFO'

# 抑制 PyTorch 分散式訓練相關警告
warnings.filterwarnings("ignore", message=".*c10d::allreduce_.*autograd kernel.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.autograd")


def setup_distributed():
    """Initialize distributed training with better error handling"""
    # 檢查是否在分布式環境中
    if 'WORLD_SIZE' not in os.environ or int(os.environ['WORLD_SIZE']) <= 1:
        print("Not running in distributed mode")
        return False
    
    # 確保所有必要的環境變數都存在
    required_env = ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']
    for env_var in required_env:
        if env_var not in os.environ:
            print(f"Missing environment variable: {env_var}")
            return False

    # 初始化分布式進程組
    try:
        # 設定timeout來避免掛起
        dist.init_process_group(
            backend='nccl',
            timeout=torch.distributed.default_pg_timeout,
            init_method=None
        )
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ['LOCAL_RANK'])
        
        torch.cuda.set_device(local_rank)
        
        # 設定CUDA設定來提升穩定性
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
        print(f"[GPU {rank}] Distributed training initialized:")
        print(f"  World size: {world_size}")
        print(f"  Rank: {rank}")
        print(f"  Local rank: {local_rank}")
        
        return True
        
    except Exception as e:
        print(f"Failed to initialize distributed training: {e}")
        return False


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def train(net_params=None):
    # Setup distributed training
    is_distributed = setup_distributed()
    
    # 如果分布式初始化失敗，嘗試單GPU模式
    if not is_distributed:
        print("Falling back to single GPU mode")
        # 設置默認環境變數以支持單GPU運行
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'

    try:
        Re = 5000   # Reynolds number
        N_neu = 80
        N_neu_1 = 40
        lam_bcs = 10
        lam_equ = 1
        N_f = 120000
        # 移除batch size限制，使用完整數據訓練
        alpha_evm = 0.03
        N_HLayer = 6
        N_HLayer_1 = 4

        PINN = psolver.PysicsInformedNeuralNetwork(
            Re=Re,
            layers=N_HLayer,
            layers_1=N_HLayer_1,
            hidden_size = N_neu,
            hidden_size_1 = N_neu_1,
            N_f = N_f,
            alpha_evm=alpha_evm,
            bc_weight=lam_bcs,
            eq_weight=lam_equ,
            net_params=net_params)

        path = './data/'
        dataloader = cavity.DataLoader(path=path, N_f=N_f, N_b=1000)

        # Set boundary data, | u, v, x, y
        boundary_data = dataloader.loading_boundary_data()
        PINN.set_boundary_data(X=boundary_data)

        # Set training data, | x, y
        training_data = dataloader.loading_training_data()
        PINN.set_eq_training_data(X=training_data)

        filename = './data/cavity_Re'+str(Re)+'_256_Uniform.mat'
        x_star, y_star, u_star, v_star, p_star = dataloader.loading_evaluate_data(filename)

        # Training stages with different alpha_evm values
        # 總共 3,000,000 epochs，分6個階段
        training_stages = [
            (0.05, 500000, 1e-3, "Stage 1"),
            (0.03, 500000, 2e-4, "Stage 2"),
            (0.01, 500000, 4e-5, "Stage 3"),
            (0.005, 500000, 1e-5, "Stage 4"),
            (0.002, 500000, 2e-6, "Stage 5"),
            (0.002, 500000, 2e-6, "Stage 6")
        ]
        
        total_epochs = sum([stage[1] for stage in training_stages])
        if not is_distributed or PINN.rank == 0:
            print(f"🚀 開始完整訓練：總共 {total_epochs:,} epochs，分 {len(training_stages)} 個階段")
            print(f"   預估完成時間將在訓練開始後計算...")
            print("=" * 60)

        for alpha, epochs, lr, stage_name in training_stages:
            if not is_distributed or PINN.rank == 0:
                print(f"\n🎯 Starting {stage_name}: alpha_evm={alpha}, epochs={epochs:,}, lr={lr:.1e}")
            
            PINN.current_stage = stage_name
            PINN.set_alpha_evm(alpha)
            PINN.train(num_epoch=epochs, lr=lr)
            
            if not is_distributed or PINN.rank == 0:
                print(f"\n📊 {stage_name} 完成，進行評估...")
                PINN.evaluate(x_star, y_star, u_star, v_star, p_star)

        if not is_distributed or PINN.rank == 0:
            # 計算總訓練時間
            if PINN.training_start_time is not None:
                total_training_time = time.time() - PINN.training_start_time
                def format_time(seconds):
                    if seconds < 3600:
                        return f"{seconds//60:.0f}分 {seconds%60:.0f}秒"
                    elif seconds < 86400:
                        return f"{seconds//3600:.0f}小時 {(seconds%3600)//60:.0f}分"
                    else:
                        return f"{seconds//86400:.0f}天 {(seconds%86400)//3600:.0f}小時"
                
                print(f"\n🎉 ===== 訓練完成！=====")
                print(f"   總訓練時間: {format_time(total_training_time)}")
                print(f"   總 epochs: {total_epochs:,}")
                print(f"   平均每 epoch: {total_training_time/total_epochs:.3f}秒")
                print("=" * 40)
            
            # 關閉TensorBoard writer
            if hasattr(PINN, 'tb_writer') and PINN.tb_writer is not None:
                PINN.tb_writer.close()
                print(f"📊 TensorBoard 日誌已保存")

    except Exception as e:
        print(f"Training failed: {e}")
        raise e
    finally:
        # Clean up distributed training
        if is_distributed:
            cleanup_distributed()


if __name__ == "__main__":
    train()
