import os
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import pinn_solver as psolver
import cavity_data as cavity
from config import ConfigManager
import argparse

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='PINN Training with Configuration Management')
    parser.add_argument('--config', type=str, default='configs/production.yaml',
                       help='配置文件路徑 (default: configs/production.yaml)')
    parser.add_argument('--resume', type=str, default=None,
                       help='從檢查點恢復訓練')
    parser.add_argument('--dry-run', action='store_true',
                       help='只顯示配置不執行訓練')
    return parser.parse_args()

def setup_distributed():
    """設置分布式訓練環境"""
    # 檢查分布式環境變數
    if 'RANK' not in os.environ:
        print("💻 單GPU模式")
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'

def cleanup_distributed():
    """清理分布式訓練環境"""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

def main():
    """主訓練函數 - 使用配置系統"""
    args = parse_args()
    
    # 設置分布式環境
    setup_distributed()
    
    try:
        # 載入配置
        print(f"📂 載入配置文件: {args.config}")
        config_manager = ConfigManager.from_file(args.config)
        
        # 驗證配置
        warnings = config_manager.validate_config()
        if warnings:
            print("⚠️  配置警告:")
            for warning in warnings:
                print(f"   - {warning}")
        
        # 顯示配置
        config_manager.print_config()
        
        if args.dry_run:
            print("🏃 Dry run模式，不執行訓練")
            return
        
        # 創建PINN實例 (使用配置)
        config = config_manager.config
        print("🚀 創建PINN實例...")
        
        PINN = psolver.PysicsInformedNeuralNetwork(
            Re=config.physics.Re,
            layers=config.network.layers,
            layers_1=config.network.layers_1,
            hidden_size=config.network.hidden_size,
            hidden_size_1=config.network.hidden_size_1,
            N_f=config.training.N_f,
            batch_size=config.training.batch_size,
            alpha_evm=config.physics.alpha_evm,
            bc_weight=config.physics.bc_weight,
            eq_weight=config.physics.eq_weight,
            checkpoint_freq=config.training.checkpoint_freq
        )
        
        # 載入數據
        print("📁 載入訓練數據...")
        path = './data/'
        dataloader = cavity.DataLoader(path=path, N_f=config.training.N_f, N_b=1000)

        # Set boundary data, | u, v, x, y
        boundary_data = dataloader.loading_boundary_data()
        PINN.set_boundary_data(X=boundary_data)

        # Set training data, | x, y
        training_data = dataloader.loading_training_data()
        PINN.set_eq_training_data(X=training_data)

        filename = f'./data/cavity_Re{config.physics.Re}_256_Uniform.mat'
        x_star, y_star, u_star, v_star, p_star = dataloader.loading_evaluate_data(filename)

        # 使用配置中的訓練階段
        training_stages = []
        for i, (alpha, epochs, lr) in enumerate(config.training.training_stages):
            stage_name = f"Stage {i+1}"
            training_stages.append((alpha, epochs, lr, stage_name))
        
        total_epochs = sum([stage[1] for stage in training_stages])
        is_distributed = int(os.environ.get('WORLD_SIZE', 1)) > 1
        
        if not is_distributed or PINN.rank == 0:
            print(f"🚀 開始完整訓練：總共 {total_epochs:,} epochs，分 {len(training_stages)} 個階段")
            print(f"   預估完成時間將在訓練開始後計算...")
            print("=" * 60)

        # 創建optimizer
        PINN.opt = torch.optim.Adam(
            list(PINN.get_model_parameters(PINN.net)) + list(PINN.get_model_parameters(PINN.net_1)),
            lr=training_stages[0][2],  # 使用第一階段的學習率
            weight_decay=0.0
        )

        # 執行分階段訓練
        for stage_idx, (alpha_evm, num_epochs, learning_rate, stage_name) in enumerate(training_stages):
            if not is_distributed or PINN.rank == 0:
                print(f"\n🔄 {stage_name}: alpha_evm={alpha_evm}, epochs={num_epochs:,}, lr={learning_rate:.2e}")
            
            # 設置階段名稱和參數
            PINN.current_stage = stage_name
            PINN.set_alpha_evm(alpha_evm)
            
            # 設置學習率
            for param_group in PINN.opt.param_groups:
                param_group['lr'] = learning_rate
            
            # 創建學習率調度器 (可選)
            scheduler = torch.optim.lr_scheduler.StepLR(
                PINN.opt, 
                step_size=num_epochs//4,  # 每1/4階段降低學習率
                gamma=0.8
            )
            
            # 訓練當前階段
            start_time = time.time()
            PINN.train(num_epoch=num_epochs, lr=learning_rate, scheduler=scheduler)
            stage_time = time.time() - start_time
            
            if not is_distributed or PINN.rank == 0:
                print(f"✅ {stage_name} 完成！耗時: {stage_time/3600:.2f} 小時")
                
                # 評估當前階段結果
                PINN.test(x_star, y_star, u_star, v_star, p_star, loop=stage_idx)
                print("-" * 60)

        if not is_distributed or PINN.rank == 0:
            print("🎉 所有訓練階段完成！")
            print("=" * 60)

    except Exception as e:
        print(f"❌ 訓練過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        raise e
    
    finally:
        # 清理分布式訓練環境
        cleanup_distributed()

if __name__ == "__main__":
    main()