import os
import sys
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

import time

def cleanup_distributed():
    """清理分布式訓練環境"""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

def setup_distributed():
    """設置分布式訓練環境"""
    # 檢查分布式環境變數
    if 'RANK' not in os.environ:
        print("💻 單GPU模式")
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'

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
    main()
