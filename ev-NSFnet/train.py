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

torch.backends.cudnn.benchmark = True

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='PINN Training with Configuration Management')
    parser.add_argument('--config', type=str, default='configs/production.yaml',
                       help='配置文件路徑 (default: configs/production.yaml)')
    parser.add_argument('--resume', type=str, default=None,
                       help='從檢查點恢復訓練')
    parser.add_argument('--dry-run', action='store_true',
                       help='只顯示配置不執行訓練')
    parser.add_argument('--lr-scheduler', type=str, default='StepLR',
                        choices=['StepLR', 'MultiStage', 'CosineAnnealing', 'Constant'],
                        help='選擇學習率調度策略 (default: StepLR)')
    return parser.parse_args()

def setup_distributed():
    """設置分布式訓練環境"""
    # 檢查分布式環境變數
    if 'RANK' not in os.environ:
        print("💻 單GPU模式")
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        return False  # 非分布式模式
    
    # 初始化分布式進程組
    try:
        if not dist.is_initialized():
            # 只有在 rank 0 時才顯示初始化信息
            rank = int(os.environ.get('RANK', 0))
            if rank == 0:
                print("🔗 初始化分布式訓練...")
                
            dist.init_process_group(backend='nccl')
            
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            local_rank = int(os.environ['LOCAL_RANK'])
            
            # 只有主進程顯示分布式信息
            if rank == 0:
                print(f"📡 分布式訓練設置完成: {world_size} GPUs")
                print(f"   - Backend: NCCL")
                print(f"   - 每個進程負責 GPU {local_rank}")
            
            # 設定CUDA設備
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                
        return True  # 分布式模式
        
    except Exception as e:
        rank = int(os.environ.get('RANK', 0))
        if rank == 0:
            print(f"❌ 分布式初始化失敗: {e}")
            print("💻 退回單GPU模式")
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        return False

def cleanup_distributed():
    """清理分布式訓練環境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def main():
    """主訓練函數 - 使用配置系統"""
    args = parse_args()
    
    # 設置分布式環境
    is_distributed = setup_distributed()
    
    # 獲取當前進程的rank（用於控制輸出）
    rank = int(os.environ.get('RANK', 0))
    
    try:
        # 只在主進程顯示配置載入信息
        if rank == 0:
            print(f"📂 載入配置文件: {args.config}")
        
        config_manager = ConfigManager.from_file(args.config)
        config = config_manager.config  # 獲取配置對象
        
        # 只在主進程顯示驗證和配置信息
        if rank == 0:
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
        
        # Dry run檢查（所有進程都需要退出）
        if args.dry_run:
            return

        # Enable anomaly detection to find the operation that failed to compute its gradient
        torch.autograd.set_detect_anomaly(False)
        
        # 只在主進程顯示PINN創建信息
        if rank == 0:
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
        
        # 只在主進程顯示數據載入信息
        if rank == 0:
            print("📁 載入訓練數據...")
        
        path = './data/'
        dataloader = cavity.DataLoader(path=path, N_f=config.training.N_f, N_b=1000)

        # Set boundary data, | u, v, x, y
        boundary_np = dataloader.loading_boundary_data()
        xb_cpu = torch.as_tensor(boundary_np[0], dtype=torch.float32).contiguous()
        yb_cpu = torch.as_tensor(boundary_np[1], dtype=torch.float32).contiguous()
        ub_cpu = torch.as_tensor(boundary_np[2], dtype=torch.float32).contiguous()
        vb_cpu = torch.as_tensor(boundary_np[3], dtype=torch.float32).contiguous()
        total_b = xb_cpu.shape[0]
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        r = rank
        if total_b < world_size:
            if r < total_b:
                b_start, b_end = r, r+1
            else:
                b_start, b_end = 0, 0
        else:
            per = total_b // world_size
            b_start = r * per
            b_end = b_start + per if r < world_size - 1 else total_b
        if dist.is_initialized():
            idx = [b_start, b_end]
            dist.broadcast_object_list(idx, src=0)
            b_start, b_end = idx
        xb = xb_cpu[b_start:b_end].to(PINN.device)
        yb = yb_cpu[b_start:b_end].to(PINN.device)
        ub = ub_cpu[b_start:b_end].to(PINN.device)
        vb = vb_cpu[b_start:b_end].to(PINN.device)
        PINN.set_boundary_data(X=(xb, yb, ub, vb))

        # Set training data, | x, y
        eq_np = dataloader.loading_training_data()
        xf_cpu = torch.as_tensor(eq_np[0], dtype=torch.float32).contiguous()
        yf_cpu = torch.as_tensor(eq_np[1], dtype=torch.float32).contiguous()
        total_f = xf_cpu.shape[0]
        if total_f < world_size:
            if r < total_f:
                f_start, f_end = r, r+1
            else:
                f_start, f_end = 0, 1
        else:
            per_f = total_f // world_size
            f_start = r * per_f
            f_end = f_start + per_f if r < world_size - 1 else total_f
        if dist.is_initialized():
            idxf = [f_start, f_end]
            dist.broadcast_object_list(idxf, src=0)
            f_start, f_end = idxf
        xf = xf_cpu[f_start:f_end].to(PINN.device)
        yf = yf_cpu[f_start:f_end].to(PINN.device)
        PINN.set_eq_training_data(X=(xf, yf))

        filename = f'./data/cavity_Re{config.physics.Re}_256_Uniform.mat'
        x_star, y_star, u_star, v_star, p_star = dataloader.loading_evaluate_data(filename)

        # 使用配置中的訓練階段
        training_stages = []
        for i, (alpha, epochs, lr) in enumerate(config.training.training_stages):
            stage_name = f"Stage {i+1}"
            training_stages.append((alpha, epochs, lr, stage_name))
        
        total_epochs = sum([stage[1] for stage in training_stages])
        
        if not is_distributed or PINN.rank == 0:
            print(f"🚀 開始完整訓練：總共 {total_epochs:,} epochs，分 {len(training_stages)} 個階段")
            print(f"   預估完成時間將在訓練開始後計算...")
            print("=" * 60)

        # 創建optimizer
        optimizer = torch.optim.Adam(
            list(PINN.get_model_parameters(PINN.net)) + list(PINN.get_model_parameters(PINN.net_1)),
            lr=training_stages[0][2],  # 使用第一階段的學習率
            weight_decay=0.0
        )
        PINN.set_optimizers(optimizer)

        # 恢復訓練狀態
        start_epoch = 0
        if args.resume:
            if rank == 0:
                print(f"🔄 正在從檢查點恢復: {args.resume}")
            start_epoch = PINN.load_checkpoint(args.resume, optimizer)
            if rank == 0:
                if start_epoch > 0:
                    print(f"✅ 成功恢復，將從 epoch {start_epoch} 開始")
                else:
                    print("⚠️ 無法載入檢查點，將從頭開始訓練")

        # 執行分階段訓練
        # Note: When resuming, training will continue from the next epoch in the sequence,
        # but will start with the stage configuration determined by the current logic.
        # This means if you resume in what was originally stage 2, it will still follow the
        # sequence from stage 1 as defined in the config.
        # A more advanced implementation might save and restore the stage index.
        for stage_idx, (alpha_evm, num_epochs, learning_rate, stage_name) in enumerate(training_stages):
            # Skip epochs that are already completed if resuming
            if start_epoch >= num_epochs:
                if rank == 0:
                    print(f"⏭️ 跳過 {stage_name} (已完成 {num_epochs} epochs，從 {start_epoch} 恢復)")
                start_epoch -= num_epochs  # Decrement for next stage
                continue
            
            epochs_to_run = num_epochs - start_epoch

            if not is_distributed or PINN.rank == 0:
                print(f"🔄 {stage_name}: alpha_evm={alpha_evm}, epochs={epochs_to_run}/{num_epochs}, lr={learning_rate:.2e}")
            
            # 設置階段名稱和參數
            PINN.current_stage = stage_name
            PINN.set_alpha_evm(alpha_evm)
            
            # 設置優化器的學習率，這是每個階段的基礎學習率
            for param_group in PINN.opt.param_groups:
                param_group['lr'] = learning_rate

            # 根據策略決定調度器
            stage_scheduler = None
            if args.lr_scheduler == 'StepLR':
                if not is_distributed or PINN.rank == 0:
                    print("   - 啟用階段內 StepLR 調度器 (每 1/4 階段衰減至 80%)")
                stage_scheduler = torch.optim.lr_scheduler.StepLR(PINN.opt, step_size=num_epochs//4, gamma=0.8)
            
            elif args.lr_scheduler == 'MultiStage':
                if not is_distributed or PINN.rank == 0:
                    print("   - 使用多階段固定學習率 (無內部衰減)")
                # 不需要調度器，保持學習率恆定

            elif args.lr_scheduler == 'Constant':
                constant_lr = training_stages[0][2]
                if not is_distributed or PINN.rank == 0:
                    print(f"   - 使用全局恆定學習率: {constant_lr:.2e}")
                for param_group in PINN.opt.param_groups:
                    param_group['lr'] = constant_lr

            elif args.lr_scheduler == 'CosineAnnealing':
                if not is_distributed or PINN.rank == 0:
                    print(f"   - 啟用分階段 CosineAnnealing 調度器")

                # 確定當前階段的最小學習率 (eta_min)
                if stage_idx < len(training_stages) - 1:
                    # 下一階段的起始學習率作為當前階段的終點
                    eta_min = training_stages[stage_idx + 1][2]
                else:
                    # 最後一個階段的終點學習率固定為 2e-6
                    eta_min = 2e-6

                if not is_distributed or PINN.rank == 0:
                    print(f"     - {stage_name}: {num_epochs} epochs CosineAnnealing (end LR: {eta_min:.2e})")
                stage_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(PINN.opt, T_max=num_epochs, eta_min=eta_min)

            # 訓練當前階段
            start_time = time.time()
            
            # 設置 Profiler
            profiler_log_dir = f"runs/profiler/{stage_name}"
            if rank == 0:
                os.makedirs(profiler_log_dir, exist_ok=True)
            
            do_profile = (start_epoch % 20000 == 0)
            if do_profile:
                with torch.profiler.profile(
                    schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
                    on_trace_ready=None,
                    record_shapes=False,
                    with_stack=False,
                    profile_memory=False
                ) as prof:
                    PINN.train(num_epoch=epochs_to_run, lr=learning_rate, scheduler=stage_scheduler, profiler=prof, start_epoch=start_epoch)
            else:
                class _Noop:
                    def step(self):
                        pass
                PINN.train(num_epoch=epochs_to_run, lr=learning_rate, scheduler=stage_scheduler, profiler=_Noop(), start_epoch=start_epoch)

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
