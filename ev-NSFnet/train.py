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
from tools import setup_device, get_cuda_info
import argparse

torch.backends.cudnn.benchmark = True

def display_supervision_setup(config_manager, rank=0):
    """顯示監督設置和監督點位置"""
    if rank != 0:  # 只在主進程顯示
        return
        
    supervision_config = config_manager.config.supervision
    
    print("🎯 監督數據配置:")
    print(f"   啟用狀態: {'✅ 啟用' if supervision_config.enabled else '❌ 關閉'}")
    print(f"   監督點數: {supervision_config.data_points}")
    print(f"   數據權重: {supervision_config.weight}")
    print(f"   隨機種子: {supervision_config.random_seed}")
    print(f"   數據路徑: {supervision_config.data_path}")
    
    if supervision_config.enabled and supervision_config.data_points > 0:
        print("\n" + "="*50)
        print("🔍 監督點位置詳細信息：")
        print("="*50)
        
        # 檢查數據文件是否存在
        if not os.path.exists(supervision_config.data_path):
            print(f"⚠️  數據文件不存在: {supervision_config.data_path}")
            return
            
        from cavity_data import DataLoader
        loader = DataLoader()
        try:
            loader.print_supervision_locations(
                supervision_config.data_path,
                supervision_config.data_points, 
                supervision_config.random_seed
            )
        except Exception as e:
            print(f"⚠️  載入監督點位置時出錯: {e}")

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
            
            # 使用統一設備管理函數
            device = setup_device(local_rank)
                
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
            
            # 顯示監督數據設置
            display_supervision_setup(config_manager, rank)
            
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
            supervised_data_weight=config.supervision.weight if hasattr(config, 'supervision') else 1.0,
            supervision_data_points=config.supervision.data_points if hasattr(config, 'supervision') else 0,
            supervision_data_path=config.supervision.data_path if hasattr(config, 'supervision') else None,
            supervision_random_seed=config.supervision.random_seed if hasattr(config, 'supervision') else 42,
            checkpoint_freq=config.training.checkpoint_freq,
            config=config
        )
        # config 已於構造時注入並在DDP前完成縮放
        
        # 只在主進程顯示數據載入信息
        if rank == 0:
            print("📁 載入訓練數據...")
        
        path = './data/'
        dataloader = cavity.DataLoader(
            path=path,
            N_f=config.training.N_f,
            N_b=1000,
            sort_by_boundary_distance=getattr(config.training, 'sort_by_boundary_distance', True)
        )

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
        xf = xf_cpu[f_start:f_end].to(PINN.device).contiguous().requires_grad_(True)
        yf = yf_cpu[f_start:f_end].to(PINN.device).contiguous().requires_grad_(True)
        PINN.set_eq_training_data(X=(xf, yf))

        # 加载监督数据（如果启用）
        if hasattr(config, 'supervision') and config.supervision.enabled and config.supervision.data_points > 0:
            if rank == 0:
                print(f"📊 载入监督数据: {config.supervision.data_points} 个数据点...")
            
            # 加载监督数据
            x_sup, y_sup, u_sup, v_sup, p_sup = dataloader.loading_supervision_data(
                config.supervision.data_path, 
                config.supervision.data_points,
                config.supervision.random_seed
            )
            
            # 转换为tensor并移到GPU
            if x_sup.shape[0] > 0:  # 确保有数据点
                # 監督座標不需要梯度，只需讓網路權重反向即可
                x_sup_tensor = torch.as_tensor(x_sup, dtype=torch.float32).to(PINN.device)
                y_sup_tensor = torch.as_tensor(y_sup, dtype=torch.float32).to(PINN.device)
                u_sup_tensor = torch.as_tensor(u_sup, dtype=torch.float32).to(PINN.device)
                v_sup_tensor = torch.as_tensor(v_sup, dtype=torch.float32).to(PINN.device)
                p_sup_tensor = torch.as_tensor(p_sup, dtype=torch.float32).to(PINN.device)
                
                # 设置监督数据到PINN
                PINN.x_sup = x_sup_tensor
                PINN.y_sup = y_sup_tensor
                PINN.u_sup = u_sup_tensor
                PINN.v_sup = v_sup_tensor
                PINN.p_sup = p_sup_tensor
                
                if rank == 0:
                    print(f"✅ 监督数据加载完成，监督点坐标: ({x_sup[0,0]:.4f}, {y_sup[0,0]:.4f})")
        else:
            if rank == 0:
                print("📊 未启用监督数据或数据点数量为0")

        filename = f'./data/cavity_Re{config.physics.Re}_256_Uniform.mat'
        x_star, y_star, u_star, v_star, p_star = dataloader.loading_evaluate_data(filename)

        # 使用配置中的訓練階段
        training_stages = []
        for i, stage in enumerate(config.training.training_stages):
            alpha, epochs, lr = stage[0], stage[1], stage[2]
            sched = stage[3] if len(stage) > 3 else 'Constant'
            stage_name = f"Stage {i+1}"
            training_stages.append((alpha, epochs, lr, sched, stage_name))
        
        total_epochs = sum([stage[1] for stage in training_stages])
        
        if not is_distributed or PINN.rank == 0:
            print(f"🚀 開始完整訓練：總共 {total_epochs:,} epochs，分 {len(training_stages)} 個階段")
            print(f"   預估完成時間將在訓練開始後計算...")
            print("=" * 60)

        # 初始化 AdamW（Stage 0 placeholder，實際每Stage重建）
        PINN.build_adamw_optimizer = getattr(PINN, 'build_adamw_optimizer')  # 保險引用
        first_lr = training_stages[0][2]
        if getattr(config.training, 'weight_decay_stages', None) is not None:
            first_wd = config.training.weight_decay_stages[0]
        else:
            first_wd = getattr(config.training, 'weight_decay', 0.0)
        PINN.build_adamw_optimizer(first_lr, first_wd)

        # 恢復訓練狀態
        start_epoch = 0
        if args.resume:
            if rank == 0:
                print(f"🔄 正在從檢查點恢復: {args.resume}")
            start_epoch = PINN.load_checkpoint(args.resume, PINN.opt)
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
        for stage_idx, (alpha_evm, num_epochs, learning_rate, sched_name, stage_name) in enumerate(training_stages):
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
            
            # 重建 AdamW 以套用該階段學習率與 weight decay
            if getattr(config.training, 'weight_decay_stages', None) is not None:
                stage_wd = config.training.weight_decay_stages[stage_idx]
            else:
                stage_wd = getattr(config.training, 'weight_decay', 0.0)
            PINN.build_adamw_optimizer(learning_rate, stage_wd)
            if rank == 0:
                print(f"   - Optimizer: AdamW (lr={learning_rate:.2e}, wd={stage_wd})")

            # 根據策略決定調度器（由配置指定）
            stage_scheduler = None
            if sched_name not in ['Constant','MultiStepLR','CosineAnnealingLR','CosineAnnealingWarmRestarts','SGDR']:
                if not is_distributed or PINN.rank == 0:
                    print(f"   - 未知調度器 {sched_name}，回退 Constant")
                sched_name = 'Constant'
            if sched_name == 'MultiStepLR':
                import math
                m1 = math.ceil(num_epochs/2)
                m2 = math.ceil(4*num_epochs/5)
                if not is_distributed or PINN.rank == 0:
                    print(f"   - 啟用 MultiStepLR 里程碑: {m1}, {m2}")
                stage_scheduler = torch.optim.lr_scheduler.MultiStepLR(PINN.opt, milestones=[m1, m2], gamma=0.5)
            elif sched_name == 'CosineAnnealingLR':
                if stage_idx < len(training_stages) - 1:
                    eta_min = training_stages[stage_idx + 1][2]
                else:
                    eta_min = max(learning_rate * 0.1, 1e-8)
                if not is_distributed or PINN.rank == 0:
                    print(f"   - 啟用 CosineAnnealingLR: T_max={num_epochs}, eta_min={eta_min:.2e}")
                stage_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(PINN.opt, T_max=num_epochs, eta_min=eta_min)
            elif sched_name in ['CosineAnnealingWarmRestarts', 'SGDR']:
                # SGDR: CosineAnnealingWarmRestarts + Warmup (LinearLR)
                sgdr_cfg = getattr(config.training, 'sgdr', None)
                # 預設暖啟動步數為階段的5%，限制在[500, 10000]
                default_warmup = max(500, int(0.05 * num_epochs))
                default_warmup = min(default_warmup, 10000)
                warmup_epochs = int(getattr(sgdr_cfg, 'warmup_epochs', default_warmup) if sgdr_cfg else default_warmup)
                # 預設第一個週期長度（不含暖啟動）
                remain = max(1, num_epochs - warmup_epochs)
                default_T0 = max(1000, int(0.25 * remain))
                T_0 = int(getattr(sgdr_cfg, 'T_0', default_T0) if sgdr_cfg else default_T0)
                T_mult = int(getattr(sgdr_cfg, 'T_mult', 2) if sgdr_cfg else 2)
                # eta_min 使用下一階段lr或當前lr的10%
                if stage_idx < len(training_stages) - 1:
                    eta_min = training_stages[stage_idx + 1][2]
                else:
                    eta_min = max(learning_rate * 0.1, 1e-8)
                eta_min = float(getattr(sgdr_cfg, 'eta_min', eta_min) if sgdr_cfg else eta_min)
                start_factor = float(getattr(sgdr_cfg, 'start_factor', 0.1) if sgdr_cfg else 0.1)
                end_factor = float(getattr(sgdr_cfg, 'end_factor', 1.0) if sgdr_cfg else 1.0)
                
                if not is_distributed or PINN.rank == 0:
                    print(f"   - 啟用 SGDR: warmup={warmup_epochs}, T_0={T_0}, T_mult={T_mult}, eta_min={eta_min:.2e}")
                    print(f"     * 基礎學習率: {learning_rate:.2e} (warmup: {start_factor} -> {end_factor})")
                
                # 建立 SequentialLR: LinearLR (warmup) -> CosineAnnealingWarmRestarts
                warmup_sched = torch.optim.lr_scheduler.LinearLR(
                    PINN.opt,
                    start_factor=start_factor,
                    end_factor=end_factor,
                    total_iters=warmup_epochs
                )
                cawr_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    PINN.opt,
                    T_0=T_0,
                    T_mult=T_mult,
                    eta_min=eta_min
                )
                stage_scheduler = torch.optim.lr_scheduler.SequentialLR(
                    PINN.opt,
                    schedulers=[warmup_sched, cawr_sched],
                    milestones=[warmup_epochs]
                )

            # 訓練當前階段
            start_time = time.time()

            # 設置 Profiler
            profiler_log_dir = f"runs/profiler/{stage_name}"
            if rank == 0:
                os.makedirs(profiler_log_dir, exist_ok=True)
            do_profile = (start_epoch % 20000 == 0)
            class _Noop:
                def step(self):
                    pass
            if False:  # 原先Stage 3混合優化關閉，改由滑窗觸發 L-BFGS
                switch_epoch = int(num_epochs * 0.6)
                if start_epoch < switch_epoch:
                    run_epochs = min(epochs_to_run, switch_epoch - start_epoch)
                    PINN.train(num_epoch=run_epochs, lr=learning_rate, scheduler=stage_scheduler, profiler=_Noop(), start_epoch=start_epoch)
                    start_epoch += run_epochs
                    epochs_to_run -= run_epochs
                if epochs_to_run > 0:
                    if not is_distributed or PINN.rank == 0:
                        print(f"🔁 {stage_name}: 切換至 L-BFGS (後40%)")
                    lbfgs_cfg = {
                        'max_iter': 50,
                        'history_size': 20,
                        'tolerance_grad': 1e-8,
                        'tolerance_change': 1e-9,
                        'line_search_fn': 'strong_wolfe'
                    }
                    PINN.train_with_lbfgs_segment(max_outer_steps=2000, lbfgs_params=lbfgs_cfg, log_interval=200)
                    if not is_distributed or PINN.rank == 0:
                        print(f"✅ {stage_name}: L-BFGS 段完成，恢復 Adam")
                remaining = num_epochs - start_epoch
                if remaining > 0:
                    PINN.train(num_epoch=remaining, lr=learning_rate, scheduler=stage_scheduler, profiler=_Noop(), start_epoch=start_epoch)
            else:
                if do_profile:
                    with torch.profiler.profile(
                        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
                        on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_log_dir),
                        record_shapes=True,
                        with_stack=True,
                        profile_memory=True
                    ) as prof:
                        PINN.train(num_epoch=epochs_to_run, lr=learning_rate, scheduler=stage_scheduler, profiler=prof, start_epoch=start_epoch)
                        if rank == 0:
                            print(f"🔧 Profiler數據已保存至: {profiler_log_dir}")
                else:
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
