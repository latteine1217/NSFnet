#!/usr/bin/env python3
"""
檢查實際訓練中SGDR調度器狀態的調試腳本
"""
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.optim as optim
from config import load_config
from pinn_solver import PysicsInformedNeuralNetwork

def test_actual_sgdr_setup():
    """測試實際訓練設置中的SGDR調度器"""
    
    # 載入配置
    config_path = "configs/production.yaml"
    config = load_config(config_path)
    
    print("=== 實際SGDR設置測試 ===")
    print(f"配置文件: {config_path}")
    
    # 檢查配置
    stage_1 = config.training.training_stages[0]
    print(f"Stage 1 配置: alpha_evm={stage_1[0]}, epochs={stage_1[1]}, lr={stage_1[2]}, scheduler={stage_1[3]}")
    
    if hasattr(config.training, 'sgdr'):
        sgdr_cfg = config.training.sgdr
        print(f"SGDR配置: warmup_epochs={sgdr_cfg.warmup_epochs}, T_0={sgdr_cfg.T_0}, T_mult={sgdr_cfg.T_mult}")
    else:
        print("❌ 配置中未找到SGDR設定")
        return
    
    # 創建PINN實例
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    
    try:
        pinn = PysicsInformedNeuralNetwork(
            Re=config.physics.Re,
            layers=config.network.layers,
            layers_1=config.network.layers_1,
            hidden_size=config.network.hidden_size,
            hidden_size_1=config.network.hidden_size_1,
            N_f=min(1000, config.training.N_f),  # 使用較少的點進行測試
            alpha_evm=config.physics.alpha_evm,
            batch_size=None,
            bc_weight=int(config.physics.bc_weight),
            eq_weight=int(config.physics.eq_weight),
            supervised_data_weight=int(getattr(config.supervision, 'weight', 1)),
            supervised_data_points=getattr(config.supervision, 'data_points', 0),
            checkpoint_freq=10000,
            device=device
        )
        
        print("✅ PINN 實例創建成功")
        
    except Exception as e:
        print(f"❌ PINN 實例創建失敗: {e}")
        return
    
    # 模擬train.py中的scheduler創建邏輯
    num_epochs = stage_1[1]  # 225000
    learning_rate = stage_1[2]  # 1e-3
    sched_name = stage_1[3]  # 'SGDR'
    
    print(f"\n=== 模擬調度器創建 ===")
    print(f"num_epochs: {num_epochs}, lr: {learning_rate}, scheduler: {sched_name}")
    
    if sched_name in ['CosineAnnealingWarmRestarts', 'SGDR']:
        # 複製train.py中的邏輯
        sgdr_cfg = getattr(config.training, 'sgdr', None)
        default_warmup = max(500, int(0.05 * num_epochs))
        default_warmup = min(default_warmup, 10000)
        warmup_epochs = int(getattr(sgdr_cfg, 'warmup_epochs', default_warmup) if sgdr_cfg else default_warmup)
        
        remain = max(1, num_epochs - warmup_epochs)
        default_T0 = max(1000, int(0.25 * remain))
        T_0 = int(getattr(sgdr_cfg, 'T_0', default_T0) if sgdr_cfg else default_T0)
        T_mult = int(getattr(sgdr_cfg, 'T_mult', 2) if sgdr_cfg else 2)
        
        eta_min = float(getattr(sgdr_cfg, 'eta_min', learning_rate * 0.1) if sgdr_cfg else learning_rate * 0.1)
        start_factor = float(getattr(sgdr_cfg, 'start_factor', 0.1) if sgdr_cfg else 0.1)
        end_factor = float(getattr(sgdr_cfg, 'end_factor', 1.0) if sgdr_cfg else 1.0)
        
        print(f"warmup_epochs: {warmup_epochs}")
        print(f"T_0: {T_0}, T_mult: {T_mult}")
        print(f"eta_min: {eta_min}, start_factor: {start_factor}, end_factor: {end_factor}")
        
        # 建立調度器
        try:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                pinn.opt,
                start_factor=start_factor,
                end_factor=end_factor,
                total_iters=warmup_epochs
            )
            cawr_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                pinn.opt,
                T_0=T_0,
                T_mult=T_mult,
                eta_min=eta_min
            )
            stage_scheduler = torch.optim.lr_scheduler.SequentialLR(
                pinn.opt,
                schedulers=[warmup_sched, cawr_sched],
                milestones=[warmup_epochs]
            )
            
            print("✅ SGDR調度器創建成功")
            
            # 測試前幾步的學習率變化
            print(f"\n=== 前10步調度器測試 ===")
            for step in range(10):
                old_lr = pinn.opt.param_groups[0]['lr']
                stage_scheduler.step()
                new_lr = pinn.opt.param_groups[0]['lr']
                print(f"Step {step}: lr {old_lr:.8f} -> {new_lr:.8f}")
                
                # 模擬一個簡單的優化步驟
                pinn.opt.zero_grad()
                loss = torch.tensor(1.0, requires_grad=True)
                loss.backward()
                pinn.opt.step()
            
            # 測試跳到warmup結束附近
            print(f"\n=== Warmup結束附近測試 (epoch {warmup_epochs-2} 到 {warmup_epochs+2}) ===")
            # 重置調度器到warmup結束前
            stage_scheduler.last_epoch = warmup_epochs - 3
            for i in range(5):
                epoch = warmup_epochs - 2 + i
                old_lr = pinn.opt.param_groups[0]['lr']
                stage_scheduler.step()
                new_lr = pinn.opt.param_groups[0]['lr']
                print(f"Epoch {epoch}: lr {old_lr:.8f} -> {new_lr:.8f}")
                
        except Exception as e:
            print(f"❌ SGDR調度器創建或測試失敗: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print(f"❌ 不支援的調度器類型: {sched_name}")

if __name__ == "__main__":
    test_actual_sgdr_setup()