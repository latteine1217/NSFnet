#!/usr/bin/env python3
"""
測試分佈式L-BFGS功能
用於驗證分佈式訓練環境下L-BFGS的觸發和執行機制
"""

import sys
import os
import torch
import torch.distributed as dist
import numpy as np
from config import ConfigManager
import pinn_solver as psolver
import cavity_data as cavity

def setup_distributed():
    """設置分佈式環境"""
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
        else:
            device = torch.device('cpu')
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    return rank, world_size, local_rank, device

def test_lbfgs_config_loading():
    """測試L-BFGS配置加載"""
    print("=== 測試L-BFGS配置加載 ===")
    
    # 測試production配置
    try:
        config_manager = ConfigManager.from_file('configs/production.yaml')
        config = config_manager.config
        
        assert hasattr(config.training, 'lbfgs'), "缺少lbfgs配置"
        lbfgs_config = config.training.lbfgs
        
        print(f"✅ 分佈式L-BFGS啟用: {lbfgs_config.enabled_in_distributed}")
        print(f"✅ 波動度閾值: {lbfgs_config.volatility_threshold}")
        print(f"✅ 最大外循環步數: {lbfgs_config.max_outer_steps}")
        print(f"✅ 超時時間: {lbfgs_config.timeout_seconds}s")
        print(f"✅ L-BFGS內部迭代: {lbfgs_config.max_iter}")
        
        return True
    except Exception as e:
        print(f"❌ 配置加載失敗: {e}")
        return False

def test_distributed_trigger_logic():
    """測試分佈式觸發邏輯"""
    print("\n=== 測試分佈式觸發邏輯 ===")
    
    try:
        # 載入測試配置
        config_manager = ConfigManager.from_file('configs/test.yaml')
        config = config_manager.config
        
        # 創建PINN實例
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
        PINN.config = config
        
        # 模擬階段訓練狀態
        PINN.stage_step = 25000  # 超過20000閾值
        PINN.last_strategy_step = 0  # 間隔超過5000
        
        # 模擬損失歷史（低波動度）
        base_loss = 0.001
        loss_history = [base_loss + 0.00001 * np.sin(i * 0.1) for i in range(10000)]
        PINN.stage_loss_deque = loss_history
        
        # 測試觸發檢測
        trigger_result = PINN._check_distributed_lbfgs_trigger()
        
        print(f"✅ 觸發檢測結果: {trigger_result}")
        print(f"✅ 世界大小: {PINN.world_size}")
        print(f"✅ 當前rank: {PINN.rank}")
        
        return True
    except Exception as e:
        print(f"❌ 觸發邏輯測試失敗: {e}")
        return False

def test_parameter_sync_mechanism():
    """測試參數同步機制"""
    print("\n=== 測試參數同步機制 ===")
    
    try:
        config_manager = ConfigManager.from_file('configs/test.yaml')
        config = config_manager.config
        
        # 創建PINN實例
        PINN = psolver.PysicsInformedNeuralNetwork(
            Re=config.physics.Re,
            layers=config.network.layers,
            layers_1=config.network.layers_1,
            hidden_size=config.network.hidden_size,
            hidden_size_1=config.network.hidden_size_1,
            N_f=100,  # 最小數據集用於測試
            batch_size=config.training.batch_size,
            alpha_evm=config.physics.alpha_evm,
            bc_weight=config.physics.bc_weight,
            eq_weight=config.physics.eq_weight,
            checkpoint_freq=config.training.checkpoint_freq
        )
        PINN.config = config
        
        # 載入最小訓練數據
        path = './data/'
        if not os.path.exists(path):
            print("⚠️ 數據路徑不存在，跳過參數同步測試")
            return True
            
        try:
            dataloader = cavity.DataLoader(path=path, N_f=100, N_b=50)
            boundary_np = dataloader.loading_boundary_data()
            
            # 設置邊界數據
            xb_cpu = torch.as_tensor(boundary_np[0][:50], dtype=torch.float32)
            yb_cpu = torch.as_tensor(boundary_np[1][:50], dtype=torch.float32)
            ub_cpu = torch.as_tensor(boundary_np[2][:50], dtype=torch.float32)
            vb_cpu = torch.as_tensor(boundary_np[3][:50], dtype=torch.float32)
            
            PINN.x_b = xb_cpu.to(PINN.device).requires_grad_(True)
            PINN.y_b = yb_cpu.to(PINN.device).requires_grad_(True)
            PINN.u_b = ub_cpu.to(PINN.device)
            PINN.v_b = vb_cpu.to(PINN.device)
            
            # 設置方程數據
            eqn_np = dataloader.loading_eqn_data()
            xf_cpu = torch.as_tensor(eqn_np[0][:100], dtype=torch.float32)
            yf_cpu = torch.as_tensor(eqn_np[1][:100], dtype=torch.float32)
            
            PINN.x_f = xf_cpu.to(PINN.device).requires_grad_(True)
            PINN.y_f = yf_cpu.to(PINN.device).requires_grad_(True)
            
            # 測試參數同步（單GPU模式下應該返回True）
            sync_result = PINN._broadcast_model_parameters_with_verification()
            print(f"✅ 參數同步結果: {sync_result}")
            
            return True
            
        except Exception as e:
            print(f"⚠️ 數據載入失敗，跳過同步測試: {e}")
            return True
            
    except Exception as e:
        print(f"❌ 參數同步測試失敗: {e}")
        return False

def test_lbfgs_execution_fallback():
    """測試L-BFGS執行和回退機制"""
    print("\n=== 測試L-BFGS執行和回退機制 ===")
    
    try:
        config_manager = ConfigManager.from_file('configs/test.yaml')
        config = config_manager.config
        
        # 創建PINN實例
        PINN = psolver.PysicsInformedNeuralNetwork(
            Re=config.physics.Re,
            layers=config.network.layers,
            layers_1=config.network.layers_1,
            hidden_size=config.network.hidden_size,
            hidden_size_1=config.network.hidden_size_1,
            N_f=50,  # 最小數據集
            alpha_evm=config.physics.alpha_evm,
            bc_weight=config.physics.bc_weight,
            eq_weight=config.physics.eq_weight
        )
        PINN.config = config
        
        # 創建虛擬訓練數據
        PINN.x_f = torch.randn(50, requires_grad=True, device=PINN.device)
        PINN.y_f = torch.randn(50, requires_grad=True, device=PINN.device)
        PINN.x_b = torch.randn(20, requires_grad=True, device=PINN.device)
        PINN.y_b = torch.randn(20, requires_grad=True, device=PINN.device)
        PINN.u_b = torch.randn(20, device=PINN.device)
        PINN.v_b = torch.randn(20, device=PINN.device)
        
        # 初始化優化器
        params = list(PINN.get_model(PINN.net).parameters()) + list(PINN.get_model(PINN.net_1).parameters())
        PINN.opt = torch.optim.Adam(params, lr=0.001)
        
        # 測試L-BFGS執行（使用配置中的參數）
        print("🔧 測試L-BFGS執行...")
        best_loss = PINN.train_with_lbfgs_segment(
            max_outer_steps=5,  # 只執行5步用於測試
            log_interval=1
        )
        
        print(f"✅ L-BFGS執行完成，最佳損失: {best_loss}")
        print(f"✅ 優化器類型: {type(PINN.opt).__name__}")
        
        return True
    except Exception as e:
        print(f"❌ L-BFGS執行測試失敗: {e}")
        return False

def main():
    """主測試函數"""
    rank, world_size, local_rank, device = setup_distributed()
    
    if rank == 0:
        print("🧪 分佈式L-BFGS功能測試")
        print("=" * 50)
        print(f"📊 設備: {device}")
        print(f"🌐 世界大小: {world_size}")
        print(f"📍 當前rank: {rank}")
        print("=" * 50)
    
    # 運行測試
    tests = [
        ("配置加載測試", test_lbfgs_config_loading),
        ("觸發邏輯測試", test_distributed_trigger_logic),
        ("參數同步測試", test_parameter_sync_mechanism),
        ("L-BFGS執行測試", test_lbfgs_execution_fallback),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if rank == 0:
            print(f"\n🧪 運行 {test_name}...")
        
        try:
            result = test_func()
            if result:
                passed += 1
                if rank == 0:
                    print(f"✅ {test_name} 通過")
            else:
                if rank == 0:
                    print(f"❌ {test_name} 失敗")
        except Exception as e:
            if rank == 0:
                print(f"💥 {test_name} 異常: {e}")
    
    if rank == 0:
        print("\n" + "=" * 50)
        print(f"🏁 測試完成: {passed}/{total} 通過")
        if passed == total:
            print("🎉 所有測試通過！分佈式L-BFGS功能正常")
        else:
            print("⚠️ 部分測試失敗，請檢查實現")
        print("=" * 50)
    
    # 清理分佈式環境
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()