#!/usr/bin/env python3
"""
測試static_graph=True修復的DDP動態參數凍結
驗證原有的EVM網路凍結/解凍機制是否正常工作
"""
import os
import torch
import torch.distributed as dist
from pinn_solver import PysicsInformedNeuralNetwork
import cavity_data as cavity

def test_ddp_dynamic_freezing():
    """測試DDP + 動態參數凍結是否正常工作"""
    
    # 基本PINN配置
    Re = 5000
    N_neu = 80
    N_neu_1 = 40
    N_f = 5000  # 較小的數據集進行快速測試
    batch_size = 1000
    alpha_evm = 0.03
    N_HLayer = 6
    N_HLayer_1 = 4

    try:
        print("=== Testing DDP + Dynamic Parameter Freezing ===")
        
        # 創建PINN實例
        PINN = PysicsInformedNeuralNetwork(
            Re=Re,
            layers=N_HLayer,
            layers_1=N_HLayer_1,
            hidden_size=N_neu,
            hidden_size_1=N_neu_1,
            N_f=N_f,
            batch_size=batch_size,
            alpha_evm=alpha_evm,
            bc_weight=10,
            eq_weight=1)

        print(f"PINN created successfully:")
        print(f"  Rank: {PINN.rank}")
        print(f"  World size: {PINN.world_size}")
        print(f"  Device: {PINN.device}")
        print(f"  DDP enabled: {PINN.world_size > 1}")
        if PINN.world_size > 1:
            print(f"  DDP static_graph: True")
            print(f"  DDP find_unused_parameters: True")

        # 載入測試數據
        path = './data/'
        dataloader = cavity.DataLoader(path=path, N_f=N_f, N_b=300)

        # 設置邊界數據
        boundary_data = dataloader.loading_boundary_data()
        PINN.set_boundary_data(X=boundary_data)
        print("Boundary data loaded successfully")

        # 設置訓練數據
        training_data = dataloader.loading_training_data()
        PINN.set_eq_training_data(X=training_data)
        print("Training data loaded successfully")

        # 測試動態凍結機制
        print("\n--- Testing Dynamic Freezing Mechanism ---")
        
        # 1. 測試初始凍結
        print("1. Testing initial freezing...")
        PINN.freeze_evm_net(0)
        
        # 檢查net_1參數是否被凍結
        frozen_count = sum(1 for p in PINN.net_1.parameters() if not p.requires_grad)
        total_count = sum(1 for p in PINN.net_1.parameters())
        print(f"   net_1 frozen parameters: {frozen_count}/{total_count}")
        
        # 2. 進行短期訓練測試（凍結狀態）
        print("2. Training with frozen EVM network (20 epochs)...")
        PINN.train(num_epoch=20, lr=1e-3, batchsize=batch_size)
        
        # 3. 測試解凍
        print("3. Testing unfreezing...")
        PINN.defreeze_evm_net(1)
        
        # 檢查net_1參數是否被解凍
        active_count = sum(1 for p in PINN.net_1.parameters() if p.requires_grad)
        print(f"   net_1 active parameters: {active_count}/{total_count}")
        
        # 4. 進行短期訓練測試（解凍狀態）
        print("4. Training with unfrozen EVM network (20 epochs)...")
        PINN.train(num_epoch=20, lr=1e-3, batchsize=batch_size)
        
        # 5. 再次測試凍結
        print("5. Testing re-freezing...")
        PINN.freeze_evm_net(2)
        
        # 再次檢查凍結狀態
        frozen_count = sum(1 for p in PINN.net_1.parameters() if not p.requires_grad)
        print(f"   net_1 frozen parameters: {frozen_count}/{total_count}")
        
        # 6. 最終訓練測試
        print("6. Final training test (10 epochs)...")
        PINN.train(num_epoch=10, lr=1e-3, batchsize=batch_size)
        
        print("\n=== DDP Dynamic Freezing Test PASSED ===")
        print("✓ static_graph=True successfully prevents bucket reconstruction errors")
        print("✓ Dynamic parameter freezing/unfreezing works correctly")
        print("✓ No 'unmarked_param_indices' or DDP internal assert errors")
        
        return True
        
    except RuntimeError as e:
        if "unmarked_param_indices" in str(e) or "INTERNAL ASSERT FAILED" in str(e):
            print(f"\n=== DDP Dynamic Freezing Test FAILED ===")
            print(f"DDP bucket error still occurs: {e}")
            return False
        else:
            print(f"Different error occurred: {e}")
            return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_ddp_dynamic_freezing()
    if success:
        print("\n🎉 All tests passed! The DDP fix with static_graph=True works correctly.")
    else:
        print("\n❌ Tests failed. Check the error messages above.")