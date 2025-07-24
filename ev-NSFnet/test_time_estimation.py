#!/usr/bin/env python3
"""
簡單測試腳本，驗證時間預估和TensorBoard功能
"""

import os
import torch
import numpy as np
import time

# 設定環境變數以支持單GPU運行
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'

import cavity_data as cavity
import pinn_solver as psolver

def test_time_estimation():
    """測試時間預估功能"""
    print("🧪 開始測試時間預估和TensorBoard功能...")
    
    # 創建小規模的PINN進行測試
    Re = 3000
    N_neu = 20  # 減小網路規模以加快測試
    N_neu_1 = 10
    lam_bcs = 10
    lam_equ = 1
    N_f = 1000  # 減少訓練點數量
    alpha_evm = 0.03
    N_HLayer = 2  # 減少層數
    N_HLayer_1 = 2

    print(f"創建測試 PINN (Re={Re}, N_f={N_f})...")
    
    PINN = psolver.PysicsInformedNeuralNetwork(
        Re=Re,
        layers=N_HLayer,
        layers_1=N_HLayer_1,
        hidden_size=N_neu,
        hidden_size_1=N_neu_1,
        N_f=N_f,
        alpha_evm=alpha_evm,
        bc_weight=lam_bcs,
        eq_weight=lam_equ)

    # 準備測試數據
    print("準備測試數據...")
    path = './data/'
    dataloader = cavity.DataLoader(path=path, N_f=N_f, N_b=100)

    boundary_data = dataloader.loading_boundary_data()
    PINN.set_boundary_data(X=boundary_data)

    training_data = dataloader.loading_training_data()
    PINN.set_eq_training_data(X=training_data)

    # 進行短時間訓練測試
    print("開始短時間訓練測試...")
    
    test_stages = [
        (0.05, 10, 1e-3, "Test Stage 1"),
        (0.03, 10, 2e-4, "Test Stage 2"),
    ]

    for alpha, epochs, lr, stage_name in test_stages:
        print(f"\n🎯 測試 {stage_name}: alpha_evm={alpha}, epochs={epochs}")
        
        PINN.current_stage = stage_name
        PINN.set_alpha_evm(alpha)
        PINN.train(num_epoch=epochs, lr=lr)

    # 關閉TensorBoard writer
    if hasattr(PINN, 'tb_writer') and PINN.tb_writer is not None:
        PINN.tb_writer.close()
        print("📊 TensorBoard 日誌已保存")

    print("✅ 測試完成！")
    print("💡 請檢查 runs/ 目錄中的 TensorBoard 日誌")
    print("💡 執行 'tensorboard --logdir=runs' 查看訓練過程")

if __name__ == "__main__":
    test_time_estimation()