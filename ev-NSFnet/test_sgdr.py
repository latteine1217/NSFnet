#!/usr/bin/env python3
"""
SGDR調度器測試 - 驗證SequentialLR組合是否正常工作
"""
import torch
import matplotlib.pyplot as plt
import numpy as np

def test_sgdr_scheduler():
    """測試SGDR調度器的實際行為"""
    
    # 創建一個簡單的模型和優化器用於測試
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 模擬Stage 1的SGDR配置
    warmup_epochs = 5000
    T_0 = 30000
    T_mult = 2
    eta_min = 1e-6
    start_factor = 0.1
    end_factor = 1.0
    total_epochs = 50000  # 測試用，比實際225000少
    
    print("=== SGDR調度器測試 ===")
    print(f"warmup_epochs: {warmup_epochs}")
    print(f"T_0: {T_0}")
    print(f"T_mult: {T_mult}")
    print(f"eta_min: {eta_min}")
    print(f"base_lr: {optimizer.param_groups[0]['lr']}")
    
    # 建立SequentialLR: LinearLR (warmup) -> CosineAnnealingWarmRestarts
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=end_factor,
        total_iters=warmup_epochs
    )
    
    cawr_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=T_0,
        T_mult=T_mult,
        eta_min=eta_min
    )
    
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cawr_sched],
        milestones=[warmup_epochs]
    )
    
    # 記錄學習率變化
    epochs = []
    learning_rates = []
    
    for epoch in range(total_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        epochs.append(epoch)
        learning_rates.append(current_lr)
        
        # 每5000步輸出一次
        if epoch % 5000 == 0:
            print(f"Epoch {epoch:6d}: lr = {current_lr:.8f}")
        
        # 模擬一個簡單的訓練步驟
        optimizer.zero_grad()
        loss = torch.sum(model.weight ** 2)  # 簡單的L2損失
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    # 繪製學習率曲線
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, learning_rates, 'b-', linewidth=1)
    plt.axvline(x=warmup_epochs, color='r', linestyle='--', alpha=0.7, label=f'Warmup結束 ({warmup_epochs})')
    plt.axvline(x=warmup_epochs + T_0, color='g', linestyle='--', alpha=0.7, label=f'第一週期結束 ({warmup_epochs + T_0})')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('SGDR Learning Rate Schedule')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('sgdr_test_lr_curve.png', dpi=150, bbox_inches='tight')
    print(f"\n學習率曲線已保存到: sgdr_test_lr_curve.png")
    
    # 檢查關鍵時間點的學習率
    key_points = [0, warmup_epochs//2, warmup_epochs-1, warmup_epochs, warmup_epochs + T_0//2, warmup_epochs + T_0]
    print("\n=== 關鍵時間點檢查 ===")
    for point in key_points:
        if point < len(learning_rates):
            print(f"Epoch {point:6d}: lr = {learning_rates[point]:.8f}")
    
    # 驗證SGDR是否正常工作
    warmup_start = learning_rates[0]
    warmup_end = learning_rates[warmup_epochs-1] if warmup_epochs < len(learning_rates) else learning_rates[-1]
    
    print(f"\n=== 驗證結果 ===")
    print(f"Warmup階段: {warmup_start:.8f} -> {warmup_end:.8f}")
    
    # 檢查warmup是否正常
    if abs(warmup_start - 1e-3 * start_factor) < 1e-8:
        print("✅ Warmup起始學習率正確")
    else:
        print(f"❌ Warmup起始學習率錯誤: 期望{1e-3 * start_factor:.8f}, 實際{warmup_start:.8f}")
    
    if abs(warmup_end - 1e-3 * end_factor) < 1e-8:
        print("✅ Warmup結束學習率正確")
    else:
        print(f"❌ Warmup結束學習率錯誤: 期望{1e-3 * end_factor:.8f}, 實際{warmup_end:.8f}")
    
    # 檢查SGDR週期是否啟動
    if warmup_epochs + 100 < len(learning_rates):
        post_warmup_lrs = learning_rates[warmup_epochs:warmup_epochs + min(1000, T_0)]
        lr_changes = [abs(post_warmup_lrs[i+1] - post_warmup_lrs[i]) for i in range(len(post_warmup_lrs)-1)]
        avg_change = np.mean(lr_changes)
        
        if avg_change > 1e-8:
            print("✅ SGDR週期正常啟動（學習率有變化）")
            print(f"   Warmup後平均學習率變化: {avg_change:.2e}")
        else:
            print("❌ SGDR週期未啟動（學習率無變化）")
            print(f"   Warmup後平均學習率變化: {avg_change:.2e}")

if __name__ == "__main__":
    test_sgdr_scheduler()