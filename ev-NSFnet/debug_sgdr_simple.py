#!/usr/bin/env python3
"""
簡化的SGDR調度器調試腳本
"""
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# 檢查當前專案中SGDR的確切問題
def debug_sgdr_behavior():
    """調試SGDR調度器的行為"""
    
    print("=== SGDR調度器調試 ===")
    
    # 使用生產配置的參數
    learning_rate = 1e-3
    warmup_epochs = 5000
    T_0 = 30000
    T_mult = 2
    eta_min = 1e-6
    start_factor = 0.1
    end_factor = 1.0
    
    # 創建一個簡單的模型來測試
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"基礎學習率: {learning_rate}")
    print(f"Warmup期間: {warmup_epochs} epochs")
    print(f"T_0: {T_0}, T_mult: {T_mult}")
    print(f"eta_min: {eta_min}")
    
    # 創建和產品代碼相同的SGDR調度器
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
    
    print("✅ 調度器創建成功")
    
    # 檢查不同階段的學習率
    test_epochs = [0, 2500, warmup_epochs-1, warmup_epochs, warmup_epochs+1000, warmup_epochs+T_0//2, warmup_epochs+T_0]
    
    lrs = []
    epochs = []
    
    print("\n=== 關鍵時間點測試 ===")
    for epoch in range(max(test_epochs) + 100):
        current_lr = optimizer.param_groups[0]['lr']
        
        if epoch in test_epochs:
            print(f"Epoch {epoch:6d}: lr = {current_lr:.8f}")
        
        epochs.append(epoch)
        lrs.append(current_lr)
        
        # 模擬訓練步驟
        optimizer.zero_grad()
        loss = torch.sum(model.weight ** 2)
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    # 檢查Warmup後的學習率變化
    warmup_end_lr = lrs[warmup_epochs]
    post_warmup_lrs = lrs[warmup_epochs:warmup_epochs+1000]
    
    # 計算學習率變化
    lr_changes = [abs(post_warmup_lrs[i+1] - post_warmup_lrs[i]) for i in range(len(post_warmup_lrs)-1)]
    avg_change = sum(lr_changes) / len(lr_changes) if lr_changes else 0
    max_change = max(lr_changes) if lr_changes else 0
    
    print(f"\n=== 行為分析 ===")
    print(f"Warmup結束時學習率: {warmup_end_lr:.8f}")
    print(f"期望學習率: {learning_rate * end_factor:.8f}")
    print(f"Warmup後1000步平均變化: {avg_change:.2e}")
    print(f"Warmup後1000步最大變化: {max_change:.2e}")
    
    # 判斷SGDR是否正常工作
    if abs(warmup_end_lr - learning_rate * end_factor) < 1e-7:
        print("✅ Warmup階段正常")
    else:
        print("❌ Warmup階段異常")
    
    if avg_change > 1e-6:
        print("✅ SGDR週期正常啟動")
    else:
        print("❌ SGDR週期未啟動，學習率保持常數")
    
    # 保存學習率曲線
    try:
        plt.figure(figsize=(12, 8))
        
        # 繪製前部分（包含Warmup）
        plt.subplot(2, 1, 1)
        warmup_range = range(warmup_epochs + 5000)
        plt.plot([epochs[i] for i in warmup_range], [lrs[i] for i in warmup_range], 'b-', linewidth=1)
        plt.axvline(x=warmup_epochs, color='r', linestyle='--', alpha=0.7, label=f'Warmup結束')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('SGDR Learning Rate - Warmup + Early SGDR')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 繪製SGDR週期部分
        plt.subplot(2, 1, 2)
        sgdr_start = warmup_epochs
        sgdr_range = range(sgdr_start, min(len(epochs), sgdr_start + T_0 + 1000))
        plt.plot([epochs[i] for i in sgdr_range], [lrs[i] for i in sgdr_range], 'g-', linewidth=1)
        plt.axvline(x=warmup_epochs + T_0, color='orange', linestyle='--', alpha=0.7, label=f'第一週期結束')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('SGDR Learning Rate - First Cycle')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sgdr_debug_analysis.png', dpi=150, bbox_inches='tight')
        print(f"\n學習率分析圖已保存到: sgdr_debug_analysis.png")
    except Exception as e:
        print(f"無法保存圖表: {e}")
    
    return avg_change > 1e-6

if __name__ == "__main__":
    sgdr_working = debug_sgdr_behavior()
    if sgdr_working:
        print("\n🎉 SGDR調度器本身工作正常！")
        print("📋 問題可能在於:")
        print("   1. 實際訓練代碼中scheduler未正確傳遞")
        print("   2. L-BFGS後scheduler重建有問題")
        print("   3. 分佈式環境下scheduler同步問題")
    else:
        print("\n❌ SGDR調度器本身有問題")