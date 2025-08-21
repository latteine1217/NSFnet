#!/usr/bin/env python3
"""
修復SGDR調度器的問題並重新測試
"""
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

def test_corrected_sgdr():
    """測試修復後的SGDR調度器"""
    
    print("=== 修復後SGDR調度器測試 ===")
    
    # 使用生產配置的參數
    learning_rate = 1e-3
    warmup_epochs = 5000
    T_0 = 30000
    T_mult = 2
    eta_min = 1e-6
    start_factor = 0.1
    end_factor = 1.0
    
    # 創建模型和優化器
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"基礎學習率: {learning_rate}")
    print(f"Warmup期間: {warmup_epochs} epochs")
    print(f"T_0: {T_0}, T_mult: {T_mult}, eta_min: {eta_min}")
    
    # 方法1：修復SequentialLR，確保CAWR從epoch 0開始計算
    print("\n=== 方法1：修復SequentialLR ===")
    optimizer1 = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer1,
        start_factor=start_factor,
        end_factor=end_factor,
        total_iters=warmup_epochs
    )
    
    # 關鍵修復：CAWR的last_epoch應該從-1開始，讓它在warmup結束後正確啟動
    cawr_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer1,
        T_0=T_0,
        T_mult=T_mult,
        eta_min=eta_min,
        last_epoch=-1  # 確保從初始狀態開始
    )
    
    scheduler1 = torch.optim.lr_scheduler.SequentialLR(
        optimizer1,
        schedulers=[warmup_sched, cawr_sched],
        milestones=[warmup_epochs]
    )
    
    # 測試前幾個關鍵點
    test_epochs = [warmup_epochs-1, warmup_epochs, warmup_epochs+1, warmup_epochs+100, warmup_epochs+1000]
    
    print("\n關鍵時間點測試:")
    for epoch in range(max(test_epochs) + 1):
        current_lr = optimizer1.param_groups[0]['lr']
        
        if epoch in test_epochs:
            print(f"Epoch {epoch:6d}: lr = {current_lr:.8f}")
        
        # 模擬訓練步驟
        optimizer1.zero_grad()
        loss = torch.sum(model.weight ** 2)
        loss.backward()
        optimizer1.step()
        scheduler1.step()
    
    # 檢查SGDR是否正常工作
    warmup_end_lr = None
    post_warmup_lrs = []
    
    # 重新運行以收集數據
    optimizer2 = torch.optim.Adam(model.parameters(), lr=learning_rate)
    warmup_sched2 = torch.optim.lr_scheduler.LinearLR(
        optimizer2, start_factor=start_factor, end_factor=end_factor, total_iters=warmup_epochs
    )
    cawr_sched2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer2, T_0=T_0, T_mult=T_mult, eta_min=eta_min, last_epoch=-1
    )
    scheduler2 = torch.optim.lr_scheduler.SequentialLR(
        optimizer2, schedulers=[warmup_sched2, cawr_sched2], milestones=[warmup_epochs]
    )
    
    epochs = []
    lrs = []
    
    for epoch in range(warmup_epochs + 5000):  # 測試warmup + 5000步
        current_lr = optimizer2.param_groups[0]['lr']
        epochs.append(epoch)
        lrs.append(current_lr)
        
        if epoch == warmup_epochs:
            warmup_end_lr = current_lr
        elif epoch > warmup_epochs:
            post_warmup_lrs.append(current_lr)
        
        optimizer2.zero_grad()
        loss = torch.sum(model.weight ** 2)
        loss.backward()
        optimizer2.step()
        scheduler2.step()
    
    # 分析結果
    if len(post_warmup_lrs) > 100:
        lr_changes = [abs(post_warmup_lrs[i+1] - post_warmup_lrs[i]) for i in range(100)]
        avg_change = sum(lr_changes) / len(lr_changes)
        max_change = max(lr_changes)
        
        print(f"\n=== 分析結果 ===")
        print(f"Warmup結束時學習率: {warmup_end_lr:.8f}")
        print(f"期望學習率: {learning_rate * end_factor:.8f}")
        print(f"Warmup後100步平均變化: {avg_change:.2e}")
        print(f"Warmup後100步最大變化: {max_change:.2e}")
        
        if avg_change > 1e-6:
            print("✅ SGDR週期正常啟動")
        else:
            print("❌ SGDR週期仍未啟動")
        
    # 方法2：嘗試手動實現SGDR邏輯
    print("\n=== 方法2：手動實現SGDR ===")
    
    class ManualSGDR:
        def __init__(self, optimizer, warmup_epochs, T_0, T_mult, eta_min, start_factor, end_factor, base_lr):
            self.optimizer = optimizer
            self.warmup_epochs = warmup_epochs
            self.T_0 = T_0
            self.T_mult = T_mult
            self.eta_min = eta_min
            self.start_factor = start_factor
            self.end_factor = end_factor
            self.base_lr = base_lr
            self.step_count = 0
            
        def step(self):
            if self.step_count < self.warmup_epochs:
                # Warmup階段
                progress = self.step_count / self.warmup_epochs
                factor = self.start_factor + (self.end_factor - self.start_factor) * progress
                lr = self.base_lr * factor
            else:
                # SGDR階段
                sgdr_step = self.step_count - self.warmup_epochs
                
                # 計算當前在哪個週期
                cycle = 0
                cycle_length = self.T_0
                remaining_steps = sgdr_step
                
                while remaining_steps >= cycle_length:
                    remaining_steps -= cycle_length
                    cycle += 1
                    cycle_length *= self.T_mult
                
                # 在當前週期內的進度
                progress = remaining_steps / cycle_length
                import math
                lr = self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            self.step_count += 1
            return lr
    
    # 測試手動實現
    optimizer3 = torch.optim.Adam(model.parameters(), lr=learning_rate)
    manual_scheduler = ManualSGDR(optimizer3, warmup_epochs, T_0, T_mult, eta_min, start_factor, end_factor, learning_rate)
    
    print("\n手動SGDR關鍵時間點:")
    manual_lrs = []
    for epoch in range(warmup_epochs + 2000):
        lr = manual_scheduler.step()
        manual_lrs.append(lr)
        
        if epoch in [warmup_epochs-1, warmup_epochs, warmup_epochs+1, warmup_epochs+100, warmup_epochs+1000]:
            print(f"Epoch {epoch:6d}: lr = {lr:.8f}")
        
        optimizer3.zero_grad()
        loss = torch.sum(model.weight ** 2)
        loss.backward()
        optimizer3.step()
    
    # 檢查手動實現的變化
    manual_post_warmup = manual_lrs[warmup_epochs:warmup_epochs+100]
    manual_changes = [abs(manual_post_warmup[i+1] - manual_post_warmup[i]) for i in range(99)]
    manual_avg_change = sum(manual_changes) / len(manual_changes)
    
    print(f"\n手動SGDR Warmup後100步平均變化: {manual_avg_change:.2e}")
    
    if manual_avg_change > 1e-6:
        print("✅ 手動SGDR正常工作")
        return True
    else:
        print("❌ 手動SGDR也有問題")
        return False

if __name__ == "__main__":
    sgdr_working = test_corrected_sgdr()
    
    if sgdr_working:
        print("\n🎉 找到SGDR問題的解決方案！")
        print("💡 建議使用手動實現的SGDR邏輯替換SequentialLR")
    else:
        print("\n🤔 需要進一步調查SGDR問題的根源")