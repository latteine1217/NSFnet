#!/usr/bin/env python3
"""
测试scheduler修复的功能
验证freeze/unfreeze后scheduler是否正常工作
"""

import torch
import torch.optim as optim

# 简化的测试：创建一个简单的模型和scheduler
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

def test_scheduler_reconstruction():
    print("🧪 测试Scheduler重建功能")
    print("=" * 50)
    
    # 创建模型和optimizer
    net = SimpleNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    
    # 创建CosineAnnealingLR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-4)
    
    print(f"初始学习率: {optimizer.param_groups[0]['lr']:.6f}")
    
    # 模拟几个step
    for i in range(5):
        scheduler.step()
        print(f"Step {i+1}: {optimizer.param_groups[0]['lr']:.6f}")
    
    print("\n📝 保存scheduler参数...")
    # 模拟保存scheduler参数的过程
    scheduler_params = {
        'class': type(scheduler),
        'T_max': getattr(scheduler, 'T_max', None),
        'eta_min': getattr(scheduler, 'eta_min', None),
        'last_epoch': scheduler.last_epoch
    }
    
    print(f"保存的参数: T_max={scheduler_params['T_max']}, eta_min={scheduler_params['eta_min']}, last_epoch={scheduler_params['last_epoch']}")
    
    # 模拟optimizer重建
    print("\n🔄 重建optimizer...")
    current_lr = optimizer.param_groups[0]['lr']
    optimizer = torch.optim.Adam(net.parameters(), lr=current_lr)
    
    # 确保有initial_lr参数 (这是修复的关键！)
    for group in optimizer.param_groups:
        group['initial_lr'] = current_lr
    
    # 重建scheduler
    print("🔄 重建scheduler...")
    new_scheduler = scheduler_params['class'](
        optimizer,
        T_max=scheduler_params['T_max'],
        eta_min=scheduler_params['eta_min'],
        last_epoch=scheduler_params['last_epoch']
    )
    
    print(f"重建后学习率: {optimizer.param_groups[0]['lr']:.6f}")
    
    # 继续几个step
    print("\n继续training:")
    for i in range(5):
        new_scheduler.step()
        print(f"Step {scheduler_params['last_epoch'] + i + 2}: {optimizer.param_groups[0]['lr']:.6f}")
    
    print("\n✅ 测试完成！")

if __name__ == "__main__":
    test_scheduler_reconstruction()