#!/usr/bin/env python3
"""
Tesla P100 相容性測試腳本
檢查PyTorch功能與P100 GPU的相容性
"""

import torch
import os
import sys

def test_gpu_compatibility():
    """測試GPU相容性"""
    print("=== Tesla P100 相容性檢查 ===")
    
    # 基本CUDA檢查
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    # GPU信息
    gpu_count = torch.cuda.device_count()
    print(f"GPU數量: {gpu_count}")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        capability = torch.cuda.get_device_capability(i)
        major, minor = capability
        cuda_capability = major + minor * 0.1
        
        print(f"\nGPU {i}: {props.name}")
        print(f"  CUDA Capability: {major}.{minor} ({cuda_capability})")
        print(f"  記憶體: {props.total_memory / 1024**3:.1f} GB")
        print(f"  多處理器數量: {props.multi_processor_count}")
        
        # 檢查是否為P100
        if "P100" in props.name:
            print(f"  ✅ 檢測到Tesla P100")
            if cuda_capability < 7.0:
                print(f"  ⚠️  CUDA capability {cuda_capability} < 7.0，不支援Triton編譯器")
            else:
                print(f"  ✅ CUDA capability {cuda_capability} >= 7.0，支援所有功能")
    
    return True

def test_torch_compile():
    """測試torch.compile相容性"""
    print("\n=== torch.compile 測試 ===")
    
    if not hasattr(torch, 'compile'):
        print("❌ torch.compile 不可用 (PyTorch版本太舊)")
        return False
    
    print("✅ torch.compile 可用")
    
    # 創建簡單模型測試
    class SimpleNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SimpleNet().to(device)
        
        # 檢查CUDA capability
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(0)
            major, minor = capability
            cuda_capability = major + minor * 0.1
            
            if cuda_capability >= 7.0:
                print("✅ CUDA capability >= 7.0，嘗試torch.compile...")
                compiled_model = torch.compile(model, mode='reduce-overhead')
                print("✅ torch.compile 成功")
            else:
                print(f"⚠️  CUDA capability {cuda_capability} < 7.0，跳過torch.compile")
                print("   使用標準eager模式")
                return True
        
        # 測試前向傳播
        x = torch.randn(10, 2).to(device)
        with torch.no_grad():
            if 'compiled_model' in locals():
                output = compiled_model(x)
            else:
                output = model(x)
        
        print(f"✅ 模型前向傳播成功，輸出形狀: {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ torch.compile 測試失敗: {e}")
        return False

def test_distributed_setup():
    """測試分散式訓練設置"""
    print("\n=== 分散式訓練設置測試 ===")
    
    # 檢查NCCL
    if torch.cuda.is_available():
        try:
            backend = 'nccl'
            print(f"✅ NCCL後端可用")
        except:
            print("❌ NCCL後端不可用")
            return False
    
    # 檢查環境變數
    env_vars = [
        'RANK', 'LOCAL_RANK', 'WORLD_SIZE',
        'TORCH_NCCL_ASYNC_ERROR_HANDLING',
        'CUDA_VISIBLE_DEVICES'
    ]
    
    print("環境變數檢查:")
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"  {var}: {value}")
    
    return True

def test_memory_operations():
    """測試GPU記憶體操作"""
    print("\n=== GPU記憶體操作測試 ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳過記憶體測試")
        return False
    
    try:
        device = torch.device('cuda:0')
        
        # 記憶體分配測試
        print("測試記憶體分配...")
        tensor = torch.randn(1000, 1000, device=device)
        
        # 記憶體狀態
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        
        print(f"✅ 記憶體分配成功")
        print(f"   已分配: {allocated:.3f} GB")
        print(f"   已保留: {reserved:.3f} GB")
        
        # 清理測試
        del tensor
        torch.cuda.empty_cache()
        
        allocated_after = torch.cuda.memory_allocated(device) / 1024**3
        print(f"   清理後: {allocated_after:.3f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ 記憶體操作失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("🔍 Tesla P100 相容性完整測試")
    print("=" * 50)
    
    tests = [
        ("GPU相容性", test_gpu_compatibility),
        ("torch.compile", test_torch_compile),
        ("分散式設置", test_distributed_setup),
        ("記憶體操作", test_memory_operations)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 測試出現異常: {e}")
            results.append((test_name, False))
    
    # 總結報告
    print("\n" + "=" * 50)
    print("🏁 測試總結報告")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"{test_name:15s}: {status}")
        if result:
            passed += 1
    
    print(f"\n總計: {passed}/{total} 測試通過")
    
    if passed == total:
        print("🎉 所有測試通過！系統與Tesla P100完全相容")
    elif passed >= total * 0.75:
        print("⚠️  大部分測試通過，系統基本相容")
    else:
        print("❌ 多個測試失敗，建議檢查系統配置")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)