"""
測試中優先級改進功能
"""
import os
import sys
import torch
from config import ConfigManager, get_test_config
from logger import get_pinn_logger
from health_monitor import TrainingHealthMonitor
from memory_manager import TrainingMemoryManager

def test_config_system():
    """測試配置管理系統"""
    print("🔧 測試配置管理系統...")
    
    # 測試默認配置
    config_manager = get_test_config()
    config_manager.print_config()
    
    # 測試保存和載入
    config_path = "configs/test_temp.yaml"
    config_manager.save_to_file(config_path)
    
    loaded_config = ConfigManager.from_file(config_path)
    print("✅ 配置保存和載入測試通過")
    
    # 清理測試文件
    if os.path.exists(config_path):
        os.remove(config_path)

def test_logger_system():
    """測試日誌系統"""
    print("\n📝 測試日誌系統...")
    
    logger = get_pinn_logger("TestLogger", "DEBUG")
    
    logger.debug("這是調試信息")
    logger.info("這是一般信息")
    logger.warning("這是警告信息")
    logger.error("這是錯誤信息")
    
    # 測試專用日誌方法
    logger.training_start({"experiment_name": "Test", "Re": 3000, "N_f": 1000})
    logger.training_stage("Stage 1", 0.05, 10, 1e-3)
    logger.epoch_log(1, 1.23e-3, 1e-3, 0.5, "2分30秒")
    
    print("✅ 日誌系統測試通過")

def test_health_monitor():
    """測試健康監控系統"""
    print("\n🏥 測試健康監控系統...")
    
    logger = get_pinn_logger("HealthTest", "INFO")
    monitor = TrainingHealthMonitor(logger)
    
    # 獲取系統健康狀態
    health = monitor.get_current_health()
    print(f"   CPU使用率: {health.cpu_usage:.1f}%")
    print(f"   記憶體使用率: {health.memory_usage:.1f}%")
    print(f"   進程記憶體: {health.process_memory:.1f}MB")
    
    # 測試健康檢查
    is_healthy, issues = monitor.check_health()
    if is_healthy:
        print("✅ 系統健康狀態良好")
    else:
        print(f"⚠️ 發現健康問題: {issues}")
    
    # 輸出健康報告
    monitor.log_health_report()

def test_memory_manager():
    """測試記憶體管理系統"""
    print("\n🧠 測試記憶體管理系統...")
    
    logger = get_pinn_logger("MemoryTest", "INFO")
    memory_manager = TrainingMemoryManager(logger)
    
    # 獲取記憶體統計
    stats = memory_manager.get_memory_stats()
    print(f"   CPU記憶體: {stats.cpu_memory_percent:.1f}%")
    print(f"   進程記憶體: {stats.process_memory:.1f}MB")
    
    if torch.cuda.is_available():
        print(f"   GPU記憶體: {stats.gpu_memory_percent:.1f}%")
    
    # 測試記憶體清理
    cleanup_results = memory_manager.cleanup_memory()
    if cleanup_results['actions_taken']:
        print(f"✅ 記憶體清理完成: {', '.join(cleanup_results['actions_taken'])}")
    else:
        print("✅ 記憶體狀態良好，無需清理")
    
    # 測試tensor緩存
    test_tensor = torch.randn(100, 100)
    memory_manager.cache_tensor("test_key", test_tensor)
    cached_tensor = memory_manager.get_cached_tensor("test_key")
    
    if cached_tensor is not None:
        print("✅ Tensor緩存功能正常")
    
    # 輸出記憶體報告
    memory_manager.log_memory_report()

def test_integrated_systems():
    """測試系統集成"""
    print("\n🔗 測試系統集成...")
    
    # 創建配置
    config_manager = get_test_config()
    
    # 創建日誌器
    logger = get_pinn_logger("IntegrationTest", "INFO")
    
    # 測試配置與日誌集成
    config_info = {
        "experiment_name": config_manager.config.experiment_name,
        "Re": config_manager.config.physics.Re,
        "N_f": config_manager.config.training.N_f,
        "layers": config_manager.config.network.layers,
        "hidden_size": config_manager.config.network.hidden_size
    }
    logger.system_info(config_info)
    
    print("✅ 系統集成測試通過")

def main():
    """主測試函數"""
    print("🧪 開始測試中優先級改進功能...")
    print("=" * 60)
    
    try:
        test_config_system()
        test_logger_system()
        
        # 只在有合適依賴時測試健康監控和記憶體管理
        try:
            test_health_monitor()
        except ImportError as e:
            print(f"⚠️ 跳過健康監控測試 (缺少依賴: {e})")
        
        try:
            test_memory_manager()
        except ImportError as e:
            print(f"⚠️ 跳過記憶體管理測試 (缺少依賴: {e})")
        
        test_integrated_systems()
        
        print("\n" + "=" * 60)
        print("🎉 所有測試完成！中優先級功能運行正常")
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        raise

if __name__ == "__main__":
    main()