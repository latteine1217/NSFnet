"""
健康檢查機制 - PINN系統狀態監控
"""
import os
import time
import psutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.distributed as dist
from logger import PINNLogger
import threading
import queue

# 可選依賴
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

try:
    import nvidia_ml_py3 as nvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False


@dataclass
class SystemHealth:
    """系統健康狀態"""
    timestamp: float
    cpu_usage: float                # CPU使用率 (%)
    memory_usage: float             # 記憶體使用率 (%)
    gpu_usage: List[float]          # GPU使用率列表 (%)
    gpu_memory: List[float]         # GPU記憶體使用率列表 (%)
    disk_usage: float              # 磁碟使用率 (%)
    network_io: Tuple[int, int]    # 網路IO (bytes_sent, bytes_recv)
    process_memory: float          # 當前進程記憶體使用量 (MB)
    
    def is_healthy(self, thresholds: 'HealthThresholds') -> Tuple[bool, List[str]]:
        """檢查系統是否健康"""
        issues = []
        
        if self.cpu_usage > thresholds.cpu_warning:
            issues.append(f"CPU使用率過高: {self.cpu_usage:.1f}%")
        
        if self.memory_usage > thresholds.memory_warning:
            issues.append(f"記憶體使用率過高: {self.memory_usage:.1f}%")
        
        for i, usage in enumerate(self.gpu_usage):
            if usage > thresholds.gpu_warning:
                issues.append(f"GPU {i} 使用率過高: {usage:.1f}%")
        
        for i, mem in enumerate(self.gpu_memory):
            if mem > thresholds.gpu_memory_warning:
                issues.append(f"GPU {i} 記憶體使用率過高: {mem:.1f}%")
        
        if self.disk_usage > thresholds.disk_warning:
            issues.append(f"磁碟使用率過高: {self.disk_usage:.1f}%")
        
        if self.process_memory > thresholds.process_memory_warning:
            issues.append(f"進程記憶體使用量過高: {self.process_memory:.1f}MB")
        
        return len(issues) == 0, issues


@dataclass 
class HealthThresholds:
    """健康檢查閾值"""
    cpu_warning: float = 80.0              # CPU警告閾值 (%)
    memory_warning: float = 85.0           # 記憶體警告閾值 (%)
    gpu_warning: float = 90.0              # GPU使用率警告閾值 (%)
    gpu_memory_warning: float = 90.0       # GPU記憶體警告閾值 (%)
    disk_warning: float = 85.0             # 磁碟使用率警告閾值 (%)
    process_memory_warning: float = 8000.0  # 進程記憶體警告閾值 (MB)


class HealthMonitor:
    """系統健康監控器"""
    
    def __init__(self, 
                 logger: PINNLogger,
                 thresholds: Optional[HealthThresholds] = None,
                 check_interval: float = 30.0,
                 history_size: int = 100):
        
        self.logger = logger
        self.thresholds = thresholds or HealthThresholds()
        self.check_interval = check_interval
        self.history_size = history_size
        
        self.health_history: List[SystemHealth] = []
        self.is_monitoring = False
        self.monitor_thread = None
        self.alert_queue = queue.Queue()
        
        # 初始化系統信息
        self._init_system_info()
    
    def _init_system_info(self):
        """初始化系統信息"""
        try:
            # CPU信息
            self.cpu_count = psutil.cpu_count()
            self.cpu_freq = psutil.cpu_freq()
            
            # 記憶體信息
            self.memory_total = psutil.virtual_memory().total / (1024**3)  # GB
            
            # GPU信息
            if HAS_GPUTIL:
                self.gpus = GPUtil.getGPUs() if GPUtil.getGPUs() else []
            else:
                self.gpus = []
            
            # 磁碟信息
            self.disk_total = psutil.disk_usage('/').total / (1024**3)  # GB
            
            self.logger.info("🔍 系統信息初始化完成")
            self.logger.info(f"   CPU: {self.cpu_count} 核心")
            self.logger.info(f"   記憶體: {self.memory_total:.1f} GB")
            self.logger.info(f"   GPU: {len(self.gpus)} 個設備")
            self.logger.info(f"   磁碟: {self.disk_total:.1f} GB")
            
        except Exception as e:
            self.logger.warning(f"系統信息初始化失敗: {e}")
    
    def get_current_health(self) -> SystemHealth:
        """獲取當前系統健康狀態"""
        try:
            # CPU使用率
            cpu_usage = psutil.cpu_percent(interval=1.0)
            
            # 記憶體使用率
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # GPU使用率和記憶體
            gpu_usage = []
            gpu_memory = []
            
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    # PyTorch GPU信息
                    props = torch.cuda.get_device_properties(i)
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    cached = torch.cuda.memory_reserved(i) / (1024**3)
                    total = props.total_memory / (1024**3)
                    
                    gpu_memory.append((allocated / total) * 100)
                    
                    # 嘗試獲取GPU使用率
                    if HAS_NVML:
                        try:
                            nvml.nvmlInit()
                            handle = nvml.nvmlDeviceGetHandleByIndex(i)
                            utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                            gpu_usage.append(utilization.gpu)
                        except:
                            gpu_usage.append(0.0)  # 無法獲取時設為0
                    else:
                        gpu_usage.append(0.0)  # 沒有nvml時設為0
            
            # 磁碟使用率
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # 網路IO
            net_io = psutil.net_io_counters()
            network_io = (net_io.bytes_sent, net_io.bytes_recv)
            
            # 當前進程記憶體
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024**2)  # MB
            
            return SystemHealth(
                timestamp=time.time(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                gpu_memory=gpu_memory,
                disk_usage=disk_usage,
                network_io=network_io,
                process_memory=process_memory
            )
            
        except Exception as e:
            self.logger.error(f"獲取系統健康狀態失敗: {e}")
            return SystemHealth(
                timestamp=time.time(),
                cpu_usage=0.0, memory_usage=0.0,
                gpu_usage=[], gpu_memory=[],
                disk_usage=0.0, network_io=(0, 0),
                process_memory=0.0
            )
    
    def check_health(self) -> Tuple[bool, List[str]]:
        """執行健康檢查"""
        health = self.get_current_health()
        
        # 添加到歷史記錄
        self.health_history.append(health)
        if len(self.health_history) > self.history_size:
            self.health_history.pop(0)
        
        # 檢查是否健康
        is_healthy, issues = health.is_healthy(self.thresholds)
        
        if not is_healthy:
            for issue in issues:
                self.logger.warning(f"🏥 健康檢查警告: {issue}")
        
        return is_healthy, issues
    
    def start_monitoring(self):
        """開始後台監控"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info(f"🔍 開始系統健康監控 (間隔: {self.check_interval}秒)")
    
    def stop_monitoring(self):
        """停止後台監控"""
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("🔍 系統健康監控已停止")
    
    def _monitoring_loop(self):
        """監控循環"""
        consecutive_unhealthy = 0
        
        while self.is_monitoring:
            try:
                is_healthy, issues = self.check_health()
                
                if not is_healthy:
                    consecutive_unhealthy += 1
                    
                    # 連續不健康達到閾值時發出警報
                    if consecutive_unhealthy >= 3:
                        self.alert_queue.put({
                            'type': 'system_unhealthy',
                            'issues': issues,
                            'consecutive_count': consecutive_unhealthy
                        })
                else:
                    consecutive_unhealthy = 0
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"監控循環錯誤: {e}")
                time.sleep(self.check_interval)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """獲取健康狀態摘要"""
        if not self.health_history:
            return {}
        
        latest = self.health_history[-1]
        
        # 計算平均值 (最近10次記錄)
        recent = self.health_history[-10:]
        avg_cpu = sum(h.cpu_usage for h in recent) / len(recent)
        avg_memory = sum(h.memory_usage for h in recent) / len(recent)
        avg_process_memory = sum(h.process_memory for h in recent) / len(recent)
        
        return {
            'timestamp': latest.timestamp,
            'current': {
                'cpu_usage': latest.cpu_usage,
                'memory_usage': latest.memory_usage,
                'gpu_usage': latest.gpu_usage,
                'gpu_memory': latest.gpu_memory,
                'disk_usage': latest.disk_usage,
                'process_memory': latest.process_memory
            },
            'average_10': {
                'cpu_usage': avg_cpu,
                'memory_usage': avg_memory,
                'process_memory': avg_process_memory
            },
            'system_info': {
                'cpu_count': self.cpu_count,
                'memory_total': self.memory_total,
                'gpu_count': len(self.gpus),
                'disk_total': self.disk_total
            }
        }
    
    def log_health_report(self):
        """記錄健康狀態報告"""
        summary = self.get_health_summary()
        
        if not summary:
            return
        
        current = summary['current']
        avg = summary['average_10']
        sys_info = summary['system_info']
        
        self.logger.info("🏥 === 系統健康報告 ===")
        self.logger.info(f"   CPU: {current['cpu_usage']:.1f}% (平均: {avg['cpu_usage']:.1f}%)")
        self.logger.info(f"   記憶體: {current['memory_usage']:.1f}% ({sys_info['memory_total']:.1f}GB 總計)")
        self.logger.info(f"   進程記憶體: {current['process_memory']:.1f}MB (平均: {avg['process_memory']:.1f}MB)")
        
        if current['gpu_usage']:
            for i, (usage, memory) in enumerate(zip(current['gpu_usage'], current['gpu_memory'])):
                self.logger.info(f"   GPU {i}: {usage:.1f}% 使用率, {memory:.1f}% 記憶體")
        
        self.logger.info(f"   磁碟: {current['disk_usage']:.1f}% ({sys_info['disk_total']:.1f}GB 總計)")
        self.logger.info("=" * 30)
    
    def emergency_cleanup(self):
        """緊急清理 - 在系統資源緊張時執行"""
        self.logger.warning("🚨 執行緊急系統清理...")
        
        try:
            # 清理GPU記憶體
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("   GPU記憶體快取已清理")
            
            # 強制垃圾回收
            import gc
            gc.collect()
            self.logger.info("   Python垃圾回收已執行")
            
            # 清理臨時文件 (可選)
            # self._cleanup_temp_files()
            
        except Exception as e:
            self.logger.error(f"緊急清理失敗: {e}")


class TrainingHealthMonitor(HealthMonitor):
    """訓練專用健康監控器"""
    
    def __init__(self, logger: PINNLogger, **kwargs):
        super().__init__(logger, **kwargs)
        
        # 訓練特定的閾值
        self.thresholds.gpu_memory_warning = 95.0  # 訓練時GPU記憶體更嚴格
        self.thresholds.process_memory_warning = 12000.0  # 12GB進程記憶體限制
        
        self.training_start_time = None
        self.last_checkpoint_time = None
        
    def start_training_monitoring(self):
        """開始訓練監控"""
        self.training_start_time = time.time()
        self.start_monitoring()
        self.logger.info("🎯 訓練健康監控已啟動")
    
    def check_training_health(self, epoch: int, loss: float) -> bool:
        """檢查訓練健康狀態"""
        is_healthy, issues = self.check_health()
        
        # 訓練特定檢查
        if self.training_start_time:
            training_time = time.time() - self.training_start_time
            
            # 檢查是否需要檢查點
            if self.last_checkpoint_time is None:
                self.last_checkpoint_time = time.time()
            elif time.time() - self.last_checkpoint_time > 3600:  # 1小時
                self.logger.warning("⏰ 建議保存檢查點 (距離上次保存超過1小時)")
        
        # 檢查損失是否異常
        if loss > 1e10:
            issues.append(f"損失值異常過大: {loss:.2e}")
            is_healthy = False
        
        return is_healthy