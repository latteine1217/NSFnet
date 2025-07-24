"""
記憶體管理優化 - 智能垃圾回收和緩存管理
"""
import gc
import torch
import psutil
import threading
import time
import weakref
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from logger import PINNLogger


@dataclass
class MemoryStats:
    """記憶體統計信息"""
    timestamp: float
    cpu_memory_used: float          # CPU記憶體使用量 (MB)
    cpu_memory_percent: float       # CPU記憶體使用率 (%)
    gpu_memory_allocated: float     # GPU已分配記憶體 (MB) 
    gpu_memory_cached: float        # GPU快取記憶體 (MB)
    gpu_memory_percent: float       # GPU記憶體使用率 (%)
    process_memory: float           # 當前進程記憶體 (MB)
    python_objects: int             # Python對象數量


class MemoryManager:
    """智能記憶體管理器"""
    
    def __init__(self, 
                 logger: PINNLogger,
                 cpu_threshold: float = 85.0,        # CPU記憶體清理閾值 (%)
                 gpu_threshold: float = 90.0,        # GPU記憶體清理閾值 (%)
                 auto_cleanup_interval: float = 300.0,  # 自動清理間隔 (秒)
                 aggressive_cleanup: bool = False):   # 激進清理模式
        
        self.logger = logger
        self.cpu_threshold = cpu_threshold
        self.gpu_threshold = gpu_threshold
        self.auto_cleanup_interval = auto_cleanup_interval
        self.aggressive_cleanup = aggressive_cleanup
        
        # 記憶體統計歷史
        self.memory_history: List[MemoryStats] = []
        self.max_history_size = 100
        
        # 自動清理線程
        self.auto_cleanup_enabled = False
        self.cleanup_thread = None
        
        # 緩存管理
        self.tensor_cache: Dict[str, torch.Tensor] = {}
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        
        # 記憶體洩漏檢測
        self.object_tracker: Dict[type, int] = {}
        self.weak_refs: List[weakref.ref] = []
        
        self.logger.info("🧠 記憶體管理器初始化完成")
    
    def get_memory_stats(self) -> MemoryStats:
        """獲取當前記憶體統計"""
        # CPU記憶體
        memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info().rss / (1024**2)  # MB
        
        # GPU記憶體
        gpu_allocated = 0.0
        gpu_cached = 0.0
        gpu_percent = 0.0
        
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
            gpu_cached = torch.cuda.memory_reserved() / (1024**2)  # MB
            
            # 計算GPU記憶體使用率
            props = torch.cuda.get_device_properties(0)
            total_memory = props.total_memory / (1024**2)  # MB
            gpu_percent = (gpu_allocated / total_memory) * 100
        
        # Python對象數量
        python_objects = len(gc.get_objects())
        
        stats = MemoryStats(
            timestamp=time.time(),
            cpu_memory_used=memory.used / (1024**2),  # MB
            cpu_memory_percent=memory.percent,
            gpu_memory_allocated=gpu_allocated,
            gpu_memory_cached=gpu_cached,
            gpu_memory_percent=gpu_percent,
            process_memory=process_memory,
            python_objects=python_objects
        )
        
        # 添加到歷史記錄
        self.memory_history.append(stats)
        if len(self.memory_history) > self.max_history_size:
            self.memory_history.pop(0)
        
        return stats
    
    def needs_cleanup(self) -> bool:
        """檢查是否需要進行記憶體清理"""
        stats = self.get_memory_stats()
        
        needs_cpu_cleanup = stats.cpu_memory_percent > self.cpu_threshold
        needs_gpu_cleanup = stats.gpu_memory_percent > self.gpu_threshold
        
        return needs_cpu_cleanup or needs_gpu_cleanup
    
    def cleanup_memory(self, force: bool = False) -> Dict[str, Any]:
        """執行記憶體清理"""
        cleanup_results = {
            'before': self.get_memory_stats(),
            'actions_taken': [],
            'errors': []
        }
        
        try:
            # 1. 清理Python垃圾回收
            collected = gc.collect()
            if collected > 0:
                cleanup_results['actions_taken'].append(f"Python GC: {collected} objects collected")
                self.logger.debug(f"Python垃圾回收: {collected} 個對象被清理")
            
            # 2. 清理GPU記憶體
            if torch.cuda.is_available():
                before_gpu = torch.cuda.memory_allocated() / (1024**2)
                torch.cuda.empty_cache()
                after_gpu = torch.cuda.memory_allocated() / (1024**2)
                
                freed_gpu = before_gpu - after_gpu
                if freed_gpu > 0:
                    cleanup_results['actions_taken'].append(f"GPU cache cleared: {freed_gpu:.1f}MB freed")
                    self.logger.debug(f"GPU快取清理: {freed_gpu:.1f}MB 已釋放")
            
            # 3. 清理tensor緩存
            if self.tensor_cache:
                cache_memory = sum(t.numel() * t.element_size() for t in self.tensor_cache.values()) / (1024**2)
                self.tensor_cache.clear()
                cleanup_results['actions_taken'].append(f"Tensor cache cleared: {cache_memory:.1f}MB")
                self.logger.debug(f"Tensor緩存清理: {cache_memory:.1f}MB")
            
            # 4. 激進清理模式
            if self.aggressive_cleanup or force:
                # 強制清理所有弱引用
                dead_refs = []
                for ref in self.weak_refs:
                    if ref() is None:
                        dead_refs.append(ref)
                
                for ref in dead_refs:
                    self.weak_refs.remove(ref)
                
                if dead_refs:
                    cleanup_results['actions_taken'].append(f"Weak references cleaned: {len(dead_refs)}")
                
                # 再次強制垃圾回收
                collected_aggressive = gc.collect()
                if collected_aggressive > 0:
                    cleanup_results['actions_taken'].append(f"Aggressive GC: {collected_aggressive} additional objects")
            
        except Exception as e:
            error_msg = f"記憶體清理錯誤: {e}"
            cleanup_results['errors'].append(error_msg)
            self.logger.error(error_msg)
        
        cleanup_results['after'] = self.get_memory_stats()
        
        # 記錄清理結果
        if cleanup_results['actions_taken']:
            self.logger.info(f"🧹 記憶體清理完成: {', '.join(cleanup_results['actions_taken'])}")
        
        return cleanup_results
    
    def start_auto_cleanup(self):
        """啟動自動記憶體清理"""
        if self.auto_cleanup_enabled:
            return
        
        self.auto_cleanup_enabled = True
        self.cleanup_thread = threading.Thread(target=self._auto_cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        self.logger.info(f"🔄 自動記憶體清理已啟動 (間隔: {self.auto_cleanup_interval}秒)")
    
    def stop_auto_cleanup(self):
        """停止自動記憶體清理"""
        self.auto_cleanup_enabled = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5.0)
        self.logger.info("🔄 自動記憶體清理已停止")
    
    def _auto_cleanup_loop(self):
        """自動清理循環"""
        while self.auto_cleanup_enabled:
            try:
                if self.needs_cleanup():
                    self.cleanup_memory()
                
                time.sleep(self.auto_cleanup_interval)
                
            except Exception as e:
                self.logger.error(f"自動清理循環錯誤: {e}")
                time.sleep(self.auto_cleanup_interval)
    
    def cache_tensor(self, key: str, tensor: torch.Tensor) -> bool:
        """緩存tensor"""
        try:
            # 檢查緩存大小限制
            cache_size = len(self.tensor_cache)
            if cache_size >= 1000:  # 限制緩存數量
                # 清理最舊的緩存項
                oldest_key = next(iter(self.tensor_cache))
                del self.tensor_cache[oldest_key]
            
            self.tensor_cache[key] = tensor.clone().detach()
            return True
            
        except Exception as e:
            self.logger.warning(f"緩存tensor失敗: {e}")
            return False
    
    def get_cached_tensor(self, key: str) -> Optional[torch.Tensor]:
        """獲取緩存的tensor"""
        if key in self.tensor_cache:
            self.cache_hit_count += 1
            return self.tensor_cache[key].clone()
        else:
            self.cache_miss_count += 1
            return None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """獲取緩存統計"""
        total_requests = self.cache_hit_count + self.cache_miss_count
        hit_rate = (self.cache_hit_count / total_requests * 100) if total_requests > 0 else 0
        
        cache_memory = 0
        if self.tensor_cache:
            cache_memory = sum(t.numel() * t.element_size() for t in self.tensor_cache.values()) / (1024**2)
        
        return {
            'cache_size': len(self.tensor_cache),
            'cache_memory_mb': cache_memory,
            'hit_count': self.cache_hit_count,
            'miss_count': self.cache_miss_count,
            'hit_rate_percent': hit_rate
        }
    
    def detect_memory_leaks(self) -> Dict[str, Any]:
        """檢測記憶體洩漏"""
        current_objects = {}
        
        # 統計各類型對象數量
        for obj in gc.get_objects():
            obj_type = type(obj)
            current_objects[obj_type] = current_objects.get(obj_type, 0) + 1
        
        # 檢測對象數量增長
        leaks = {}
        for obj_type, count in current_objects.items():
            if obj_type in self.object_tracker:
                growth = count - self.object_tracker[obj_type]
                if growth > 100:  # 閾值：增長超過100個對象
                    leaks[obj_type.__name__] = {
                        'previous': self.object_tracker[obj_type],
                        'current': count,
                        'growth': growth
                    }
        
        # 更新追蹤記錄
        self.object_tracker = current_objects
        
        if leaks:
            self.logger.warning(f"🔍 檢測到潛在記憶體洩漏:")
            for obj_name, info in leaks.items():
                self.logger.warning(f"   {obj_name}: {info['previous']} -> {info['current']} (+{info['growth']})")
        
        return leaks
    
    def log_memory_report(self):
        """記錄記憶體使用報告"""
        stats = self.get_memory_stats()
        cache_stats = self.get_cache_stats()
        
        self.logger.info("🧠 === 記憶體使用報告 ===")
        self.logger.info(f"   CPU記憶體: {stats.cpu_memory_percent:.1f}% ({stats.cpu_memory_used:.0f}MB)")
        self.logger.info(f"   進程記憶體: {stats.process_memory:.0f}MB")
        
        if torch.cuda.is_available():
            self.logger.info(f"   GPU記憶體: {stats.gpu_memory_percent:.1f}% ({stats.gpu_memory_allocated:.0f}MB 已分配)")
            self.logger.info(f"   GPU快取: {stats.gpu_memory_cached:.0f}MB")
        
        self.logger.info(f"   Python對象: {stats.python_objects:,}")
        self.logger.info(f"   Tensor緩存: {cache_stats['cache_size']} 項 ({cache_stats['cache_memory_mb']:.1f}MB)")
        self.logger.info(f"   緩存命中率: {cache_stats['hit_rate_percent']:.1f}%")
        self.logger.info("=" * 30)
    
    def optimize_for_training(self):
        """為訓練優化記憶體設置"""
        self.logger.info("🎯 啟動訓練記憶體優化...")
        
        # 設置更嚴格的閾值
        self.cpu_threshold = 80.0
        self.gpu_threshold = 85.0
        
        # 啟用激進清理
        self.aggressive_cleanup = True
        
        # 減少自動清理間隔
        self.auto_cleanup_interval = 180.0  # 3分鐘
        
        # 清理並開始監控
        self.cleanup_memory(force=True)
        self.start_auto_cleanup()
        
        self.logger.info("   記憶體優化設置已應用")


class TrainingMemoryManager(MemoryManager):
    """訓練專用記憶體管理器"""
    
    def __init__(self, logger: PINNLogger, **kwargs):
        super().__init__(logger, **kwargs)
        
        # 訓練專用設置
        self.cpu_threshold = 80.0
        self.gpu_threshold = 85.0
        self.auto_cleanup_interval = 300.0  # 5分鐘
        self.aggressive_cleanup = True
        
        # 訓練狀態追蹤
        self.peak_memory = 0.0
        self.cleanup_count = 0
        
    def monitor_training_memory(self, epoch: int, loss: float) -> bool:
        """監控訓練時的記憶體使用"""
        stats = self.get_memory_stats()
        
        # 追蹤記憶體峰值
        current_memory = stats.gpu_memory_allocated if torch.cuda.is_available() else stats.process_memory
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
        
        # 檢查是否需要清理
        needs_cleanup = self.needs_cleanup()
        
        if needs_cleanup:
            self.logger.warning(f"⚡ Epoch {epoch}: 記憶體使用過高，執行清理...")
            cleanup_results = self.cleanup_memory()
            self.cleanup_count += 1
            
            # 如果清理後仍然過高，警告用戶
            after_stats = cleanup_results['after']
            if after_stats.gpu_memory_percent > 95.0 or after_stats.cpu_memory_percent > 95.0:
                self.logger.error("🚨 記憶體清理後仍然過高，建議減少批次大小或模型複雜度")
                return False
        
        return True
    
    def get_training_memory_summary(self) -> Dict[str, Any]:
        """獲取訓練記憶體摘要"""
        current_stats = self.get_memory_stats()
        cache_stats = self.get_cache_stats()
        
        return {
            'current_memory_percent': current_stats.gpu_memory_percent if torch.cuda.is_available() else current_stats.cpu_memory_percent,
            'peak_memory_mb': self.peak_memory,
            'cleanup_count': self.cleanup_count,
            'cache_hit_rate': cache_stats['hit_rate_percent'],
            'total_objects': current_stats.python_objects
        }