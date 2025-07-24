"""
統一日誌系統 - PINN專案日誌管理
"""
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import torch.distributed as dist


class ColoredFormatter(logging.Formatter):
    """帶顏色的日誌格式化器"""
    
    # 顏色定義
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 綠色 
        'WARNING': '\033[33m',    # 黃色
        'ERROR': '\033[31m',      # 紅色
        'CRITICAL': '\033[35m',   # 紫色
        'RESET': '\033[0m'        # 重置顏色
    }
    
    def format(self, record):
        # 添加顏色
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        
        # 添加emoji
        emoji_map = {
            'DEBUG': '🔍',
            'INFO': '💡', 
            'WARNING': '⚠️',
            'ERROR': '🚨',
            'CRITICAL': '💀'
        }
        record.emoji = emoji_map.get(record.levelname.strip('\033[0m\033[32m\033[33m\033[31m\033[35m\033[36m'), '📝')
        
        return super().format(record)


class DistributedFilter(logging.Filter):
    """分布式訓練日誌過濾器 - 只顯示rank 0的日誌"""
    
    def filter(self, record):
        # 在分布式模式下，只讓rank 0的進程輸出日誌
        if dist.is_initialized():
            return dist.get_rank() == 0
        return True


class PINNLogger:
    """PINN專用日誌器"""
    
    def __init__(self, 
                 name: str = "PINN",
                 level: str = "INFO",
                 log_dir: str = "logs",
                 enable_file_logging: bool = True,
                 enable_console_logging: bool = True,
                 rank: int = 0):
        
        self.name = name
        self.rank = rank
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # 清除已有的handlers
        self.logger.handlers.clear()
        
        # 創建日誌目錄
        if enable_file_logging:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            
            # 創建文件handler
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_path / f"{name}_{timestamp}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)  # 文件記錄所有等級
            
            # 文件格式 (無顏色)
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            
            # 添加分布式過濾器
            file_handler.addFilter(DistributedFilter())
            self.logger.addHandler(file_handler)
        
        # 創建控制台handler
        if enable_console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, level.upper()))
            
            # 控制台格式 (帶顏色和emoji)
            console_formatter = ColoredFormatter(
                '%(emoji)s %(asctime)s | %(levelname)s | %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            
            # 添加分布式過濾器
            console_handler.addFilter(DistributedFilter())
            self.logger.addHandler(console_handler)
    
    def debug(self, msg: str, **kwargs):
        """調試信息"""
        self.logger.debug(msg, **kwargs)
    
    def info(self, msg: str, **kwargs):
        """一般信息"""
        self.logger.info(msg, **kwargs)
    
    def warning(self, msg: str, **kwargs):
        """警告信息"""
        self.logger.warning(msg, **kwargs)
    
    def error(self, msg: str, **kwargs):
        """錯誤信息"""
        self.logger.error(msg, **kwargs)
    
    def critical(self, msg: str, **kwargs):
        """嚴重錯誤"""
        self.logger.critical(msg, **kwargs)
    
    def training_start(self, config: Dict[str, Any]):
        """訓練開始日誌"""
        self.info("=" * 60)
        self.info(f"🚀 開始訓練: {config.get('experiment_name', 'Unknown')}")
        self.info(f"📋 Reynolds數: {config.get('Re', 'Unknown')}")
        self.info(f"🔢 訓練點數: {config.get('N_f', 'Unknown'):,}")
        self.info(f"🧠 網路架構: {config.get('layers', 'Unknown')}層 × {config.get('hidden_size', 'Unknown')}神經元")
        self.info("=" * 60)
    
    def training_stage(self, stage_name: str, alpha: float, epochs: int, lr: float):
        """訓練階段日誌"""
        self.info(f"🎯 {stage_name}: α_EVM={alpha}, epochs={epochs:,}, lr={lr:.1e}")
    
    def epoch_log(self, epoch: int, loss: float, lr: float, time_per_epoch: float, eta: str):
        """Epoch訓練日誌"""
        self.info(f"Epoch {epoch:6d} | Loss: {loss:.6e} | LR: {lr:.1e} | Time: {time_per_epoch:.3f}s | ETA: {eta}")
    
    def memory_warning(self, memory_gb: float, limit_gb: float):
        """記憶體警告"""
        self.warning(f"GPU記憶體使用量高: {memory_gb:.2f}GB / {limit_gb:.2f}GB")
    
    def checkpoint_saved(self, checkpoint_path: str, epoch: int):
        """檢查點保存日誌"""
        self.info(f"💾 檢查點已保存: {checkpoint_path} (epoch {epoch})")
    
    def evaluation_result(self, metrics: Dict[str, float]):
        """評估結果日誌"""
        self.info("📊 評估結果:")
        for metric, value in metrics.items():
            self.info(f"   {metric}: {value:.6e}")
    
    def training_complete(self, total_time: float, total_epochs: int):
        """訓練完成日誌"""
        self.info("🎉 ===== 訓練完成！=====")
        self.info(f"   總訓練時間: {self._format_time(total_time)}")
        self.info(f"   總 epochs: {total_epochs:,}")
        self.info(f"   平均每 epoch: {total_time/total_epochs:.3f}秒")
        self.info("=" * 40)
    
    def ddp_error(self, error_msg: str, attempt: int, max_attempts: int):
        """DDP錯誤日誌"""
        self.warning(f"DDP錯誤 (嘗試 {attempt}/{max_attempts}): {error_msg[:100]}...")
    
    def loss_validation_error(self, epoch: int, loss_value: float, loss_type: str = "main"):
        """損失驗證錯誤"""
        self.error(f"損失值異常 at epoch {epoch}: {loss_type} loss = {loss_value}")
    
    def system_info(self, info: Dict[str, Any]):
        """系統信息日誌"""
        self.info("💻 系統信息:")
        for key, value in info.items():
            self.info(f"   {key}: {value}")
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """格式化時間顯示"""
        if seconds < 60:
            return f"{seconds:.1f}秒"
        elif seconds < 3600:
            return f"{seconds//60:.0f}分 {seconds%60:.0f}秒"
        elif seconds < 86400:
            return f"{seconds//3600:.0f}小時 {(seconds%3600)//60:.0f}分"
        else:
            return f"{seconds//86400:.0f}天 {(seconds%86400)//3600:.0f}小時"


class LoggerFactory:
    """日誌工廠類"""
    
    _loggers: Dict[str, PINNLogger] = {}
    
    @classmethod
    def get_logger(cls, 
                   name: str = "PINN",
                   level: str = "INFO",
                   log_dir: str = "logs",
                   enable_file_logging: bool = True,
                   enable_console_logging: bool = True,
                   rank: int = 0) -> PINNLogger:
        """獲取或創建日誌器"""
        
        logger_key = f"{name}_{rank}"
        
        if logger_key not in cls._loggers:
            cls._loggers[logger_key] = PINNLogger(
                name=name,
                level=level,
                log_dir=log_dir,
                enable_file_logging=enable_file_logging,
                enable_console_logging=enable_console_logging,
                rank=rank
            )
        
        return cls._loggers[logger_key]
    
    @classmethod
    def configure_from_config(cls, config_manager) -> PINNLogger:
        """從配置管理器創建日誌器"""
        config = config_manager.config
        
        return cls.get_logger(
            name=config.experiment_name,
            level=config.system.log_level,
            log_dir="logs",
            enable_file_logging=True,
            enable_console_logging=True,
            rank=int(os.environ.get('RANK', 0))
        )


# 便捷函數
def get_pinn_logger(name: str = "PINN", level: str = "INFO") -> PINNLogger:
    """獲取PINN日誌器的便捷函數"""
    return LoggerFactory.get_logger(name=name, level=level)

# 預設日誌器
default_logger = get_pinn_logger()