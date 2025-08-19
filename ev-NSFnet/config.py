"""
配置管理系統 - 統一管理PINN訓練參數
"""
import os
import json
import yaml
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
import torch

# 常量定義
RESULTS_PATH = "results"  # 結果輸出路徑

@dataclass
class NetworkConfig:
    """神經網路架構配置"""
    layers: int = 6                    # 主網路層數
    layers_1: int = 4                  # EVM網路層數
    hidden_size: int = 80              # 主網路神經元數
    hidden_size_1: int = 40            # EVM網路神經元數
    # 首/末層縮放（主網/EVM）與EVM輸出激活
    first_layer_scale_main: float = 2.0
    last_layer_scale_main: float = 0.5
    first_layer_scale_evm: float = 1.2
    last_layer_scale_evm: float = 0.1
    evm_output_activation: str = "softplus_cap"  # 可選：softplus_cap | abs_cap

@dataclass
class LBFGSConfig:
    """L-BFGS優化器配置"""
    enabled_in_distributed: bool = True    # 分佈式模式下啟用L-BFGS
    volatility_threshold: float = 0.01     # （舊）波動度閾值（保留向後相容）
    # 觸發條件（新）：分階段視窗與改善率門檻
    trigger_window_per_stage: Optional[List[int]] = None     # 例如 [5000, 7500, 10000]
    min_improve_pct_per_stage: Optional[List[float]] = None  # 例如 [0.02, 0.01, 0.005]
    ema_gamma: float = 0.95                        # 改善率分母的 EMA 平滑係數
    grad_median_abs_thresh: float = 1e-3           # 梯度絕對門檻（fp32友善）
    grad_relative_factor: float = 0.01             # 相對門檻：< factor × g_base
    grad_cos_ema_thresh: float = 0.9               # 梯度方向穩定閾值
    cooldown_steps: int = 5000                     # 兩段 L-BFGS 之間冷卻步數
    freeze_evm_during_lbfgs: bool = True           # 段內凍結 EVM
    # 段參數（建議值）：
    max_outer_steps: int = 200                     # 外循環步數上限
    timeout_seconds: int = 600                     # 執行超時(秒)
    max_iter: int = 25                             # L-BFGS內部迭代數
    history_size: int = 20                         # 歷史大小
    tolerance_grad: float = 1e-6                   # 梯度容差（fp32）
    tolerance_change: float = 1e-8                 # 變化容差
    line_search_fn: str = "strong_wolfe"           # 線搜索方法
    early_stop_patience: int = 8                   # 內迭代連續停滯次數
    early_stop_min_delta: float = 1e-4             # 內迭代最小改善
    checkpoint_before_lbfgs: bool = True           # L-BFGS前自動保存

@dataclass
class TrainingConfig:
    """訓練參數配置"""
    N_f: int = 120000                  # 方程點數量
    batch_size: Optional[int] = None   # 批次大小 (None = 全批次)
    checkpoint_freq: int = 5000        # 檢查點保存頻率
    log_tips: bool = True              # 訓練過程提示訊息
    sort_by_boundary_distance: bool = True  # 是否按距離邊界遠近排序方程點
    pde_distance_weighting: bool = True     # 是否啟用PDE距離權重 w(d)
    pde_distance_w_min: float = 0.2         # 權重下限，避免遠區權重為0
    pde_distance_tau: float = 0.1           # 指數權重尺度參數 tau
    
    # 訓練階段配置 (alpha_evm, epochs, learning_rate[, scheduler])
    training_stages: Optional[List[Tuple]] = None
    
    # L-BFGS配置
    lbfgs: Optional[LBFGSConfig] = None
    
    def __post_init__(self):
        if self.training_stages is None:
            # 默認6階段訓練配置
            self.training_stages = [
                (0.05, 350000, 1e-3),   # Stage 1
                (0.03, 350000, 2e-4),   # Stage 2  
                (0.01, 350000, 4e-5),   # Stage 3
                (0.005, 350000, 1e-5),   # Stage 4
                (0.002, 350000, 2e-6),   # Stage 5
                (0.002, 350000, 2e-6)   # Stage 6
            ]
        if self.lbfgs is None:
            self.lbfgs = LBFGSConfig()
        # 預設觸發視窗與門檻（若未設定）
        if self.lbfgs.trigger_window_per_stage is None:
            self.lbfgs.trigger_window_per_stage = [5000, 7500, 10000]
        if self.lbfgs.min_improve_pct_per_stage is None:
            self.lbfgs.min_improve_pct_per_stage = [0.02, 0.01, 0.005]

@dataclass
class PhysicsConfig:
    """物理參數配置"""
    Re: int = 5000                     # Reynolds number
    alpha_evm: float = 0.03            # 初始EVM係數
    beta: float = 1.0                  # 人工粘滯度上限係數
    bc_weight: float = 10.0            # 邊界條件權重
    eq_weight: float = 1.0             # 方程權重

@dataclass
class SupervisionConfig:
    """监督数据配置"""
    enabled: bool = True               # 是否启用监督数据
    data_points: int = 0               # 监督数据点数量，0表示不使用
    data_path: str = "data/cavity_Re5000_256_Uniform.mat"  # 数据文件路径
    weight: float = 1.0                # 监督数据权重
    random_seed: int = 42              # 随机种子，确保可重现性

@dataclass
class SystemConfig:
    """系統配置"""
    device: str = "auto"               # 設備選擇 (auto/cpu/cuda)
    precision: str = "float32"         # 精度設置
    tensorboard_enabled: bool = True   # TensorBoard啟用
    log_level: str = "INFO"            # 日誌等級
    memory_limit_gb: float = 14.0      # GPU記憶體限制(GB)
    
    # 性能優化配置
    gradient_clip_norm: float = 1.0    # 梯度裁剪
    memory_cleanup_freq: int = 100     # 記憶體清理頻率
    epoch_times_limit: int = 1000      # epoch時間記錄限制

@dataclass
class ExperimentConfig:
    """完整實驗配置"""
    experiment_name: str = "NSFnet_Re5000"
    description: str = "Physics-Informed Neural Network for Lid-Driven Cavity Flow"
    
    # 子配置
    network: Optional[NetworkConfig] = None
    training: Optional[TrainingConfig] = None  
    physics: Optional[PhysicsConfig] = None
    supervision: Optional[SupervisionConfig] = None
    system: Optional[SystemConfig] = None
    
    def __post_init__(self):
        # 初始化子配置
        if self.network is None:
            self.network = NetworkConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.physics is None:
            self.physics = PhysicsConfig()
        if self.supervision is None:
            self.supervision = SupervisionConfig()
        if self.system is None:
            self.system = SystemConfig()

class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self.config = ExperimentConfig()
        
    @classmethod
    def from_file(cls, config_path: str) -> 'ConfigManager':
        """從配置文件載入"""
        manager = cls()
        
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path}")
        
        manager.load_from_dict(config_dict)
        return manager
    
    def load_from_dict(self, config_dict: Dict[str, Any]):
        """從字典載入配置"""
        # 正規化 training_stages，支援 [alpha, epochs, lr, scheduler]
        if 'training' in config_dict and 'training_stages' in config_dict['training']:
            processed_stages = []
            for stage in config_dict['training']['training_stages']:
                alpha = float(stage[0])
                epochs = int(stage[1])
                lr = float(stage[2])
                sched = stage[3] if len(stage) > 3 else 'Constant'
                processed_stages.append((alpha, epochs, lr, str(sched)))
            config_dict['training']['training_stages'] = processed_stages
        
        # 更新各個子配置
        if 'network' in config_dict:
            self.config.network = NetworkConfig(**config_dict['network'])
        if 'training' in config_dict:
            training_config = config_dict['training'].copy()
            
            # 處理L-BFGS配置
            if 'lbfgs' in training_config:
                lbfgs_config = LBFGSConfig(**training_config['lbfgs'])
                training_config.pop('lbfgs')
                self.config.training = TrainingConfig(**training_config)
                self.config.training.lbfgs = lbfgs_config
            else:
                self.config.training = TrainingConfig(**training_config)
        if 'physics' in config_dict:
            self.config.physics = PhysicsConfig(**config_dict['physics'])
        if 'supervision' in config_dict:
            self.config.supervision = SupervisionConfig(**config_dict['supervision'])
        if 'system' in config_dict:
            self.config.system = SystemConfig(**config_dict['system'])
            
        # 更新主配置
        for key in ['experiment_name', 'description']:
            if key in config_dict:
                setattr(self.config, key, config_dict[key])
    
    def save_to_file(self, config_path: str):
        """保存配置到文件"""
        config_dict = asdict(self.config)
        
        # 將tuple轉換為list (YAML序列化兼容性)
        if 'training' in config_dict and 'training_stages' in config_dict['training']:
            config_dict['training']['training_stages'] = [
                list(stage) for stage in config_dict['training']['training_stages']
            ]
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
        elif config_path.endswith('.json'):
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported config file format: {config_path}")
    
    def get_device(self) -> torch.device:
        """獲取設備配置"""
        if self.config.system.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(self.config.system.device)
    
    def get_precision_dtype(self) -> torch.dtype:
        """獲取精度配置"""
        if self.config.system.precision == "float64":
            return torch.float64
        elif self.config.system.precision == "float16":
            return torch.float16
        else:
            return torch.float32
    
    def validate_config(self) -> List[str]:
        """驗證配置合法性"""
        warnings = []
        
        # 檢查網路配置
        if self.config.network.layers < 1:
            warnings.append("Network layers should be >= 1")
        if self.config.network.hidden_size < 1:
            warnings.append("Hidden size should be >= 1")
            
        # 檢查訓練配置  
        if self.config.training.N_f <= 0:
            warnings.append("N_f should be > 0")
        if self.config.training.checkpoint_freq <= 0:
            warnings.append("Checkpoint frequency should be > 0")
            
        # 檢查物理配置
        if self.config.physics.Re <= 0:
            warnings.append("Reynolds number should be > 0")
        if self.config.physics.alpha_evm <= 0:
            warnings.append("Alpha EVM should be > 0")
            
        return warnings
    
    def print_config(self):
        """打印配置摘要"""
        print("=" * 60)
        print(f"🔧 實驗配置: {self.config.experiment_name}")
        print(f"📝 描述: {self.config.description}")
        print("=" * 60)
        
        print(f"🧠 網路架構:")
        print(f"   主網路: {self.config.network.layers} 層 × {self.config.network.hidden_size} 神經元")
        print(f"   EVM網路: {self.config.network.layers_1} 層 × {self.config.network.hidden_size_1} 神經元")
        
        print(f"🎯 訓練設定:")
        print(f"   方程點數: {self.config.training.N_f:,}")
        print(f"   批次大小: {'全批次' if self.config.training.batch_size is None else self.config.training.batch_size}")
        print(f"   依距離排序: {'是' if self.config.training.sort_by_boundary_distance else '否'}")
        print(f"   PDE距離權重: {'啟用' if self.config.training.pde_distance_weighting else '關閉'} (w_min={self.config.training.pde_distance_w_min}, tau={self.config.training.pde_distance_tau})")
        print(f"   總階段數: {len(self.config.training.training_stages)}")
        for i, st in enumerate(self.config.training.training_stages):
            try:
                a,e,l,s = st
            except Exception:
                a,e,l = st[:3]
                s = 'Constant'
            print(f"   - Stage {i+1}: alpha={a}, epochs={e}, lr={l}, sched={s}")
        
        print(f"⚡ 物理參數:")
        print(f"   Reynolds數: {self.config.physics.Re}")
        print(f"   初始α_EVM: {self.config.physics.alpha_evm}")
        
        print(f"💻 系統配置:")
        print(f"   設備: {self.config.system.device}")
        print(f"   精度: {self.config.system.precision}")
        print(f"   TensorBoard: {'啟用' if self.config.system.tensorboard_enabled else '關閉'}")
        print("=" * 60)

# 預設配置實例
default_config = ConfigManager()

# 高性能配置 (適用於服務器)
def get_server_config() -> ConfigManager:
    """獲取服務器高性能配置"""
    config = ConfigManager()
    
    # 調整為高性能設置
    config.config.training.N_f = 120000
    config.config.system.memory_limit_gb = 14.0
    config.config.system.tensorboard_enabled = True
    
    return config

# 測試配置 (適用於快速測試)
def get_test_config() -> ConfigManager:
    """獲取測試配置"""
    config = ConfigManager()
    
    # 調整為快速測試設置
    config.config.training.N_f = 1000
    config.config.training.training_stages = [
        (0.05, 10, 1e-3),
        (0.03, 10, 5e-4)
    ]
    config.config.system.memory_limit_gb = 2.0
    
    return config
