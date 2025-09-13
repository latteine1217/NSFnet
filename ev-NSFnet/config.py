"""
Minimal configuration system for restored 41c5563 baseline with modern output style.
保留必要欄位：physics / network / training，並加入 supervision（可選）。
"""
from dataclasses import dataclass, field
from typing import List, Tuple
import yaml

@dataclass
class PhysicsConfig:
    Re: int = 5000
    alpha_evm: float = 0.05  # initial stage alpha (will be overridden by stages)
    bc_weight: float = 10.0
    eq_weight: float = 1.0

@dataclass
class NetworkConfig:
    layers: int = 6
    layers_1: int = 4
    hidden_size: int = 80
    hidden_size_1: int = 40

@dataclass
class TrainingStage:
    alpha: float
    epochs: int
    lr: float
    name: str

@dataclass
class TrainingConfig:
    N_f: int = 120000
    sort_by_boundary_distance: bool = True  # 是否按距離邊界遠近排序方程點
    normalize_coordinates: bool = False     # 是否將 (x,y) 映射到 [-1, 1]
    # PDE 距離權重配置（參考 ev-NSFnet copy）
    pde_distance_weighting: bool = False   # 啟用PDE距離權重 w(d)
    pde_distance_w_min: float = 0.8        # 權重下限，避免遠區為0
    pde_distance_tau: float = 0.2          # 距離尺度 (越小越聚焦邊界)
    log_interval: int = 1000  # epoch interval for logging
    enable_tensorboard: bool = True  # toggle TensorBoard logging
    tb_log_dir: str = 'runs'  # base directory for tensorboard logs
    training_stages: List[TrainingStage] = field(default_factory=lambda: [
        TrainingStage(0.05, 500000, 1e-3,  "Stage 1"),
        TrainingStage(0.03, 500000, 2e-4,  "Stage 2"),
        TrainingStage(0.01, 500000, 4e-5,  "Stage 3"),
        TrainingStage(0.005,500000, 1e-5,  "Stage 4"),
        TrainingStage(0.002,500000, 2e-6,  "Stage 5"),
        TrainingStage(0.002,500000, 2e-6,  "Stage 6"),
    ])

@dataclass
class SupervisionConfig:
    enabled: bool = False              # 是否啟用監督數據
    data_points: int = 0               # 監督點數量（0 表示不使用）
    data_path: str = 'data/cavity_Re5000_256_Uniform.mat'  # 監督數據來源 .mat
    weight: float = 2.0                # 監督損失整體權重（傳遞至 solver）
    random_seed: int = 42              # 隨機種子（固定抽樣）

@dataclass
class AppConfig:
    physics: PhysicsConfig = PhysicsConfig()
    network: NetworkConfig = NetworkConfig()
    training: TrainingConfig = TrainingConfig()
    supervision: SupervisionConfig = SupervisionConfig()
    experiment_name: str = "NSFnet_Restore"
    description: str = "Restored baseline with modern logging"

class ConfigManager:
    def __init__(self, config: AppConfig | None = None):
        self.config = config or AppConfig()

    @classmethod
    def from_file(cls, path: str) -> 'ConfigManager':
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        cfg = AppConfig()
        # physics
        if 'physics' in data:
            for k,v in data['physics'].items():
                if hasattr(cfg.physics, k):
                    setattr(cfg.physics, k, v)
        # network
        if 'network' in data:
            for k,v in data['network'].items():
                if hasattr(cfg.network, k):
                    setattr(cfg.network, k, v)
        # training stages
        if 'training' in data:
            tr = data['training']
            if 'N_f' in tr:
                cfg.training.N_f = int(tr['N_f'])
            if 'sort_by_boundary_distance' in tr:
                cfg.training.sort_by_boundary_distance = bool(tr['sort_by_boundary_distance'])
            if 'normalize_coordinates' in tr:
                cfg.training.normalize_coordinates = bool(tr['normalize_coordinates'])
            if 'pde_distance_weighting' in tr:
                cfg.training.pde_distance_weighting = bool(tr['pde_distance_weighting'])
            if 'pde_distance_w_min' in tr:
                cfg.training.pde_distance_w_min = float(tr['pde_distance_w_min'])
            if 'pde_distance_tau' in tr:
                cfg.training.pde_distance_tau = float(tr['pde_distance_tau'])
            if 'log_interval' in tr:
                cfg.training.log_interval = int(tr['log_interval'])
            if 'enable_tensorboard' in tr:
                cfg.training.enable_tensorboard = bool(tr['enable_tensorboard'])
            if 'tb_log_dir' in tr:
                cfg.training.tb_log_dir = str(tr['tb_log_dir'])
            if 'training_stages' in tr:
                stages = []
                for st in tr['training_stages']:
                    # support dict or list
                    if isinstance(st, dict):
                        stages.append(TrainingStage(
                            alpha=float(st['alpha']),
                            epochs=int(st['epochs']),
                            lr=float(st['lr']),
                            name=str(st.get('name','Stage'))
                        ))
                    elif isinstance(st, (list, tuple)) and len(st) >= 4:
                        stages.append(TrainingStage(float(st[0]), int(st[1]), float(st[2]), str(st[3])))
                if stages:
                    cfg.training.training_stages = stages
        # supervision (optional)
        if 'supervision' in data:
            sup = data['supervision'] or {}
            if isinstance(sup, dict):
                for k, v in sup.items():
                    if hasattr(cfg.supervision, k):
                        setattr(cfg.supervision, k, v)
        # meta
        if 'experiment_name' in data:
            cfg.experiment_name = str(data['experiment_name'])
        if 'description' in data:
            cfg.description = str(data['description'])
        return cls(cfg)

    def print_config(self):
        c = self.config
        print("="*60)
        print(f"🔧 Experiment: {c.experiment_name}")
        print(f"📝 Description: {c.description}")
        print("="*60)
        print("🧠 Network:")
        print(f"  Main: {c.network.layers} layers × {c.network.hidden_size} neurons")
        print(f"  EVM : {c.network.layers_1} layers × {c.network.hidden_size_1} neurons")
        print("⚡ Physics:")
        print(f"  Re={c.physics.Re}, bc_weight={c.physics.bc_weight}, eq_weight={c.physics.eq_weight}")
        print("🎯 Training:")
        print(f"  N_f={c.training.N_f:,}")
        print(f"  sort_by_boundary_distance={c.training.sort_by_boundary_distance}")
        print(f"  normalize_coordinates={c.training.normalize_coordinates}")
        print(f"  PDE distance weighting: {c.training.pde_distance_weighting} (w_min={c.training.pde_distance_w_min}, tau={c.training.pde_distance_tau})")
        print(f"  Stages={len(c.training.training_stages)}")
        for i, st in enumerate(c.training.training_stages, 1):
            print(f"    - {i}: {st.name} | alpha={st.alpha} | epochs={st.epochs:,} | lr={st.lr:.2e}")
        print(f"  TensorBoard={'ON' if c.training.enable_tensorboard else 'OFF'} dir={c.training.tb_log_dir}")
        print("🧪 Supervision:")
        print(f"  enabled={c.supervision.enabled} data_points={c.supervision.data_points} weight={c.supervision.weight}")
        print(f"  data_path={c.supervision.data_path} seed={c.supervision.random_seed}")
        print("="*60)

    def validate_config(self):
        warnings = []
        if self.config.physics.Re <= 0:
            warnings.append("Re must be > 0")
        if self.config.training.N_f <= 0:
            warnings.append("N_f must be > 0")
        return warnings
