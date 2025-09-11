"""
Minimal configuration system for restored 41c5563 baseline with modern output style.
只保留必要欄位：physics / network / training。無 supervision、無進階 scheduler。
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
    log_interval: int = 1000  # epoch interval for logging
    training_stages: List[TrainingStage] = field(default_factory=lambda: [
        TrainingStage(0.05, 500000, 1e-3,  "Stage 1"),
        TrainingStage(0.03, 500000, 2e-4,  "Stage 2"),
        TrainingStage(0.01, 500000, 4e-5,  "Stage 3"),
        TrainingStage(0.005,500000, 1e-5,  "Stage 4"),
        TrainingStage(0.002,500000, 2e-6,  "Stage 5"),
        TrainingStage(0.002,500000, 2e-6,  "Stage 6"),
    ])

@dataclass
class AppConfig:
    physics: PhysicsConfig = PhysicsConfig()
    network: NetworkConfig = NetworkConfig()
    training: TrainingConfig = TrainingConfig()
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
        print(f"  Stages={len(c.training.training_stages)}")
        for i, st in enumerate(c.training.training_stages, 1):
            print(f"    - {i}: {st.name} | alpha={st.alpha} | epochs={st.epochs:,} | lr={st.lr:.2e}")
        print("="*60)

    def validate_config(self):
        warnings = []
        if self.config.physics.Re <= 0:
            warnings.append("Re must be > 0")
        if self.config.training.N_f <= 0:
            warnings.append("N_f must be > 0")
        return warnings
