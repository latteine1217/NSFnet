# Copyright (c) 2023 scien42.tech, Se42 Authors. All Rights Reserved.
#
        
        
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Author: Zhicheng Wang, Hui Xiang
# Created: 08.03.2023
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import scipy.io
import numpy as np
import math
from net import FCNet
from laaf import LAAFScalar, compute_laaf_regularization
from tools import setup_device, get_cuda_info
from typing import Dict, List, Set, Optional, Union, Callable, Any, Tuple
import warnings
import time
import datetime
from logger import LoggerFactory, PINNLogger
from torch.nn import Module
from torch import Tensor
# from health_monitor import TrainingHealthMonitor, HealthThresholds
# from memory_manager import TrainingMemoryManager



# 抑制 PyTorch 分散式訓練的 autograd 警告
warnings.filterwarnings("ignore", message=".*c10d::allreduce_.*autograd kernel.*")

class PysicsInformedNeuralNetwork:
    # 類型註解
    net: Union[FCNet, DDP]
    net_1: Union[FCNet, DDP]
    device: torch.device
    logger: PINNLogger
    tb_writer: Optional[SummaryWriter]
    x_f: Optional[Tensor]
    y_f: Optional[Tensor]
    x_b: Optional[Tensor]
    y_b: Optional[Tensor]
    x_i: Optional[Tensor]
    y_i: Optional[Tensor]
    x_o: Optional[Tensor]
    y_o: Optional[Tensor]
    opt: Optional[torch.optim.Optimizer]
    opt_1: Optional[torch.optim.Optimizer]
    
    def _param_name_map(self, model: Union[Module, DDP]) -> Dict[int, str]:
        return {id(p): n for n, p in model.named_parameters()}

    def _safe_optimizer_state_dict(self, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        try:
            state = optimizer.state
            param_ids = set(id(p) for g in optimizer.param_groups for p in g.get('params', []))
            safe_state = {}
            for k, v in state.items():
                if k in param_ids:
                    safe_state[k] = v
            pg = []
            for g in optimizer.param_groups:
                pg.append({k: v for k, v in g.items() if k != 'params'})
            return {'state': safe_state, 'param_groups': pg}
        except Exception:
            return {}

    def _load_optimizer_state_dict_safe(self, optimizer: torch.optim.Optimizer, opt_state: Optional[Dict[str, Any]]) -> None:
        try:
            if not opt_state:
                return
            optimizer.load_state_dict(opt_state)
            for state in optimizer.state.values():
                for k, v in list(state.items()):
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
        except Exception:
            pass
    # Initialize the class
    def __init__(self,
                 opt=None,
                 Re = 1000,
                 layers=6,
                 layers_1=4,
                 hidden_size=80,
                 hidden_size_1=40,
                 N_f = 100000,
                 batch_size = None,
                 alpha_evm=0.03,
                 learning_rate=0.001,
                 outlet_weight=1,
                 bc_weight=10,
                 eq_weight=1,
                 ic_weight=0.1,
                 num_ins=2,
                 num_outs=3,
                 num_outs_1=1,
                 supervised_data_weight=1,
                 supervision_data_points=0,
                 supervision_data_path=None,
                 supervision_random_seed=42,
                 net_params=None,
                 net_params_1=None,
                 checkpoint_freq=2000,
                 config=None):
        # Initialize distributed training identifiers first
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))

        # Initialize logger ASAP to avoid use-before-init warnings
        self.logger = LoggerFactory.get_logger(
            name=f"PINN_Re{Re}",
            level="INFO",
            rank=self.rank
        )

        # 使用統一設備管理函數
        self.device = setup_device(self.local_rank, self.logger)

        self.evm = None
        self.Re = Re
        self.vis_t0 = 20.0/self.Re
        self.beta = None

        self.layers = layers
        self.layers_1 = layers_1
        self.hidden_size = hidden_size
        self.hidden_size_1 = hidden_size_1
        self.N_f = N_f
        self.batch_size = batch_size if batch_size is not None else N_f
        self.current_stage = ' '

        self.checkpoint_freq = checkpoint_freq
        # 提示控制
        self._tips_last_step = {}
        self.prev_strategy_step = -10**9

        # TensorBoard設定（支援頻率控制與可關閉）
        sys_cfg = getattr(self, 'config', None)
        sys_cfg = getattr(sys_cfg, 'system', None) if sys_cfg is not None else None
        tb_enabled = bool(getattr(sys_cfg, 'tensorboard_enabled', True)) if sys_cfg is not None else True
        self.tb_interval = int(getattr(sys_cfg, 'tensorboard_interval', 1000)) if sys_cfg is not None else 1000
        if self.rank == 0 and tb_enabled:
            log_dir = f"runs/NSFnet_Re{Re}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.tb_writer = SummaryWriter(log_dir=log_dir)
            self.logger.info(f"📊 TensorBoard log directory: {log_dir} (interval={self.tb_interval})")
        else:
            self.tb_writer = None

        # 時間追蹤相關變數
        self.epoch_start_time = None
        self.epoch_times = []
        self.stage_start_time = None
        self.training_start_time = None
        self.global_step_offset = 0  # 用於計算跨階段的global step

        # 健康監控系統
        self.health_monitor = None
        self.memory_manager = None

        self.alpha_evm = alpha_evm
        self.alpha_b = bc_weight
        self.alpha_e = eq_weight
        self.alpha_i = ic_weight
        self.alpha_o = outlet_weight
        self.alpha_s = supervised_data_weight
        self.loss_i = self.loss_o = self.loss_b = self.loss_e = self.loss_s = 0.0

        # 监督数据参数
        self.supervision_data_points = supervision_data_points
        self.supervision_data_path = supervision_data_path
        self.supervision_random_seed = supervision_random_seed
        
        # 监督数据存储变量 (将在数据加载时初始化)
        self.x_sup = None
        self.y_sup = None
        self.u_sup = None
        self.v_sup = None  
        self.p_sup = None

        # Persist config early for post-init scaling, if provided
        self.config = config if config is not None else getattr(self, 'config', None)

        # initialize NN
        self.net = self.initialize_NN(
                num_ins=num_ins, num_outs=num_outs, num_layers=layers, hidden_size=hidden_size, is_evm=False).to(self.device)
        self.net_1 = self.initialize_NN(
                num_ins=num_ins, num_outs=num_outs_1, num_layers=layers_1, hidden_size=hidden_size_1, is_evm=True).to(self.device)

        

        # 確保所有張量使用 float32 精度
        self.net = self.net.float()
        self.net_1 = self.net_1.float()

        # 在DDP包裹之前依配置施加首/末層縮放，避免跨rank權重差異
        self.apply_config_post_init()

        # 優化：初始化vis_t相關變數，避免重複檢查
        self.vis_t_minus_gpu = None  # GPU版本的vis_t_minus

        # Wrap models with DDP only if in distributed mode
        if self.world_size > 1:
            # 從配置讀取 DDP broadcast_buffers 切換（預設 False）
            ddp_broadcast_buffers = False
            if self.config is not None and hasattr(self.config, 'system'):
                try:
                    ddp_broadcast_buffers = bool(getattr(self.config.system, 'ddp_broadcast_buffers', False))
                except Exception:
                    ddp_broadcast_buffers = False
            # 固定前向路徑以關閉未用參數掃描
            self.net = DDP(self.net, 
                           device_ids=[self.local_rank], 
                           output_device=self.local_rank,
                           find_unused_parameters=False,                           
                           broadcast_buffers=ddp_broadcast_buffers,        # 可配置的buffer同步
                           gradient_as_bucket_view=True)  # 提升記憶體效率
            self.net_1 = DDP(self.net_1, 
                             device_ids=[self.local_rank], 
                             output_device=self.local_rank,
                             find_unused_parameters=False,                             
                             broadcast_buffers=ddp_broadcast_buffers,        # 可配置的buffer同步
                             gradient_as_bucket_view=True)  # 提升記憶體效率

        if net_params:
            self.logger.info(f"Loading net params from {net_params}")
            self.load(net_params)

        # 顯示分布式訓練信息
        self.logger.info("Distributed training setup:")
        self.logger.info(f"  World size: {self.world_size}")
        self.logger.info(f"  Rank: {self.rank}")
        self.logger.info(f"  Local rank: {self.local_rank}")

        # 輸出初始化信息
        config_info = {
            "Reynolds數": self.Re,
            "主網路": f"{self.layers} 層 × {self.hidden_size} 神經元",
            "EVM網路": f"{self.layers_1} 層 × {self.hidden_size_1} 神經元",
            "訓練點數": f"{self.N_f:,}",
            "設備": str(self.device),
            "批次大小": "全批次" if self.batch_size == self.N_f else str(self.batch_size)
        }
        self.logger.system_info(config_info)

    # ========= Config-driven setup helpers =========
    def _apply_layer_scales(self, model: torch.nn.Module, first_scale: float, last_scale: float) -> None:
        """Apply scaling to the first and last Linear layers' weights.

        This should be called after Xavier init and is deterministic across ranks
        when executed on each process.
        """
        linear_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        if not linear_layers:
            return
        # 首層
        linear_layers[0].weight.data.mul_(float(first_scale))
        # 末層
        linear_layers[-1].weight.data.mul_(float(last_scale))

    def apply_config_post_init(self) -> None:
        """Apply configuration-dependent behaviors after construction.

        - Layer scaling for main/EVM nets
        """
        cfg = getattr(self, 'config', None)
        if cfg is None:
            return
        # 讀取縮放因子（若缺省則採用預設）
        ncfg = cfg.network
        try:
            main_first = float(getattr(ncfg, 'first_layer_scale_main', 2.0))
            main_last = float(getattr(ncfg, 'last_layer_scale_main', 0.5))
            evm_first = float(getattr(ncfg, 'first_layer_scale_evm', 1.2))
            evm_last = float(getattr(ncfg, 'last_layer_scale_evm', 0.1))
        except Exception:
            main_first, main_last, evm_first, evm_last = 2.0, 0.5, 1.2, 0.1

        # 對底層模型操作（兼容DDP）
        main_model = self.get_model(self.net)
        evm_model = self.get_model(self.net_1)
        self._apply_layer_scales(main_model, main_first, main_last)
        self._apply_layer_scales(evm_model, evm_first, evm_last)

    def get_model_parameters(self, model):
        """Get model parameters considering DDP wrapper"""
        if hasattr(model, 'module'):
            return model.module.parameters()
        else:
            return model.parameters()

    def get_model(self, model: Union[FCNet, DDP]) -> FCNet:
        """Get underlying model considering DDP wrapper"""
        if isinstance(model, DDP):
            return model.module
        else:
            return model

    def get_checkpoint_dir(self):
        """Generates the directory path for saving checkpoints and results."""
        Re_folder = f'Re{self.Re}'
        # Ensure integer conversion for folder names
        n_f_k = int(self.N_f / 1000)
        
        # Format stage name for path
        stage_name = self.current_stage.replace(' ', '_')

        nn_size = f'{self.layers}x{self.hidden_size}_Nf{n_f_k}k'
        params = f'lamB{int(self.alpha_b)}_alpha{self.alpha_evm}{stage_name}'
        
        # Use os.path.join for robust path construction
        base_path = os.path.expanduser('~/NSFnet/ev-NSFnet/results')
        return os.path.join(base_path, Re_folder, f"{nn_size}_{params}")

    def save_checkpoint(self, epoch, optimizer):
        """Saves a comprehensive checkpoint."""
        if self.rank != 0:
            return

        checkpoint_dir = self.get_checkpoint_dir()
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        
        # Ensure we are saving the underlying model state
        net_state = self.get_model(self.net).state_dict()
        net_1_state = self.get_model(self.net_1).state_dict()

        # 解決 torch.compile 在儲存 optimizer state dict 時的 KeyError
        # 直接調用基類的方法以繞過編譯後的函數
        checkpoint = {
            'epoch': epoch,
            'net_state_dict': net_state,
            'net_1_state_dict': net_1_state,
            # safe optimizer state_dict to avoid KeyError when params changed
            'optimizer_state_dict': self._safe_optimizer_state_dict(optimizer),
            'Re': self.Re,
            'alpha_evm': self.alpha_evm,
            'current_stage': self.current_stage,
            'global_step_offset': self.global_step_offset,
            'current_weight_decay': getattr(self, 'current_weight_decay', 0.0)
        }

        try:
            torch.save(checkpoint, checkpoint_path)
            self.logger.checkpoint_saved(checkpoint_path, epoch)
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint to {checkpoint_path}: {e}")

    def load_checkpoint(self, checkpoint_path, optimizer):
        """Loads a checkpoint to resume training with optimizer structure auto-repair.

        增強內容:
        1. 自動檢查 optimizer.param_groups 結構是否符合 AdamW(decay + nodecay) 規範
        2. 若檢測到 legacy / 不一致結構 → 依照保存的 current_weight_decay 重建
        3. 嘗試將舊 state 中的 exp_avg / exp_avg_sq / step 遷移至新參數 (以 id 匹配)
        4. 若遷移失敗不終止，記錄警告並以新狀態繼續
        """
        def _needs_rebuild(opt: torch.optim.Optimizer) -> bool:
            try:
                if opt is None or not opt.param_groups:
                    return True
                # 規範: 1~2 groups; 若 >2 代表舊格式或手動 group
                if len(opt.param_groups) > 2:
                    return True
                # 若只有1組但 weight_decay=0 且存在可 decay 參數 → 允許, 但不強制重建
                # 驗證 group 欄位完整性
                for g in opt.param_groups:
                    if 'params' not in g:
                        return True
                return False
            except Exception:
                return True
        def _migrate_state(old_state: dict, new_opt: torch.optim.Optimizer):
            try:
                # old_state: optimizer.state (k=id(param))
                if not old_state:
                    return 0,0
                transferred = 0
                skipped = 0
                new_param_ids = {id(p): p for pg in new_opt.param_groups for p in pg['params']}
                for pid, s in old_state.items():
                    if pid in new_param_ids:
                        try:
                            new_state_slot = new_opt.state[new_param_ids[pid]]
                            for k,v in s.items():
                                if isinstance(v, torch.Tensor):
                                    if k in new_state_slot and new_state_slot[k].shape == v.shape:
                                        new_state_slot[k].copy_(v.to(new_state_slot[k].device))
                                    else:
                                        new_state_slot[k] = v.to(new_state_slot.get(k, v).device)
                                else:
                                    new_state_slot[k] = v
                            transferred += 1
                        except Exception:
                            skipped += 1
                    else:
                        skipped += 1
                if self.rank == 0 and transferred>0:
                    print(f"   🔄 Optimizer state migrated: {transferred} tensors (skipped {skipped})")
            except Exception as e:
                if self.rank == 0:
                    print(f"   ⚠️ Optimizer state migration failed: {e}")
        if not os.path.exists(checkpoint_path):
            self.logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return 0 # Return 0 to indicate training should start from epoch 0

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            # 寬鬆載入：允許多出的鍵（例如舊版 LAAF 的 a_raw 參數）
            res_main = self.get_model(self.net).load_state_dict(checkpoint['net_state_dict'], strict=False)
            res_evm  = self.get_model(self.net_1).load_state_dict(checkpoint['net_1_state_dict'], strict=False)
            # 記錄不相容鍵，便於診斷
            if res_main.unexpected_keys or res_main.missing_keys:
                self.logger.warning(f"Main net state_dict mismatches. unexpected={len(res_main.unexpected_keys)}, missing={len(res_main.missing_keys)}")
                # 常見：activation 層的 LAAF 參數 a_raw 來自舊 checkpoint
                laaf_unexpected = [k for k in res_main.unexpected_keys if 'a_raw' in k]
                if laaf_unexpected:
                    self.logger.info(f"Ignored legacy LAAF keys in main net: {laaf_unexpected[:4]}{' ...' if len(laaf_unexpected)>4 else ''}")
            if res_evm.unexpected_keys or res_evm.missing_keys:
                self.logger.warning(f"EVM net state_dict mismatches. unexpected={len(res_evm.unexpected_keys)}, missing={len(res_evm.missing_keys)}")

            # 先讀取保存的 wd
            saved_wd = checkpoint.get('current_weight_decay', 0.0)
            self.current_weight_decay = saved_wd

            # 嘗試載入舊 optimizer state（若結構不符會後續重建）
            opt_state = checkpoint.get('optimizer_state_dict', None)
            legacy_state = None
            if opt_state and optimizer is not None:
                try:
                    # 嘗試直接載入 (可能失敗或結構不符)
                    self._load_optimizer_state_dict_safe(optimizer, opt_state)
                    # 保留原始 state 用於後續遷移
                    legacy_state = {k:v for k,v in optimizer.state.items()}
                except Exception:
                    legacy_state = None

            # 決定是否需要重建 param groups
            if _needs_rebuild(optimizer):
                if self.rank == 0:
                    print("   ⚠️ Detected legacy/invalid optimizer param_groups → rebuilding AdamW")
                # 獲取 lr（從 opt_state 或 fallback）
                lr_guess = 1e-3
                try:
                    if optimizer and optimizer.param_groups:
                        lr_guess = optimizer.param_groups[0].get('lr', lr_guess)
                except Exception:
                    pass
                # 重建 AdamW (這會覆寫 self.opt)
                self.build_adamw_optimizer(lr_guess, saved_wd)
                if optimizer is not self.opt:
                    optimizer = self.opt
                # 遷移狀態
                if legacy_state:
                    _migrate_state(legacy_state, optimizer)
            else:
                # 若結構正常，確保 current_weight_decay 與 group 一致
                try:
                    has_decay = False
                    for pg in optimizer.param_groups:
                        if pg.get('weight_decay',0.0)>0:
                            has_decay = True
                            if abs(pg['weight_decay']-saved_wd)>1e-12:
                                if self.rank==0:
                                    print(f"   ⚠️ Mismatch wd(group={pg['weight_decay']}) vs saved({saved_wd}), syncing to saved")
                                pg['weight_decay']=saved_wd
                    if not has_decay and saved_wd>0:
                        if self.rank==0:
                            print("   ⚠️ Saved checkpoint had weight_decay>0 but current groups have none; rebuilding")
                        lr_guess = optimizer.param_groups[0].get('lr',1e-3)
                        self.build_adamw_optimizer(lr_guess, saved_wd)
                except Exception:
                    pass

            start_epoch = checkpoint['epoch'] + 1
            self.global_step_offset = checkpoint.get('global_step_offset', 0)
            self.Re = checkpoint.get('Re', self.Re)
            self.alpha_evm = checkpoint.get('alpha_evm', self.alpha_evm)
            if self.rank == 0:
                print(f"   Restored weight decay: {self.current_weight_decay}")
            self.logger.info(f"✅ Resumed training from checkpoint: {checkpoint_path} at epoch {start_epoch}")
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            # Scheduler 可能先前尚未構建（需訓練循環注入），此處僅保留 wd
            return start_epoch
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            return 0 # Start from scratch if loading fails
        # (legacy duplicate implementation removed)
            
    def init_vis_t(self):
        """優化版本：避免不必要的CPU轉換"""
        (_,_,_,e_raw) = self.neural_net_u(self.x_f, self.y_f)
        # 使用配置激活映射得到非負的 EVM 粘滯（未加上限）
        nu_e = self._compute_nu_e(e_raw)
        cap_val = float(self.beta) / float(self.Re) if self.beta is not None else (self.vis_t0)
        cap = torch.full_like(nu_e, cap_val)
        self.vis_t_minus_gpu = torch.minimum(self.alpha_evm * nu_e.detach(), cap)

    def set_boundary_data(self, X=None, time=False):
        # 接受已切片且在正確裝置上的張量，直接賦值
        if X is None:
            self.logger.warning("邊界數據為None，跳過設置")
            return
        
        if len(X) < 4:
            self.logger.error(f"邊界數據格式錯誤，期望至少4個元素，得到{len(X)}個")
            return
            
        self.x_b, self.y_b, self.u_b, self.v_b = X[:4]
        if time and len(X) > 4:
            self.t_b = X[4]
        total_points = (self.x_b.shape[0] if isinstance(self.x_b, torch.Tensor) else 0)
        
    def set_eq_training_data(self, X=None, time=False):
        # 接受已切片且在正確裝置上的張量，直接賦值
        if X is None:
            self.logger.warning("訓練數據為None，跳過設置")
            return
            
        if len(X) < 2:
            self.logger.error(f"訓練數據格式錯誤，期望至少2個元素，得到{len(X)}個")
            return
            
        self.x_f, self.y_f = X[:2]
        if time and len(X) > 2:
            self.t_f = X[2]
        # Ensure gradients for PDE points
        if isinstance(self.x_f, torch.Tensor):
            self.x_f.requires_grad_(True)
        if isinstance(self.y_f, torch.Tensor):
            self.y_f.requires_grad_(True)
        total_points = (self.x_f.shape[0] if isinstance(self.x_f, torch.Tensor) else 0)
        if self.rank == 0:
            print(f"GPU {self.rank}: Processing {total_points} equation points")
        if hasattr(self, 'config') and hasattr(self.config, 'physics') and hasattr(self.config.physics, 'beta'):
            self.beta = float(self.config.physics.beta)
        else:
            self.beta = 1.0
        self.init_vis_t()

        # === 預計算 PDE 距離權重 w(d)（僅在啟用時，並固定於當前等式點集） ===
        # 好處：避免每個 epoch 重複計算 exp/min/normalize，降低前向耗時與抖動
        try:
            enable_weight = True
            w_min = 0.2
            tau = 0.1
            if hasattr(self, 'config') and hasattr(self.config, 'training'):
                tr = self.config.training
                enable_weight = bool(getattr(tr, 'pde_distance_weighting', True))
                w_min = float(getattr(tr, 'pde_distance_w_min', 0.2))
                tau = float(getattr(tr, 'pde_distance_tau', 0.1))

            if enable_weight and isinstance(self.x_f, torch.Tensor) and isinstance(self.y_f, torch.Tensor):
                with torch.no_grad():
                    d_x = torch.minimum(self.x_f + 1.0, 1.0 - self.x_f)
                    d_y = torch.minimum(self.y_f + 1.0, 1.0 - self.y_f)
                    d = torch.minimum(d_x, d_y)
                    w = w_min + (1.0 - w_min) * torch.exp(-d / max(tau, 1e-6))
                    # 均值歸一，並固定為常數張量（避免進入計算圖）
                    w = (w / (w.mean() + 1e-12)).detach()
                self.w_f = w
            else:
                self.w_f = None
        except Exception:
            # 保守回退
            self.w_f = None

    def set_optimizers(self, opt):
        self.opt = opt

    # ================= AdamW / Weight Decay 支援 =================
    def _build_param_groups(self, weight_decay: float) -> list:
        """建立 AdamW 參數分組：可訓練參數中，維度>1且非 bias 施加 decay。"""
        params_decay = []
        params_nodecay = []
        for model in [self.get_model(self.net), self.get_model(self.net_1)]:
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if p.dim() > 1 and not name.endswith('bias'):
                    params_decay.append(p)
                else:
                    params_nodecay.append(p)
        groups = []
        if params_decay:
            groups.append({'params': params_decay, 'weight_decay': weight_decay})
        if params_nodecay:
            groups.append({'params': params_nodecay, 'weight_decay': 0.0})
        return groups

    def print_optimizer_groups(self):
        """診斷輸出目前 AdamW 參數組資訊。"""
        if self.opt is None:
            if self.rank == 0:
                print("   (optimizer not initialized)")
            return
        if self.rank != 0:
            return
        try:
            total_params = 0
            print("--- Optimizer Param Groups (AdamW) ---")
            for i, g in enumerate(self.opt.param_groups):
                params = g.get('params', [])
                count = sum(p.numel() for p in params if isinstance(p, torch.Tensor))
                total_params += count
                wd = g.get('weight_decay', 0.0)
                sample_names = []
                # 取前3個名稱
                name_map = self._param_name_map(self.get_model(self.net)) | self._param_name_map(self.get_model(self.net_1))
                for p in params[:3]:
                    n = name_map.get(id(p), 'unknown')
                    sample_names.append(n)
                print(f"Group {i}: params={count}, wd={wd}, sample={sample_names}")
            print(f"Total trainable params: {total_params}")
            print(f"Current weight_decay tracked: {getattr(self,'current_weight_decay',0.0)}")
            print("--------------------------------------")
        except Exception as e:
            if self.rank == 0:
                print(f"   ⚠️ print_optimizer_groups error: {e}")

    def build_adamw_optimizer(self, lr: float, weight_decay: float):
        """建立新的 AdamW 並更新 self.opt，同時保留 scheduler 連續性資訊。"""
        from torch.optim import AdamW
        groups = self._build_param_groups(weight_decay)
        self.opt = AdamW(groups, lr=lr, betas=(0.9, 0.999))
        for pg in self.opt.param_groups:
            pg['initial_lr'] = lr
        self.current_weight_decay = weight_decay
        # 重建 scheduler 以綁定新 optimizer（若已有記錄）
        try:
            self._rebuild_scheduler()
        except Exception:
            pass
        if self.rank == 0:
            print(f"🔧 構建 AdamW: lr={lr:.2e}, wd={weight_decay}, groups={len(self.opt.param_groups)}")
        return self.opt

    def rebuild_after_structure_change(self):
        """在 freeze/unfreeze 或 L-BFGS 後重建 AdamW（保持 lr / wd）。"""
        if not hasattr(self, 'current_weight_decay'):
            self.current_weight_decay = 0.0
        lr = 1e-3
        if self.opt is not None and self.opt.param_groups:
            lr = self.opt.param_groups[0].get('lr', lr)
        self.build_adamw_optimizer(lr, self.current_weight_decay)

    def set_alpha_evm(self, alpha):
        self.alpha_evm = alpha

    def _check_gradients(self):
        """檢查梯度狀態，避免梯度爆炸"""
        total_norm = 0
        param_count = 0
        for p in list(self.net.parameters()) + list(self.net_1.parameters()):
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
                param_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            return total_norm
        return 0.0

    def check_tanh_saturation(self, epoch_id):
        """檢測激活函數飽和情況（tanh 或 LAAF+tnah）使用全域步數串接所有stage。

        對於 LAAF，飽和條件以 |a * pre| > 2 估算。
        """
        if epoch_id % 1000 == 0 and self.rank == 0:  # 頻率控制仍沿用呼叫端 + 1000步節流
            # 全域步數（跨 stage 單調遞增）避免TensorBoard覆寫
            global_step = getattr(self, 'global_step_offset', 0) + epoch_id
            saturation_info = []
            
            # 檢查主網絡
            with torch.no_grad():
                test_input = torch.cat([self.x_f[:100], self.y_f[:100]], dim=1)
                layer_count = 0
                # 使用Sequential結構配對 Linear -> activation
                main_layers = getattr(self.get_model(self.net), 'layers', None)
                if isinstance(main_layers, torch.nn.Sequential):
                    idx = 0
                    keys = list(main_layers._modules.keys())
                    while idx < len(keys):
                        key = keys[idx]
                        mod = main_layers._modules[key]
                        if isinstance(mod, torch.nn.Linear):
                            pre = torch.matmul(test_input, mod.weight.T) + mod.bias
                            # 尋找下一個激活
                            sat_ratio = 0.0
                            next_act = None
                            if idx + 1 < len(keys):
                                next_act = main_layers._modules[keys[idx + 1]]
                            if isinstance(next_act, torch.nn.Tanh):
                                sat_ratio = (pre.abs() > 2.0).float().mean().item()
                                test_input = torch.tanh(pre)
                            elif isinstance(next_act, LAAFScalar):
                                # 飽和條件：|a*pre| > 2
                                a = next_act.a
                                sat_ratio = ((a * pre).abs() > 2.0).float().mean().item()
                                test_input = next_act(pre)
                            else:
                                # 未知激活，退化為tanh估算
                                sat_ratio = (pre.abs() > 2.0).float().mean().item()
                                test_input = torch.tanh(pre)
                            saturation_info.append((f"主網絡_Layer{layer_count}", sat_ratio))
                            layer_count += 1
                            idx += 2
                        else:
                            idx += 1
                else:
                    # 後備方案：與舊邏輯一致
                    for name, module in self.get_model(self.net).named_modules():
                        if isinstance(module, torch.nn.Linear):
                            pre = torch.matmul(test_input, module.weight.T) + module.bias
                            sat_ratio = (pre.abs() > 2.0).float().mean().item()
                            saturation_info.append((f"主網絡_Layer{layer_count}", sat_ratio))
                            test_input = torch.tanh(pre)
                            layer_count += 1
                
                # 檢查EVM網絡
                test_input_evm = torch.cat([self.x_f[:100], self.y_f[:100]], dim=1)
                layer_count = 0
                evm_layers = getattr(self.get_model(self.net_1), 'layers', None)
                if isinstance(evm_layers, torch.nn.Sequential):
                    idx = 0
                    keys = list(evm_layers._modules.keys())
                    while idx < len(keys):
                        key = keys[idx]
                        mod = evm_layers._modules[key]
                        if isinstance(mod, torch.nn.Linear):
                            pre = torch.matmul(test_input_evm, mod.weight.T) + mod.bias
                            next_act = None
                            if idx + 1 < len(keys):
                                next_act = evm_layers._modules[keys[idx + 1]]
                            if isinstance(next_act, torch.nn.Tanh):
                                sat_ratio = (pre.abs() > 2.0).float().mean().item()
                                test_input_evm = torch.tanh(pre)
                            elif isinstance(next_act, LAAFScalar):
                                a = next_act.a
                                sat_ratio = ((a * pre).abs() > 2.0).float().mean().item()
                                test_input_evm = next_act(pre)
                            else:
                                sat_ratio = (pre.abs() > 2.0).float().mean().item()
                                test_input_evm = torch.tanh(pre)
                            saturation_info.append((f"EVM網絡_Layer{layer_count}", sat_ratio))
                            layer_count += 1
                            idx += 2
                        else:
                            idx += 1
                else:
                    for name, module in self.get_model(self.net_1).named_modules():
                        if isinstance(module, torch.nn.Linear):
                            pre = torch.matmul(test_input_evm, module.weight.T) + module.bias
                            sat_ratio = (pre.abs() > 2.0).float().mean().item()
                            saturation_info.append((f"EVM網絡_Layer{layer_count}", sat_ratio))
                            test_input_evm = torch.tanh(pre)
                            layer_count += 1
            
            # 輸出診斷信息
            high_saturation_layers = [(name, ratio) for name, ratio in saturation_info if ratio > 0.3]
            if high_saturation_layers:
                self.logger.warning(f"⚠️  高飽和層 (>30%): {high_saturation_layers}")
            
            # 記錄到TensorBoard（使用global_step避免覆寫）
            if self.tb_writer is not None:
                for name, ratio in saturation_info:
                    self.safe_tensorboard_log(f"NetworkHealth/Saturation_{name}", ratio, global_step)
            
            # 梯度分析 (增強診斷)
            grad_norms = []
            avg_grad_norm = 0.0
            if hasattr(self, 'opt') and self.opt is not None:
                for param_group in self.opt.param_groups:
                    for param in param_group['params']:
                        if param.grad is not None:
                            grad_norms.append(param.grad.norm().item())
                if grad_norms:
                    avg_grad_norm = sum(grad_norms) / len(grad_norms)
                    max_grad_norm = max(grad_norms)
                    if self.tb_writer is not None:
                        self.safe_tensorboard_log('NetworkHealth/Avg_Grad_Norm', avg_grad_norm, global_step)
                        self.safe_tensorboard_log('NetworkHealth/Max_Grad_Norm', max_grad_norm, global_step)
                    if avg_grad_norm < 1e-6:
                        self.logger.warning(f"🔻 梯度異常小: {avg_grad_norm:.2e} (可能梯度消失)")
                    elif avg_grad_norm > 1e2:
                        self.logger.warning(f"🔺 梯度異常大: {avg_grad_norm:.2e} (可能梯度爆炸)")
            
            # 輸出量級檢查
            with torch.no_grad():
                sample_input = torch.cat([self.x_f[:50], self.y_f[:50]], dim=1)
                main_output = self.net(sample_input)
                evm_output = self.net_1(sample_input)
                velocity_max = main_output[:, :2].abs().max().item()
                evm_max = evm_output.abs().max().item()
                if self.tb_writer is not None:
                    self.safe_tensorboard_log('NetworkHealth/Velocity_Output_Max', velocity_max, global_step)
                    self.safe_tensorboard_log('NetworkHealth/EVM_Output_Max', evm_max, global_step)
                if velocity_max > 2.0:
                    self.logger.warning(f"🌊 速度輸出過大: {velocity_max:.3f} (建議<2.0)")
                if evm_max > 0.1:
                    self.logger.warning(f"💨 EVM輸出過大: {evm_max:.3f} (建議<0.1)")
            
            # 計算平均飽和率
            avg_saturation = sum(ratio for _, ratio in saturation_info) / len(saturation_info) if saturation_info else 0.0
            
            # 整體健康狀態評估
            health_issues = []
            if hasattr(self, 'opt') and grad_norms:
                if avg_grad_norm < 1e-6:
                    health_issues.append("梯度消失")
                elif avg_grad_norm > 1e2:
                    health_issues.append("梯度爆炸")
            if 'velocity_max' in locals() and velocity_max > 2.0:
                health_issues.append("速度輸出過大")
            if 'evm_max' in locals() and evm_max > 0.1:
                health_issues.append("EVM輸出過大")
            if health_issues:
                self.logger.warning(f"🏥 網路健康警告: {'; '.join(health_issues)}")
            else:
                self.logger.info(f"✅ 網路健康狀態良好 (飽和率: {avg_saturation*100:.1f}%)")
            if avg_saturation > 0.2:
                self.logger.warning(f"🔥 平均飽和率: {avg_saturation*100:.1f}% (建議<20%)")

    def initialize_NN(self,
                      num_ins=3,
                      num_outs=3,
                      num_layers=10,
                      hidden_size=50,
                      is_evm: bool = False):
        """建立主網/副網，根據配置選擇激活函數（支援 LAAF）和神經元數量。"""
        activation_factory = torch.nn.Tanh
        hidden_sizes = None
        
        cfg = getattr(self, 'config', None)
        if cfg is not None and hasattr(cfg, 'network'):
            ncfg = cfg.network
            
            # 獲取每層神經元配置
            if is_evm:
                # EVM網路配置
                if hasattr(ncfg, 'hidden_sizes_1') and ncfg.hidden_sizes_1 is not None:
                    hidden_sizes = ncfg.hidden_sizes_1
                    if len(hidden_sizes) != num_layers:
                        self.logger.warning(f"EVM hidden_sizes_1長度({len(hidden_sizes)})與layers_1({num_layers})不符，使用hidden_size_1")
                        hidden_sizes = None
            else:
                # 主網路配置  
                if hasattr(ncfg, 'hidden_sizes') and ncfg.hidden_sizes is not None:
                    hidden_sizes = ncfg.hidden_sizes
                    if len(hidden_sizes) != num_layers:
                        self.logger.warning(f"主網 hidden_sizes長度({len(hidden_sizes)})與layers({num_layers})不符，使用hidden_size")
                        hidden_sizes = None
            
            # 選擇 main 或 evm 的 activation 設定
            act_name = (ncfg.activation_evm if is_evm else ncfg.activation_main).strip().lower()
            if act_name == 'laaf':
                init_scale = float(getattr(ncfg, 'laaf_init_scale', 1.0))
                max_scale = float(getattr(ncfg, 'laaf_max_scale', 20.0))
                # 以偏函式方式提供 layer-wise 參數
                def _factory():
                    return LAAFScalar(init_scale=init_scale, max_scale=max_scale)
                activation_factory = _factory
            else:
                activation_factory = torch.nn.Tanh
                
        return FCNet(num_ins=num_ins,
                     num_outs=num_outs,
                     num_layers=num_layers,
                     hidden_size=hidden_size,
                     hidden_sizes=hidden_sizes,
                     activation=activation_factory)

    def set_eq_training_func(self, train_data_func):
        self.train_data_func = train_data_func

    def neural_net_u(self, x, y):
        X = torch.cat((x, y), dim=1)
        
        # 確保輸入張量在正確的設備上
        X = X.to(self.device)
        
        # 使用上下文管理器確保梯度正確傳播
        with torch.enable_grad():
            uvp = self.net(X)
            ee = self.net_1(X)
        
        u = uvp[:, 0]
        v = uvp[:, 1]
        p = uvp[:, 2:3]
        e = ee[:, 0:1]
        return u, v, p, e  # 返回原始殘差預測 e_raw

    def neural_net_equations(self, x, y):
        """優化版本：減少重複計算和批量化梯度計算"""
        X = torch.cat((x, y), dim=1)
        
        # 確保輸入張量在正確的設備上
        X = X.to(self.device)
        
        # 使用上下文管理器確保梯度正確傳播
        with torch.enable_grad():
            uvp = self.net(X)
            ee = self.net_1(X)

        u = uvp[:, 0:1]
        v = uvp[:, 1:2]
        p = uvp[:, 2:3]
        e_raw = ee[:, 0:1]
        self.evm = e_raw

        # 優化：批量計算所有一階梯度
        outputs = [u, v, p]
        grads = self.compute_gradients_batch(outputs, [x, y])
        
        u_x, u_y = grads[0]
        v_x, v_y = grads[1]
        p_x, p_y = grads[2]
        
        # 優化：批量計算二階梯度
        second_order_outputs = [u_x, u_y, v_x, v_y]
        second_order_inputs = [x, y, x, y]
        second_grads = self.compute_second_gradients_batch(second_order_outputs, second_order_inputs)
        
        u_xx, u_yy, v_xx, v_yy = second_grads

        # 計算非負的EVM粘滯（帶上限）
        batch_size = x.shape[0]
        # 基於上一輪的 vis_t_minus，與 vis_t0 及 beta/Re 共同裁切
        self.vis_t = self._compute_vis_t_optimized(batch_size, e_raw)
            
        # 更新 vis_t_minus (移到GPU上避免CPU-GPU轉換)
        cap_val = float(self.beta) / float(self.Re) if self.beta is not None else (1.0 / float(self.Re))
        nu_e_now = self._compute_nu_e(e_raw)
        if not getattr(self, 'lock_vis_t_minus', False):
            self.vis_t_minus_gpu = torch.minimum(self.alpha_evm * nu_e_now.detach(), torch.full_like(nu_e_now, cap_val))

        # 定義域已經是 [-1,1]²，梯度即為物理梯度，無需額外縮放
        # 直接使用計算的梯度
        u_x_phys = u_x
        u_y_phys = u_y
        v_x_phys = v_x
        v_y_phys = v_y
        p_x_phys = p_x
        p_y_phys = p_y
        
        u_xx_phys = u_xx
        u_yy_phys = u_yy
        v_xx_phys = v_xx
        v_yy_phys = v_yy

        # NS equations - 使用物理座標的導數
        vis_total = (1.0/self.Re + self.vis_t)
        
        eq1 = (u*u_x_phys + v*u_y_phys) + p_x_phys - vis_total*(u_xx_phys + u_yy_phys)
        eq2 = (u*v_x_phys + v*v_y_phys) + p_y_phys - vis_total*(v_xx_phys + v_yy_phys)
        eq3 = u_x_phys + v_y_phys

        # 保持熵殘差方程使用原始 e_raw（帶符號）
        residual = (eq1*(u-0.5)+eq2*(v-0.5))-e_raw
        return eq1, eq2, eq3, residual

    def _compute_nu_e(self, e_raw: torch.Tensor) -> torch.Tensor:
        """Compute nonnegative EVM contribution before scaling/capping."""
        return torch.abs(e_raw)

    def compute_gradients_batch(self, outputs: List[torch.Tensor], inputs: List[torch.Tensor]) -> List[List[torch.Tensor]]:
        """
        批量計算多個輸出對多個輸入的梯度，減少autograd調用次數
        """
        batch_gradients = []
        
        for output in outputs:
            grad_outputs = [torch.ones_like(output, device=output.device)]
            grads = torch.autograd.grad(
                [output],
                inputs,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )
            # 處理None梯度
            processed_grads = [g if g is not None else torch.zeros_like(inputs[i]) for i, g in enumerate(grads)]
            batch_gradients.append(processed_grads)
            
        return batch_gradients
    
    def compute_second_gradients_batch(self, first_grads: List[torch.Tensor], inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        批量計算二階梯度
        """
        second_grads = []
        
        for i, grad in enumerate(first_grads):
            input_tensor = inputs[i]
            grad_outputs = [torch.ones_like(grad, device=grad.device)]
            second_grad = torch.autograd.grad(
                [grad],
                [input_tensor],
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            
            if second_grad is None:
                second_grad = torch.zeros_like(input_tensor)
            second_grads.append(second_grad)
            
        return second_grads
    
    def _compute_vis_t_optimized(self, batch_size: int, e: torch.Tensor) -> torch.Tensor:
        """
        優化的vis_t計算，避免CPU-GPU轉換和numpy操作
        """
        if hasattr(self, 'vis_t_minus_gpu') and self.vis_t_minus_gpu is not None:
            # 確保尺寸匹配
            if self.vis_t_minus_gpu.shape[0] != batch_size:
                if self.vis_t_minus_gpu.shape[0] > batch_size:
                    vis_t_minus_batch = self.vis_t_minus_gpu[:batch_size]
                else:
                    # GPU上的重複操作
                    repeat_times = (batch_size + self.vis_t_minus_gpu.shape[0] - 1) // self.vis_t_minus_gpu.shape[0]
                    vis_t_minus_batch = self.vis_t_minus_gpu.repeat(repeat_times, 1)[:batch_size]
            else:
                vis_t_minus_batch = self.vis_t_minus_gpu
            
            # 在GPU上計算minimum
            vis_t0_tensor = torch.full_like(vis_t_minus_batch, self.vis_t0)
            beta_cap = torch.full_like(vis_t_minus_batch, (float(self.beta) / float(self.Re)) if self.beta is not None else self.vis_t0)
            vis_t = torch.minimum(torch.minimum(vis_t0_tensor, vis_t_minus_batch), beta_cap)
        else:
            # 首次運行或沒有前一步數據
            vis_t = torch.full((batch_size, 1), self.vis_t0, device=self.device, dtype=torch.float32)
            
        return vis_t

    def autograd(self, y: torch.Tensor, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        計算梯度的函數 (保留原函數以兼容性)
        """
        grad_outputs: List[torch.Tensor] = [torch.ones_like(y, device=y.device)]
        grad = torch.autograd.grad(
            [y],
            x,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )

        if grad is None:
            grad = [torch.zeros_like(xx) for xx in x]
        assert grad is not None
        grad = [g if g is not None else torch.zeros_like(x[i]) for i, g in enumerate(grad)]
        return grad

    def predict(self, net_params, X):
        x, y = X
        return self.neural_net_u(x, y)

    def shuffle(self, tensor):
        tensor_to_numpy = tensor.detach().cpu()
        shuffle_numpy = np.random.shuffle(tensor_to_numpy)
        return torch.tensor(tensor_to_numpy, requires_grad=True).float()

    def fwd_computing_loss_2d(self, loss_mode='MSE'):
        # boundary data
        (self.u_pred_b, self.v_pred_b, _, _) = self.neural_net_u(self.x_b, self.y_b)

        # BC loss - 處理空邊界數據的情況
        if loss_mode == 'MSE':
            if self.x_b.shape[0] > 0:  # 檢查是否有邊界數據
                # 確保張量形狀匹配
                u_b_flat = self.u_b.view(-1)
                v_b_flat = self.v_b.view(-1)
                u_pred_b_flat = self.u_pred_b.view(-1)
                v_pred_b_flat = self.v_pred_b.view(-1)
                
                # 檢查張量大小是否匹配
                if u_b_flat.shape[0] != u_pred_b_flat.shape[0]:
                    print(f"ERROR: Boundary tensor size mismatch: {u_b_flat.shape} vs {u_pred_b_flat.shape}")
                    # 使用較小的尺寸
                    min_size = min(u_b_flat.shape[0], u_pred_b_flat.shape[0])
                    u_b_flat = u_b_flat[:min_size]
                    v_b_flat = v_b_flat[:min_size]
                    u_pred_b_flat = u_pred_b_flat[:min_size]
                    v_pred_b_flat = v_pred_b_flat[:min_size]
                
                self.loss_b = torch.mean(torch.square(u_b_flat - u_pred_b_flat)) + \
                              torch.mean(torch.square(v_b_flat - v_pred_b_flat))
            else:
                # 沒有邊界數據時設置損失為0，但保持在計算圖中
                # 確保兩個網路都參與計算圖
                # 獲取模型（處理DDP包裝）
                net_model = self.get_model(self.net)
                net1_model = self.get_model(self.net_1)
                
                # 獲取第一個線性層
                first_layer = None
                first_layer_1 = None
                for layer in net_model.layers:
                    if isinstance(layer, torch.nn.Linear):
                        first_layer = layer
                        break
                for layer in net1_model.layers:
                    if isinstance(layer, torch.nn.Linear):
                        first_layer_1 = layer
                        break
                
                if first_layer is not None and first_layer_1 is not None:
                    dummy_loss_net = torch.sum(first_layer.weight * 0.0)
                    dummy_loss_net1 = torch.sum(first_layer_1.weight * 0.0)
                    self.loss_b = dummy_loss_net + dummy_loss_net1
                else:
                    self.loss_b = torch.tensor(0.0, device=self.device)

        # equation
        assert self.x_f is not None and self.y_f is not None

        (self.eq1_pred, self.eq2_pred, self.eq3_pred, self.eq4_pred) = self.neural_net_equations(self.x_f, self.y_f)
    
        if loss_mode == 'MSE':
            # 使用預計算的距離權重（若有）；否則使用常數1.0
            w = getattr(self, 'w_f', None)
            if w is None:
                w = 1.0

            eq1_sq = torch.square(self.eq1_pred.view(-1))
            eq2_sq = torch.square(self.eq2_pred.view(-1))
            eq3_sq = torch.square(self.eq3_pred.view(-1))
            eq4_sq = torch.square(self.eq4_pred.view(-1))

            if isinstance(w, torch.Tensor):
                w_flat = w.view(-1)
                self.loss_eq1 = torch.mean(w_flat * eq1_sq)
                self.loss_eq2 = torch.mean(w_flat * eq2_sq)
                self.loss_eq3 = torch.mean(w_flat * eq3_sq)
                self.loss_eq4 = torch.mean(w_flat * eq4_sq)
            else:
                self.loss_eq1 = torch.mean(eq1_sq)
                self.loss_eq2 = torch.mean(eq2_sq)
                self.loss_eq3 = torch.mean(eq3_sq)
                self.loss_eq4 = torch.mean(eq4_sq)

            self.loss_e = self.loss_eq1 + self.loss_eq2 + self.loss_eq3 + 0.1 * self.loss_eq4

        # supervision loss
        if self.x_sup is not None and self.x_sup.shape[0] > 0:
            # 计算监督点的预测值
            (u_pred_sup, v_pred_sup, p_pred_sup, _) = self.neural_net_u(self.x_sup, self.y_sup)
            
            # 计算监督损失
            if loss_mode == 'MSE':
                u_sup_flat = self.u_sup.view(-1)
                v_sup_flat = self.v_sup.view(-1)
                p_sup_flat = self.p_sup.view(-1)
                u_pred_sup_flat = u_pred_sup.view(-1)
                v_pred_sup_flat = v_pred_sup.view(-1)
                p_pred_sup_flat = p_pred_sup.view(-1)
                
                self.loss_s = torch.mean(torch.square(u_pred_sup_flat - u_sup_flat)) + \
                              torch.mean(torch.square(v_pred_sup_flat - v_sup_flat)) + \
                              torch.mean(torch.square(p_pred_sup_flat - p_sup_flat))
        else:
            # 没有监督数据时，设置损失为0但保持在计算图中
            if hasattr(self.net, 'module'):
                dummy_loss_net = torch.sum(self.net.module.layers[0].weight * 0.0)
                dummy_loss_net1 = torch.sum(self.net_1.module.layers[0].weight * 0.0)
            else:
                dummy_loss_net = torch.sum(self.net.layers[0].weight * 0.0)
                dummy_loss_net1 = torch.sum(self.net_1.layers[0].weight * 0.0)
            self.loss_s = dummy_loss_net + dummy_loss_net1

        # 跨GPU聚合損失以獲得全局損失值（僅用於日誌：使用 reduce 匯總到 rank 0）
        if self.world_size > 1:
            # 確保loss變數是tensor類型
            loss_b_tensor = self.loss_b if isinstance(self.loss_b, torch.Tensor) else torch.tensor(self.loss_b, device=self.device)
            loss_e_tensor = self.loss_e if isinstance(self.loss_e, torch.Tensor) else torch.tensor(self.loss_e, device=self.device)
            loss_s_tensor = self.loss_s if isinstance(self.loss_s, torch.Tensor) else torch.tensor(self.loss_s, device=self.device)
            
            loss_vec = torch.stack([
                loss_b_tensor.detach(),
                loss_e_tensor.detach(),
                loss_s_tensor.detach()
            ])
            dist.reduce(loss_vec, dst=0, op=dist.ReduceOp.SUM)
            if self.rank == 0:
                loss_vec = loss_vec / self.world_size
                # 用於日誌顯示的平均損失（rank 0）
                self.loss_b_avg = loss_vec[0]
                self.loss_e_avg = loss_vec[1]
                self.loss_s_avg = loss_vec[2]
            else:
                # 非rank 0不需要全域平均，保留本地值供必要時使用
                self.loss_b_avg = self.loss_b
                self.loss_e_avg = self.loss_e
                self.loss_s_avg = self.loss_s
        else:
            self.loss_b_avg = self.loss_b
            self.loss_e_avg = self.loss_e
            self.loss_s_avg = self.loss_s

        # 計算總損失（保持梯度追踪），包含监督损失
        self.loss = self.alpha_b * self.loss_b + self.alpha_e * self.loss_e + self.alpha_s * self.loss_s
        
        # 分佈式模式下的 dummy L2：當無實際 weight decay 時才啟用，避免與 AdamW 重複
        if self.world_size > 1:
            use_dummy = True
            if hasattr(self, 'current_weight_decay') and getattr(self, 'current_weight_decay', 0.0) > 0:
                use_dummy = False
            else:
                if self.opt is not None:
                    for pg in self.opt.param_groups:
                        if pg.get('weight_decay', 0.0) > 0:
                            use_dummy = False
                            break
            if use_dummy:
                regularization_weight = 1e-8
                if hasattr(self.net, 'module'):
                    params_main = self.net.module.parameters()
                    params_evm = self.net_1.module.parameters()
                else:
                    params_main = self.net.parameters()
                    params_evm = self.net_1.parameters()
                net_reg = sum(p.pow(2).sum() for p in params_main)
                net1_reg = sum(p.pow(2).sum() for p in params_evm)
                self.loss = self.loss + regularization_weight * (net_reg + net1_reg)

        # LAAF 正則化（可選）
        try:
            ncfg = getattr(self.config, 'network', None)
            laaf_lambda = float(getattr(ncfg, 'laaf_reg_lambda', 0.0)) if ncfg is not None else 0.0
        except Exception:
            laaf_lambda = 0.0
        if laaf_lambda > 0.0:
            try:
                reg_main = compute_laaf_regularization(self.get_model(self.net), target=1.0)
                reg_evm = compute_laaf_regularization(self.get_model(self.net_1), target=1.0)
                self.loss = self.loss + laaf_lambda * (reg_main + reg_evm)
            except Exception:
                pass


        # 創建用於backward的loss（保持梯度）
        loss_for_backward = self.loss
        
        # 創建用於日誌記錄的detached數值
        if hasattr(self, 'loss_e_avg'):
            loss_e_log = self.loss_e_avg.detach().item() if isinstance(self.loss_e_avg, torch.Tensor) else float(self.loss_e_avg)
        else:
            loss_e_log = self.loss_e.detach().item() if isinstance(self.loss_e, torch.Tensor) else float(self.loss_e)
            
        if hasattr(self, 'loss_b_avg'):
            loss_b_log = self.loss_b_avg.detach().item() if isinstance(self.loss_b_avg, torch.Tensor) else float(self.loss_b_avg)
        else:
            loss_b_log = self.loss_b.detach().item() if isinstance(self.loss_b, torch.Tensor) else float(self.loss_b)
            
        if hasattr(self, 'loss_s_avg'):
            loss_s_log = self.loss_s_avg.detach().item() if isinstance(self.loss_s_avg, torch.Tensor) else float(self.loss_s_avg)
        else:
            loss_s_log = self.loss_s.detach().item() if isinstance(self.loss_s, torch.Tensor) else float(self.loss_s)

        return loss_for_backward, [loss_e_log, loss_b_log, loss_s_log, self.loss_eq1.detach().item(), self.loss_eq2.detach().item(), self.loss_eq3.detach().item(), self.loss_eq4.detach().item()]

    def train(self,
              num_epoch=1,
              lr=1e-4,
              optimizer=None,
              scheduler=None,
              batchsize=None,
              profiler=None,
              start_epoch=0):
        if self.opt is not None:
            # 對於SGDR (SequentialLR)，需要特殊處理以確保新階段的學習率正確設置
            if scheduler is not None and hasattr(scheduler, '_schedulers'):
                # SequentialLR情況：需要更新基礎學習率但保持scheduler狀態
                current_lr = self.opt.param_groups[0]['lr']
                
                # 更新基礎學習率和initial_lr，確保scheduler從正確的基礎開始
                for param_group in self.opt.param_groups:
                    param_group['initial_lr'] = lr
                
                # 重置scheduler狀態以使用新的基礎學習率
                scheduler.last_epoch = -1
                
                if self.rank == 0:
                    print(f"🔧 檢測到SequentialLR scheduler (SGDR)，更新基礎lr: {current_lr:.6f} -> {lr:.6f}")
                    print(f"   重置scheduler以從新基礎學習率開始")
            else:
                # 無scheduler或非SequentialLR：正常設置學習率
                self.opt.param_groups[0]['lr'] = lr
        return self.solve_Adam(self.fwd_computing_loss_2d, num_epoch, batchsize, scheduler, profiler, start_epoch)

    def _stage_group_index(self) -> int:
        """將 Stage 1-6 分為三段：0:(1-2), 1:(3-4), 2:(5-6)"""
        try:
            name = str(self.current_stage)
            if 'Stage' in name:
                idx = int(name.split()[-1])
            else:
                idx = 1
        except Exception:
            idx = 1
        if idx <= 2:
            return 0
        elif idx <= 4:
            return 1
        return 2

    def _compute_ema(self, seq, gamma: float) -> float:
        """對序列做EMA平滑（返回最後EMA）"""
        ema = None
        for v in seq:
            if ema is None:
                ema = float(v)
            else:
                ema = gamma * ema + (1.0 - gamma) * float(v)
        return float(ema) if ema is not None else 0.0

    def _check_distributed_lbfgs_trigger(self) -> bool:
        """分佈式L-BFGS觸發檢測（改為EMA相對改善 + 梯度/物理條件 + 冷卻）"""
        trigger_lbfgs = False

        # 配置與分佈式開關
        cfg = getattr(self, 'config', None)
        lbfgs_cfg = getattr(cfg.training, 'lbfgs', None) if cfg and hasattr(cfg, 'training') else None
        if lbfgs_cfg and not lbfgs_cfg.enabled_in_distributed and self.world_size > 1:
            return False

        # 階段檢查：只在Stage 3+才允許L-BFGS觸發
        group_idx = self._stage_group_index()
        enable_from_stage = getattr(lbfgs_cfg, 'enable_from_stage', 3) if lbfgs_cfg else 3
        current_stage_num = 1
        try:
            name = str(self.current_stage)
            if 'Stage' in name:
                current_stage_num = int(name.split()[-1])
        except Exception:
            current_stage_num = 1
        
        if current_stage_num < enable_from_stage:
            return False

         # 冷卻檢查（使用相對步數解決階段切換問題）
        cooldown = getattr(lbfgs_cfg, 'cooldown_steps', 5000) if lbfgs_cfg else 5000
        
        # 獲取當前階段起始步數，確保相對步數計算正確
        stage_start_step = getattr(self, 'stage_start_step', 0)
        current_relative_step = self.stage_step - stage_start_step
        last_strategy_relative_step = self.last_strategy_step - stage_start_step
        
        # 使用相對步數檢查冷卻
        if (current_relative_step - last_strategy_relative_step) < cooldown:
            # Debug: 偶爾輸出冷卻狀態（每5000步一次）
            if self.rank == 0 and self.stage_step % 5000 == 0:
                print(f"[L-BFGS冷卻] 當前相對步數:{current_relative_step}, 上次觸發相對步數:{last_strategy_relative_step}, 冷卻需求:{cooldown}")
            return False

        # 需要足夠的滑窗
        group_idx = self._stage_group_index()
        windows = getattr(lbfgs_cfg, 'trigger_window_per_stage', [5000, 7500, 10000]) if lbfgs_cfg else [5000, 7500, 10000]
        min_improves = getattr(lbfgs_cfg, 'min_improve_pct_per_stage', [0.02, 0.01, 0.005]) if lbfgs_cfg else [0.02, 0.01, 0.005]
        W = int(windows[min(group_idx, len(windows)-1)])
        min_r = float(min_improves[min(group_idx, len(min_improves)-1)])
        gamma = float(getattr(lbfgs_cfg, 'ema_gamma', 0.95)) if lbfgs_cfg else 0.95

        if len(self.stage_loss_deque) < max(W, 50):
            return False

        if self.rank == 0:
            losses = list(self.stage_loss_deque)
            L_t = float(losses[-1])
            L_w = float(losses[-W])
            denom = max(self._compute_ema(losses[-W:], gamma), 1e-12)
            r = (L_w - L_t) / denom

            # 梯度條件（簡化）
            use_simple_grad_check = getattr(lbfgs_cfg, 'use_simple_grad_check', True) if lbfgs_cfg else True
            
            if use_simple_grad_check:
                # 簡化的梯度檢查：只需要滿足絕對值或相對改善條件
                grad_med = float(getattr(self, 'grad_median', 1e9))
                g_base = float(getattr(self, 'grad_baseline', grad_med))
                rel_ok = grad_med < (getattr(lbfgs_cfg, 'grad_relative_factor', 0.02) * g_base) if lbfgs_cfg else False  # 放寬到2%
                abs_ok = grad_med < (getattr(lbfgs_cfg, 'grad_median_abs_thresh', 2e-3) if lbfgs_cfg else 2e-3)  # 放寬到2e-3
                grad_ok = abs_ok or rel_ok
            else:
                # 原始複雜梯度檢查
                grad_med = float(getattr(self, 'grad_median', 1e9))
                grad_iqr = float(getattr(self, 'grad_iqr', 0.0))
                grad_iqr_ratio = grad_iqr / (grad_med + 1e-12)
                g_base = float(getattr(self, 'grad_baseline', grad_med))
                rel_ok = grad_med < (getattr(lbfgs_cfg, 'grad_relative_factor', 0.01) * g_base) if lbfgs_cfg else False
                abs_ok = grad_med < (getattr(lbfgs_cfg, 'grad_median_abs_thresh', 1e-3) if lbfgs_cfg else 1e-3)
                cos_ema = float(getattr(self, 'grad_cos_ema', 0.0))
                cos_ok = cos_ema > (getattr(lbfgs_cfg, 'grad_cos_ema_thresh', 0.9) if lbfgs_cfg else 0.9)
                grad_ok = (abs_ok or rel_ok) and (grad_iqr_ratio < 5.0 or cos_ok)

            # 物理條件（放寬）
            alpha_threshold = getattr(lbfgs_cfg, 'alpha_evm_threshold', 0.02) if lbfgs_cfg else 0.02  # 放寬到0.02
            alpha_ok = self.alpha_evm <= alpha_threshold
            cap_ratio_p95 = float(getattr(self, 'vis_cap_p95', 0.0))  # 由訓練循環維護
            cap_threshold = getattr(lbfgs_cfg, 'cap_ratio_threshold', 0.7) if lbfgs_cfg else 0.7  # 放寬到0.7
            phys_ok = cap_ratio_p95 < cap_threshold

            trigger_lbfgs = (r <= min_r) and grad_ok and alpha_ok and phys_ok

        # 廣播
        if self.world_size > 1:
            try:
                data = [trigger_lbfgs]
                dist.broadcast_object_list(data, src=0)
                trigger_lbfgs = data[0]
            except Exception as e:
                if self.rank == 0:
                    print(f"🚨 L-BFGS觸發廣播失敗: {e}")
                trigger_lbfgs = False

        return trigger_lbfgs

    def _log_tip_once(self, key: str, msg: str):
        """在rank0輸出一次提示，每1萬步同類不重複"""
        try:
            if self.rank != 0:
                return
            if not (hasattr(self, 'config') and hasattr(self.config, 'training') and getattr(self.config.training, 'log_tips', True)):
                return
            now = int(getattr(self, 'stage_step', 0))
            last = int(self._tips_last_step.get(key, -10**9))
            if now - last >= 10000:
                print("--- Tips ---\n" + msg)
                self._tips_last_step[key] = now
        except Exception:
            pass

    def _calculate_parameter_checksum(self, net_state, net1_state):
        """計算參數校驗碼"""
        checksum = 0.0
        for state_dict in [net_state, net1_state]:
            for param in state_dict.values():
                checksum += torch.sum(param).item()
        return checksum

    def _broadcast_model_parameters_with_verification(self):
        """參數廣播並驗證一致性"""
        if self.world_size <= 1:
            return True
            
        try:
            if self.rank == 0:
                # 準備參數數據並計算校驗碼
                net_state = {k: v.cpu() for k, v in self.get_model(self.net).state_dict().items()}
                net1_state = {k: v.cpu() for k, v in self.get_model(self.net_1).state_dict().items()}
                
                # 計算參數校驗碼
                checksum = self._calculate_parameter_checksum(net_state, net1_state)
                payload = [net_state, net1_state, checksum]
            else:
                payload = [None, None, None]
            
            # 廣播參數
            dist.broadcast_object_list(payload, src=0)
            
            # 非master rank載入參數並驗證
            if self.rank != 0:
                self.get_model(self.net).load_state_dict(payload[0], strict=True)
                self.get_model(self.net_1).load_state_dict(payload[1], strict=True)
                
                # 驗證參數校驗碼
                local_checksum = self._calculate_parameter_checksum(payload[0], payload[1])
                if abs(local_checksum - payload[2]) > 1e-10:
                    raise RuntimeError(f"Rank {self.rank}: 參數校驗失敗 (local={local_checksum:.6e}, expected={payload[2]:.6e})")
            
            # GPU同步
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            return True
            
        except Exception as e:
            if self.rank == 0:
                print(f"🚨 參數同步失敗: {e}")
            return False

    def train_with_lbfgs_segment(self, max_outer_steps=None, lbfgs_params=None, log_interval=200, timeout_seconds=None):
        """增強版分佈式L-BFGS訓練段"""
        import copy
        
        # 從配置中獲取參數
        lbfgs_config = getattr(self, 'config', None)
        if lbfgs_config and hasattr(lbfgs_config.training, 'lbfgs'):
            config_lbfgs = lbfgs_config.training.lbfgs
            if max_outer_steps is None:
                max_outer_steps = config_lbfgs.max_outer_steps
            if timeout_seconds is None:
                timeout_seconds = config_lbfgs.timeout_seconds
            if lbfgs_params is None:
                lbfgs_params = {
                    'max_iter': config_lbfgs.max_iter,
                    'history_size': config_lbfgs.history_size,
                    'tolerance_grad': config_lbfgs.tolerance_grad,
                    'tolerance_change': config_lbfgs.tolerance_change,
                    'line_search_fn': config_lbfgs.line_search_fn
                }
        
        # 使用默認值如果沒有配置
        if max_outer_steps is None:
            max_outer_steps = 2000
        if timeout_seconds is None:
            timeout_seconds = 600
        if lbfgs_params is None:
            lbfgs_params = {
                'max_iter': 50,
                'history_size': 20,
                'tolerance_grad': 1e-8,
                'tolerance_change': 1e-9,
                'line_search_fn': 'strong_wolfe'
            }
        
        mode = "分佈式" if self.world_size > 1 else "單GPU"
        if self.rank == 0:
            print(f"=== 進入 {mode} L-BFGS 段 ===")
        
        # 保存當前Adam狀態
        adam_state_backup = None
        if self.opt is not None:
            try:
                adam_state_backup = copy.deepcopy(self.opt.state_dict())
            except:
                pass
        
        success = False
        best_loss = float('inf')
        
        try:
            # 段內策略：可選凍結EVM、鎖定vis_t_minus
            if lbfgs_config and hasattr(lbfgs_config.training, 'lbfgs') and lbfgs_config.training.lbfgs.freeze_evm_during_lbfgs:
                self.freeze_evm_net(self.stage_step)
            self.lock_vis_t_minus = True

            if self.opt is not None:
                self.opt.zero_grad(set_to_none=True)
            params = list(self.get_model(self.net).parameters()) + list(self.get_model(self.net_1).parameters())
            lbfgs = torch.optim.LBFGS(params,
                                      max_iter=lbfgs_params.get('max_iter', 50),
                                      history_size=lbfgs_params.get('history_size', 20),
                                      tolerance_grad=lbfgs_params.get('tolerance_grad', 1e-8),
                                      tolerance_change=lbfgs_params.get('tolerance_change', 1e-9),
                                      line_search_fn=lbfgs_params.get('line_search_fn', 'strong_wolfe'))
            
            stagnation = 0
            start_time = time.time()
            patience = int(lbfgs_config.training.lbfgs.early_stop_patience) if lbfgs_config and hasattr(lbfgs_config.training, 'lbfgs') else 8
            min_delta = float(lbfgs_config.training.lbfgs.early_stop_min_delta) if lbfgs_config and hasattr(lbfgs_config.training, 'lbfgs') else 1e-4
            
            def closure():
                lbfgs.zero_grad(set_to_none=True)
                loss, _ = self.fwd_computing_loss_2d()
                if isinstance(loss, torch.Tensor):
                    loss.backward()
                else:
                    # 如果是標量，轉換為tensor
                    loss = torch.tensor(loss, requires_grad=True, device=self.device)
                    loss.backward()
                return loss
            
            stop_reason = "done"
            if self.world_size > 1:
                # 分佈式模式：僅rank 0執行L-BFGS
                if self.rank == 0:
                    for step in range(max_outer_steps):
                        try:
                            loss_prev = best_loss
                            loss_val = lbfgs.step(closure).item()
                            if step % log_interval == 0:
                                print(f"[分佈式 L-BFGS] step={step} loss={loss_val:.3e}")
                            
                            if loss_val + 1e-12 < best_loss:
                                best_loss = loss_val
                                stagnation = 0
                            else:
                                if (loss_prev - loss_val) < min_delta:
                                    stagnation += 1
                                else:
                                    stagnation = 0
                            
                            if best_loss < 1e-8 or stagnation >= patience:
                                stop_reason = "early_stop"
                                break
                            if time.time() - start_time > timeout_seconds:
                                print("⏱️ L-BFGS 段達到超時限制，提前結束")
                                stop_reason = "timeout"
                                break
                        except Exception as e:
                            print(f"🚨 L-BFGS步驟失敗 (step={step}): {e}")
                            stop_reason = "error"
                            break
                
                # 使用增強的參數同步機制
                sync_success = self._broadcast_model_parameters_with_verification()
                if not sync_success:
                    raise RuntimeError("參數同步失敗")
                
                success = True
                
            else:
                # 單GPU模式
                for step in range(max_outer_steps):
                    try:
                        loss_prev = best_loss
                        loss_val = lbfgs.step(closure).item()
                        if step % log_interval == 0:
                            print(f"[L-BFGS] step={step} loss={loss_val:.3e}")
                        
                        if loss_val + 1e-12 < best_loss:
                            best_loss = loss_val
                            stagnation = 0
                        else:
                            if (loss_prev - loss_val) < min_delta:
                                stagnation += 1
                            else:
                                stagnation = 0
                        
                        if best_loss < 1e-8 or stagnation >= patience:
                            stop_reason = "early_stop"
                            break
                        if time.time() - start_time > timeout_seconds:
                            if self.rank == 0:
                                print("⏱️ L-BFGS 段達到超時限制，提前結束")
                            stop_reason = "timeout"
                            break
                    except Exception as e:
                        if self.rank == 0:
                            print(f"🚨 L-BFGS步驟失敗 (step={step}): {e}")
                        stop_reason = "error"
                        break
                
                success = True
                
        except Exception as e:
            if self.rank == 0:
                print(f"🚨 L-BFGS執行失敗: {e}")
            success = False
        
        # 恢復AdamW優化器（保持 stage lr / weight decay）
        current_lr = self.opt.param_groups[0]['lr'] if self.opt is not None else 1e-4
        wd = getattr(self, 'current_weight_decay', 0.0)
        try:
            self.build_adamw_optimizer(current_lr, wd)
        except Exception:
            # 回退普通Adam
            self.opt = torch.optim.Adam(list(self.get_model(self.net).parameters()) + list(self.get_model(self.net_1).parameters()), lr=current_lr)
            for group in self.opt.param_groups:
                group['initial_lr'] = current_lr
        
        # 段後解鎖/解凍：先解凍但暫不重建，接著統一以 AdamW + scheduler 重建
        self.lock_vis_t_minus = False
        try:
            self.defreeze_evm_net(self.stage_step, rebuild=False)
        except Exception:
            pass
        # AdamW 已於上方 build_adamw_optimizer 重建；此處只需 scheduler 重建
        self._rebuild_scheduler()
        
        # 如果L-BFGS失敗且有備份，嘗試恢復Adam狀態
        if adam_state_backup is not None and not success:
            try:
                self.opt.load_state_dict(adam_state_backup)
                if self.rank == 0:
                    print("🔄 已恢復Adam優化器狀態")
            except Exception as e:
                if self.rank == 0:
                    print(f"⚠️ 恢復Adam狀態失敗: {e}")
        
        if self.rank == 0:
            status = "成功" if success else "失敗，已回退"
            print(f"=== 離開 {mode} L-BFGS 段 ({status}) ===")
            # 訊息提示
            if hasattr(self, 'config') and hasattr(self.config, 'training') and getattr(self.config.training, 'log_tips', True):
                if stop_reason == "timeout":
                    self._log_tip_once('lbfgs_timeout', "L-BFGS 段因 timeout 結束；建議 max_iter→20 或縮短 timeout_seconds。")
                elif stop_reason == "early_stop":
                    self._log_tip_once('lbfgs_early_stop', "L-BFGS 連續多次改善很小而早停；可調整 tolerance 或增加 cooldown 減少頻度。")
                elif stop_reason == "error":
                    self._log_tip_once('lbfgs_error', "L-BFGS 段出現錯誤；確保段內 FP32、可暫停 line search 或降低 max_iter。")
        
        return best_loss

    def solve_Adam(self, loss_func, num_epoch=1000, batchsize=None, scheduler=None, profiler=None, start_epoch=0):
        # 存储当前scheduler以便在optimizer重建时使用
        self.current_scheduler = scheduler
        self.current_scheduler_params = None
        # 記錄scheduler名稱（None 視為 Constant），供重建時判斷是否需要靜默跳過
        try:
            self.current_scheduler_name = type(scheduler).__name__ if scheduler is not None else 'Constant'
        except Exception:
            self.current_scheduler_name = 'Constant'
        if scheduler is not None:
            # 存储scheduler的类型和参数以便重建
            self.current_scheduler_params = {
                'class': type(scheduler),
                'T_max': getattr(scheduler, 'T_max', None),
                'eta_min': getattr(scheduler, 'eta_min', None),
                'milestones': getattr(scheduler, 'milestones', None),
                'gamma': getattr(scheduler, 'gamma', None),
                'last_epoch': getattr(scheduler, 'last_epoch', -1)
            }
            # 額外支援：CosineAnnealingWarmRestarts 與 SequentialLR（暖啟動）
            try:
                import torch as _torch
                # CosineAnnealingWarmRestarts 參數
                if isinstance(scheduler, _torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    self.current_scheduler_params.update({
                        'T_0': getattr(scheduler, 'T_0', None),
                        'T_mult': getattr(scheduler, 'T_mult', 1),
                        'eta_min': getattr(scheduler, 'eta_min', 0.0)
                    })
                # SequentialLR: 保存子scheduler配置
                if isinstance(scheduler, _torch.optim.lr_scheduler.SequentialLR):
                    sub_schedulers = getattr(scheduler, '_schedulers', [])
                    children = []
                    for sub in sub_schedulers:
                        entry = {'class': type(sub)}
                        if isinstance(sub, _torch.optim.lr_scheduler.LinearLR):
                            entry.update({
                                'start_factor': getattr(sub, 'start_factor', 1.0),
                                'end_factor': getattr(sub, 'end_factor', 1.0),
                                'total_iters': getattr(sub, 'total_iters', 0)
                            })
                        elif isinstance(sub, _torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                            entry.update({
                                'T_0': getattr(sub, 'T_0', None),
                                'T_mult': getattr(sub, 'T_mult', 1),
                                'eta_min': getattr(sub, 'eta_min', 0.0)
                            })
                        elif isinstance(sub, _torch.optim.lr_scheduler.CosineAnnealingLR):
                            entry.update({
                                'T_max': getattr(sub, 'T_max', None),
                                'eta_min': getattr(sub, 'eta_min', 0.0)
                            })
                        elif isinstance(sub, _torch.optim.lr_scheduler.MultiStepLR):
                            entry.update({
                                'milestones': list(getattr(sub, 'milestones', [])),
                                'gamma': getattr(sub, 'gamma', 0.1)
                            })
                        children.append(entry)
                    self.current_scheduler_params.update({
                        'sequential': True,
                        'children': children,
                        'milestones': list(getattr(scheduler, 'milestones', [])) or []
                    })
            except Exception:
                pass
            
            # Debug輸出確認scheduler參數
            if self.rank == 0:
                scheduler_name = type(scheduler).__name__
                if scheduler_name == 'CosineAnnealingLR':
                    print(f"🔧 Scheduler初始化: {scheduler_name}")
                    print(f"   - T_max: {scheduler.T_max}")
                    print(f"   - eta_min: {scheduler.eta_min:.2e}")
                    print(f"   - 初始lr: {self.opt.param_groups[0]['lr']:.2e}")
                elif scheduler_name == 'MultiStepLR':
                    print(f"🔧 Scheduler初始化: {scheduler_name}")
                    print(f"   - milestones: {scheduler.milestones}")
                    print(f"   - gamma: {scheduler.gamma}")
                elif scheduler_name == 'CosineAnnealingWarmRestarts':
                    print(f"🔧 Scheduler初始化: {scheduler_name}")
                    print(f"   - T_0: {getattr(scheduler,'T_0', None)}  T_mult: {getattr(scheduler,'T_mult', 1)}  eta_min: {getattr(scheduler,'eta_min', 0.0):.2e}")
                elif scheduler_name == 'SequentialLR':
                    print(f"🔧 Scheduler初始化: {scheduler_name} (包含warmup/SGDR)")
                else:
                    print(f"🔧 Scheduler初始化: {scheduler_name}")
        
        # 啟用初始凍結
        self.freeze_evm_net(0)
        
        # 使用完整數據進行訓練（不使用批次處理）
        actual_data_points = self.x_f.shape[0]
        
        # 記錄階段開始時間和啟動健康監控
        if self.rank == 0:
            self.stage_start_time = time.time()
            
            # 設置訓練開始時間（只在第一次調用時）
            if self.training_start_time is None:
                self.training_start_time = time.time()
                
            # 啟動健康監控

        
        if self.rank == 0:
            training_info = {
                "階段": self.current_stage,
                "訓練點總數": f"{self.N_f:,}",
                "實際GPU數據點": f"{actual_data_points:,}",
                "訓練模式": "全批次 (無DataLoader)",
                "總epochs": f"{num_epoch:,}",
                "DDP模式": "啟用" if self.world_size > 1 else "關閉",
                "數值精度": "Float32 (完整精度)"
            }
            self.logger.info("=== 訓練配置 (全批次) ===")
            for key, value in training_info.items():
                self.logger.info(f"{key}: {value}")
            
            # GPU記憶體信息
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
                self.logger.info(f"GPU記憶體 - 已分配: {memory_allocated:.2f}GB, 已保留: {memory_reserved:.2f}GB")
            self.logger.info("=" * 50)
        
        # 時間估算相關變數
        # 計時同步頻率（減少頻繁同步造成的停頓）
        timing_sync_interval = 1000
        try:
            if hasattr(self, 'config') and hasattr(self.config, 'system'):
                timing_sync_interval = int(getattr(self.config.system, 'timing_sync_interval', 1000))
        except Exception:
            timing_sync_interval = 1000
        estimate_frequency = max(500, timing_sync_interval)  # 與同步頻率對齊，避免頻繁估算
        
        # 滑窗與步數計數
        if not hasattr(self, 'global_step'):
            self.global_step = 0
        # 關鍵修復：每個新階段都重置stage_step，並記錄階段起始步數
        self.stage_step = start_epoch  # 從start_epoch開始，而不是0
        self.stage_start_step = start_epoch  # 記錄當前階段起始步數，用於相對步數計算
        
        from collections import deque
        if not hasattr(self, 'stage_loss_deque') or self.stage_step == 0:
            self.stage_loss_deque = deque(maxlen=20000)
        if not hasattr(self, 'last_strategy_step'):
            self.last_strategy_step = -999999
        
        for epoch_id in range(start_epoch, num_epoch):
            # 記錄epoch開始時間（僅在需要精確計時時同步GPU）
            if self.rank == 0:
                # 確定是否需要精確計時（可配置）
                need_precise_timing = (
                    epoch_id % timing_sync_interval == 0 or
                    epoch_id == 0 or                     # 首個epoch
                    epoch_id == num_epoch - 1 or         # 最後epoch
                    (epoch_id + 1) % timing_sync_interval == 0
                )
                
                if need_precise_timing and torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize(self.device)
                    except Exception:
                        pass
                self.epoch_start_time = time.time()
            
            # 修改後的動態凍結邏輯：EVM在epoch 1-9999保持凍結
            if epoch_id != 0 and epoch_id % 10000 == 0:
                self.defreeze_evm_net(epoch_id)
            if epoch_id > 10000 and (epoch_id - 1) % 10000 == 0:
                self.freeze_evm_net(epoch_id)

            # 清除上一個epoch的梯度
            self.opt.zero_grad(set_to_none=True)
            # 使用標準float32精度進行計算
            loss, losses = loss_func()
            
            # 損失值驗證和GPU記憶體檢查
            if not self.validate_loss_and_memory(loss, losses, epoch_id):
                self.logger.critical(f"Critical error at epoch {epoch_id}, stopping training...")
                return
            
            # 標準精度梯度回傳
            loss.backward()
            
            # 記錄損失值
            epoch_loss = loss.detach().item()
            epoch_losses = [
                losses[i] if isinstance(losses[i], (int, float)) else losses[i].detach().item()
                for i in range(len(losses))
            ]
            
            # 清理張量引用
            del loss
            
            # 梯度裁剪避免梯度爆炸
            torch.nn.utils.clip_grad_norm_(
                list(self.net.parameters()) + list(self.net_1.parameters()),
                max_norm=1.0
            )
            
            # 更新參數

            
            # 檢查DDP狀態並嘗試恢復
            max_retry_attempts = 3
            retry_count = 0
            
            while retry_count < max_retry_attempts:
                try:
                    self.opt.step()
                    break  # 成功執行，跳出重試循環
                    
                except RuntimeError as e:
                    retry_count += 1
                    error_msg = str(e)
                    
                    self.logger.ddp_error(error_msg, retry_count, max_retry_attempts)
                    
                    # 檢查特定的DDP錯誤類型
                    if any(keyword in error_msg for keyword in [
                        "INTERNAL ASSERT FAILED", 
                        "unmarked_param_indices",
                        "bucket_boundaries_",
                        "DDP bucket",
                        "find_unused_parameters"
                    ]):
                        if retry_count < max_retry_attempts:
                            self.logger.info("   Attempting DDP recovery...")
                            # 嘗試重建optimizer參數群組與重新同步
                            try:
                                self.rebuild_optimizer_groups()
                                self.opt.zero_grad(set_to_none=True)
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                            except Exception as recovery_e:
                                self.logger.error(f"   DDP recovery failed: {recovery_e}")
                        else:
                            self.logger.error(f"DDP recovery failed after {max_retry_attempts} attempts, skipping step...")
                            break
                    else:
                        # 非DDP相關錯誤，直接拋出
                        raise e
                        
                except Exception as e:
                    self.logger.error(f"Unexpected error during optimizer step: {e}")
                    raise e
            
            # Scheduler步進 - 必須在optimizer.step()之後
            # 優先使用傳入的scheduler，避免freeze/unfreeze後的scheduler失效問題
            # 使用當前重建後的scheduler優先，避免freeze/unfreeze後仍引用舊scheduler
            active_scheduler = self.current_scheduler if self.current_scheduler is not None else scheduler
            if active_scheduler:
                old_lr = self.opt.param_groups[0]['lr']
                active_scheduler.step()
                new_lr = self.opt.param_groups[0]['lr']
                
                # Debug輸出檢查scheduler是否正常工作（降頻）
                if self.rank == 0 and epoch_id % timing_sync_interval == 0:
                    scheduler_name = type(active_scheduler).__name__
                    print(f"🔧 {scheduler_name} step {epoch_id}: lr {old_lr:.6f} -> {new_lr:.6f}")
                
                # 關鍵：保存scheduler的完整狀態
                if self.current_scheduler_params is not None:
                    self.current_scheduler_params['last_epoch'] = active_scheduler.last_epoch
                    # 對於CosineAnnealingLR，額外保存當前的學習率狀態
                    if hasattr(active_scheduler, 'T_max'):
                        self.current_scheduler_params['current_lr'] = new_lr
            
            # 步數與滑窗
            self.global_step += 1
            self.stage_step += 1
            self.stage_loss_deque.append(epoch_loss)
            
            # 監測：梯度分佈與方向穩定性（每 N 步）
            try:
                monitor_interval = 1000
                try:
                    if hasattr(self, 'config') and hasattr(self.config, 'system'):
                        monitor_interval = int(getattr(self.config.system, 'monitor_interval', 1000))
                except Exception:
                    monitor_interval = 1000
                if monitor_interval <= 0:
                    monitor_interval = 1000
                if (self.stage_step % monitor_interval) == 0:
                    grad_norms = []
                    flat_list = []
                    for p in list(self.get_model(self.net).parameters()) + list(self.get_model(self.net_1).parameters()):
                        if p.grad is not None:
                            g = p.grad.detach()
                            grad_norms.append(g.norm().item())
                            # 僅在監測步才做 CPU 拷貝
                            flat_list.append(g.view(-1).float().cpu())
                    if grad_norms:
                        import numpy as _np
                        med = float(_np.median(grad_norms))
                        q1 = float(_np.percentile(grad_norms, 25))
                        q3 = float(_np.percentile(grad_norms, 75))
                        self.grad_median = med
                        self.grad_iqr = max(q3 - q1, 0.0)
                        if not hasattr(self, 'grad_baseline'):
                            self.grad_baseline = med
                        else:
                            if self.stage_step < 5000:
                                self.grad_baseline = 0.99 * self.grad_baseline + 0.01 * med
                        if flat_list:
                            g_flat = torch.cat(flat_list)
                            if hasattr(self, 'prev_grad_flat') and self.prev_grad_flat is not None:
                                denom = (g_flat.norm() * self.prev_grad_flat.norm()).item() + 1e-12
                                cos = float((g_flat @ self.prev_grad_flat).item() / denom)
                                self.grad_cos_ema = 0.9 * float(getattr(self, 'grad_cos_ema', 0.0)) + 0.1 * cos
                            self.prev_grad_flat = g_flat
            except Exception:
                pass

            # 監測：人工黏滯上限命中率（P95）（每 N 步）
            try:
                monitor_interval = 1000
                try:
                    if hasattr(self, 'config') and hasattr(self.config, 'system'):
                        monitor_interval = int(getattr(self.config.system, 'monitor_interval', 1000))
                except Exception:
                    monitor_interval = 1000
                if monitor_interval <= 0:
                    monitor_interval = 1000
                if (self.stage_step % monitor_interval) == 0:
                    if hasattr(self, 'vis_t_minus_gpu') and self.vis_t_minus_gpu is not None:
                        cap_val = float(self.beta) / float(self.Re) if self.beta is not None else (1.0 / float(self.Re))
                        if cap_val > 0:
                            sl = min(self.vis_t_minus_gpu.shape[0], 4096)
                            ratio = (self.vis_t_minus_gpu[:sl] / cap_val).clamp(max=1.0).detach().float().cpu()
                            self.vis_cap_p95 = float(torch.quantile(ratio.view(-1), 0.95).item())
            except Exception:
                pass
            # 分佈式L-BFGS觸發檢測
            trigger_lbfgs = self._check_distributed_lbfgs_trigger()
            # 提示：長時間未觸發 or 人工黏滯偏高
            try:
                cfg = getattr(self, 'config', None)
                lb = getattr(cfg.training, 'lbfgs', None) if cfg and hasattr(cfg, 'training') else None
                group_idx = self._stage_group_index()
                W_list = getattr(lb, 'trigger_window_per_stage', [5000, 7500, 10000]) if lb else [5000, 7500, 10000]
                W = int(W_list[min(group_idx, len(W_list)-1)])
                cooldown = int(getattr(lb, 'cooldown_steps', 5000)) if lb else 5000
                # 長時間未觸發
                if (self.stage_step - self.last_strategy_step) > (2 * W):
                    self._log_tip_once('not_trigger', f"L-BFGS 未觸發已超過 {2*W} 步；可降低 min_improve_pct 或縮短 cooldown ({cooldown}).")
                # 人工黏滯偏高
                if float(getattr(self, 'vis_cap_p95', 0.0)) > 0.7:
                    # 積累2000步以上再提示
                    if not hasattr(self, '_vis_high_steps'):
                        self._vis_high_steps = 0
                    self._vis_high_steps += 1
                    if self._vis_high_steps > 2000:
                        self._log_tip_once('vis_cap_high', "人工黏滯使用率偏高（P95>0.7）；建議將 last_layer_scale_evm 調至 0.05 或降低 α_evm。")
                else:
                    self._vis_high_steps = 0
            except Exception:
                pass
            if trigger_lbfgs:
                # 冷卻記錄
                self.prev_strategy_step = self.last_strategy_step
                self.last_strategy_step = self.stage_step
                # 使用配置參數啟動L-BFGS段
                self.train_with_lbfgs_segment()
                if self.rank == 0:
                    print("✅ 離開 L-BFGS 段，恢復 Adam")
                # 提示：過於頻繁（使用相對步數）
                try:
                    cfg = getattr(self, 'config', None)
                    lb = getattr(cfg.training, 'lbfgs', None) if cfg and hasattr(cfg, 'training') else None
                    cooldown = int(getattr(lb, 'cooldown_steps', 5000)) if lb else 5000
                    stage_start_step = getattr(self, 'stage_start_step', 0)
                    current_relative_step = self.last_strategy_step - stage_start_step
                    prev_relative_step = self.prev_strategy_step - stage_start_step
                    if (current_relative_step - prev_relative_step) < (2 * cooldown):
                        self._log_tip_once('too_frequent', f"L-BFGS 觸發過於頻繁；建議提高 min_improve_pct 或增大 cooldown（目前 {cooldown}）。")
                except Exception:
                    pass
                # 段後優化器與scheduler已在段內恢復
            # 時間追蹤和預估（只在rank 0執行；僅在需要精確計時時同步GPU）
            if self.rank == 0:
                # 確定是否需要精確計時（與開始時相同的邏輯）
                need_precise_timing = (
                    epoch_id % 100 == 0 or               # 每100 epochs進行時間預估
                    epoch_id == 0 or                     # 首個epoch
                    epoch_id == num_epoch - 1 or         # 最後epoch
                    (epoch_id + 1) % 1000 == 0           # console輸出時需要精確時間
                )
                
                if need_precise_timing and torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize(self.device)
                    except Exception:
                        pass
                epoch_end_time = time.time()
                epoch_time = epoch_end_time - self.epoch_start_time

            # 時間追蹤和預估（只在rank 0執行；僅在需要精確計時時同步GPU）
            if self.rank == 0:
                # 健康檢查條件判斷 (需要提前以確定是否需要精確計時)
                should_monitor = False
                if epoch_id <= 100 and epoch_id % 10 == 0:  # 前100個epoch密集監測
                    should_monitor = True
                elif epoch_id in [300000, 600000, 900000, 1200000, 1500000]:  # 階段轉換點
                    should_monitor = True
                elif epoch_id > 1000 and epoch_id % 10000 == 0:  # 定期檢查
                    should_monitor = True
                
                # 確定是否需要精確計時
                need_precise_timing = (
                    epoch_id % 100 == 0 or               # 每100 epochs進行時間預估
                    epoch_id == 0 or                     # 首個epoch
                    epoch_id == num_epoch - 1 or         # 最後epoch
                    (epoch_id + 1) % 1000 == 0 or        # console輸出時需要精確時間
                    should_monitor                       # 健康檢查時
                )
                
                if need_precise_timing and torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize(self.device)
                    except Exception:
                        pass
                epoch_end_time = time.time()
                epoch_time = epoch_end_time - self.epoch_start_time
                
                # 限制epoch_times大小以防記憶體洩漏
                self.epoch_times.append(epoch_time)
                if len(self.epoch_times) > 1000:  # 只保留最近1000個epoch的時間
                    self.epoch_times = self.epoch_times[-500:]  # 刪除一半舊數據，保持高效
                
                # 記錄到TensorBoard（降頻寫入）
                if self.tb_writer is not None and (epoch_id % max(1, self.tb_interval) == 0 or epoch_id in (0, num_epoch-1)):
                    global_step = self.global_step_offset + epoch_id
                    
                    self.safe_tensorboard_log('Loss/Total', epoch_loss, global_step)
                    self.safe_tensorboard_log('Loss/Equation_Combined', epoch_losses[0], global_step)
                    self.safe_tensorboard_log('Loss/Boundary', epoch_losses[1], global_step)
                    self.safe_tensorboard_log('Loss/Supervised', epoch_losses[2], global_step)
                    self.safe_tensorboard_log('Loss/Equation_NS_X', epoch_losses[3], global_step)
                    self.safe_tensorboard_log('Loss/Equation_NS_Y', epoch_losses[4], global_step)
                    self.safe_tensorboard_log('Loss/Equation_Continuity', epoch_losses[5], global_step)
                    self.safe_tensorboard_log('Loss/Equation_EntropyResidual', epoch_losses[6], global_step)
                    self.safe_tensorboard_log('Training/LearningRate', self.opt.param_groups[0]['lr'], global_step)
                    # 記錄當前 weight decay（主參數組）
                    try:
                        wd_val = float(getattr(self, 'current_weight_decay', 0.0))
                        if wd_val <= 0.0:
                            # 若 current_weight_decay 未設或為0，嘗試從參數組檢測
                            for _pg in self.opt.param_groups:
                                if _pg.get('weight_decay', 0.0) > 0:
                                    wd_val = float(_pg['weight_decay'])
                                    break
                        self.safe_tensorboard_log('Training/WeightDecay', wd_val, global_step)
                    except Exception:
                        pass
                    self.safe_tensorboard_log('Training/EpochTime', epoch_time, global_step)
                    self.safe_tensorboard_log('Training/Alpha_EVM', self.alpha_evm, global_step)
                    
                    # GPU記憶體使用（降頻）
                    if torch.cuda.is_available():
                        memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                        self.safe_tensorboard_log('System/GPU_Memory_GB', memory_allocated, global_step)
                
                # 健康檢查（降頻）
                if should_monitor and (epoch_id % (10 * timing_sync_interval) == 0 or epoch_id in (0, num_epoch-1)):
                    self.logger.info(f"🔍 Enhanced Health Check - Epoch {epoch_id}")
                    self.check_tanh_saturation(epoch_id)

                # 記憶體監控 (每50個epoch檢查一次)

            
            # 每1000個epoch輸出一次訓練狀況，首個epoch也要輸出
            if self.rank == 0 and (epoch_id == 0 or (epoch_id + 1) % 1000 == 0 or epoch_id == num_epoch - 1):
                self.print_log_full_batch_with_time_estimate(epoch_loss, epoch_losses, epoch_id, num_epoch, actual_data_points)
                
                # 每1000個epoch輸出健康和記憶體報告
                
                # 每1000個epoch檢查tanh飽和度
                self.check_tanh_saturation(epoch_id)


            # Save checkpoint
            if self.rank == 0 and (epoch_id > 0 and epoch_id % self.checkpoint_freq == 0 or epoch_id == num_epoch - 1):
                self.save_checkpoint(epoch_id, self.opt)


        # 階段結束後更新global step offset
        if self.rank == 0:
            self.global_step_offset += num_epoch
            
            # 階段結束時的最終清理和統計

    
    def freeze_evm_net(self, epoch_id):
        """
        凍結EVM網路參數 - 保持scheduler連續性
        """
        if self.rank == 0:
            print(f"[Epoch {epoch_id}] Freezing EVM network parameters (保持scheduler連續性)")
        
        # 凍結net_1的所有參數
        for param in self.net_1.parameters():
            param.requires_grad = False
        # 重建 AdamW 以移除凍結參數
        try:
            self.rebuild_after_structure_change()
        except Exception:
            pass
        if self.rank == 0 and self.opt is not None:
            total_trainable = sum(p.numel() for g in self.opt.param_groups for p in g['params'])
            print(f"  Active parameters (net only): {total_trainable}")

    def defreeze_evm_net(self, epoch_id, rebuild: bool = True):
        """
        解凍EVM網路參數
        rebuild=True 時會呼叫 rebuild_after_structure_change 以重建 AdamW / scheduler
        """
        if self.rank == 0:
            print(f"[Epoch {epoch_id}] Unfreezing EVM network parameters (保持scheduler連續性)")
        # 解凍 net_1 參數
        for param in self.net_1.parameters():
            param.requires_grad = True
        if rebuild:
            try:
                self.rebuild_after_structure_change()
            except Exception:
                pass
        if self.rank == 0 and self.opt is not None:
            total_trainable = sum(p.numel() for p in list(self.get_model(self.net).parameters()) if p.requires_grad) + \
                               sum(p.numel() for p in list(self.get_model(self.net_1).parameters()) if p.requires_grad)
            print(f"  Active parameters (net + net_1): {total_trainable}")

    def _rebuild_scheduler(self):
        """重建scheduler以绑定新的optimizer，確保學習率連續性"""
        if self.current_scheduler is None or self.current_scheduler_params is None:
            # 若為 Constant（未配置scheduler），靜默跳過並保持目前lr，不視為警告
            if getattr(self, 'current_scheduler_name', 'Constant') == 'Constant':
                if self.rank == 0:
                    print("  ℹ️ 跳過scheduler重建：未配置scheduler（Constant），保持lr連續性")
                return
            else:
                if self.rank == 0:
                    print("  Warning: 無法重建scheduler - 缺少參數")
                return
            
        try:
            # 保存當前學習率以確保連續性
            current_lr = self.opt.param_groups[0]['lr']
            
            # 确保optimizer参数组有initial_lr
            for group in self.opt.param_groups:
                if 'initial_lr' not in group:
                    group['initial_lr'] = group['lr']
            
            scheduler_class = self.current_scheduler_params['class']
            
            # 關鍵修復：重建時不改變當前學習率，保持原有進度
            if scheduler_class.__name__ == 'CosineAnnealingLR':
                # 創建新scheduler，但立即設置到當前學習率
                self.current_scheduler = scheduler_class(
                    self.opt, 
                    T_max=self.current_scheduler_params['T_max'],
                    eta_min=self.current_scheduler_params['eta_min'],
                    last_epoch=-1  # 讓scheduler從初始狀態開始
                )
                
                # 手動設置學習率保持連續性
                for group in self.opt.param_groups:
                    group['lr'] = current_lr
                    
                # 更新scheduler的last_epoch以匹配當前進度
                stage_epoch = self.stage_step if hasattr(self, 'stage_step') else 0
                self.current_scheduler.last_epoch = stage_epoch - 1
                
                if self.rank == 0:
                    print(f"  ✅ 重建CosineAnnealingLR: 保持lr={current_lr:.6f}, stage_epoch={stage_epoch}")
                    
            elif scheduler_class.__name__ == 'MultiStepLR':
                # 創建新scheduler並設置當前學習率
                self.current_scheduler = scheduler_class(
                    self.opt,
                    milestones=self.current_scheduler_params['milestones'],
                    gamma=self.current_scheduler_params['gamma'],
                    last_epoch=-1
                )
                
                # 手動設置學習率保持連續性
                for group in self.opt.param_groups:
                    group['lr'] = current_lr
                    
                stage_epoch = self.stage_step if hasattr(self, 'stage_step') else 0
                self.current_scheduler.last_epoch = stage_epoch - 1
                
                if self.rank == 0:
                    print(f"  ✅ 重建MultiStepLR: 保持lr={current_lr:.6f}, stage_epoch={stage_epoch}")
            elif scheduler_class.__name__ == 'CosineAnnealingWarmRestarts':
                # 重建CAWR，保持學習率連續性
                T_0 = self.current_scheduler_params.get('T_0', 1000)
                T_mult = self.current_scheduler_params.get('T_mult', 1)
                eta_min = self.current_scheduler_params.get('eta_min', 0.0)
                self.current_scheduler = scheduler_class(
                    self.opt,
                    T_0=T_0,
                    T_mult=T_mult,
                    eta_min=eta_min
                )
                # 手動設置學習率保持連續性
                for group in self.opt.param_groups:
                    group['lr'] = current_lr
                stage_epoch = self.stage_step if hasattr(self, 'stage_step') else 0
                self.current_scheduler.last_epoch = stage_epoch - 1
                if self.rank == 0:
                    print(f"  ✅ 重建CosineAnnealingWarmRestarts: 保持lr={current_lr:.6f}, stage_epoch={stage_epoch}")
            elif scheduler_class.__name__ == 'SequentialLR':
                # 重建順序調度器：支援 LinearLR -> CosineAnnealingWarmRestarts 組合
                children = self.current_scheduler_params.get('children', [])
                rebuilt = []
                import torch as _torch
                for ch in children:
                    cls = ch.get('class')
                    name = cls.__name__ if hasattr(cls, '__name__') else str(cls)
                    if name == 'LinearLR':
                        rebuilt.append(_torch.optim.lr_scheduler.LinearLR(
                            self.opt,
                            start_factor=ch.get('start_factor', 1.0),
                            end_factor=ch.get('end_factor', 1.0),
                            total_iters=ch.get('total_iters', 0)
                        ))
                    elif name == 'CosineAnnealingWarmRestarts':
                        rebuilt.append(_torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                            self.opt,
                            T_0=ch.get('T_0', 1000),
                            T_mult=ch.get('T_mult', 1),
                            eta_min=ch.get('eta_min', 0.0)
                        ))
                    elif name == 'CosineAnnealingLR':
                        rebuilt.append(_torch.optim.lr_scheduler.CosineAnnealingLR(
                            self.opt,
                            T_max=ch.get('T_max', 1000),
                            eta_min=ch.get('eta_min', 0.0)
                        ))
                    elif name == 'MultiStepLR':
                        rebuilt.append(_torch.optim.lr_scheduler.MultiStepLR(
                            self.opt,
                            milestones=ch.get('milestones', []),
                            gamma=ch.get('gamma', 0.1)
                        ))
                    else:
                        # 不認識的子scheduler，回退為恆定
                        rebuilt.append(_torch.optim.lr_scheduler.ConstantLR(self.opt, factor=1.0, total_iters=1))
                # 修正milestones邏輯：確保符合PyTorch SequentialLR要求
                ms = self.current_scheduler_params.get('milestones', [])
                if len(ms) == 0 and len(rebuilt) >= 2:
                    # SGDR組合：warmup + main scheduler，需要正確的milestone
                    if hasattr(rebuilt[0], 'total_iters'):
                        ms = [rebuilt[0].total_iters]
                    else:
                        # 回退預設值：假設warmup佔前10%
                        total_epochs = getattr(self, 'epochs_per_stage', 300000)
                        ms = [int(0.1 * total_epochs)]
                elif len(ms) == 0:
                    # 單scheduler情況，不需要milestone
                    ms = []
                
                # 驗證milestones數量：len(schedulers) = len(milestones) + 1
                if len(rebuilt) != len(ms) + 1:
                    if self.rank == 0:
                        print(f"  ⚠️ Milestone調整: schedulers={len(rebuilt)}, milestones={len(ms)} -> {len(rebuilt)-1}")
                    ms = ms[:len(rebuilt)-1] if len(ms) >= len(rebuilt) else ms + [1000] * (len(rebuilt) - len(ms) - 1)
                
                self.current_scheduler = _torch.optim.lr_scheduler.SequentialLR(
                    self.opt,
                    schedulers=rebuilt,
                    milestones=ms
                )
                # 保持當前學習率與進度
                for group in self.opt.param_groups:
                    group['lr'] = current_lr
                stage_epoch = self.stage_step if hasattr(self, 'stage_step') else 0
                self.current_scheduler.last_epoch = stage_epoch - 1
                if self.rank == 0:
                    print(f"  ✅ 重建SequentialLR: 保持lr={current_lr:.6f}, stage_epoch={stage_epoch}")
            else:
                # 对于其他类型的scheduler，尝试通用重建
                self.current_scheduler = scheduler_class(self.opt)
                
                # 保持當前學習率
                for group in self.opt.param_groups:
                    group['lr'] = current_lr
                    
                if self.rank == 0:
                    print(f"  ✅ 重建{scheduler_class.__name__}: 保持lr={current_lr:.6f}")
                
        except Exception as e:
            if self.rank == 0:
                print(f"  ❌ Scheduler重建失敗: {e}")
            self.current_scheduler = None

    def safe_tensorboard_log(self, tag, value, global_step):
        """安全的TensorBoard記錄函數with錯誤處理"""
        if self.tb_writer is not None:
            try:
                # 檢查值是否有效
                if value is None or not isinstance(value, (int, float)):
                    self.logger.warning(f"Invalid value for TensorBoard tag '{tag}': {value}")
                    return
                
                # 檢查是否為NaN或Inf
                if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                    self.logger.warning(f"NaN/Inf value detected for TensorBoard tag '{tag}': {value}")
                    return
                
                # 記錄到TensorBoard
                self.tb_writer.add_scalar(tag, value, global_step)
                
            except Exception as e:
                self.logger.warning(f"TensorBoard logging error for tag '{tag}': {e}")

    def validate_loss_and_memory(self, loss, losses, epoch_id):
        """損失值驗證和GPU記憶體檢查"""
        try:
            # 檢查主損失值
            loss_value = loss.detach().item() if hasattr(loss, 'detach') else loss
            
            if math.isnan(loss_value) or math.isinf(loss_value):
                self.logger.loss_validation_error(epoch_id, loss_value, "main")
                return False
            
            if loss_value > 1e10:  # 損失值過大
                self.logger.warning(f"Extremely large loss detected at epoch {epoch_id}: {loss_value}")
            
            # 檢查各個損失組件
            for i, component_loss in enumerate(losses):
                comp_value = component_loss.detach().item() if hasattr(component_loss, 'detach') else component_loss
                
                if math.isnan(comp_value) or math.isinf(comp_value):
                    self.logger.loss_validation_error(epoch_id, comp_value, f"component_{i}")
                    return False
            
            # GPU記憶體檢查
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
                
                # 檢查記憶體使用是否過高（超過可用記憶體的90%）
                total_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
                if memory_allocated > total_memory * 0.9:
                    self.logger.memory_warning(memory_allocated, total_memory)
                    self.logger.info("   Attempting memory cleanup...")
                    
                    # 嘗試清理GPU記憶體
                    torch.cuda.empty_cache()
                    
                    # 再次檢查
                    memory_allocated_after = torch.cuda.memory_allocated(self.device) / 1024**3
                    if memory_allocated_after > total_memory * 0.95:
                        self.logger.critical(f"Critical GPU memory usage: {memory_allocated_after:.2f}GB / {total_memory:.2f}GB")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Error in loss/memory validation at epoch {epoch_id}: {e}")
            return True  # 驗證錯誤時繼續訓練

    def rebuild_optimizer_groups(self):
        """統一重建 AdamW 參數群組 (DDP 恢復 / 結構變更)。"""
        try:
            if self.opt is None or not hasattr(self, 'current_weight_decay'):
                # 若尚未初始化，使用預設 lr / wd
                self.build_adamw_optimizer(1e-3, getattr(self, 'current_weight_decay', 0.0))
                return
            current_lr = self.opt.param_groups[0].get('lr', 1e-3)
            wd = getattr(self, 'current_weight_decay', 0.0)
            self.build_adamw_optimizer(current_lr, wd)
            if self.rank == 0:
                print(f"   ✅ DDP恢復: 重新構建 AdamW (lr={current_lr:.2e}, wd={wd})")
        except Exception as e:
            if self.rank == 0:
                print(f"   ❌ DDP恢復重建失敗: {e}")
            raise e

    def print_log_full_batch_with_time_estimate(self, loss, losses, epoch_id, num_epoch, data_points):
        """打印訓練日誌包含詳細時間預估和收斂分析"""
        current_lr = self.opt.param_groups[0]['lr']
        
        # 計算時間統計
        if len(self.epoch_times) > 10:  # 至少需要10個epoch來計算可靠的預估
            # 使用最近50個epoch的平均時間，更準確反映當前速度
            recent_epochs = min(50, len(self.epoch_times))
            avg_epoch_time = np.mean(self.epoch_times[-recent_epochs:])
            
            # 預估剩餘時間
            remaining_epochs = num_epoch - (epoch_id + 1)
            estimated_remaining_time = remaining_epochs * avg_epoch_time
            
            # 計算階段總時間預估
            stage_elapsed = time.time() - self.stage_start_time
            stage_progress = (epoch_id + 1) / num_epoch
            stage_total_estimated = stage_elapsed / stage_progress if stage_progress > 0 else 0
            stage_eta = stage_total_estimated - stage_elapsed
            
            # 計算整個訓練的進度（如果是多階段訓練）
            if hasattr(self, 'training_start_time') and self.training_start_time:
                total_training_time = time.time() - self.training_start_time
            else:
                total_training_time = stage_elapsed
            
            # 計算epoch處理速度
            epochs_per_minute = 60.0 / avg_epoch_time if avg_epoch_time > 0 else 0
            
            # 損失收斂分析
            convergence_info = self._analyze_convergence_trend(losses)
            
            # 格式化時間顯示
            def format_time(seconds):
                if seconds < 60:
                    return f"{seconds:.1f}s"
                elif seconds < 3600:
                    return f"{seconds//60:.0f}m {seconds%60:.0f}s"
                elif seconds < 86400:
                    hours = seconds // 3600
                    minutes = (seconds % 3600) // 60
                    return f"{hours:.0f}h {minutes:.0f}m"
                else:
                    days = seconds // 86400
                    hours = (seconds % 86400) // 3600
                    return f"{days:.0f}d {hours:.0f}h"
            
            # 顯示詳細訓練報告
            print(f"\n{'='*100}")
            print(f"🔥 {self.current_stage} | 訓練進度報告")
            print(f"{'='*100}")
            
            # 進度信息
            progress_bar_length = 40
            filled_length = int(progress_bar_length * (epoch_id + 1) / num_epoch)
            bar = '█' * filled_length + '-' * (progress_bar_length - filled_length)
            
            print(f"📊 進度: [{bar}] {(epoch_id + 1)/num_epoch*100:.1f}%")
            print(f"   Epoch: {epoch_id + 1:,} / {num_epoch:,}")
            print(f"   資料點: {data_points:,} | 學習率: {current_lr:.2e}")
            
            # 損失信息
            print(f"\n📈 損失狀況:")
            print(f"   總損失:   {loss:.3e} {convergence_info['trend_symbol']}")
            print(f"   方程總損失: {losses[0]:.3e}")
            print(f"   監督損失: {losses[2]:.3e}")
            print(f"   邊界損失: {losses[1]:.3e}")
            print(f"   Navier-Stokes X損失: {losses[3]:.3e}")
            print(f"   Navier-Stokes Y損失: {losses[4]:.3e}")
            print(f"   連續性方程損失: {losses[5]:.3e}")
            print(f"   熵殘差損失: {losses[6]:.3e}")
            print(f"   收斂趨勢: {convergence_info['description']}")
            
            # 時間分析
            print(f"\n⏰ 時間分析:")
            print(f"   單epoch平均: {avg_epoch_time:.2f}s ({epochs_per_minute:.1f} epochs/min)")
            print(f"   階段已耗時: {format_time(stage_elapsed)}")
            print(f"   階段預估剩餘: {format_time(stage_eta)}")
            print(f"   階段總預估: {format_time(stage_total_estimated)}")
            print(f"   累計訓練時間: {format_time(total_training_time)}")
            
            # 系統狀態
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
                total_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
                memory_usage_percent = (memory_allocated / total_memory) * 100
                
                memory_status = "🟢 正常" if memory_usage_percent < 70 else "🟡 中等" if memory_usage_percent < 85 else "🔴 高"
                
                print(f"\n💾 系統狀態:")
                print(f"   GPU記憶體: {memory_allocated:.2f}GB / {total_memory:.2f}GB ({memory_usage_percent:.1f}%) {memory_status}")
                print(f"   保留記憶體: {memory_reserved:.2f}GB")
            
            # 訓練效率指標
            data_points_per_second = data_points / avg_epoch_time if avg_epoch_time > 0 else 0
            print(f"\n🚀 效率指標:")
            print(f"   資料處理速度: {data_points_per_second:,.0f} points/sec")
            
            # 物理參數診斷 - 計算等效雷諾數
            vis_t_mean = getattr(self, 'vis_t', torch.tensor(0.0)).mean().item()
            base_visc = 1.0/self.Re
            Re_eff = 1.0 / (base_visc + vis_t_mean) if vis_t_mean > 0 else self.Re
            vis_ratio = vis_t_mean / base_visc if base_visc > 0 else 0
            
            print(f"\n🔬 物理參數診斷:")
            print(f"   目標雷諾數: {self.Re}")
            print(f"   等效雷諾數: {Re_eff:.1f}")
            print(f"   Alpha EVM: {self.alpha_evm:.4f}")
            print(f"   EVM放大倍數: {vis_ratio:.2f}x")
            if Re_eff < 1000:
                print(f"   ⚠️  警告: Re_eff過低可能導致Couette流!")
            
            print(f"{'='*100}\n")
            
        else:
            # 初始幾個epoch，資訊較少
            print(f"\n{'='*80}")
            print(f"🔥 {self.current_stage} - 初始化階段")
            print(f"   Epoch: {epoch_id + 1:,} / {num_epoch:,}")
            print(f"   學習率: {current_lr:.2e} | 資料點: {data_points:,}")
            print(f"   損失 - 總: {loss:.3e} | 方程: {losses[0]:.3e} | 監督: {losses[2]:.3e} | 邊界: {losses[1]:.3e}")
            
            # 物理參數診斷 - 計算等效雷諾數
            vis_t_mean = getattr(self, 'vis_t', torch.tensor(0.0)).mean().item()
            base_visc = 1.0/self.Re
            Re_eff = 1.0 / (base_visc + vis_t_mean) if vis_t_mean > 0 else self.Re
            vis_ratio = vis_t_mean / base_visc if base_visc > 0 else 0
            
            print(f"   🔬 Re_eff: {Re_eff:.1f} | α_EVM: {self.alpha_evm:.4f} | EVM: {vis_ratio:.1f}x")
            if Re_eff < 1000:
                print(f"   ⚠️  警告: Re_eff={Re_eff:.1f} 過低，可能導致Couette流!")
            
            print(f"   (時間預估將在第10個epoch後提供)")
            print(f"{'='*80}\n")

    def _analyze_convergence_trend(self, current_losses):
        """分析損失收斂趨勢"""
        # 如果歷史數據不足，返回默認信息
        if not hasattr(self, 'loss_history'):
            self.loss_history = []
        
        # 記錄當前損失
        current_total_loss = current_losses[0] + current_losses[1]
        self.loss_history.append(current_total_loss)
        
        # 保持最近100個損失記錄
        if len(self.loss_history) > 100:
            self.loss_history = self.loss_history[-100:]
        
        if len(self.loss_history) < 10:
            return {"trend_symbol": "📊", "description": "收集數據中..."}
        
        # 分析最近的趨勢
        recent_losses = self.loss_history[-10:]
        earlier_losses = self.loss_history[-20:-10] if len(self.loss_history) >= 20 else self.loss_history[:-10]
        
        if len(earlier_losses) > 0:
            recent_avg = np.mean(recent_losses)
            earlier_avg = np.mean(earlier_losses)
            
            improvement_ratio = (earlier_avg - recent_avg) / earlier_avg if earlier_avg > 0 else 0
            
            if improvement_ratio > 0.1:
                return {"trend_symbol": "📉", "description": "快速收斂中"}
            elif improvement_ratio > 0.01:
                return {"trend_symbol": "📊", "description": "穩定收斂中"}
            elif improvement_ratio > -0.01:
                return {"trend_symbol": "➡️", "description": "緩慢收斂/平穩"}
            else:
                return {"trend_symbol": "📈", "description": "可能發散，需注意"}
        
        return {"trend_symbol": "📊", "description": "趨勢分析中..."}

    def print_log_full_batch(self, loss, losses, epoch_id, num_epoch, data_points):
        current_lr = self.opt.param_groups[0]['lr']
        print('current lr is {}'.format(current_lr))
        print('epoch/num_epoch: {:6d} / {:d} data_points: {:d} avg_loss[Adam]: {:.3e} avg_eq_combined_loss: {:.3e} avg_bc_loss: {:.3e} avg_sup_loss: {:.3e} avg_eq1_loss: {:.3e} avg_eq2_loss: {:.3e} avg_eq3_loss: {:.3e} avg_eq4_loss: {:.3e}'.format(
            epoch_id + 1, num_epoch, data_points, loss, losses[0], losses[1], losses[2], losses[3], losses[4], losses[5], losses[6]))

    def print_log_batch(self, loss, losses, epoch_id, num_epoch, batch_size, steps_per_epoch):
        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        coverage_percent = 100.0  # 循環覆蓋確保100%覆蓋
        print("current lr is {}".format(get_lr(self.opt)))
        print("epoch/num_epoch: ", epoch_id + 1, "/", num_epoch,
              "batch_size:", batch_size,
              "steps/epoch:", steps_per_epoch,
              "coverage: {:.1f}%".format(coverage_percent),
              "avg_loss[Adam]: %.3e" %(loss),
              "avg_eq_combined_loss: %.3e" %(losses[0] if len(losses) > 0 else 0),
              "avg_bc_loss: %.3e" %(losses[1] if len(losses) > 1 else 0),
              "avg_sup_loss: %.3e" %(losses[2] if len(losses) > 2 else 0),
              "avg_eq1_loss: %.3e" %(losses[3] if len(losses) > 3 else 0),
              "avg_eq2_loss: %.3e" %(losses[4] if len(losses) > 4 else 0),
              "avg_eq3_loss: %.3e" %(losses[5] if len(losses) > 5 else 0),
              "avg_eq4_loss: %.3e" %(losses[6] if len(losses) > 6 else 0))

    def print_log(self, loss, losses, epoch_id, num_epoch):
        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        print("current lr is {}".format(get_lr(self.opt)))
        print("epoch/num_epoch: ", epoch_id + 1, "/", num_epoch,
              "loss[Adam]: %.3e" %(loss.detach().cpu().item()),
              "eq_combined_loss: %.3e" %(losses[0]),
              "bc_loss: %.3e" %(losses[1]),
              "sup_loss: %.3e" %(losses[2]),
              "eq1_loss: %.3e" %(losses[3]),
              "eq2_loss: %.3e" %(losses[4]),
              "eq3_loss: %.3e" %(losses[5]),
              "eq4_loss: %.3e" %(losses[6]))

    def evaluate(self, x, y, u, v, p):
        """ testing all points in the domain """
        x_test = x.reshape(-1,1)
        y_test = y.reshape(-1,1)
        u_test = u.reshape(-1,1)
        v_test = v.reshape(-1,1)
        p_test = p.reshape(-1,1)

        x_test = torch.tensor(x_test).float().to(self.device)
        y_test = torch.tensor(y_test).float().to(self.device)
        u_pred, v_pred, p_pred, _= self.neural_net_u(x_test, y_test)
        u_pred = u_pred.detach().cpu().numpy().reshape(-1,1)
        v_pred = v_pred.detach().cpu().numpy().reshape(-1,1)
        p_pred = p_pred.detach().cpu().numpy().reshape(-1,1)
        
        mask_p = ~np.isnan(p_test)
        # Error
        error_u = 100*np.linalg.norm(u_test-u_pred,2)/np.linalg.norm(u_test,2)
        error_v = 100*np.linalg.norm(v_test-v_pred,2)/np.linalg.norm(v_test,2)
        error_p = 100*np.linalg.norm(p_test[mask_p]-p_pred[mask_p], 2) / np.linalg.norm(p_test[mask_p], 2)
        if self.rank == 0:
            print('------------------------')
            print('Error u: %.2f %%' % (error_u))
            print('Error v: %.2f %%' % (error_v))
            print('Error p: %.2f %%' % (error_p))

    def test(self, x, y, u, v, p, loop=None, custom_save_dir=None):
        """ testing all points in the domain """
        x_test = x.reshape(-1,1)
        y_test = y.reshape(-1,1)
        u_test = u.reshape(-1,1)
        v_test = v.reshape(-1,1)
        p_test = p.reshape(-1,1)
        # Prediction
        x_test = torch.tensor(x_test).float().to(self.device)
        y_test = torch.tensor(y_test).float().to(self.device)
        u_pred, v_pred, p_pred, e_pred= self.neural_net_u(x_test, y_test)
        u_pred = u_pred.detach().cpu().numpy().reshape(-1,1)
        v_pred = v_pred.detach().cpu().numpy().reshape(-1,1)
        p_pred = p_pred.detach().cpu().numpy().reshape(-1,1)
        e_pred = e_pred.detach().cpu().numpy().reshape(-1,1)
        
        mask_p = ~np.isnan(p_test)
        # Error
        error_u = 100*np.linalg.norm(u_test-u_pred,2)/np.linalg.norm(u_test,2)
        error_v = 100*np.linalg.norm(v_test-v_pred,2)/np.linalg.norm(v_test,2)
        error_p = 100*np.linalg.norm(p_test[mask_p]-p_pred[mask_p], 2) / np.linalg.norm(p_test[mask_p], 2)
        if self.rank == 0:
            print('------------------------')
            print('Error u: %.3f %%' % (error_u))
            print('Error v: %.3f %%' % (error_v))
            print('Error p: %.3f %%' % (error_p))
            print('------------------------')

            u_pred = u_pred.reshape(257,257)
            v_pred = v_pred.reshape(257,257)
            p_pred = p_pred.reshape(257,257)
            e_pred = e_pred.reshape(257,257)

            Re_folder = 'Re'+str(self.Re)
            NNsize = str(self.layers) + 'x' + str(self.hidden_size) + '_Nf'+str(np.int32(self.N_f/1000)) + 'k'
            lambdas = 'lamB'+str(self.alpha_b) + '_alpha'+str(self.alpha_evm) + str(self.current_stage)
            
            if custom_save_dir:
                # 使用自定義保存目錄
                relative_path = custom_save_dir
                filename = f'test_result_epoch_{loop:07d}.mat'  # 7位數字，方便排序
            else:
                # 使用原來的邏輯
                # 從config.py讀取基礎路徑
                try:
                    from config import RESULTS_PATH
                    base_path = RESULTS_PATH
                except ImportError:
                    base_path = 'results'

                relative_path = os.path.join(base_path, Re_folder, f"{NNsize}_{lambdas}")
                filename = f'cavity_result_loop_{loop}.mat'

            if not os.path.exists(relative_path):
                os.makedirs(relative_path, exist_ok=True)

            file_path = os.path.join(relative_path, filename)

            scipy.io.savemat(file_path,
                        {'U_pred':u_pred,
                         'V_pred':v_pred,
                         'P_pred':p_pred,
                         'E_pred':e_pred,
                         'error_u':error_u,
                         'error_v':error_v,
                         'error_p':error_p,
                         'lam_bcs':self.alpha_b,
                         'lam_equ':self.alpha_e,
                         'global_epoch':loop,  # 添加全局epoch信息
                         'stage_info': getattr(self, 'current_stage', 'unknown')})  # 添加stage信息

    def save(self, filename, directory=None, N_HLayer=None, N_neu=None, N_f=None):
        Re_folder = 'Re'+str(self.Re)
        NNsize = str(N_HLayer) + 'x' + str(N_neu) + '_Nf'+str(np.int32(N_f/1000)) + 'k'
        lambdas = 'lamB'+str(self.alpha_b) + '_alpha'+str(self.alpha_evm) + str(self.current_stage)

        relative_path = '/results/' +  Re_folder + '/' + NNsize + '_' + lambdas + '/'

        if not directory:
            directory = os.getcwd()
        save_results_to = directory + relative_path
        if not os.path.exists(save_results_to):
            os.makedirs(save_results_to)

        # Save model state dict without DDP wrapper
        torch.save(self.get_model(self.net).state_dict(), save_results_to+filename)
        torch.save(self.get_model(self.net_1).state_dict(), save_results_to+filename+'_evm')

    def divergence(self, x_star, y_star):
        (self.eq1_pred, self.eq2_pred,
         self.eq3_pred, self.eq4_pred) = self.neural_net_equations(x_star, y_star)
        div = self.eq3_pred
        return div
