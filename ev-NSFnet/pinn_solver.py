# Copyright (c) 2023 scien42.tech, Se42 Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
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
from typing import Dict, List, Set, Optional, Union, Callable
import warnings
import time
import datetime
from logger import LoggerFactory, PINNLogger
from health_monitor import TrainingHealthMonitor, HealthThresholds
from memory_manager import TrainingMemoryManager



# 抑制 PyTorch 分散式訓練的 autograd 警告
warnings.filterwarnings("ignore", message=".*c10d::allreduce_.*autograd kernel.*")

class PysicsInformedNeuralNetwork:
    # Initialize the class
    def __init__(self,
                 opt=None,
                 Re = 1000,
                 layers=6,
                 layers_1=6,
                 hidden_size=80,
                 hidden_size_1=20,
                 N_f = 100000,
                 batch_size = None,
                 alpha_evm=0.03,
                 learning_rate=0.001,
                 weight_decay=0.9,
                 outlet_weight=1,
                 bc_weight=10,
                 eq_weight=1,
                 ic_weight=0.1,
                 num_ins=2,
                 num_outs=3,
                 num_outs_1=1,
                 supervised_data_weight=1,
                 net_params=None,
                 net_params_1=None,
                 checkpoint_freq=2000):

        # Initialize distributed training
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))

        # Set device for current process
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.local_rank}')
            try:
                torch.cuda.set_device(self.local_rank)
            except (RuntimeError, AttributeError):
                # 處理CUDA設備設置失敗的情況
                self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                self.logger.warning(f"Failed to set CUDA device {self.local_rank}, using {self.device}")
        else:
            self.device = torch.device('cpu')
            self.logger.warning("CUDA not available, using CPU")

        self.evm = None
        self.Re = Re
        self.vis_t0 = 20.0/self.Re

        self.layers = layers
        self.layers_1 = layers_1
        self.hidden_size = hidden_size
        self.hidden_size_1 = hidden_size_1
        self.N_f = N_f
        self.batch_size = batch_size if batch_size is not None else N_f
        self.current_stage = ' '

        self.checkpoint_freq = checkpoint_freq

        # 日誌系統設定
        self.logger = LoggerFactory.get_logger(
            name=f"PINN_Re{Re}",
            level="INFO",
            rank=self.rank
        )

        # TensorBoard設定
        if self.rank == 0:  # 只在主進程創建TensorBoard writer
            log_dir = f"runs/NSFnet_Re{Re}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.tb_writer = SummaryWriter(log_dir=log_dir)
            self.logger.info(f"📊 TensorBoard log directory: {log_dir}")
        else:
            self.tb_writer = None

        # 時間追蹤相關變數
        self.epoch_start_time = None
        self.epoch_times = []
        self.stage_start_time = None
        self.training_start_time = None
        self.global_step_offset = 0  # 用於計算跨階段的global step

        # 健康監控系統
        if self.rank == 0:  # 只在主進程啟用健康監控
            health_thresholds = HealthThresholds(
                gpu_memory_warning=90.0,
                process_memory_warning=12000.0,  # 12GB
                cpu_warning=85.0
            )
            self.health_monitor = TrainingHealthMonitor(
                logger=self.logger,
                thresholds=health_thresholds,
                check_interval=60.0  # 每分鐘檢查一次
            )
            
            # 記憶體管理系統
            self.memory_manager = TrainingMemoryManager(
                logger=self.logger,
                cpu_threshold=80.0,
                gpu_threshold=85.0,
                auto_cleanup_interval=300.0  # 5分鐘自動清理
            )
        else:
            self.health_monitor = None
            self.memory_manager = None

        self.alpha_evm = alpha_evm
        self.alpha_b = bc_weight
        self.alpha_e = eq_weight
        self.alpha_i = ic_weight
        self.alpha_o = outlet_weight
        self.alpha_s = supervised_data_weight
        self.loss_i = self.loss_o = self.loss_b = self.loss_e = self.loss_s = 0.0

        # initialize NN
        self.net = self.initialize_NN(
                num_ins=num_ins, num_outs=num_outs, num_layers=layers, hidden_size=hidden_size).to(self.device)
        self.net_1 = self.initialize_NN(
                num_ins=num_ins, num_outs=num_outs_1, num_layers=layers_1, hidden_size=hidden_size_1).to(self.device)

        

        # 確保所有張量使用 float32 精度
        self.net = self.net.float()
        self.net_1 = self.net_1.float()

        # 優化：初始化vis_t相關變數，避免重複檢查
        self.vis_t_minus_gpu = None  # GPU版本的vis_t_minus

        # Wrap models with DDP only if in distributed mode
        if self.world_size > 1:
            # 使用find_unused_parameters=True支援動態參數凍結，避免使用static_graph避免警告
            self.net = DDP(self.net, 
                           device_ids=[self.local_rank], 
                           output_device=self.local_rank,
                           find_unused_parameters=True,   # 支援動態凍結功能
                           broadcast_buffers=True,        # 確保buffer同步
                           gradient_as_bucket_view=True)  # 提升記憶體效率
            self.net_1 = DDP(self.net_1, 
                             device_ids=[self.local_rank], 
                             output_device=self.local_rank,
                             find_unused_parameters=True,   # 支援動態凍結功能
                             broadcast_buffers=True,        # 確保buffer同步
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

    def get_model_parameters(self, model):
        """Get model parameters considering DDP wrapper"""
        if hasattr(model, 'module'):
            return model.module.parameters()
        else:
            return model.parameters()

    def get_model(self, model):
        """Get underlying model considering DDP wrapper"""
        if hasattr(model, 'module'):
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
            'optimizer_state_dict': torch.optim.Optimizer.state_dict(optimizer),
            'Re': self.Re,
            'alpha_evm': self.alpha_evm,
            'current_stage': self.current_stage,
            'global_step_offset': self.global_step_offset
        }

        try:
            torch.save(checkpoint, checkpoint_path)
            self.logger.checkpoint_saved(checkpoint_path, epoch)
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint to {checkpoint_path}: {e}")

    def load_checkpoint(self, checkpoint_path, optimizer):
        """Loads a checkpoint to resume training."""
        if not os.path.exists(checkpoint_path):
            self.logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return 0 # Return 0 to indicate training should start from epoch 0

        try:
            # Load checkpoint to the same device as the model
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Load model weights
            self.get_model(self.net).load_state_dict(checkpoint['net_state_dict'])
            self.get_model(self.net_1).load_state_dict(checkpoint['net_1_state_dict'])

            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load training state
            start_epoch = checkpoint['epoch'] + 1
            self.global_step_offset = checkpoint.get('global_step_offset', 0)
            
            # Restore key parameters to ensure consistency
            self.Re = checkpoint.get('Re', self.Re)
            self.alpha_evm = checkpoint.get('alpha_evm', self.alpha_evm)
            
            self.logger.info(f"✅ Resumed training from checkpoint: {checkpoint_path} at epoch {start_epoch}")
            
            # Move optimizer states to the correct device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

            return start_epoch

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            return 0 # Start from scratch if loading fails
            
    def init_vis_t(self):
        """優化版本：避免不必要的CPU轉換"""
        (_,_,_,e) = self.neural_net_u(self.x_f, self.y_f)
        self.vis_t_minus_gpu = self.alpha_evm*torch.abs(e).detach()  # 保持在GPU上

    def set_boundary_data(self, X=None, time=False):
        # Split boundary data across GPUs
        total_points = X[0].shape[0]
        
        # 確保每個GPU至少有一些數據點
        if total_points < self.world_size:
            if self.rank == 0:
                print(f"WARNING: Only {total_points} boundary points for {self.world_size} GPUs")
            # 如果數據點少於GPU數量，只有部分GPU處理數據
            if self.rank < total_points:
                start_idx = self.rank
                end_idx = self.rank + 1
            else:
                # 沒有數據的GPU使用空張量
                start_idx = 0
                end_idx = 0
        else:
            points_per_gpu = total_points // self.world_size
            start_idx = self.rank * points_per_gpu
            end_idx = start_idx + points_per_gpu if self.rank < self.world_size - 1 else total_points

        # 檢查索引邊界
        start_idx = max(0, min(start_idx, total_points))
        end_idx = max(start_idx, min(end_idx, total_points))
        
        # 邊界條件數據不需要梯度，只需要進行前向計算
        coord_requires_grad = False  # 邊界座標不需要梯度
        target_requires_grad = False # u, v 標準答案不需要梯度
        
        # 如果沒有數據點，創建空張量
        if start_idx >= end_idx:
            self.x_b = torch.empty(0, 1, requires_grad=coord_requires_grad).float().to(self.device)
            self.y_b = torch.empty(0, 1, requires_grad=coord_requires_grad).float().to(self.device)
            self.u_b = torch.empty(0, 1, requires_grad=target_requires_grad).float().to(self.device)
            self.v_b = torch.empty(0, 1, requires_grad=target_requires_grad).float().to(self.device)
        else:
            self.x_b = torch.tensor(X[0][start_idx:end_idx], requires_grad=coord_requires_grad).float().to(self.device)
            self.y_b = torch.tensor(X[1][start_idx:end_idx], requires_grad=coord_requires_grad).float().to(self.device)
            self.u_b = torch.tensor(X[2][start_idx:end_idx], requires_grad=target_requires_grad).float().to(self.device)
            self.v_b = torch.tensor(X[3][start_idx:end_idx], requires_grad=target_requires_grad).float().to(self.device)
            
        if time:
            if start_idx >= end_idx:
                self.t_b = torch.empty(0, 1, requires_grad=coord_requires_grad).float().to(self.device)
            else:
                self.t_b = torch.tensor(X[4][start_idx:end_idx], requires_grad=coord_requires_grad).float().to(self.device)

        if self.rank == 0:
            print(f"GPU {self.rank}: Processing {end_idx - start_idx} boundary points out of {total_points} total")

    def set_eq_training_data(self,
                             X=None,
                             time=False):
        # Split equation training data across GPUs
        total_points = X[0].shape[0]
        
        # 確保每個GPU至少有一些數據點
        if total_points < self.world_size:
            if self.rank == 0:
                print(f"WARNING: Only {total_points} training points for {self.world_size} GPUs")
            # 如果數據點少於GPU數量，只有部分GPU處理數據
            if self.rank < total_points:
                start_idx = self.rank
                end_idx = self.rank + 1
            else:
                # 沒有數據的GPU使用最小集合
                start_idx = 0
                end_idx = 1
        else:
            points_per_gpu = total_points // self.world_size
            start_idx = self.rank * points_per_gpu
            end_idx = start_idx + points_per_gpu if self.rank < self.world_size - 1 else total_points

        # 檢查索引邊界
        start_idx = max(0, min(start_idx, total_points))
        end_idx = max(start_idx + 1, min(end_idx, total_points))  # 確保至少有1個點

        requires_grad = True
        self.x_f = torch.tensor(X[0][start_idx:end_idx], requires_grad=requires_grad).float().to(self.device)
        self.y_f = torch.tensor(X[1][start_idx:end_idx], requires_grad=requires_grad).float().to(self.device)
        if time:
            self.t_f = torch.tensor(X[2][start_idx:end_idx], requires_grad=requires_grad).float().to(self.device)

        if self.rank == 0:
            print(f"GPU {self.rank}: Processing {end_idx - start_idx} equation points out of {total_points} total")

        self.init_vis_t()

    def set_optimizers(self, opt):
        self.opt = opt

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

    def initialize_NN(self,
                      num_ins=3,
                      num_outs=3,
                      num_layers=10,
                      hidden_size=50):
        return FCNet(num_ins=num_ins,
                     num_outs=num_outs,
                     num_layers=num_layers,
                     hidden_size=hidden_size,
                     activation=torch.nn.Tanh)

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
        return u, v, p, e

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
        e = ee[:, 0:1]
        self.evm = e

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

        # Get the minum between (vis_t0, vis_t_mius(calculated with last step e))
        batch_size = x.shape[0]
        self.vis_t = self._compute_vis_t_optimized(batch_size, e)
            
        # 更新 vis_t_minus (移到GPU上避免CPU-GPU轉換)
        self.vis_t_minus_gpu = self.alpha_evm * torch.abs(e).detach()

        # NS equations - 優化：避免重複的乘法運算
        vis_total = (1.0/self.Re + self.vis_t)
        
        eq1 = (u*u_x + v*u_y) + p_x - vis_total*(u_xx + u_yy)
        eq2 = (u*v_x + v*v_y) + p_y - vis_total*(v_xx + v_yy)
        eq3 = u_x + v_y

        residual = (eq1*(u-0.5)+eq2*(v-0.5))-e
        return eq1, eq2, eq3, residual

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
                vis_t_minus_batch = self.vis_t_minus_gpu.clone()
            
            # 在GPU上計算minimum
            vis_t0_tensor = torch.full_like(vis_t_minus_batch, self.vis_t0)
            vis_t = torch.minimum(vis_t0_tensor, vis_t_minus_batch)
        else:
            # 首次運行或沒有前一步數據
            vis_t = torch.full((batch_size, 1), self.vis_t0, device=self.device, dtype=torch.float32)
            
        return vis_t

    def autograd(self, y: torch.Tensor, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        計算梯度的函數 (保留原函數以兼容性)
        """
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y, device=y.device)]
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
                if hasattr(self.net, 'module'):
                    dummy_loss_net = torch.sum(self.net.module.layers[0].weight * 0.0)
                    dummy_loss_net1 = torch.sum(self.net_1.module.layers[0].weight * 0.0)
                else:
                    dummy_loss_net = torch.sum(self.net.layers[0].weight * 0.0)
                    dummy_loss_net1 = torch.sum(self.net_1.layers[0].weight * 0.0)
                self.loss_b = dummy_loss_net + dummy_loss_net1

        # equation
        assert self.x_f is not None and self.y_f is not None

        (self.eq1_pred, self.eq2_pred, self.eq3_pred, self.eq4_pred) = self.neural_net_equations(self.x_f, self.y_f)
    
        if loss_mode == 'MSE':
            self.loss_eq1 = torch.mean(torch.square(self.eq1_pred.view(-1)))
            self.loss_eq2 = torch.mean(torch.square(self.eq2_pred.view(-1)))
            self.loss_eq3 = torch.mean(torch.square(self.eq3_pred.view(-1)))
            self.loss_eq4 = torch.mean(torch.square(self.eq4_pred.view(-1)))
            self.loss_e = self.loss_eq1 + self.loss_eq2 + self.loss_eq3 + 0.1 * self.loss_eq4

        # 跨GPU聚合損失以獲得全局損失值
        if self.world_size > 1:
            # 聚合邊界損失 - 使用 detach() 避免 autograd 警告
            loss_b_detached = self.loss_b.detach().clone()
            dist.all_reduce(loss_b_detached, op=dist.ReduceOp.SUM)
            loss_b_avg = loss_b_detached / self.world_size
            
            # 聚合方程損失 - 使用 detach() 避免 autograd 警告  
            loss_e_detached = self.loss_e.detach().clone()
            dist.all_reduce(loss_e_detached, op=dist.ReduceOp.SUM)
            loss_e_avg = loss_e_detached / self.world_size
            
            # 用於日誌顯示的平均損失
            self.loss_b_avg = loss_b_avg
            self.loss_e_avg = loss_e_avg
        else:
            self.loss_b_avg = self.loss_b
            self.loss_e_avg = self.loss_e

        # 計算總損失（保持梯度追踪），確保兩個網路都參與
        self.loss = self.alpha_b * self.loss_b + self.alpha_e * self.loss_e

        # 添加一個小的正則化項確保兩個網路都參與梯度計算
        # 這不會影響訓練結果，但確保DDP工作正常
        if self.world_size > 1:
            regularization_weight = 1e-8
            if hasattr(self.net, 'module'):
                net_reg = torch.sum(torch.stack([torch.sum(p**2) for p in self.net.module.parameters()]))
                net1_reg = torch.sum(torch.stack([torch.sum(p**2) for p in self.net_1.module.parameters()]))
            else:
                net_reg = torch.sum(torch.stack([torch.sum(p**2) for p in self.net.parameters()]))
                net1_reg = torch.sum(torch.stack([torch.sum(p**2) for p in self.net_1.parameters()]))
            
            self.loss = self.loss + regularization_weight * (net_reg + net1_reg)

        # 創建用於backward的loss（保持梯度）
        loss_for_backward = self.loss
        
        # 創建用於日誌記錄的detached數值
        if hasattr(self, 'loss_e_avg'):
            loss_e_log = self.loss_e_avg.detach().item()
        else:
            loss_e_log = self.loss_e.detach().item()
            
        if hasattr(self, 'loss_b_avg'):
            loss_b_log = self.loss_b_avg.detach().item()
        else:
            loss_b_log = self.loss_b.detach().item()

        return loss_for_backward, [loss_e_log, loss_b_log, self.loss_eq1.detach().item(), self.loss_eq2.detach().item(), self.loss_eq3.detach().item(), self.loss_eq4.detach().item()]

    def train(self,
              num_epoch=1,
              lr=1e-4,
              optimizer=None,
              scheduler=None,
              batchsize=None,
              profiler=None,
              start_epoch=0):
        if self.opt is not None:
            self.opt.param_groups[0]['lr'] = lr
        return self.solve_Adam(self.fwd_computing_loss_2d, num_epoch, batchsize, scheduler, profiler, start_epoch)

    def solve_Adam(self, loss_func, num_epoch=1000, batchsize=None, scheduler=None, profiler=None, start_epoch=0):
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
            if self.health_monitor and not self.health_monitor.is_monitoring:
                self.health_monitor.start_training_monitoring()
                
            # 啟動記憶體管理
            if self.memory_manager:
                self.memory_manager.optimize_for_training()
        
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
        estimate_frequency = 100  # 每100個epoch計算一次預估時間
        
        for epoch_id in range(start_epoch, num_epoch):
            # 記錄epoch開始時間
            if self.rank == 0:
                self.epoch_start_time = time.time()
            
            # 恢復原有的動態凍結邏輯
            if epoch_id != 0 and epoch_id % 10000 == 0:
                self.defreeze_evm_net(epoch_id)
            if (epoch_id - 1) % 10000 == 0:
                self.freeze_evm_net(epoch_id)

            # 清除上一個epoch的梯度
            self.opt.zero_grad()

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
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
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
                            
                            # 嘗試重建optimizer參數群組
                            try:
                                self.rebuild_optimizer_groups()
                                # 清空梯度並重新同步
                                self.opt.zero_grad()
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
            
            if scheduler:
                scheduler.step()

            if profiler:
                profiler.step()

            # 時間追蹤和預估（只在rank 0執行）
            if self.rank == 0:
                epoch_end_time = time.time()
                epoch_time = epoch_end_time - self.epoch_start_time
                
                # 限制epoch_times大小以防記憶體洩漏
                self.epoch_times.append(epoch_time)
                if len(self.epoch_times) > 1000:  # 只保留最近1000個epoch的時間
                    self.epoch_times = self.epoch_times[-500:]  # 刪除一半舊數據，保持高效
                
                # 記錄到TensorBoard
                if self.tb_writer is not None:
                    global_step = self.global_step_offset + epoch_id
                    
                    self.safe_tensorboard_log('Loss/Total', epoch_loss, global_step)
                    self.safe_tensorboard_log('Loss/Equation_Combined', epoch_losses[0], global_step)
                    self.safe_tensorboard_log('Loss/Boundary', epoch_losses[1], global_step)
                    self.safe_tensorboard_log('Loss/Equation_NS_X', epoch_losses[2], global_step)
                    self.safe_tensorboard_log('Loss/Equation_NS_Y', epoch_losses[3], global_step)
                    self.safe_tensorboard_log('Loss/Equation_Continuity', epoch_losses[4], global_step)
                    self.safe_tensorboard_log('Loss/Equation_EntropyResidual', epoch_losses[5], global_step)
                    self.safe_tensorboard_log('Training/LearningRate', self.opt.param_groups[0]['lr'], global_step)
                    self.safe_tensorboard_log('Training/EpochTime', epoch_time, global_step)
                    self.safe_tensorboard_log('Training/Alpha_EVM', self.alpha_evm, global_step)
                    
                    # GPU記憶體使用
                    if torch.cuda.is_available():
                        memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                        self.safe_tensorboard_log('System/GPU_Memory_GB', memory_allocated, global_step)
                
                # 健康檢查 (每100個epoch檢查一次)
                if self.health_monitor and epoch_id % 100 == 0:
                    is_healthy = self.health_monitor.check_training_health(epoch_id, epoch_loss)
                    if not is_healthy:
                        self.logger.warning("⚕️ 系統健康檢查發現問題，考慮調整訓練參數")
                        
                        # 緊急情況下執行清理
                        try:
                            self.health_monitor.emergency_cleanup()
                        except Exception as cleanup_error:
                            self.logger.error(f"緊急清理失敗: {cleanup_error}")
                
                # 記憶體監控 (每50個epoch檢查一次)
                if self.memory_manager and epoch_id % 50 == 0:
                    memory_ok = self.memory_manager.monitor_training_memory(epoch_id, epoch_loss)
                    if not memory_ok:
                        self.logger.critical("💾 記憶體使用危險，建議調整訓練參數或重啟訓練")
            
            # 每1000個epoch輸出一次訓練狀況，首個epoch也要輸出
            if self.rank == 0 and (epoch_id == 0 or (epoch_id + 1) % 1000 == 0 or epoch_id == num_epoch - 1):
                self.print_log_full_batch_with_time_estimate(epoch_loss, epoch_losses, epoch_id, num_epoch, actual_data_points)
                
                # 每1000個epoch輸出健康和記憶體報告
                if epoch_id == 0 or epoch_id % 1000 == 0:
                    if self.health_monitor:
                        self.health_monitor.log_health_report()
                    if self.memory_manager:
                        self.memory_manager.log_memory_report()

            # Save checkpoint
            if self.rank == 0 and (epoch_id > 0 and epoch_id % self.checkpoint_freq == 0 or epoch_id == num_epoch - 1):
                self.save_checkpoint(epoch_id, self.opt)
                if self.health_monitor:
                    self.health_monitor.last_checkpoint_time = time.time()

        # 階段結束後更新global step offset
        if self.rank == 0:
            self.global_step_offset += num_epoch
            
            # 階段結束時的最終清理和統計
            if self.memory_manager:
                self.logger.info("🧹 訓練階段結束，執行最終記憶體清理...")
                final_cleanup = self.memory_manager.cleanup_memory(force=True)
                
                # 記憶體洩漏檢測
                leaks = self.memory_manager.detect_memory_leaks()
                if leaks:
                    self.logger.warning("🔍 檢測到記憶體洩漏，請檢查代碼")
                
                # 輸出訓練記憶體摘要
                memory_summary = self.memory_manager.get_training_memory_summary()
                self.logger.info(f"📊 訓練記憶體摘要: 峰值={memory_summary['peak_memory_mb']:.1f}MB, "
                               f"清理次數={memory_summary['cleanup_count']}, "
                               f"緩存命中率={memory_summary['cache_hit_rate']:.1f}%")
    
    def freeze_evm_net(self, epoch_id):
        """
        凍結EVM網路參數 - 使用static_graph=True避免DDP bucket重建問題
        """
        if self.rank == 0:
            print(f"[Epoch {epoch_id}] Freezing EVM network parameters (DDP-compatible)")
        
        # 凍結net_1的所有參數
        for param in self.net_1.parameters():
            param.requires_grad = False
    
        # 更新optimizer的參數組 - 僅包含需要梯度的參數
        active_params = []
        for param in self.net.parameters():
            if param.requires_grad:
                active_params.append(param)
        
        if len(active_params) > 0:
            # 更新參數列表，但保持學習率不變
            current_lr = self.opt.param_groups[0]['lr']
            self.opt.param_groups[0]['params'] = active_params
            self.opt.param_groups[0]['lr'] = current_lr
        
        if self.rank == 0:
            print(f"  Active parameters: {len(active_params)} (net only)")

    def defreeze_evm_net(self, epoch_id):
        """
        解凍EVM網路參數 - 使用static_graph=True避免DDP bucket重建問題
        """
        if self.rank == 0:
            print(f"[Epoch {epoch_id}] Unfreezing EVM network parameters (DDP-compatible)")
        
        # 解凍net_1的所有參數
        for param in self.net_1.parameters():
            param.requires_grad = True
    
        # 更新optimizer的參數組包含所有參數
        all_params = []
        for param in list(self.net.parameters()) + list(self.net_1.parameters()):
            if param.requires_grad:
                all_params.append(param)
        
        if len(all_params) > 0:
            # 更新參數列表，保持當前學習率
            current_lr = self.opt.param_groups[0]['lr']
            self.opt.param_groups[0]['params'] = all_params
            self.opt.param_groups[0]['lr'] = current_lr
        
        if self.rank == 0:
            print(f"  Active parameters: {len(all_params)} (net + net_1)")

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
        """重建optimizer參數群組以修復DDP問題"""
        try:
            # 獲取所有需要梯度的參數
            all_params = []
            for param in self.net.parameters():
                if param.requires_grad:
                    all_params.append(param)
            for param in self.net_1.parameters():
                if param.requires_grad:
                    all_params.append(param)
            
            if len(all_params) > 0:
                # 保存當前學習率和其他配置
                current_lr = self.opt.param_groups[0]['lr']
                current_config = {k: v for k, v in self.opt.param_groups[0].items() if k != 'params'}
                
                # 更新參數列表
                self.opt.param_groups[0]['params'] = all_params
                
                # 恢復其他配置
                for key, value in current_config.items():
                    self.opt.param_groups[0][key] = value
                    
                if self.rank == 0:
                    print(f"   Rebuilt optimizer with {len(all_params)} parameters")
                    
        except Exception as e:
            if self.rank == 0:
                print(f"   Failed to rebuild optimizer groups: {e}")
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
            print(f"   Navier-Stokes X損失: {losses[2]:.3e}")
            print(f"   Navier-Stokes Y損失: {losses[3]:.3e}")
            print(f"   連續性方程損失: {losses[4]:.3e}")
            print(f"   熵殘差損失: {losses[5]:.3e}")
            print(f"   邊界損失: {losses[1]:.3e}")
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
            print(f"   Alpha EVM: {self.alpha_evm}")
            
            print(f"{'='*100}\n")
            
        else:
            # 初始幾個epoch，資訊較少
            print(f"\n{'='*80}")
            print(f"🔥 {self.current_stage} - 初始化階段")
            print(f"   Epoch: {epoch_id + 1:,} / {num_epoch:,}")
            print(f"   學習率: {current_lr:.2e} | 資料點: {data_points:,}")
            print(f"   損失 - 總: {loss:.3e} | 方程: {losses[0]:.3e} | 邊界: {losses[1]:.3e}")
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
        print('epoch/num_epoch: {:6d} / {:d} data_points: {:d} avg_loss[Adam]: {:.3e} avg_eq_combined_loss: {:.3e} avg_eq1_loss: {:.3e} avg_eq2_loss: {:.3e} avg_eq3_loss: {:.3e} avg_eq4_loss: {:.3e} avg_bc_loss: {:.3e}'.format(
            epoch_id + 1, num_epoch, data_points, loss, losses[0], losses[2], losses[3], losses[4], losses[5], losses[1]))

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
              "avg_eq1_loss: %.3e" %(losses[2] if len(losses) > 2 else 0),
              "avg_eq2_loss: %.3e" %(losses[3] if len(losses) > 3 else 0),
              "avg_eq3_loss: %.3e" %(losses[4] if len(losses) > 4 else 0),
              "avg_eq4_loss: %.3e" %(losses[5] if len(losses) > 5 else 0),
              "avg_bc_loss: %.3e" %(losses[1] if len(losses) > 1 else 0))

    def print_log(self, loss, losses, epoch_id, num_epoch):
        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        print("current lr is {}".format(get_lr(self.opt)))
        print("epoch/num_epoch: ", epoch_id + 1, "/", num_epoch,
              "loss[Adam]: %.3e"
              %(loss.detach().cpu().item()),
              "eq_combined_loss: %.3e " %(losses[0]),
              "eq1_loss: %.3e " %(losses[2]),
              "eq2_loss: %.3e " %(losses[3]),
              "eq3_loss: %.3e " %(losses[4]),
              "eq4_loss: %.3e " %(losses[5]),
              "bc_loss: %.3e" %(losses[1]))

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

    def test(self, x, y, u, v, p, loop=None):
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
            
            # 從config.py讀取基礎路徑
            try:
                from config import RESULTS_PATH
                base_path = RESULTS_PATH
            except ImportError:
                base_path = 'results'

            relative_path = os.path.join(base_path, Re_folder, f"{NNsize}_{lambdas}")

            if not os.path.exists(relative_path):
                os.makedirs(relative_path)

            file_path = os.path.join(relative_path, 'cavity_result_loop_%d.mat'%(loop))

            scipy.io.savemat(file_path,
                        {'U_pred':u_pred,
                         'V_pred':v_pred,
                         'P_pred':p_pred,
                         'E_pred':e_pred,
                         'error_u':error_u,
                         'error_v':error_v,
                         'error_p':error_p,
                         'lam_bcs':self.alpha_b,
                         'lam_equ':self.alpha_e})

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