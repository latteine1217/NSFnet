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
import scipy.io
import numpy as np
from net import FCNet
from typing import Dict, List, Set, Optional, Union, Callable
import warnings

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
        self.device = torch.device(f'cuda:{self.local_rank}')
        torch.cuda.set_device(self.local_rank)

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

        # Wrap models with DDP only if in distributed mode
        if self.world_size > 1:
            # 使用static_graph=True會自動檢測未使用參數，不需要find_unused_parameters=True
            self.net = DDP(self.net, 
                           device_ids=[self.local_rank], 
                           output_device=self.local_rank,
                           broadcast_buffers=True,        # 確保buffer同步
                           gradient_as_bucket_view=True,  # 提升記憶體效率
                           static_graph=True)             # 防止動態凍結時bucket重建
            self.net_1 = DDP(self.net_1, 
                             device_ids=[self.local_rank], 
                             output_device=self.local_rank,
                             broadcast_buffers=True,        # 確保buffer同步
                             gradient_as_bucket_view=True,  # 提升記憶體效率
                             static_graph=True)             # 防止動態凍結時bucket重建

        if net_params:
            if self.rank == 0:
                print(f"Loading net params from {net_params}")
            load_params = torch.load(net_params, map_location=self.device)
            if hasattr(self.net, 'module'):
                self.net.module.load_state_dict(load_params)
            else:
                self.net.load_state_dict(load_params)

        if net_params_1:
            if self.rank == 0:
                print(f"Loading net_1 params from {net_params_1}")
            load_params_1 = torch.load(net_params_1, map_location=self.device)
            if hasattr(self.net_1, 'module'):
                self.net_1.module.load_state_dict(load_params_1)
            else:
                self.net_1.load_state_dict(load_params_1)

        # 初始化 vis_t 相關變數
        self.vis_t = None
        self.vis_t_minus = None

        self.opt = torch.optim.Adam(
            list(self.get_model_parameters(self.net))+list(self.get_model_parameters(self.net_1)),
            lr=learning_rate,
            weight_decay=0.0) if not opt else opt

        if self.rank == 0:
            print(f"Distributed training setup:")
            print(f"  World size: {self.world_size}")
            print(f"  Rank: {self.rank}")
            print(f"  Local rank: {self.local_rank}")

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
            print(f"  Device: {self.device}")
            print(f"  DDP find_unused_parameters: True (enabled for dynamic network freezing)")

    def init_vis_t(self):
        (_,_,_,e) = self.neural_net_u(self.x_f, self.y_f)
        self.vis_t_minus = self.alpha_evm*torch.abs(e).detach().cpu().numpy()

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

        u_x, u_y = self.autograd(u, [x,y])
        u_xx = self.autograd(u_x, [x])[0]
        u_yy = self.autograd(u_y, [y])[0]

        v_x, v_y = self.autograd(v, [x,y])
        v_xx = self.autograd(v_x, [x])[0]
        v_yy = self.autograd(v_y, [y])[0]

        p_x, p_y = self.autograd(p, [x,y])

        # Get the minum between (vis_t0, vis_t_mius(calculated with last step e))
        batch_size = x.shape[0]
        if self.vis_t_minus is not None and self.vis_t_minus.shape[0] > 0:
            # 確保 vis_t_minus 的形狀匹配當前批次大小
            if self.vis_t_minus.shape[0] != batch_size:
                # 如果尺寸不匹配，使用 vis_t0 填充或截斷
                if self.vis_t_minus.shape[0] > batch_size:
                    vis_t_minus_batch = self.vis_t_minus[:batch_size]
                else:
                    # 重複填充到當前批次大小
                    if self.vis_t_minus.shape[0] > 0:
                        repeat_times = (batch_size + self.vis_t_minus.shape[0] - 1) // self.vis_t_minus.shape[0]
                        vis_t_minus_extended = np.tile(self.vis_t_minus, (repeat_times, 1))
                        vis_t_minus_batch = vis_t_minus_extended[:batch_size]
                    else:
                        # 如果 vis_t_minus 為空，直接使用 vis_t0
                        vis_t_minus_batch = np.full((batch_size, 1), self.vis_t0)
            else:
                vis_t_minus_batch = self.vis_t_minus
            
            vis_t0_batch = np.full_like(vis_t_minus_batch, self.vis_t0)
            self.vis_t = torch.tensor(
                    np.minimum(vis_t0_batch, vis_t_minus_batch)).float().to(self.device)
        else:
            # 創建與批次大小匹配的 vis_t0 張量
            vis_t0_batch = np.full((batch_size, 1), self.vis_t0)
            self.vis_t = torch.tensor(vis_t0_batch).float().to(self.device)
            
        # Save vis_t_minus for computing vis_t in the next step
        self.vis_t_minus = self.alpha_evm*torch.abs(e).detach().cpu().numpy()

        # NS
        eq1 = (u*u_x + v*u_y) + p_x - (1.0/self.Re+self.vis_t)*(u_xx + u_yy)
        eq2 = (u*v_x + v*v_y) + p_y - (1.0/self.Re+self.vis_t)*(v_xx + v_yy)
        eq3 = u_x + v_y

        residual = (eq1*(u-0.5)+eq2*(v-0.5))-e
        return eq1, eq2, eq3, residual

    def autograd(self, y: torch.Tensor, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        計算梯度的函數 (移除 @torch.jit.script 裝飾器)
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

        return loss_for_backward, [loss_e_log, loss_b_log]

    def train(self,
              num_epoch=1,
              lr=1e-4,
              optimizer=None,
              scheduler=None,
              batchsize=None):
        if self.opt is not None:
            self.opt.param_groups[0]['lr'] = lr
        return self.solve_Adam(self.fwd_computing_loss_2d, num_epoch, batchsize, scheduler)

    def solve_Adam(self, loss_func, num_epoch=1000, batchsize=None, scheduler=None):
        # 啟用初始凍結 - 使用static_graph=True避免DDP錯誤
        self.freeze_evm_net(0)
        
        # 如果指定了 batchsize，使用批次訓練
        if batchsize is not None:
            self.batch_size = batchsize
        
        # 創建DataLoader進行批次處理
        actual_data_points = self.x_f.shape[0]
        
        # 創建TensorDataset
        dataset = TensorDataset(self.x_f, self.y_f)
        
        # 創建DataLoader with distributed support
        if self.world_size > 1:
            # 分散式訓練使用DistributedSampler
            sampler = DistributedSampler(
                dataset, 
                num_replicas=self.world_size, 
                rank=self.rank,
                shuffle=True,
                drop_last=True  # 避免最後批次大小不一致
            )
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                drop_last=True,
                pin_memory=True,
                num_workers=0  # 避免多進程問題
            )
        else:
            # 單GPU模式
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                pin_memory=True,
                num_workers=0
            )
        
        effective_batch_size = min(self.batch_size, actual_data_points)
        steps_per_epoch = len(dataloader)
        
        if self.rank == 0:
            print(f"=== Training Configuration (DataLoader) ===")
            print(f"Total training points (N_f): {self.N_f}")
            print(f"Actual GPU data points: {actual_data_points}")
            print(f"Effective batch size: {effective_batch_size}")
            print(f"Steps per epoch: {steps_per_epoch}")
            print(f"Total epochs: {num_epoch}")
            print(f"DDP Mode: {'Enabled' if self.world_size > 1 else 'Disabled'}")
            print("=" * 45)
        
        for epoch_id in range(num_epoch):
            # 恢復原有的動態凍結邏輯
            if epoch_id != 0 and epoch_id % 10000 == 0:
                self.defreeze_evm_net(epoch_id)
            if (epoch_id - 1) % 10000 == 0:
                self.freeze_evm_net(epoch_id)

            # 設置epoch對於DistributedSampler
            if self.world_size > 1:
                dataloader.sampler.set_epoch(epoch_id)

            # 清除上一個epoch的梯度
            self.opt.zero_grad()

            epoch_loss = 0.0
            epoch_losses = [0.0, 0.0]
            
            # 使用DataLoader進行批次處理
            for step, (batch_x_f, batch_y_f) in enumerate(dataloader):
                # 設置當前批次數據
                self.x_f = batch_x_f.to(self.device)
                self.y_f = batch_y_f.to(self.device)
                
                # 計算損失
                with torch.enable_grad():
                    loss, losses = loss_func()
                
                # 正規化損失進行梯度累積
                normalized_loss = loss / steps_per_epoch
                
                # 梯度累積
                try:
                    normalized_loss.backward()
                except RuntimeError as e:
                    if "backward through the graph a second time" in str(e):
                        if self.rank == 0:
                            print(f"Warning: Graph reuse detected at step {step}, skipping this batch")
                        del loss, normalized_loss
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
                
                # 記錄損失值
                epoch_loss += loss.detach().item()
                epoch_losses[0] += losses[0] if isinstance(losses[0], (int, float)) else losses[0].detach().item()
                epoch_losses[1] += losses[1] if isinstance(losses[1], (int, float)) else losses[1].detach().item()
                
                # 清理張量引用
                del loss, normalized_loss
                
                # 定期清理GPU緩存
                if step % max(1, steps_per_epoch // 4) == 0 and step > 0:
                    torch.cuda.empty_cache()
            
            # 梯度裁剪避免梯度爆炸
            if self.world_size > 1:
                # DDP環境下的梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    list(self.net.parameters()) + list(self.net_1.parameters()),
                    max_norm=1.0
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    list(self.net.parameters()) + list(self.net_1.parameters()),
                    max_norm=1.0
                )
            
            # 一個epoch結束後統一更新參數
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # 檢查DDP狀態並嘗試恢復
            try:
                self.opt.step()
                self.opt.zero_grad()
            except RuntimeError as e:
                if "INTERNAL ASSERT FAILED" in str(e) or "unmarked_param_indices" in str(e):
                    if self.rank == 0:
                        print(f"Warning: DDP sync error at epoch {epoch_id}, attempting recovery...")
                    self.opt.zero_grad()
                    continue
                else:
                    raise e
            
            # 計算平均損失
            epoch_loss /= steps_per_epoch
            epoch_losses[0] /= steps_per_epoch
            epoch_losses[1] /= steps_per_epoch
            
            if scheduler:
                scheduler.step()

            # 只在rank 0打印和保存
            if self.rank == 0 and (epoch_id == 0 or (epoch_id + 1) % 100 == 0):
                self.print_log_batch(epoch_loss, epoch_losses, epoch_id, num_epoch, effective_batch_size, steps_per_epoch)

            if self.rank == 0 and (epoch_id == 0 or epoch_id % self.checkpoint_freq == 0):
                saved_ckpt = 'model_cavity_loop%d.pth' % (epoch_id)
                layers = self.layers
                hidden_size = self.hidden_size
                N_f = self.N_f
                self.save(saved_ckpt, N_HLayer=layers, N_neu=hidden_size, N_f=N_f)
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
              "avg_eq1_loss: %.3e " %(losses[0] if len(losses) > 0 else 0),
              "avg_bc_loss: %.3e" %(losses[1] if len(losses) > 1 else 0))

    def print_log(self, loss, losses, epoch_id, num_epoch):
        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        print("current lr is {}".format(get_lr(self.opt)))
        if isinstance(losses[0], int):
            eq_loss = losses[0]
        else:
            eq_loss = losses[0].detach().cpu().item()

        print("epoch/num_epoch: ", epoch_id + 1, "/", num_epoch,
              "loss[Adam]: %.3e"
              %(loss.detach().cpu().item()),
              "eq1_loss: %.3e " %(self.loss_eq1.detach().cpu().item()),
              "eq2_loss: %.3e " %(self.loss_eq2.detach().cpu().item()),
              "eq3_loss: %.3e " %(self.loss_eq3.detach().cpu().item()),
              "eq4_loss: %.3e " %(self.loss_eq4.detach().cpu().item()),
              "bc_loss: %.3e" %(losses[1].detach().cpu().item()))

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

            scipy.io.savemat('./NSFnet/ev-NSFnet/results/Re5000/test_result/cavity_result_loop_%d.mat'%(loop),
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
