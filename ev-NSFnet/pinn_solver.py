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
import scipy.io
import numpy as np
import time
from net import FCNet
from training_enhancer import DNSDataEnhancer
from typing import Dict, List, Set, Optional, Union, Callable

class PysicsInformedNeuralNetwork:
    # TensorBoard: 外部可注入 writer (rank0)，未注入則不記錄
    tb_writer = None
    global_step = 0
    # Initialize the class
    def __init__(self,
                 opt=None,
                 Re = 1000,
                 layers=6,
                 layers_1=6,
                 hidden_size=80,
                 hidden_size_1=20,
                 N_f = 100000,
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
                 checkpoint_freq=2000,
                 checkpoint_path='./checkpoint/',
                 # 坐標歸一化
                 normalize_coordinates=False,
                 # 新增參數
                 enable_distance_weighting=False,
                 distance_weight_function='inverse',
                 enable_point_sorting=False,
                 sort_method='distance_to_boundary',
                 enable_dns_enhancement=False,
                 dns_points_count=1000,
                 # PDE 距離權重（參考 ev-NSFnet copy）
                 pde_distance_weighting=False,
                 pde_distance_w_min=0.8,
                 pde_distance_tau=0.2):

        # Initialize distributed training
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))

        # Set device for current process
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.local_rank}')
            try:
                torch.cuda.set_device(self.local_rank)
            except (AttributeError, RuntimeError):
                # Fallback for compatibility
                self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        self.evm = None
        self.Re = Re
        self.vis_t0 = 20.0/self.Re

        self.layers = layers
        self.layers_1 = layers_1
        self.hidden_size = hidden_size
        self.hidden_size_1 = hidden_size_1
        self.N_f = N_f
        self.current_stage = ' '

        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_path = checkpoint_path

        self.alpha_evm = alpha_evm
        self.alpha_b = bc_weight
        self.alpha_e = eq_weight
        self.alpha_i = ic_weight
        self.alpha_o = outlet_weight
        self.alpha_s = supervised_data_weight
        self.loss_i = self.loss_o = self.loss_b = self.loss_e = self.loss_s = 0.0
        # 座標歸一化設定
        self.normalize_coordinates = bool(normalize_coordinates)

        # initialize NN
        self.net = self.initialize_NN(
                num_ins=num_ins, num_outs=num_outs, num_layers=layers, hidden_size=hidden_size).to(self.device)
        self.net_1 = self.initialize_NN(
                num_ins=num_ins, num_outs=num_outs_1, num_layers=layers_1, hidden_size=hidden_size_1).to(self.device)

        # Conditionally wrap with DDP only when distributed is initialized
        self.is_distributed = dist.is_available() and dist.is_initialized() and self.world_size > 1
        if self.is_distributed:
            self.net = DDP(self.net, device_ids=[self.local_rank], output_device=self.local_rank)
            self.net_1 = DDP(self.net_1, device_ids=[self.local_rank], output_device=self.local_rank)

        if net_params:
            if self.rank == 0:
                print(f"Loading net params from {net_params}")
            load_params = torch.load(net_params, map_location=self.device)
            target_net = self.net.module if self.is_distributed else self.net
            target_net.load_state_dict(load_params)

        if net_params_1:
            if self.rank == 0:
                print(f"Loading net_1 params from {net_params_1}")
            load_params_1 = torch.load(net_params_1, map_location=self.device)
            target_net1 = self.net_1.module if self.is_distributed else self.net_1
            target_net1.load_state_dict(load_params_1)

        # 初始化 vis_t 相關變數
        self.vis_t = None
        self.vis_t_minus = None

        # 初始化（移除 TrainingEnhancer；僅保留 DNS 增強）
        self.enable_distance_weighting = enable_distance_weighting  # deprecated, not used
        self.enable_point_sorting = enable_point_sorting            # deprecated, not used
        self.enable_dns_enhancement = enable_dns_enhancement
        
        if self.enable_dns_enhancement:
            self.dns_enhancer = DNSDataEnhancer(device=self.device)
            self.dns_points_count = dns_points_count
            self.dns_data = None  # 將在set_dns_data中設置
            self.dns_loss_weight = supervised_data_weight

        # PDE 距離權重設定（鏡像 ev-NSFnet copy），預設關閉
        self.pde_distance_weighting = bool(pde_distance_weighting)
        self.pde_distance_w_min = float(pde_distance_w_min)
        self.pde_distance_tau = float(pde_distance_tau)
        self.w_f = None

        self.opt = torch.optim.Adam(
            list(self.net.parameters())+list(self.net_1.parameters()),
            lr=learning_rate,
            weight_decay=0.0) if not opt else opt

        if self.rank == 0:
            print(f"Distributed training setup:")
            print(f"  World size: {self.world_size}")
            print(f"  Rank: {self.rank}")
            print(f"  Local rank: {self.local_rank}")
            print(f"  Device: {self.device}")

    def init_vis_t(self):
        (_,_,_,e) = self.neural_net_u(self.x_f, self.y_f)
        self.vis_t_minus = self.alpha_evm*torch.abs(e).detach().cpu().numpy()

    def set_boundary_data(self, X=None, time=False):
        # Split boundary data across GPUs
        total_points = X[0].shape[0]
        points_per_gpu = total_points // self.world_size
        start_idx = self.rank * points_per_gpu
        end_idx = start_idx + points_per_gpu if self.rank < self.world_size - 1 else total_points

        requires_grad = False
        self.x_b = torch.tensor(X[0][start_idx:end_idx], requires_grad=requires_grad).float().to(self.device)
        self.y_b = torch.tensor(X[1][start_idx:end_idx], requires_grad=requires_grad).float().to(self.device)
        self.u_b = torch.tensor(X[2][start_idx:end_idx], requires_grad=requires_grad).float().to(self.device)
        self.v_b = torch.tensor(X[3][start_idx:end_idx], requires_grad=requires_grad).float().to(self.device)
        if time:
            self.t_b = torch.tensor(X[4][start_idx:end_idx], requires_grad=requires_grad).float().to(self.device)

        if self.rank == 0:
            print(f"GPU {self.rank}: Processing {end_idx - start_idx} boundary points out of {total_points} total")

    def set_eq_training_data(self,
                             X=None,
                             time=False):
        # Split equation training data across GPUs
        total_points = X[0].shape[0]
        points_per_gpu = total_points // self.world_size
        start_idx = self.rank * points_per_gpu
        end_idx = start_idx + points_per_gpu if self.rank < self.world_size - 1 else total_points

        requires_grad = True
        self.x_f = torch.tensor(X[0][start_idx:end_idx], requires_grad=requires_grad).float().to(self.device)
        self.y_f = torch.tensor(X[1][start_idx:end_idx], requires_grad=requires_grad).float().to(self.device)
        if time:
            self.t_f = torch.tensor(X[2][start_idx:end_idx], requires_grad=requires_grad).float().to(self.device)

        # 點排序由 DataLoader 控制；此處不再排序

        # 預計算 PDE 距離權重（若啟用），使用解析的邊界距離（避免 cdist 計算開銷）
        try:
            if self.pde_distance_weighting and isinstance(self.x_f, torch.Tensor) and isinstance(self.y_f, torch.Tensor):
                with torch.no_grad():
                    # 計算到邊界的最小距離（域為 [0,1]^2）
                    d_x = torch.minimum(self.x_f, 1.0 - self.x_f)
                    d_y = torch.minimum(self.y_f, 1.0 - self.y_f)
                    d = torch.minimum(d_x, d_y)
                    w = self.pde_distance_w_min + (1.0 - self.pde_distance_w_min) * torch.exp(-d / max(self.pde_distance_tau, 1e-6))
                    w = (w / (w.mean() + 1e-12)).detach()
                self.w_f = w
            else:
                self.w_f = None
        except Exception:
            self.w_f = None

        # 舊版的距離權重（透過 TrainingEnhancer）已移除

        if self.rank == 0:
            print(f"GPU {self.rank}: Processing {end_idx - start_idx} equation points out of {total_points} total")

        self.init_vis_t()

    def set_optimizers(self, opt):
        self.opt = opt

    def set_alpha_evm(self, alpha):
        self.alpha_evm = alpha

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
        # 確保輸入張量有正確的形狀
        if x.dim() == 1:
            x = x.view(-1, 1)
        if y.dim() == 1:
            y = y.view(-1, 1)
            
        # 視設定決定是否將座標線性映射到 [-1, 1]（不覆蓋原始座標，保持鏈式法則於方程處正確）
        if self.normalize_coordinates:
            x_in = 2.0 * x - 1.0
            y_in = 2.0 * y - 1.0
        else:
            x_in, y_in = x, y
        X = torch.cat((x_in, y_in), dim=1)
        uvp = self.net(X)
        ee = self.net_1(X)
        u = uvp[:, 0]
        v = uvp[:, 1]
        p = uvp[:, 2:3]
        e =  ee[:,0:1]
        return u, v, p, e

    def neural_net_equations(self, x, y):
        # 確保輸入張量有正確的形狀
        if x.dim() == 1:
            x = x.view(-1, 1)
        if y.dim() == 1:
            y = y.view(-1, 1)
            
        # 視設定決定是否將座標線性映射到 [-1, 1]；保持 x,y 為原始物理座標以正確計算偏導
        if self.normalize_coordinates:
            x_in = 2.0 * x - 1.0
            y_in = 2.0 * y - 1.0
        else:
            x_in, y_in = x, y
        X = torch.cat((x_in, y_in), dim=1)
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
        if self.vis_t_minus is not None:
            self.vis_t = torch.tensor(
                    np.minimum(self.vis_t0, self.vis_t_minus)).float().to(self.device)
        else:
            self.vis_t = torch.tensor(self.vis_t0).float().to(self.device)
            
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

        # BC loss
        if loss_mode == 'MSE':
                u_loss = torch.square(self.u_b.reshape([-1]) - self.u_pred_b.reshape([-1]))
                v_loss = torch.square(self.v_b.reshape([-1]) - self.v_pred_b.reshape([-1]))
                self.loss_b = torch.mean(u_loss) + torch.mean(v_loss)

        # equation
        assert self.x_f is not None and self.y_f is not None

        (self.eq1_pred, self.eq2_pred, self.eq3_pred, self.eq4_pred) = self.neural_net_equations(self.x_f, self.y_f)
    
        if loss_mode == 'MSE':
                eq1_loss = torch.square(self.eq1_pred.reshape([-1]))
                eq2_loss = torch.square(self.eq2_pred.reshape([-1]))
                eq3_loss = torch.square(self.eq3_pred.reshape([-1]))
                eq4_loss = torch.square(self.eq4_pred.reshape([-1]))
                
                # 應用 PDE 距離權重（若可用）
                if isinstance(getattr(self, 'w_f', None), torch.Tensor) and self.w_f.shape[0] == eq1_loss.shape[0]:
                    w = self.w_f.view(-1)
                    self.loss_eq1 = torch.mean(w * eq1_loss)
                    self.loss_eq2 = torch.mean(w * eq2_loss)
                    self.loss_eq3 = torch.mean(w * eq3_loss)
                    self.loss_eq4 = torch.mean(w * eq4_loss)
                else:
                    self.loss_eq1 = torch.mean(eq1_loss)
                    self.loss_eq2 = torch.mean(eq2_loss)
                    self.loss_eq3 = torch.mean(eq3_loss)
                    self.loss_eq4 = torch.mean(eq4_loss)
                    
                self.loss_e = self.loss_eq1 + self.loss_eq2 + self.loss_eq3 + 0.1 * self.loss_eq4

        # DNS監督數據損失（如果啟用）
        if self.enable_dns_enhancement and self.dns_data is not None:
            dns_loss = self.compute_dns_loss()
            self.loss = self.alpha_b * self.loss_b + self.alpha_e * self.loss_e + self.dns_loss_weight * dns_loss
        else:
            self.loss = self.alpha_b * self.loss_b + self.alpha_e * self.loss_e

        # 跨GPU聚合損失以獲得全局損失值
        if self.world_size > 1:
                # 聚合邊界損失
                dist.all_reduce(self.loss_b, op=dist.ReduceOp.SUM)
                self.loss_b /= self.world_size
        
                # 聚合方程損失
                dist.all_reduce(self.loss_e, op=dist.ReduceOp.SUM)
                self.loss_e /= self.world_size

        return self.loss, [self.loss_e, self.loss_b]

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
        # 記錄階段與累積起始時間（每個 stage 呼叫一次）
        if not hasattr(self, 'cumulative_start_time'):
            self.cumulative_start_time = time.time()
        self._epoch_start_wall = time.time()  # stage 起始
        self._last_log_time = self._epoch_start_wall
        self._last_log_epoch = 0
        # 若外部未設置，提供回退值
        if not hasattr(self, 'log_interval'):
            self.log_interval = 100
        if not hasattr(self, 'progress_bar_width'):
            self.progress_bar_width = 30
        self.freeze_evm_net(0)
        
        if not hasattr(self, 'global_step'):
            self.global_step = 0
        for epoch_id in range(num_epoch):
            self.global_step += 1
            # train evm net every 10000 step
            if epoch_id != 0 and epoch_id % 10000 == 0:
                self.defreeze_evm_net(epoch_id)
            if (epoch_id - 1) % 10000 == 0:
                self.freeze_evm_net(epoch_id)

            # 計算損失
            loss, losses = loss_func()
            
            # 反向傳播
            self.opt.zero_grad()
            loss.backward()
            
            # 更新參數
            self.opt.step()
            
            if scheduler:
                scheduler.step()

            # 只在 rank 0 打印與保存
            interval = self.log_interval if self.log_interval > 0 else 100
            if self.rank == 0 and (epoch_id == 0 or (epoch_id + 1) % interval == 0 or epoch_id == num_epoch - 1):
                self.print_log(loss, losses, epoch_id, num_epoch)

            if self.rank == 0 and (epoch_id == 0 or epoch_id % 10000 == 0):
                saved_ckpt = 'model_cavity_loop%d.pth' % (epoch_id)
                layers = self.layers
                hidden_size = self.hidden_size
                N_f = self.N_f
                self.save(saved_ckpt, N_HLayer=layers, N_neu=hidden_size, N_f=N_f)

    def freeze_evm_net(self, epoch_id):
        """凍結EVM網絡參數"""
        for para in self.net_1.parameters():
                para.requires_grad = False
    
        # 重新創建優化器只包含需要訓練的參數
        self.opt = torch.optim.Adam(
                [p for p in self.net.parameters() if p.requires_grad],
                lr=self.opt.param_groups[0]['lr'],
                weight_decay=0.0
                )

    def defreeze_evm_net(self, epoch_id):
        """解凍EVM網絡參數"""
        for para in self.net_1.parameters():
                para.requires_grad = True
    
        # 重新創建優化器包含所有參數
        self.opt = torch.optim.Adam(
                list(self.net.parameters()) + list(self.net_1.parameters()),
                lr=self.opt.param_groups[0]['lr'],
                weight_decay=0.0
                )

    def print_log(self, loss, losses, epoch_id, num_epoch):
        # 多行輸出：進度條 / 損失細節 / 時間與GPU / 物理量
        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']
        now = time.time()
        if not hasattr(self, '_epoch_start_wall'):
            self._epoch_start_wall = now
        if not hasattr(self, 'cumulative_start_time'):
            self.cumulative_start_time = self._epoch_start_wall
        if not hasattr(self, '_last_log_time'):
            self._last_log_time = self._epoch_start_wall
        if not hasattr(self, '_last_log_epoch'):
            self._last_log_epoch = 0

        # 時間統計
        stage_elapsed = now - self._epoch_start_wall
        total_elapsed = now - self.cumulative_start_time
        avg_it_s = (epoch_id + 1) / stage_elapsed if stage_elapsed > 0 else 0.0
        interval_epochs = (epoch_id - self._last_log_epoch) if (epoch_id - self._last_log_epoch) > 0 else 1
        interval_time = now - self._last_log_time
        interval_it_s = interval_epochs / interval_time if interval_time > 0 else 0.0
        remain = num_epoch - (epoch_id + 1)
        eta_sec = remain / avg_it_s if avg_it_s > 0 else float('inf')

        # vis_t 與等效 Re (保持現有公式)
        if self.vis_t is not None:
            vis_t_mean = self.vis_t.detach().mean().item()
            Re_eff = 1.0 / (1.0 / self.Re + vis_t_mean)
        else:
            vis_t_mean = float('nan')
            Re_eff = float('nan')

        lr = get_lr(self.opt)
        # 損失
        loss_total = loss.detach().cpu().item()
        eq1 = self.loss_eq1.detach().cpu().item()
        eq2 = self.loss_eq2.detach().cpu().item()
        eq3 = self.loss_eq3.detach().cpu().item()
        eq4 = self.loss_eq4.detach().cpu().item()
        bc_loss = losses[1].detach().cpu().item()

        # 進度條
        width = getattr(self, 'progress_bar_width', 30)
        progress = (epoch_id + 1) / num_epoch
        filled = int(progress * width)
        bar = '█' * filled + ' ' * (width - filled)

        def fmt_t(sec):
            if sec == float('inf'):
                return 'INF'
            if sec < 60:
                return f"{sec:.1f}s"
            m, s = divmod(sec, 60)
            if m < 60:
                return f"{int(m)}m{s:04.1f}s"
            h, m = divmod(m, 60)
            return f"{int(h)}h{int(m)}m"

        # GPU 記憶體
        try:
            mem_alloc = torch.cuda.memory_allocated(self.device) / 1024**2
            mem_reserved = torch.cuda.memory_reserved(self.device) / 1024**2
            mem_total = torch.cuda.get_device_properties(self.device).total_memory / 1024**2
        except Exception:
            mem_alloc = mem_reserved = mem_total = float('nan')

        # 每區間 throughput（點/秒）: 邊界點 + 方程點 (單GPU當前區塊)
        try:
            pts = 0
            if hasattr(self, 'x_f'):
                pts += self.x_f.shape[0]
            if hasattr(self, 'x_b'):
                pts += self.x_b.shape[0]
            # interval throughput 以 interval_it_s (epochs/sec) * pts (每 epoch 處理點) 計
            throughput = interval_it_s * pts
        except Exception:
            throughput = float('nan')

        # 放大因子 (保持目前公式: 未定義 -> 省略, 可後續加入)
        amplification = 'N/A'

        header = f"[{self.current_stage}] {epoch_id+1:>7d}/{num_epoch:<7d} {progress*100:6.2f}% |{bar}|"
        line_loss = (f"  損失: total={loss_total:.3e}  方程總={self.loss_e.detach().cpu().item():.3e}  邊界={bc_loss:.3e}\n"
                     f"        eq1={eq1:.2e} eq2={eq2:.2e} eq3={eq3:.2e} eq4(熵殘差)={eq4:.2e}")
        line_time = (f"  時間: 本階段={fmt_t(stage_elapsed)}  平均/epoch={stage_elapsed/(epoch_id+1):.2f}s  interval_it/s={interval_it_s:.2f}  平均it/s={avg_it_s:.2f}\n"
                     f"        剩餘預估={fmt_t(eta_sec)}  累積總時長={fmt_t(total_elapsed)}")
        line_gpu = (f"  GPU : mem={mem_alloc:.1f}MB/{mem_total:.0f}MB (res {mem_reserved:.1f}MB)  throughput={throughput:.1f} pts/s  lr={lr:.2e}")
        line_phys = (f"  物理: 目標Re={self.Re}  Re_eff={Re_eff:.1f}  alpha_evm={self.alpha_evm}  放大因子={amplification}")

        print(header)
        print(line_loss)
        print(line_time)
        print(line_gpu)
        print(line_phys)
        print('-'*100)

        # TensorBoard 紀錄
        if hasattr(self, 'tb_writer') and self.tb_writer is not None:
            try:
                self.tb_writer.add_scalar('loss/total', loss_total, self.global_step)
                self.tb_writer.add_scalar('loss/boundary', bc_loss, self.global_step)
                self.tb_writer.add_scalar('loss/eq_total', self.loss_e.detach().cpu().item(), self.global_step)
                self.tb_writer.add_scalar('loss/eq1', eq1, self.global_step)
                self.tb_writer.add_scalar('loss/eq2', eq2, self.global_step)
                self.tb_writer.add_scalar('loss/eq3', eq3, self.global_step)
                self.tb_writer.add_scalar('loss/eq4_entropy', eq4, self.global_step)
                self.tb_writer.add_scalar('physics/Re_eff', Re_eff, self.global_step)
                self.tb_writer.add_scalar('physics/alpha_evm', self.alpha_evm, self.global_step)
                self.tb_writer.add_scalar('perf/throughput_pts_per_s', throughput, self.global_step)
                self.tb_writer.add_scalar('perf/avg_iter_s', avg_it_s, self.global_step)
                self.tb_writer.add_scalar('perf/interval_iter_s', interval_it_s, self.global_step)
                self.tb_writer.add_scalar('lr', lr, self.global_step)
            except Exception:
                pass

        # 更新區間狀態
        self._last_log_time = now
        self._last_log_epoch = epoch_id

    def get_runtime_stats(self, epoch_id: int, num_epoch: int):
        """回傳當前訓練速度與等效 Re 統計，供外部擴充使用。"""
        now = time.time()
        if not hasattr(self, '_epoch_start_wall'):
            return {}
        elapsed = now - self._epoch_start_wall
        avg_it_s = (epoch_id + 1) / elapsed if elapsed > 0 else 0.0
        remain = num_epoch - (epoch_id + 1)
        eta_sec = remain / avg_it_s if avg_it_s > 0 else float('inf')
        if self.vis_t is not None:
            vis_t_mean = self.vis_t.detach().mean().item()
            Re_eff = 1.0 / (1.0 / self.Re + vis_t_mean)
        else:
            vis_t_mean = float('nan'); Re_eff = float('nan')
        return dict(avg_it_s=avg_it_s, eta_seconds=eta_sec, vis_t_mean=vis_t_mean, Re_eff=Re_eff)


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

            save_dir = './results/Re5000/test_result'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            scipy.io.savemat(os.path.join(save_dir, f'cavity_result_loop_{loop}.mat'),
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

        # Save model state dict (unwrap if DDP)
        net_to_save = self.net.module if self.is_distributed else self.net
        net1_to_save = self.net_1.module if self.is_distributed else self.net_1
        torch.save(net_to_save.state_dict(), save_results_to+filename)
        torch.save(net1_to_save.state_dict(), save_results_to+filename+'_evm')

    def divergence(self, x_star, y_star):
        (self.eq1_pred, self.eq2_pred,
         self.eq3_pred, self.eq4_pred) = self.neural_net_equations(x_star, y_star)
        div = self.eq3_pred
        return div

    def set_dns_data(self, dns_points=None, dns_values=None, custom_weights=None):
        """
        設置DNS數據用於監督學習
        
        Args:
            dns_points: DNS座標點 [N, 2] 或 None（將生成隨機點）
            dns_values: DNS對應的物理量值 [N, num_vars]
            custom_weights: 自定義權重 [N] 或 標量
        """
        if not self.enable_dns_enhancement:
            if self.rank == 0:
                print("DNS enhancement not enabled. Call with enable_dns_enhancement=True")
            return
            
        if dns_points is None:
            # 生成隨機座標點
            dns_points = self.dns_enhancer.generate_random_points(
                self.dns_points_count,
                domain_bounds=[(0.0, 1.0), (0.0, 1.0)],
                distribution='uniform')
            if self.rank == 0:
                print(f"Generated {self.dns_points_count} random DNS points")
        else:
            dns_points = torch.tensor(dns_points, dtype=torch.float32).to(self.device)
            
        if dns_values is None:
            if self.rank == 0:
                print("Warning: No DNS values provided. DNS loss will be zero.")
            dns_values = torch.zeros((dns_points.shape[0], 3), device=self.device)  # u, v, p
        else:
            dns_values = torch.tensor(dns_values, dtype=torch.float32).to(self.device)
            
        if custom_weights is not None:
            self.dns_enhancer.set_custom_loss_weights(custom_weights, dns_points.shape[0])
        else:
            # 使用預設權重
            self.dns_enhancer.set_custom_loss_weights(1.0, dns_points.shape[0])
        
        self.dns_data = {
            'points': dns_points,
            'values': dns_values,
            'weights': self.dns_enhancer.custom_weights
        }
        
        if self.rank == 0:
            print(f"DNS data set: {dns_points.shape[0]} points with {dns_values.shape[1]} variables")
            
    def compute_dns_loss(self):
        """
        計算DNS監督數據損失
        """
        if self.dns_data is None:
            return torch.tensor(0.0, device=self.device)
            
        dns_points = self.dns_data['points']
        dns_values = self.dns_data['values']  # [N, num_vars] 假設包含 u, v, p
        dns_weights = self.dns_data['weights']
        
        # 拆分座標
        x_dns = dns_points[:, 0:1]
        y_dns = dns_points[:, 1:2]
        
        # 預測值
        u_pred, v_pred, p_pred, _ = self.neural_net_u(x_dns, y_dns)
        
        # 計算各項損失
        u_dns_loss = torch.square(dns_values[:, 0] - u_pred.reshape(-1))
        v_dns_loss = torch.square(dns_values[:, 1] - v_pred.reshape(-1))
        
        if dns_values.shape[1] > 2:  # 包含壓力
            p_dns_loss = torch.square(dns_values[:, 2] - p_pred.reshape(-1))
            total_dns_loss = u_dns_loss + v_dns_loss + p_dns_loss
        else:
            total_dns_loss = u_dns_loss + v_dns_loss
        
        # 應用權重
        weighted_dns_loss = total_dns_loss * dns_weights
        
        return torch.mean(weighted_dns_loss)
        
    def update_dns_weights(self, new_weights):
        """
        更新DNS數據權重
        
        Args:
            new_weights: 新權重值
        """
        if self.dns_data is not None:
            self.dns_enhancer.set_custom_loss_weights(new_weights, len(self.dns_data['points']))
            self.dns_data['weights'] = self.dns_enhancer.custom_weights
            if self.rank == 0:
                print(f"DNS weights updated. Mean weight: {torch.mean(self.dns_data['weights']):.4f}")
        
    def get_training_enhancement_stats(self):
        """
        獲取訓練增強統計資訊
        """
        stats = {}

        if self.enable_dns_enhancement and self.dns_data is not None:
            stats['dns_data'] = {
                'num_points': len(self.dns_data['points']),
                'mean_weight': float(torch.mean(self.dns_data['weights'])),
                'loss_weight_factor': float(self.dns_loss_weight)
            }
            
        return stats
