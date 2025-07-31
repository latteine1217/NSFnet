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
import numpy as np
import scipy.io
import torch
from tools import *

class DataLoader:
    def __init__(self, path=None, N_f=20000, N_b=1000, device='cpu'):
        self.N_b = N_b
        self.x_min = 0.0
        self.x_max = 1.0
        self.y_min = 0.0
        self.y_max = 1.0
        self.N_f = N_f
        self.device = device

        # --- 一次性生成並快取所有數據 ---
        self._generate_and_cache_data()

    def _generate_and_cache_data(self):
        # 1. 生成邊界數據 (Boundary Data)
        Nx = 513
        Ny = 513
        r_const = 50

        upper_x = np.linspace(self.x_min, self.x_max, num=Nx)
        u_upper = 1 - np.cosh(r_const * (upper_x - 0.5)) / np.cosh(r_const * 0.5)

        x_b_np = np.concatenate([
            np.linspace(self.x_min, self.x_max, num=Nx),
            np.linspace(self.x_min, self.x_max, num=Nx),
            self.x_min * np.ones([Ny]),
            self.x_max * np.ones([Ny])
        ]).reshape([-1, 1])

        y_b_np = np.concatenate([
            self.y_min * np.ones([Nx]),
            self.y_max * np.ones([Nx]),
            np.linspace(self.y_min, self.y_max, num=Ny),
            np.linspace(self.y_min, self.y_max, num=Ny)
        ]).reshape([-1, 1])

        u_b_np = np.concatenate([
            np.zeros([Nx]),
            u_upper,
            np.zeros([Ny]),
            np.zeros([Ny])
        ]).reshape([-1, 1])

        v_b_np = np.zeros_like(x_b_np)

        # 將 NumPy 陣列轉換為 PyTorch 張量並直接移動到目標設備
        self.x_b_tensor = torch.from_numpy(x_b_np).float().to(self.device)
        self.y_b_tensor = torch.from_numpy(y_b_np).float().to(self.device)
        self.u_b_tensor = torch.from_numpy(u_b_np).float().to(self.device)
        self.v_b_tensor = torch.from_numpy(v_b_np).float().to(self.device)

        # 2. 生成內部訓練數據 (Collocation Points)
        pts_bc_np = np.hstack((x_b_np, y_b_np))
        xye_np = LHSample(2, [[self.x_min, self.x_max], [self.y_min, self.y_max]], self.N_f)
        
        xye_sorted_np, _ = sort_pts(xye_np, pts_bc_np)
        
        x_f_np = xye_sorted_np[:, 0:1]
        y_f_np = xye_sorted_np[:, 1:2]

        # 將 NumPy 陣列轉換為 PyTorch 張量並直接移動到目標設備
        self.x_f_tensor = torch.from_numpy(x_f_np).float().to(self.device)
        self.y_f_tensor = torch.from_numpy(y_f_np).float().to(self.device)

        print('-----------------------------')
        print(f'N_train_bcs: {self.x_b_tensor.shape[0]}')
        print(f'N_train_equ: {self.x_f_tensor.shape[0]}')
        print('--- Data cached as Tensors --')
        print('-----------------------------')

    def loading_boundary_data(self):
        # 直接返回快取的張量
        return self.x_b_tensor, self.y_b_tensor, self.u_b_tensor, self.v_b_tensor

    def loading_training_data(self):
        # 直接返回快取的張量
        return self.x_f_tensor, self.y_f_tensor

    def loading_evaluate_data(self, filename):
        data = scipy.io.loadmat(filename)
        x = data['X_ref']
        y = data['Y_ref']
        u = data['U_ref']
        v = data['V_ref']
        p = data['P_ref']
        x_star = x.reshape(-1, 1)
        y_star = y.reshape(-1, 1)
        u_star = u.reshape(-1, 1)
        v_star = v.reshape(-1, 1)
        p_star = p.reshape(-1, 1)
        return x_star, y_star, u_star, v_star, p_star
    
