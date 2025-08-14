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
from tools import *
#from scipy.stats import qmc


class DataLoader:
    def __init__(self, path=None, N_f=20000, N_b=1000):

        '''
        N_f: Num of residual points
        N_b: Num of boundary points
        '''
        self.N_b = N_b
        self.x_min = 0.0
        self.x_max = 1.0
        self.y_min = 0.0
        self.y_max = 1.0
        self.N_f = N_f # equation points
        self.pts_bc = None

    def loading_boundary_data(self):
        # boundary points
        Nx = 513
        Ny = 513
        dx = 1.0/(Nx-1)
        r_const = 50

        upper_x = np.linspace(self.x_min, self.x_max, num=Nx)
        u_upper = 1 -  np.cosh(r_const*(upper_x-0.5)) / np.cosh(r_const*0.5)
        #  lower upper left right
        x_b = np.concatenate([np.linspace(self.x_min, self.x_max, num=Nx),
                              np.linspace(self.x_min, self.x_max, num=Nx),
                              self.x_min * np.ones([Ny]),
                              self.x_max * np.ones([Ny])], 
                              axis=0).reshape([-1, 1])
        y_b = np.concatenate([self.y_min * np.ones([Nx]),
                              self.y_max * np.ones([Nx]),
                              np.linspace(self.y_min, self.y_max, num=Ny),
                              np.linspace(self.y_min, self.y_max, num=Ny)],
                              axis=0).reshape([-1, 1])
        u_b = np.concatenate([np.zeros([Nx]),
                              u_upper,
                              np.zeros([Ny]),
                              np.zeros([Ny])],
                              axis=0).reshape([-1, 1])
        v_b = np.zeros([x_b.shape[0]]).reshape([-1, 1])

        x_pbc = np.linspace(self.x_min, self.x_max, num=Nx).reshape([-1, 1]);
        y_pbc = np.zeros(x_pbc.shape[0]).reshape([-1,1]);
        p_pbc = np.zeros(x_pbc.shape[0]).reshape([-1,1]);

        self.pts_bc = np.hstack((x_b,y_b))
      
        N_train_bcs = x_b.shape[0]
        print('-----------------------------')
        print('N_train_bcs: ' + str(N_train_bcs) )
        print('N_train_equ: ' + str(self.N_f) )
        print('-----------------------------')     
        return x_b, y_b, u_b, v_b 

    def loading_training_data(self):
        #idx = np.random.choice(x_star.shape[0], N_f, replace=False)
        #x_train_f = x_star[idx,:]
        #y_train_f = y_star[idx,:]
        xye = LHSample(2, [[self.x_min, self.x_max], [self.y_min, self.y_max]], self.N_f)
        #sampler = qmc.Halton(d=2)
        #xye = sampler.random(n=N_f)
        if self.pts_bc is not None:
            xye_sorted, _ = sort_pts(xye, self.pts_bc)
        else:
            print("need to load boundary data first!")
            raise 
        x_train_f = xye_sorted[:, 0:1]
        y_train_f = xye_sorted[:, 1:2]
        return x_train_f, y_train_f

    def loading_evaluate_data(self, filename):
        """ preparing training data """
        data = scipy.io.loadmat(filename)
        x = data['X_ref']
        y = data['Y_ref']
        u = data['U_ref']
        v = data['V_ref']
        p = data['P_ref']
        x_star = x.reshape(-1,1)
        y_star = y.reshape(-1,1)
        u_star = u.reshape(-1,1)
        v_star = v.reshape(-1,1)
        p_star = p.reshape(-1,1)
        return x_star, y_star, u_star, v_star, p_star

    def loading_supervision_data(self, filename, num_points=1, random_seed=42):
        """
        从真实数据中随机采样固定数量的监督点
        
        Args:
            filename: .mat文件路径
            num_points: 采样点数量，默认1个点
            random_seed: 随机种子，确保可重现性
        
        Returns:
            x_sup, y_sup, u_sup, v_sup, p_sup: 监督数据点的坐标和物理量
        """
        if num_points <= 0:
            # 返回空张量，但保持正确的维度
            empty_tensor = np.empty((0, 1))
            return empty_tensor, empty_tensor, empty_tensor, empty_tensor, empty_tensor
        
        # 设置随机种子确保可重现性
        np.random.seed(random_seed)
        
        # 加载数据
        data = scipy.io.loadmat(filename)
        x = data['X_ref']
        y = data['Y_ref']  
        u = data['U_ref']
        v = data['V_ref']
        p = data['P_ref']
        
        # 展平数据
        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)
        u_flat = u.reshape(-1)
        v_flat = v.reshape(-1)
        p_flat = p.reshape(-1)
        
        # 随机采样指定数量的点
        total_points = x_flat.shape[0]
        indices = np.random.choice(total_points, size=num_points, replace=False)
        
        # 提取采样点数据
        x_sup = x_flat[indices].reshape(-1, 1)
        y_sup = y_flat[indices].reshape(-1, 1)
        u_sup = u_flat[indices].reshape(-1, 1)
        v_sup = v_flat[indices].reshape(-1, 1)
        p_sup = p_flat[indices].reshape(-1, 1)
        
        print(f'-----------------------------')
        print(f'Supervision data loaded: {num_points} points')
        if num_points > 0:
            print(f'Sample point coordinates: x={x_sup[0,0]:.4f}, y={y_sup[0,0]:.4f}')
            print(f'Sample point values: u={u_sup[0,0]:.4f}, v={v_sup[0,0]:.4f}, p={p_sup[0,0]:.4f}')
        print(f'-----------------------------')
        
        return x_sup, y_sup, u_sup, v_sup, p_sup
    
