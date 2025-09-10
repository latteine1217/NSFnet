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
from tools import normalize_coordinates, LHSample, sort_pts


class DataLoader:
    def __init__(self, path=None, N_f=20000, N_b=1000, sort_by_boundary_distance: bool = True, derivative_rescale_config=None):

        '''
        N_f: Num of residual points
        N_b: Num of boundary points
        derivative_rescale_config: 導數縮放配置字典 (可選)
        '''
        self.N_b = N_b
        self.x_min = 0.0
        self.x_max = 1.0
        self.y_min = 0.0
        self.y_max = 1.0
        self.N_f = N_f # equation points
        self.pts_bc = None
        self.sort_by_boundary_distance = sort_by_boundary_distance
        self.derivative_rescale_config = derivative_rescale_config

    def loading_boundary_data(self):
        # boundary points
        Nx = 513
        Ny = 513
        dx = 1.0/(Nx-1)
        r_const = 10

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
        if self.pts_bc is None:
            print("need to load boundary data first!")
            raise
        if self.sort_by_boundary_distance:
            xye_sorted, _ = sort_pts(xye, self.pts_bc)
        else:
            # 跳過距離排序，直接使用LHS樣本
            xye_sorted = xye
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
        
        # 座標變換: [0,1] → [-1,1]
        #x, y = normalize_coordinates(x, y, from_range=(0, 1), to_range=(-1, 1), 
        #                           derivative_rescale_config=self.derivative_rescale_config)
        
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
        
        # 座標變換: [0,1] → [-1,1]
        #x, y = normalize_coordinates(x, y, from_range=(0, 1), to_range=(-1, 1), 
        #                           derivative_rescale_config=self.derivative_rescale_config)
        
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
            print(f'Sample point coordinates: x={x_sup[0,0]:.6f}, y={y_sup[0,0]:.6f}')
            print(f'Sample point values: u={u_sup[0,0]:.6f}, v={v_sup[0,0]:.6f}, p={p_sup[0,0]:.6f}')
            print(f'💡 使用 loader.print_supervision_locations() 查看所有監督點詳情')
        print(f'-----------------------------')
        
        return x_sup, y_sup, u_sup, v_sup, p_sup

    def print_supervision_locations(self, filename, num_points=1, random_seed=42):
        """
        打印所有監督點的詳細位置信息
        
        Args:
            filename: .mat文件路径
            num_points: 监督点数量
            random_seed: 随机种子，确保可重现性
        """
        print(f'====== 監督數據點位置詳情 ======')
        print(f'數據文件: {filename}')
        print(f'監督點數: {num_points}')
        print(f'隨機種子: {random_seed}')
        print(f'--------------------------------')
        
        if num_points <= 0:
            print('⚠️  未使用監督數據點')
            print(f'================================')
            return
        
        # 載入監督數據
        x_sup, y_sup, u_sup, v_sup, p_sup = self.loading_supervision_data(
            filename, num_points, random_seed)
        
        # 打印每個監督點的詳細信息
        for i in range(num_points):
            print(f'📍 監督點 {i+1:>2}:')
            print(f'   座標: x = {x_sup[i,0]:>8.6f}, y = {y_sup[i,0]:>8.6f}')
            print(f'   速度: u = {u_sup[i,0]:>8.6f}, v = {v_sup[i,0]:>8.6f}')
            print(f'   壓力: p = {p_sup[i,0]:>8.6f}')
            
            # 計算與計算域中心和邊界的距離
            center_dist = np.sqrt((x_sup[i,0] - 0.0)**2 + (y_sup[i,0] - 0.0)**2)
            boundary_dist = min(x_sup[i,0]+1, 1-x_sup[i,0], y_sup[i,0]+1, 1-y_sup[i,0])
            
            print(f'   距中心: {center_dist:>8.6f}')
            print(f'   距邊界: {boundary_dist:>8.6f}')
            
            # 區域標識
            if boundary_dist < 0.1:
                region = "邊界區"
            elif center_dist < 0.2:
                region = "中心區"
            else:
                region = "主流區"
            print(f'   區域位置: {region}')
            
            if i < num_points - 1:
                print(f'   ................................')
        
        print(f'================================')
    
