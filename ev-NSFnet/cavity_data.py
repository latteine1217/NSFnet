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
from scipy.spatial import cKDTree
from tools import *
#from scipy.stats import qmc


class DataLoader:
    def __init__(self, path=None, N_f=20000, N_b=1000, sort_training_points=True, sdf_weighting=None, coord_transform=False):

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
        self.sort_training_points = sort_training_points
        self.sdf_config = sdf_weighting
        self.sdf_enabled = bool(getattr(self.sdf_config, 'enabled', False)) if self.sdf_config is not None else False
        self.sdf_weights = None
        self._bc_tree = None
        self.coord_transform = coord_transform
        self.coord_scale = 2.0 if self.coord_transform else 1.0

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

        pts = np.hstack((x_b,y_b))
        if self.coord_transform:
            pts = self._to_centered_coords(pts)
            x_b = pts[:, 0:1]
            y_b = pts[:, 1:2]
            self.x_min, self.x_max = -1.0, 1.0
            self.y_min, self.y_max = -1.0, 1.0
        self.pts_bc = pts
        if self.sdf_enabled:
            self._bc_tree = cKDTree(self.pts_bc)
      
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
        if self.coord_transform:
            xye = self._to_centered_coords(xye)
        #sampler = qmc.Halton(d=2)
        #xye = sampler.random(n=N_f)
        if self.pts_bc is None:
            print("need to load boundary data first!")
            raise 
        if self.sort_training_points:
            xye, _ = sort_pts(xye, self.pts_bc)
        if self.sdf_enabled:
            self._compute_sdf_weights(xye)
        else:
            self.sdf_weights = None
        x_train_f = xye[:, 0:1]
        y_train_f = xye[:, 1:2]
        return x_train_f, y_train_f

    def _compute_sdf_weights(self, pts):
        if self._bc_tree is None:
            self._bc_tree = cKDTree(self.pts_bc)
        dists, _ = self._bc_tree.query(pts)
        min_w = float(getattr(self.sdf_config, 'min_weight', 0.2)) if self.sdf_config else 0.2
        decay = float(getattr(self.sdf_config, 'decay', 5.0)) if self.sdf_config else 5.0
        min_w = max(1e-6, min(min_w, 1.0))
        decay = max(0.0, decay)
        weights = min_w + (1.0 - min_w) * np.exp(-decay * dists)
        mean_w = np.mean(weights)
        if mean_w > 0:
            weights = weights / mean_w
        self.sdf_weights = weights.astype(np.float32)

    def get_sdf_weights(self):
        return self.sdf_weights

    def _to_centered_coords(self, pts):
        return pts * 2.0 - 1.0

    def _to_centered_values(self, values):
        return values * 2.0 - 1.0

    def get_coord_scale(self):
        return self.coord_scale

    def loading_evaluate_data(self, filename):
        """ preparing training data """
        data = scipy.io.loadmat(filename)
        x = data['X_ref']
        y = data['Y_ref']
        u = data['U_ref']
        v = data['V_ref']
        p = data['P_ref']
        if self.coord_transform:
            x = self._to_centered_values(x)
            y = self._to_centered_values(y)
        x_star = x.reshape(-1,1)
        y_star = y.reshape(-1,1)
        u_star = u.reshape(-1,1)
        v_star = v.reshape(-1,1)
        p_star = p.reshape(-1,1)
        return x_star, y_star, u_star, v_star, p_star
    
