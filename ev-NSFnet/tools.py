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
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 17:00:36 2022

@author: Shengze Cai
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import os


def normalize_coordinates(x, y, from_range=(0, 1), to_range=(-1, 1), derivative_rescale_config=None):
    """
    統一座標變換函數：將座標從一個範圍轉換到另一個範圍
    支持導數縮放配置檢查和提示
    
    Args:
        x, y: 輸入座標 (numpy array 或 tensor)
        from_range: 原始範圍 tuple (min, max)
        to_range: 目標範圍 tuple (min, max) 
        derivative_rescale_config: 導數縮放配置字典 (可選)
        
    Returns:
        x_norm, y_norm: 變換後的座標
    """
    from_min, from_max = from_range
    to_min, to_max = to_range
    
    # 計算縮放因子
    scale_factor = (to_max - to_min) / (from_max - from_min)
    
    # 檢查導數縮放配置
    if derivative_rescale_config is not None and isinstance(derivative_rescale_config, dict):
        is_enabled = derivative_rescale_config.get('enable', False)
        first_order_scale = derivative_rescale_config.get('first_order_scale', 1.0)
        second_order_scale = derivative_rescale_config.get('second_order_scale', 4.0)
        
        if is_enabled:
            print(f"🔄 導數縮放已啟用: 座標縮放因子={scale_factor:.2f}")
            print(f"   一階導數縮放: {first_order_scale:.1f}x")  
            print(f"   二階導數縮放: {second_order_scale:.1f}x")
        else:
            print(f"⚪ 導數縮放已關閉: 使用標準座標變換 (縮放因子={scale_factor:.2f})")
    
    # 標準化到 [0,1]
    x_norm = (x - from_min) / (from_max - from_min)
    y_norm = (y - from_min) / (from_max - from_min)
    
    # 變換到目標範圍
    x_norm = x_norm * (to_max - to_min) + to_min
    y_norm = y_norm * (to_max - to_min) + to_min
    
    return x_norm, y_norm


def setup_device(local_rank=None, logger=None):
    """
    統一設備管理函數：自動檢測和設置CUDA設備
    
    Args:
        local_rank: 本地rank編號，用於分布式訓練
        logger: 日誌記錄器（可選）
        
    Returns:
        device: torch.device對象
    """
    try:
        if torch.cuda.is_available():
            if local_rank is not None:
                # 分布式訓練模式
                device = torch.device(f'cuda:{local_rank}')
                try:
                    torch.cuda.set_device(local_rank)
                    if logger:
                        logger.info(f"✅ 設置CUDA設備: cuda:{local_rank}")
                except (RuntimeError, AttributeError) as e:
                    # 設備設置失敗，回退到默認CUDA設備
                    device = torch.device('cuda:0')
                    if logger:
                        logger.warning(f"⚠️ CUDA設備 {local_rank} 設置失敗，使用 cuda:0: {e}")
            else:
                # 單GPU模式
                device = torch.device('cuda')
                if logger:
                    logger.info("✅ 使用CUDA設備")
        else:
            # CPU模式
            device = torch.device('cpu')
            if logger:
                logger.warning("⚠️ CUDA不可用，使用CPU")
            else:
                print("⚠️ CUDA不可用，使用CPU")
                
        return device
        
    except Exception as e:
        # 異常情況，回退到CPU
        device = torch.device('cpu')
        if logger:
            logger.error(f"❌ 設備設置失敗，回退到CPU: {e}")
        else:
            print(f"❌ 設備設置失敗，回退到CPU: {e}")
        return device


def get_cuda_info(device, logger=None):
    """
    獲取CUDA設備信息
    
    Args:
        device: torch.device對象
        logger: 日誌記錄器（可選）
        
    Returns:
        info_dict: 包含設備信息的字典
    """
    info = {}
    
    if device.type == 'cuda' and torch.cuda.is_available():
        try:
            info['device_name'] = torch.cuda.get_device_name(device)
            info['total_memory'] = torch.cuda.get_device_properties(device).total_memory / 1024**3
            info['allocated_memory'] = torch.cuda.memory_allocated(device) / 1024**3
            info['reserved_memory'] = torch.cuda.memory_reserved(device) / 1024**3
            info['device_index'] = device.index if device.index is not None else 0
            
            if logger:
                logger.info(f"📱 GPU信息: {info['device_name']}")
                logger.info(f"💾 總記憶體: {info['total_memory']:.2f}GB")
                logger.info(f"🔧 已分配: {info['allocated_memory']:.2f}GB")
                logger.info(f"📦 已保留: {info['reserved_memory']:.2f}GB")
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ 無法獲取CUDA信息: {e}")
            info['error'] = str(e)
    else:
        info['device_type'] = 'cpu'
        if logger:
            logger.info("📱 使用CPU設備")
    
    return info


def LHSample(D, bounds, N):
    """
    Latin Hypercube Sampling
    
    Args:
        D: Number of parameters
        bounds: [[min_1, max_1],[min_2, max_2],[min_3, max_3]](list)
        N: Number of samples
    Returns:
        result: Samples
    """
    result = np.empty([N, D])
    temp = np.empty([N])
    d = 1.0 / N
    for i in range(D):
        for j in range(N):
            temp[j] = np.random.uniform(low=j * d, high=(j + 1) * d, size=1)[0]
        np.random.shuffle(temp)
        for j in range(N):
            result[j, i] = temp[j]
    # Stretching the sampling
    b = np.array(bounds)
    lower_bounds = b[:, 0]
    upper_bounds = b[:, 1]
    if np.any(lower_bounds > upper_bounds):
        print('Wrong value bound')
        return None
    #   sample * (upper_bound - lower_bound) + lower_bound
    np.add(np.multiply(result, (upper_bounds - lower_bounds), out=result),
           lower_bounds,
           out=result)
    return result


def distance(p1, p2):
    """Return the distance between two points"""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def minDistance(pt, pts2):
    """Return the min distance between one point and a set of points"""
    dists = [distance(pt, i) for i in pts2]
    return min(dists)


def sort_pts(pts1, pts2, flag_reverse=False):
    """Sort a set of points based on their distances to another set of points"""
    minDists = []
    for pt in pts1:
        minDists.append(minDistance(pt, pts2))
    minDists = np.array(minDists).reshape(1, -1)
    
    dists_sorted = np.sort(minDists).reshape(-1, 1)
    sort_index = np.argsort(minDists)
    if flag_reverse:
        sort_index = sort_index.reshape(-1, 1)
        sort_index = sort_index[::-1].reshape(1, -1)
        dists_sorted = dists_sorted[::-1]
    pts1_sorted = pts1[sort_index, :]
    pts1_sorted = np.squeeze(pts1_sorted)
    return pts1_sorted, dists_sorted
