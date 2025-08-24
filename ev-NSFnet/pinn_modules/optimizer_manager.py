# Copyright (c) 2023 scien42.tech, Se42 Authors. All Rights Reserved.
#
# PINN 優化器和排程器管理模組

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    LinearLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, 
    MultiStepLR, SequentialLR
)
from typing import Dict, List, Any, Optional, Union

class OptimizerSchedulerManager:
    """
    PINN 優化器和排程器管理器
    統一管理 Adam、L-BFGS 優化器和各種學習率排程器
    """
    
    def __init__(self, logger=None):
        self.logger = logger
        self.optimizer = None
        self.scheduler = None
        self.scheduler_params = {}
        
    def setup_optimizer(self, parameters, lr=1e-3, betas=(0.9, 0.999)):
        """設置 Adam 優化器"""
        self.optimizer = optim.Adam(parameters, lr=lr, betas=betas)
        if self.logger:
            self.logger.info(f"✅ Adam optimizer initialized with lr={lr}")
        return self.optimizer
    
    def setup_lbfgs_optimizer(self, parameters, lr=1.0, max_iter=20):
        """設置 L-BFGS 優化器"""
        lbfgs_optimizer = optim.LBFGS(
            parameters, 
            lr=lr, 
            max_iter=max_iter,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            history_size=100
        )
        if self.logger:
            self.logger.info(f"✅ L-BFGS optimizer initialized with lr={lr}, max_iter={max_iter}")
        return lbfgs_optimizer
    
    def setup_scheduler(self, optimizer, scheduler_type, scheduler_params=None):
        """設置學習率排程器"""
        if scheduler_params is None:
            scheduler_params = {}
        
        self.scheduler_params = scheduler_params.copy()
        
        if scheduler_type == 'Constant':
            self.scheduler = None
            if self.logger:
                self.logger.info("📈 Using constant learning rate")
        
        elif scheduler_type == 'LinearLR':
            params = {
                'start_factor': scheduler_params.get('start_factor', 1.0),
                'end_factor': scheduler_params.get('end_factor', 0.1),
                'total_iters': scheduler_params.get('total_iters', 1000)
            }
            self.scheduler = LinearLR(optimizer, **params)
            if self.logger:
                self.logger.info(f"📈 LinearLR scheduler setup: {params}")
        
        elif scheduler_type == 'CosineAnnealingWarmRestarts':
            params = {
                'T_0': scheduler_params.get('T_0', 100),
                'T_mult': scheduler_params.get('T_mult', 2),
                'eta_min': scheduler_params.get('eta_min', 1e-8)
            }
            self.scheduler = CosineAnnealingWarmRestarts(optimizer, **params)
            if self.logger:
                self.logger.info(f"📈 CosineAnnealingWarmRestarts scheduler setup: {params}")
        
        elif scheduler_type == 'CosineAnnealingLR':
            params = {
                'T_max': scheduler_params.get('T_max', 1000),
                'eta_min': scheduler_params.get('eta_min', 1e-8)
            }
            self.scheduler = CosineAnnealingLR(optimizer, **params)
            if self.logger:
                self.logger.info(f"📈 CosineAnnealingLR scheduler setup: {params}")
        
        elif scheduler_type == 'MultiStepLR':
            params = {
                'milestones': scheduler_params.get('milestones', [100, 200]),
                'gamma': scheduler_params.get('gamma', 0.1)
            }
            self.scheduler = MultiStepLR(optimizer, **params)
            if self.logger:
                self.logger.info(f"📈 MultiStepLR scheduler setup: {params}")
        
        elif scheduler_type == 'SGDR':
            # SGDR: LinearLR + CosineAnnealingWarmRestarts 組合
            linear_params = {
                'start_factor': scheduler_params.get('start_factor', 1.0),
                'end_factor': scheduler_params.get('end_factor', 1.0),
                'total_iters': scheduler_params.get('total_iters', 1000)
            }
            cosine_params = {
                'T_0': scheduler_params.get('T_0', 100),
                'T_mult': scheduler_params.get('T_mult', 2),
                'eta_min': scheduler_params.get('eta_min', 1e-8)
            }
            
            schedulers = [
                LinearLR(optimizer, **linear_params),
                CosineAnnealingWarmRestarts(optimizer, **cosine_params)
            ]
            milestones = [linear_params['total_iters']]
            
            self.scheduler = SequentialLR(optimizer, schedulers, milestones)
            if self.logger:
                self.logger.info(f"📈 SGDR scheduler setup - Linear: {linear_params}, Cosine: {cosine_params}")
        
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
        return self.scheduler
    
    def rebuild_scheduler(self, optimizer):
        """
        重建排程器 - 用於 EVM 網路凍結/解凍時重建
        這是修復 SequentialLR 問題的核心方法
        """
        if self.scheduler is None:
            return None
        
        scheduler_class_name = self.scheduler.__class__.__name__
        
        if scheduler_class_name == 'SequentialLR':
            # 處理 SequentialLR 重建
            return self._rebuild_sequential_scheduler(optimizer)
        else:
            # 處理其他類型的排程器
            return self._rebuild_simple_scheduler(optimizer, scheduler_class_name)
    
    def _rebuild_sequential_scheduler(self, optimizer):
        """重建 SequentialLR (SGDR) 排程器"""
        try:
            # 提取原始排程器資訊
            if hasattr(self.scheduler, '_schedulers'):
                original_schedulers = self.scheduler._schedulers
            else:
                # 備用方案：使用保存的參數重建
                return self._rebuild_sgdr_from_params(optimizer)
            
            if hasattr(self.scheduler, 'milestones'):
                original_milestones = self.scheduler.milestones
            else:
                original_milestones = []
            
            # 重建子排程器
            new_schedulers = []
            for sched in original_schedulers:
                sched_type = type(sched).__name__
                
                if sched_type == 'LinearLR' and hasattr(sched, 'total_iters'):
                    new_schedulers.append(LinearLR(
                        optimizer,
                        start_factor=getattr(sched, 'start_factor', 1.0),
                        end_factor=getattr(sched, 'end_factor', 1.0),
                        total_iters=sched.total_iters
                    ))
                elif sched_type == 'CosineAnnealingWarmRestarts' and hasattr(sched, 'T_0'):
                    new_schedulers.append(CosineAnnealingWarmRestarts(
                        optimizer,
                        T_0=sched.T_0,
                        T_mult=getattr(sched, 'T_mult', 1),
                        eta_min=getattr(sched, 'eta_min', 0)
                    ))
                elif sched_type == 'CosineAnnealingLR' and hasattr(sched, 'T_max'):
                    new_schedulers.append(CosineAnnealingLR(
                        optimizer,
                        T_max=sched.T_max,
                        eta_min=getattr(sched, 'eta_min', 0)
                    ))
                elif sched_type == 'MultiStepLR' and hasattr(sched, 'milestones'):
                    new_schedulers.append(MultiStepLR(
                        optimizer,
                        milestones=list(sched.milestones),
                        gamma=sched.gamma
                    ))
            
            # 智能重建 milestones
            if not original_milestones:
                # 如果原始 milestones 為空，從 LinearLR 中提取 total_iters
                if new_schedulers and hasattr(new_schedulers[0], 'total_iters'):
                    milestones = [new_schedulers[0].total_iters]
                else:
                    milestones = [1000]  # 默認值
                
                if self.logger:
                    self.logger.info(f"🔄 SequentialLR milestone 重建: {milestones}")
            else:
                milestones = list(original_milestones)
            
            # 創建新的 SequentialLR
            new_scheduler = SequentialLR(optimizer, new_schedulers, milestones)
            
            if self.logger:
                self.logger.info("✅ SequentialLR 重建成功")
            
            self.scheduler = new_scheduler
            return new_scheduler
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ SequentialLR 重建失敗: {e}")
            # 嘗試備用方案
            return self._rebuild_sgdr_from_params(optimizer)
    
    def _rebuild_sgdr_from_params(self, optimizer):
        """從保存的參數重建 SGDR 排程器"""
        try:
            linear_params = {
                'start_factor': self.scheduler_params.get('start_factor', 1.0),
                'end_factor': self.scheduler_params.get('end_factor', 1.0),
                'total_iters': self.scheduler_params.get('total_iters', 1000)
            }
            cosine_params = {
                'T_0': self.scheduler_params.get('T_0', 100),
                'T_mult': self.scheduler_params.get('T_mult', 2),
                'eta_min': self.scheduler_params.get('eta_min', 1e-8)
            }
            
            schedulers = [
                LinearLR(optimizer, **linear_params),
                CosineAnnealingWarmRestarts(optimizer, **cosine_params)
            ]
            milestones = [linear_params['total_iters']]
            
            new_scheduler = SequentialLR(optimizer, schedulers, milestones)
            
            if self.logger:
                self.logger.info("✅ SGDR 從參數重建成功")
            
            self.scheduler = new_scheduler
            return new_scheduler
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ SGDR 參數重建失敗: {e}")
            return None
    
    def _rebuild_simple_scheduler(self, optimizer, scheduler_class_name):
        """重建簡單排程器 (非 SequentialLR)"""
        try:
            if scheduler_class_name == 'LinearLR':
                params = {
                    'start_factor': self.scheduler_params.get('start_factor', 1.0),
                    'end_factor': self.scheduler_params.get('end_factor', 0.1),
                    'total_iters': self.scheduler_params.get('total_iters', 1000)
                }
                new_scheduler = LinearLR(optimizer, **params)
            
            elif scheduler_class_name == 'CosineAnnealingWarmRestarts':
                params = {
                    'T_0': self.scheduler_params.get('T_0', 100),
                    'T_mult': self.scheduler_params.get('T_mult', 2),
                    'eta_min': self.scheduler_params.get('eta_min', 1e-8)
                }
                new_scheduler = CosineAnnealingWarmRestarts(optimizer, **params)
            
            elif scheduler_class_name == 'CosineAnnealingLR':
                params = {
                    'T_max': self.scheduler_params.get('T_max', 1000),
                    'eta_min': self.scheduler_params.get('eta_min', 1e-8)
                }
                new_scheduler = CosineAnnealingLR(optimizer, **params)
            
            elif scheduler_class_name == 'MultiStepLR':
                params = {
                    'milestones': self.scheduler_params.get('milestones', [100, 200]),
                    'gamma': self.scheduler_params.get('gamma', 0.1)
                }
                new_scheduler = MultiStepLR(optimizer, **params)
            
            else:
                if self.logger:
                    self.logger.warning(f"Unknown scheduler type: {scheduler_class_name}")
                return None
            
            if self.logger:
                self.logger.info(f"✅ {scheduler_class_name} 重建成功")
            
            self.scheduler = new_scheduler
            return new_scheduler
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ {scheduler_class_name} 重建失敗: {e}")
            return None
    
    def step_scheduler(self):
        """執行排程器步進"""
        if self.scheduler is not None:
            self.scheduler.step()
    
    def get_current_lr(self):
        """獲取當前學習率"""
        if self.optimizer is None:
            return None
        return self.optimizer.param_groups[0]['lr']
    
    def zero_grad(self):
        """清零梯度"""
        if self.optimizer is not None:
            self.optimizer.zero_grad()
    
    def step_optimizer(self):
        """執行優化器步進"""
        if self.optimizer is not None:
            self.optimizer.step()