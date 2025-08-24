# Copyright (c) 2023 scien42.tech, Se42 Authors. All Rights Reserved.
#
# PINN 檢查點管理模組

import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, Any, Optional, Union
from torch.nn import Module

class CheckpointManager:
    """
    PINN 檢查點管理器
    處理模型的保存和載入功能
    """
    
    def __init__(self, model, evm_model=None, rank=0, logger=None):
        self.model = model
        self.evm_model = evm_model
        self.rank = rank
        self.logger = logger
    
    def _param_name_map(self, model: Union[Module, DDP]) -> Dict[int, str]:
        """映射參數ID到參數名稱"""
        return {id(p): n for n, p in model.named_parameters()}
    
    def _safe_optimizer_state_dict(self, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """
        安全獲取optimizer state dict，避免DDP參數映射問題
        """
        try:
            return optimizer.state_dict()
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to get optimizer state dict: {e}")
            return {}
    
    def _load_optimizer_state_dict_safe(self, optimizer: torch.optim.Optimizer, opt_state: Optional[Dict[str, Any]]) -> None:
        """
        安全載入optimizer state dict
        """
        if opt_state is None or not opt_state:
            if self.logger:
                self.logger.info("No optimizer state to load")
            return
        
        try:
            optimizer.load_state_dict(opt_state)
            if self.logger:
                self.logger.info("Successfully loaded optimizer state")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to load optimizer state: {e}")
    
    def get_checkpoint_dir(self):
        """獲取檢查點目錄"""
        checkpoint_dir = "./checkpoints"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        return checkpoint_dir
    
    def save_checkpoint(self, epoch, optimizer):
        """保存檢查點"""
        if self.rank != 0:  # 只有 rank 0 保存
            return
            
        try:
            checkpoint_dir = self.get_checkpoint_dir()
            
            # 獲取模型 state_dict
            net_state = self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict()
            net1_state = None
            if self.evm_model is not None:
                net1_state = self.evm_model.module.state_dict() if isinstance(self.evm_model, DDP) else self.evm_model.state_dict()
            
            # 獲取 optimizer state_dict
            opt_state = self._safe_optimizer_state_dict(optimizer) if optimizer else None
            
            checkpoint = {
                'epoch': epoch,
                'net_state_dict': net_state,
                'net1_state_dict': net1_state,
                'optimizer_state_dict': opt_state,
                'loss': None  # 可以在調用時傳入
            }
            
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, checkpoint_path)
            
            if self.logger:
                self.logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_path, optimizer):
        """載入檢查點"""
        try:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 載入模型狀態
            net_state = checkpoint.get('net_state_dict')
            net1_state = checkpoint.get('net1_state_dict')
            
            if net_state:
                model_to_load = self.model.module if isinstance(self.model, DDP) else self.model
                model_to_load.load_state_dict(net_state)
            
            if net1_state and self.evm_model is not None:
                evm_to_load = self.evm_model.module if isinstance(self.evm_model, DDP) else self.evm_model
                evm_to_load.load_state_dict(net1_state)
            
            # 載入 optimizer 狀態
            if optimizer:
                opt_state = checkpoint.get('optimizer_state_dict')
                self._load_optimizer_state_dict_safe(optimizer, opt_state)
            
            start_epoch = checkpoint.get('epoch', 0) + 1
            
            if self.logger:
                self.logger.info(f"✅ Checkpoint loaded: {checkpoint_path}")
                self.logger.info(f"📅 Resuming from epoch: {start_epoch}")
            
            return start_epoch
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Failed to load checkpoint: {e}")
            raise
    
    def save_model(self, filename, directory=None, N_HLayer=None, N_neu=None, N_f=None):
        """保存模型 (與原始 save 方法兼容)"""
        try:
            if directory is None:
                directory = self.get_checkpoint_dir()
            
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            # 獲取實際模型 (處理 DDP 包裝)
            net_to_save = self.model.module if isinstance(self.model, DDP) else self.model
            net1_to_save = None
            if self.evm_model is not None:
                net1_to_save = self.evm_model.module if isinstance(self.evm_model, DDP) else self.evm_model
            
            save_data = {
                'net_state_dict': net_to_save.state_dict(),
                'net1_state_dict': net1_to_save.state_dict() if net1_to_save else None,
                'N_HLayer': N_HLayer,
                'N_neu': N_neu,
                'N_f': N_f
            }
            
            filepath = os.path.join(directory, filename)
            torch.save(save_data, filepath)
            
            if self.logger:
                self.logger.info(f"💾 Model saved: {filepath}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Failed to save model: {e}")
            raise