# Copyright (c) 2025 NVIDIA Corporation. All Rights Reserved.
import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
from omegaconf import DictConfig
from physicsnemo.constants import tf_dt
from physicsnemo.models.module import Module
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.io import ValidateInput

from physicsnemo_net import CombinedPhysicsNeMoNet
from physicsnemo_equations import NavierStokesEVM, EVMConstraint
from physicsnemo_data import CavityDataset


@ValidateInput(tf_dt)
class PhysicsNeMoPINNSolver(Module):
    """
    PhysicsNeMo-based PINN solver for cavity flow with eddy viscosity modeling
    
    Parameters
    ----------
    cfg : DictConfig
        Configuration dictionary containing model parameters
    """
    
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        self.cfg = cfg
        
        # Initialize distributed manager
        self.dist = DistributedManager()
        
        # Model configuration
        self.reynolds_number = cfg.reynolds_number
        self.alpha_evm = cfg.alpha_evm
        self.alpha_boundary = cfg.alpha_boundary
        self.alpha_equation = cfg.alpha_equation
        self.alpha_evm_constraint = cfg.alpha_evm_constraint
        
        # Initialize networks
        self._init_networks()
        
        # Initialize equations
        self._init_equations()
        
        # Initialize dataset
        self._init_dataset()
        
        # Training parameters
        self.current_stage = ""
        self.vis_t0 = 20.0 / self.reynolds_number
        self.vis_t_minus = None
        
    def _init_networks(self):
        """Initialize neural networks"""
        
        main_net_config = {
            "input_keys": ["x", "y"],
            "output_keys": ["u", "v", "p"],
            "nr_layers": self.cfg.main_net.nr_layers,
            "layer_size": self.cfg.main_net.layer_size,
            "activation_fn": self.cfg.main_net.activation_fn,
        }
        
        evm_net_config = {
            "input_keys": ["x", "y"],
            "output_keys": ["evm"],
            "nr_layers": self.cfg.evm_net.nr_layers,
            "layer_size": self.cfg.evm_net.layer_size,
            "activation_fn": self.cfg.evm_net.activation_fn,
        }
        
        self.model = CombinedPhysicsNeMoNet(
            main_net_config=main_net_config,
            evm_net_config=evm_net_config
        )
        
        # Move to appropriate device
        self.model = self.model.to(self.dist.device)
        
        # Wrap with distributed training if needed
        if self.dist.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.device,
            )
            
    def _init_equations(self):
        """Initialize PDE equations"""
        
        self.navier_stokes = NavierStokesEVM(
            nu=1.0/self.reynolds_number,
            rho=1.0,
            dim=2,
            time=False
        )
        
        self.evm_constraint = EVMConstraint(
            alpha_evm=self.alpha_evm,
            dim=2
        )
        
    def _init_dataset(self):
        """Initialize dataset"""
        
        self.dataset = CavityDataset(
            data_dir=self.cfg.data_dir,
            num_samples=self.cfg.num_interior_points,
            num_boundary_samples=self.cfg.num_boundary_points,
            reynolds_number=self.reynolds_number,
        )
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        return self.model(batch)
        
    def loss(self, invar: Dict[str, torch.Tensor], pred_outvar: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute physics-informed loss
        
        Parameters
        ----------
        invar : Dict[str, torch.Tensor]
            Input variables
        pred_outvar : Dict[str, torch.Tensor] 
            Predicted output variables
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Loss dictionary
        """
        
        losses = {}
        
        # Boundary loss
        if "boundary_u" in invar and "boundary_v" in invar:
            boundary_loss_u = torch.mean((pred_outvar["u"] - invar["boundary_u"]) ** 2)
            boundary_loss_v = torch.mean((pred_outvar["v"] - invar["boundary_v"]) ** 2)
            losses["boundary_loss"] = boundary_loss_u + boundary_loss_v
        
        # PDE residual losses
        if "x" in invar and "y" in invar:
            # Compute derivatives and PDE residuals
            residuals = self._compute_pde_residuals(invar, pred_outvar)
            
            # Navier-Stokes residuals
            losses["continuity_loss"] = torch.mean(residuals["continuity"] ** 2)
            losses["momentum_x_loss"] = torch.mean(residuals["momentum_x"] ** 2) 
            losses["momentum_y_loss"] = torch.mean(residuals["momentum_y"] ** 2)
            
            # EVM constraint loss
            losses["evm_constraint_loss"] = torch.mean(residuals["evm_constraint"] ** 2)
        
        # Total loss
        total_loss = (
            self.alpha_boundary * losses.get("boundary_loss", 0.0) +
            self.alpha_equation * (
                losses.get("continuity_loss", 0.0) +
                losses.get("momentum_x_loss", 0.0) +
                losses.get("momentum_y_loss", 0.0)
            ) +
            self.alpha_evm_constraint * losses.get("evm_constraint_loss", 0.0)
        )
        
        losses["total_loss"] = total_loss
        
        return losses
        
    def _compute_pde_residuals(
        self, 
        invar: Dict[str, torch.Tensor], 
        pred_outvar: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute PDE residuals using automatic differentiation"""
        
        x = invar["x"].requires_grad_(True)
        y = invar["y"].requires_grad_(True)
        
        # Forward pass to get predictions
        input_dict = {"x": x, "y": y}
        output_dict = self.model(input_dict)
        
        u = output_dict["u"]
        v = output_dict["v"] 
        p = output_dict["p"]
        evm = output_dict["evm"]
        
        # Compute derivatives
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
        
        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        
        # Update eddy viscosity with clipping
        if self.vis_t_minus is not None:
            vis_t = torch.clamp(
                torch.tensor(self.vis_t_minus, device=evm.device),
                max=self.vis_t0
            )
        else:
            vis_t = torch.tensor(self.vis_t0, device=evm.device)
            
        # Store for next iteration
        self.vis_t_minus = self.alpha_evm * torch.abs(evm).detach().cpu().numpy()
        
        # Total viscosity
        total_viscosity = 1.0/self.reynolds_number + vis_t
        
        # PDE residuals
        residuals = {}
        
        # Continuity equation
        residuals["continuity"] = u_x + v_y
        
        # Momentum equations
        residuals["momentum_x"] = (
            u * u_x + v * u_y + p_x - total_viscosity * (u_xx + u_yy)
        )
        
        residuals["momentum_y"] = (
            u * v_x + v * v_y + p_y - total_viscosity * (v_xx + v_yy)
        )
        
        # EVM constraint
        residuals["evm_constraint"] = (
            (u - 0.5) * (u * u_x + v * u_y) +
            (v - 0.5) * (u * v_x + v * v_y)
        ) - evm
        
        return residuals
        
    def set_alpha_evm(self, alpha: float):
        """Set alpha_evm parameter"""
        self.alpha_evm = alpha
        
    def freeze_evm_net(self):
        """Freeze EVM network parameters"""
        if hasattr(self.model, 'module'):  # DDP wrapped
            for param in self.model.module.evm_net.parameters():
                param.requires_grad = False
        else:
            for param in self.model.evm_net.parameters():
                param.requires_grad = False
                
    def unfreeze_evm_net(self):
        """Unfreeze EVM network parameters"""
        if hasattr(self.model, 'module'):  # DDP wrapped
            for param in self.model.module.evm_net.parameters():
                param.requires_grad = True
        else:
            for param in self.model.evm_net.parameters():
                param.requires_grad = True
                
    def evaluate(self, x: torch.Tensor, y: torch.Tensor, u_ref: torch.Tensor, 
                 v_ref: torch.Tensor, p_ref: torch.Tensor) -> Dict[str, float]:
        """Evaluate model against reference data"""
        
        with torch.no_grad():
            input_dict = {"x": x, "y": y}
            pred_dict = self.model(input_dict)
            
            u_pred = pred_dict["u"]
            v_pred = pred_dict["v"]
            p_pred = pred_dict["p"]
            
            # Compute relative L2 errors
            error_u = torch.norm(u_ref - u_pred) / torch.norm(u_ref) * 100
            error_v = torch.norm(v_ref - v_pred) / torch.norm(v_ref) * 100
            
            # Handle pressure with NaN masking
            mask_p = ~torch.isnan(p_ref)
            if mask_p.any():
                error_p = torch.norm(p_ref[mask_p] - p_pred[mask_p]) / torch.norm(p_ref[mask_p]) * 100
            else:
                error_p = torch.tensor(float('nan'))
                
        return {
            "error_u": error_u.item(),
            "error_v": error_v.item(), 
            "error_p": error_p.item() if not torch.isnan(error_p) else 0.0
        }