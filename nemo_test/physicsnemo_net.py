# Copyright (c) 2025 NVIDIA Corporation. All Rights Reserved.
import torch
import torch.nn as nn
from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.models.layers.activation import Activation
from typing import List, Optional


class PhysicsNeMoNet(nn.Module):
    """PhysicsNeMo compatible neural network for PINN cavity flow simulation"""
    
    def __init__(
        self,
        input_keys: List[str] = ["x", "y"],
        output_keys: List[str] = ["u", "v", "p"],
        nr_layers: int = 6,
        layer_size: int = 80,
        activation_fn: str = "tanh",
    ):
        super().__init__()
        
        self.input_keys = input_keys
        self.output_keys = output_keys
        
        # Main network for velocity and pressure
        self.net = FullyConnected(
            in_features=len(input_keys),
            out_features=len(output_keys),
            num_layers=nr_layers,
            layer_size=layer_size,
            activation_fn=Activation(activation_fn),
            adaptive_activations=False,
            weight_norm=True,
        )
        
    def forward(self, input_dict):
        """Forward pass compatible with PhysicsNeMo"""
        # Concatenate inputs
        x = torch.cat([input_dict[key] for key in self.input_keys], dim=-1)
        
        # Forward through network
        output = self.net(x)
        
        # Split outputs into dictionary
        output_dict = {}
        for i, key in enumerate(self.output_keys):
            output_dict[key] = output[:, i:i+1]
            
        return output_dict


class PhysicsNeMoEVMNet(nn.Module):
    """PhysicsNeMo compatible eddy viscosity model network"""
    
    def __init__(
        self,
        input_keys: List[str] = ["x", "y"],
        output_keys: List[str] = ["evm"],
        nr_layers: int = 4,
        layer_size: int = 40,
        activation_fn: str = "tanh",
    ):
        super().__init__()
        
        self.input_keys = input_keys
        self.output_keys = output_keys
        
        # EVM network
        self.net = FullyConnected(
            in_features=len(input_keys),
            out_features=len(output_keys),
            num_layers=nr_layers,
            layer_size=layer_size,
            activation_fn=Activation(activation_fn),
            adaptive_activations=False,
            weight_norm=True,
        )
        
    def forward(self, input_dict):
        """Forward pass for EVM network"""
        # Concatenate inputs
        x = torch.cat([input_dict[key] for key in self.input_keys], dim=-1)
        
        # Forward through network
        output = self.net(x)
        
        # Split outputs into dictionary
        output_dict = {}
        for i, key in enumerate(self.output_keys):
            output_dict[key] = output[:, i:i+1]
            
        return output_dict


class CombinedPhysicsNeMoNet(nn.Module):
    """Combined PhysicsNeMo network for PINN with EVM"""
    
    def __init__(
        self,
        main_net_config: dict,
        evm_net_config: dict,
    ):
        super().__init__()
        
        self.main_net = PhysicsNeMoNet(**main_net_config)
        self.evm_net = PhysicsNeMoEVMNet(**evm_net_config)
        
    def forward(self, input_dict):
        """Combined forward pass"""
        main_output = self.main_net(input_dict)
        evm_output = self.evm_net(input_dict)
        
        # Combine outputs
        output_dict = {**main_output, **evm_output}
        return output_dict