# Copyright (c) 2025 NVIDIA Corporation. All Rights Reserved.
import torch
import numpy as np
from typing import Dict, List, Union
from physicsnemo.datapipes.benchmarks.dataset import Dataset
from physicsnemo.utils.io import csv_to_dict, dict_to_csv


class CavityDataset(Dataset):
    """
    PhysicsNeMo dataset for lid-driven cavity flow with boundary conditions
    """
    
    def __init__(
        self,
        name: str = "CavityDataset",
        data_dir: str = "./data",
        split: str = "train",
        num_samples: int = 10000,
        num_boundary_samples: int = 1000,
        reynolds_number: float = 5000.0,
        domain_bounds: Dict[str, List[float]] = None,
    ):
        """
        Parameters
        ----------
        name : str
            Name of the dataset
        data_dir : str  
            Directory containing data files
        split : str
            Dataset split (train/test/val)
        num_samples : int
            Number of interior samples for training
        num_boundary_samples : int
            Number of boundary samples
        reynolds_number : float
            Reynolds number for the flow
        domain_bounds : Dict[str, List[float]]
            Domain boundaries {"x": [x_min, x_max], "y": [y_min, y_max]}
        """
        
        super().__init__(name=name, data_dir=data_dir)
        
        self.split = split
        self.num_samples = num_samples
        self.num_boundary_samples = num_boundary_samples
        self.reynolds_number = reynolds_number
        
        if domain_bounds is None:
            self.domain_bounds = {"x": [0.0, 1.0], "y": [0.0, 1.0]}
        else:
            self.domain_bounds = domain_bounds
            
        # Generate data
        self._generate_data()
        
    def _generate_data(self):
        """Generate training and boundary data"""
        
        # Generate interior points for PDE residual
        self.interior_data = self._generate_interior_points()
        
        # Generate boundary data
        self.boundary_data = self._generate_boundary_data()
        
        # Load reference data if available
        self.reference_data = self._load_reference_data()
        
    def _generate_interior_points(self) -> Dict[str, torch.Tensor]:
        """Generate interior collocation points using Latin Hypercube Sampling"""
        
        # Latin Hypercube Sampling
        n_dims = len(self.domain_bounds)
        samples = np.zeros((self.num_samples, n_dims))
        
        for i, (key, bounds) in enumerate(self.domain_bounds.items()):
            # LHS sampling
            intervals = np.linspace(0, 1, self.num_samples + 1)
            samples[:, i] = np.random.uniform(intervals[:-1], intervals[1:])
            np.random.shuffle(samples[:, i])
            
            # Scale to domain bounds
            samples[:, i] = samples[:, i] * (bounds[1] - bounds[0]) + bounds[0]
        
        # Convert to dictionary format
        data_dict = {}
        for i, key in enumerate(self.domain_bounds.keys()):
            data_dict[key] = torch.tensor(samples[:, i:i+1], dtype=torch.float32)
            
        return data_dict
    
    def _generate_boundary_data(self) -> Dict[str, torch.Tensor]:
        """Generate boundary condition data"""
        
        # Number of points per boundary
        n_per_boundary = self.num_boundary_samples // 4
        
        x_min, x_max = self.domain_bounds["x"]
        y_min, y_max = self.domain_bounds["y"]
        
        # Bottom boundary (y = y_min)
        x_bottom = np.linspace(x_min, x_max, n_per_boundary)
        y_bottom = np.full_like(x_bottom, y_min)
        u_bottom = np.zeros_like(x_bottom)
        v_bottom = np.zeros_like(x_bottom)
        
        # Top boundary (y = y_max) - moving lid
        x_top = np.linspace(x_min, x_max, n_per_boundary)
        y_top = np.full_like(x_top, y_max)
        r_const = 50.0
        u_top = 1.0 - np.cosh(r_const * (x_top - 0.5)) / np.cosh(r_const * 0.5)
        v_top = np.zeros_like(x_top)
        
        # Left boundary (x = x_min)
        y_left = np.linspace(y_min, y_max, n_per_boundary)
        x_left = np.full_like(y_left, x_min)
        u_left = np.zeros_like(y_left)
        v_left = np.zeros_like(y_left)
        
        # Right boundary (x = x_max)
        y_right = np.linspace(y_min, y_max, n_per_boundary)
        x_right = np.full_like(y_right, x_max)
        u_right = np.zeros_like(y_right)
        v_right = np.zeros_like(y_right)
        
        # Combine all boundaries
        x_bc = np.concatenate([x_bottom, x_top, x_left, x_right])
        y_bc = np.concatenate([y_bottom, y_top, y_left, y_right])
        u_bc = np.concatenate([u_bottom, u_top, u_left, u_right])
        v_bc = np.concatenate([v_bottom, v_top, v_left, v_right])
        
        boundary_dict = {
            "x": torch.tensor(x_bc.reshape(-1, 1), dtype=torch.float32),
            "y": torch.tensor(y_bc.reshape(-1, 1), dtype=torch.float32),
            "u": torch.tensor(u_bc.reshape(-1, 1), dtype=torch.float32),
            "v": torch.tensor(v_bc.reshape(-1, 1), dtype=torch.float32),
        }
        
        return boundary_dict
    
    def _load_reference_data(self) -> Dict[str, torch.Tensor]:
        """Load reference data for validation if available"""
        
        try:
            import scipy.io
            filename = f"{self.data_dir}/cavity_Re{int(self.reynolds_number)}_256_Uniform.mat"
            data = scipy.io.loadmat(filename)
            
            x_ref = torch.tensor(data['X_ref'].reshape(-1, 1), dtype=torch.float32)
            y_ref = torch.tensor(data['Y_ref'].reshape(-1, 1), dtype=torch.float32)
            u_ref = torch.tensor(data['U_ref'].reshape(-1, 1), dtype=torch.float32)
            v_ref = torch.tensor(data['V_ref'].reshape(-1, 1), dtype=torch.float32)
            p_ref = torch.tensor(data['P_ref'].reshape(-1, 1), dtype=torch.float32)
            
            reference_dict = {
                "x": x_ref,
                "y": y_ref,
                "u": u_ref,
                "v": v_ref,
                "p": p_ref,
            }
            
            return reference_dict
            
        except (FileNotFoundError, KeyError):
            print(f"Reference data not found for Re={self.reynolds_number}")
            return {}
    
    def __len__(self) -> int:
        """Return dataset length"""
        if self.split == "train":
            return self.num_samples
        else:
            return len(self.reference_data.get("x", torch.empty(0)))
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item"""
        
        if self.split == "train":
            # Return interior point for training
            data_dict = {}
            for key, values in self.interior_data.items():
                data_dict[key] = values[idx:idx+1]
            return data_dict
        else:
            # Return reference data for validation
            data_dict = {}
            for key, values in self.reference_data.items():
                data_dict[key] = values[idx:idx+1]
            return data_dict
    
    def get_boundary_data(self) -> Dict[str, torch.Tensor]:
        """Get all boundary condition data"""
        return self.boundary_data
    
    def get_interior_data(self) -> Dict[str, torch.Tensor]:
        """Get all interior collocation points"""
        return self.interior_data