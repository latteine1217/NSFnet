# Copyright (c) 2025 NVIDIA Corporation. All Rights Reserved.
import torch
from typing import Dict, List
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.eq.derivatives import gradient


class NavierStokesEVM(PDE):
    """
    NavierStokes equation with Eddy Viscosity Model for PhysicsNeMo
    
    Parameters
    ==========
    nu : float, str
        The kinematic viscosity. Default is 0.01.
    rho : float, str  
        The density. Default is 1.0.
    dim : int
        Dimension of the NavierStokes (2 or 3). Default is 3.
    time : bool
        If time-dependent equations or not. Default is True.
    mixed_form: bool
        If True, use the mixed formulation of the NavierStokes.
    """

    name = "NavierStokesEVM"

    def __init__(self, nu=0.01, rho=1.0, dim=3, time=True, mixed_form=False):
        # set params
        self.nu = nu
        self.rho = rho
        self.dim = dim
        self.time = time
        self.mixed_form = mixed_form

        # coordinates
        x, y, z = self.make_coord("x"), self.make_coord("y"), self.make_coord("z")

        # time
        t = self.make_coord("t")

        # make input variables
        if self.dim == 2:
            input_variables = {"x": x, "y": y}
            if self.time:
                input_variables["t"] = t
        elif self.dim == 3:
            input_variables = {"x": x, "y": y, "z": z}
            if self.time:
                input_variables["t"] = t
        else:
            raise ValueError("dim should be 2 or 3")

        # velocity components
        u = self.make_function("u", input_variables)
        v = self.make_function("v", input_variables)
        if self.dim == 3:
            w = self.make_function("w", input_variables)

        # pressure
        p = self.make_function("p", input_variables)
        
        # eddy viscosity
        evm = self.make_function("evm", input_variables)

        # set equations
        self.equations = {}

        # continuity equation
        if self.dim == 2:
            self.equations["continuity"] = u.diff(x) + v.diff(y)
        elif self.dim == 3:
            self.equations["continuity"] = u.diff(x) + v.diff(y) + w.diff(z)

        # momentum equations with eddy viscosity
        if self.dim == 2:
            # x-momentum 
            momentum_x = (
                u * u.diff(x) + v * u.diff(y) 
                + (1/self.rho) * p.diff(x)
                - (self.nu + evm) * (u.diff(x, 2) + u.diff(y, 2))
            )
            if self.time:
                momentum_x += u.diff(t)
            self.equations["momentum_x"] = momentum_x

            # y-momentum
            momentum_y = (
                u * v.diff(x) + v * v.diff(y)
                + (1/self.rho) * p.diff(y) 
                - (self.nu + evm) * (v.diff(x, 2) + v.diff(y, 2))
            )
            if self.time:
                momentum_y += v.diff(t)
            self.equations["momentum_y"] = momentum_y

        elif self.dim == 3:
            # x-momentum
            momentum_x = (
                u * u.diff(x) + v * u.diff(y) + w * u.diff(z)
                + (1/self.rho) * p.diff(x)
                - (self.nu + evm) * (u.diff(x, 2) + u.diff(y, 2) + u.diff(z, 2))
            )
            if self.time:
                momentum_x += u.diff(t)
            self.equations["momentum_x"] = momentum_x

            # y-momentum
            momentum_y = (
                u * v.diff(x) + v * v.diff(y) + w * v.diff(z)
                + (1/self.rho) * p.diff(y)
                - (self.nu + evm) * (v.diff(x, 2) + v.diff(y, 2) + v.diff(z, 2))
            )
            if self.time:
                momentum_y += v.diff(t)
            self.equations["momentum_y"] = momentum_y

            # z-momentum
            momentum_z = (
                u * w.diff(x) + v * w.diff(y) + w * w.diff(z)
                + (1/self.rho) * p.diff(z)
                - (self.nu + evm) * (w.diff(x, 2) + w.diff(y, 2) + w.diff(z, 2))
            )
            if self.time:
                momentum_z += w.diff(t)
            self.equations["momentum_z"] = momentum_z


class EVMConstraint(PDE):
    """
    Eddy Viscosity Model constraint equation for PhysicsNeMo
    
    This implements the residual constraint for the EVM network
    """
    
    name = "EVMConstraint"
    
    def __init__(self, alpha_evm=0.03, dim=2):
        self.alpha_evm = alpha_evm
        self.dim = dim
        
        # coordinates
        x, y = self.make_coord("x"), self.make_coord("y")
        if self.dim == 3:
            z = self.make_coord("z")
            input_variables = {"x": x, "y": y, "z": z}
        else:
            input_variables = {"x": x, "y": y}
            
        # velocity components
        u = self.make_function("u", input_variables)
        v = self.make_function("v", input_variables)
        if self.dim == 3:
            w = self.make_function("w", input_variables)
            
        # eddy viscosity
        evm = self.make_function("evm", input_variables)
        
        # EVM constraint equation
        if self.dim == 2:
            residual_term = (u - 0.5) * (
                u * u.diff(x) + v * u.diff(y)
            ) + (v - 0.5) * (
                u * v.diff(x) + v * v.diff(y)
            )
        else:  # dim == 3
            residual_term = (u - 0.5) * (
                u * u.diff(x) + v * u.diff(y) + w * u.diff(z)
            ) + (v - 0.5) * (
                u * v.diff(x) + v * v.diff(y) + w * v.diff(z)
            ) + (w - 0.5) * (
                u * w.diff(x) + v * w.diff(y) + w * w.diff(z)
            )
            
        self.equations = {
            "evm_constraint": residual_term - evm
        }