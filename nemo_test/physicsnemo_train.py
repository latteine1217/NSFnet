# Copyright (c) 2025 NVIDIA Corporation. All Rights Reserved.
import os
import torch
import hydra
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import LaunchLogger, PythonLogger, initialize_loggers
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint

from physicsnemo_solver import PhysicsNeMoPINNSolver


def create_trainer(cfg: DictConfig, solver: PhysicsNeMoPINNSolver):
    """Create PhysicsNeMo trainer with distributed support"""
    
    # Get boundary and interior data
    boundary_data = solver.dataset.get_boundary_data()
    interior_data = solver.dataset.get_interior_data()
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        solver.model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay
    )
    
    # Setup scheduler if specified
    scheduler = None
    if cfg.scheduler.enabled:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=cfg.scheduler.gamma
        )
    
    return optimizer, scheduler, boundary_data, interior_data


def training_step(
    solver: PhysicsNeMoPINNSolver,
    boundary_data: dict,
    interior_data: dict,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: DictConfig
):
    """Single training step"""
    
    solver.model.train()
    optimizer.zero_grad()
    
    # Boundary loss
    boundary_pred = solver.forward(boundary_data)
    boundary_input = {
        "x": boundary_data["x"],
        "y": boundary_data["y"],
        "boundary_u": boundary_data["u"],
        "boundary_v": boundary_data["v"]
    }
    boundary_losses = solver.loss(boundary_input, boundary_pred)
    
    # Interior PDE loss
    interior_pred = solver.forward(interior_data)
    interior_losses = solver.loss(interior_data, interior_pred)
    
    # Combine losses
    total_loss = boundary_losses["boundary_loss"] + interior_losses["total_loss"]
    
    # Backward pass
    total_loss.backward()
    
    # Gradient synchronization for distributed training
    if solver.dist.distributed:
        torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
        total_loss /= solver.dist.world_size
    
    optimizer.step()
    
    return {
        "total_loss": total_loss.item(),
        "boundary_loss": boundary_losses["boundary_loss"].item(),
        "continuity_loss": interior_losses.get("continuity_loss", torch.tensor(0.0)).item(),
        "momentum_x_loss": interior_losses.get("momentum_x_loss", torch.tensor(0.0)).item(),
        "momentum_y_loss": interior_losses.get("momentum_y_loss", torch.tensor(0.0)).item(),
        "evm_constraint_loss": interior_losses.get("evm_constraint_loss", torch.tensor(0.0)).item(),
    }


def train_stage(
    solver: PhysicsNeMoPINNSolver,
    cfg: DictConfig,
    stage_config: dict,
    logger: PythonLogger
):
    """Train a single stage with specific alpha_evm"""
    
    stage_name = stage_config["name"]
    alpha_evm = stage_config["alpha_evm"]
    num_epochs = stage_config["num_epochs"]
    learning_rate = stage_config["lr"]
    
    if solver.dist.rank == 0:
        logger.info(f"Starting {stage_name}: alpha_evm={alpha_evm}, lr={learning_rate}")
    
    # Update solver parameters
    solver.set_alpha_evm(alpha_evm)
    solver.current_stage = stage_name
    
    # Create trainer components
    optimizer, scheduler, boundary_data, interior_data = create_trainer(cfg, solver)
    optimizer.param_groups[0]['lr'] = learning_rate
    
    # Training loop
    for epoch in range(num_epochs):
        
        # EVM network freezing/unfreezing schedule
        if epoch == 0:
            solver.freeze_evm_net()
        elif epoch % 10000 == 0 and epoch > 0:
            solver.unfreeze_evm_net()
        elif epoch % 10000 == 1 and epoch > 1:
            solver.freeze_evm_net()
        
        # Training step
        losses = training_step(
            solver, boundary_data, interior_data, optimizer, epoch, cfg
        )
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        # Logging
        if solver.dist.rank == 0 and (epoch % cfg.log_freq == 0 or epoch == 0):
            log_str = f"Epoch {epoch+1}/{num_epochs}, Stage: {stage_name}"
            log_str += f", Total Loss: {losses['total_loss']:.3e}"
            log_str += f", Boundary: {losses['boundary_loss']:.3e}"
            log_str += f", Continuity: {losses['continuity_loss']:.3e}"
            log_str += f", Momentum X: {losses['momentum_x_loss']:.3e}"
            log_str += f", Momentum Y: {losses['momentum_y_loss']:.3e}"
            log_str += f", EVM: {losses['evm_constraint_loss']:.3e}"
            logger.info(log_str)
        
        # Checkpointing
        if solver.dist.rank == 0 and epoch % cfg.checkpoint_freq == 0:
            checkpoint_path = os.path.join(
                cfg.checkpoint_dir, 
                f"model_{stage_name.replace(' ', '_')}_epoch_{epoch}.pth"
            )
            save_checkpoint(
                solver.model.state_dict(),
                optimizer.state_dict(),
                epoch,
                checkpoint_path
            )
            
    # Evaluation at end of stage
    if solver.dist.rank == 0 and len(solver.dataset.reference_data) > 0:
        ref_data = solver.dataset.reference_data
        errors = solver.evaluate(
            ref_data["x"], ref_data["y"],
            ref_data["u"], ref_data["v"], ref_data["p"]
        )
        logger.info(f"Stage {stage_name} completed - Errors: U={errors['error_u']:.2f}%, V={errors['error_v']:.2f}%, P={errors['error_p']:.2f}%")


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function"""
    
    # Initialize distributed training
    DistributedManager.initialize()
    dist = DistributedManager()
    
    # Initialize loggers
    initialize_loggers()
    logger = PythonLogger("train")
    
    if dist.rank == 0:
        logger.info("Starting PhysicsNeMo PINN training")
        logger.info(f"Distributed training: {dist.distributed}")
        logger.info(f"World size: {dist.world_size}")
        logger.info(f"Device: {dist.device}")
    
    try:
        # Initialize solver
        solver = PhysicsNeMoPINNSolver(cfg)
        
        # Load checkpoint if specified
        if cfg.checkpoint_path:
            checkpoint = load_checkpoint(cfg.checkpoint_path)
            solver.model.load_state_dict(checkpoint['model_state_dict'])
            if dist.rank == 0:
                logger.info(f"Loaded checkpoint from {cfg.checkpoint_path}")
        
        # Training stages
        training_stages = [
            {"name": "Stage 1", "alpha_evm": 0.05, "num_epochs": 500000, "lr": 1e-3},
            {"name": "Stage 2", "alpha_evm": 0.03, "num_epochs": 500000, "lr": 2e-4},
            {"name": "Stage 3", "alpha_evm": 0.01, "num_epochs": 500000, "lr": 4e-5},
            {"name": "Stage 4", "alpha_evm": 0.005, "num_epochs": 500000, "lr": 1e-5},
            {"name": "Stage 5", "alpha_evm": 0.002, "num_epochs": 500000, "lr": 2e-6},
            {"name": "Stage 6", "alpha_evm": 0.002, "num_epochs": 500000, "lr": 2e-6},
        ]
        
        # Train each stage
        for stage_config in training_stages:
            train_stage(solver, cfg, stage_config, logger)
        
        if dist.rank == 0:
            logger.info("Training completed successfully!")
            
    except Exception as e:
        if dist.rank == 0:
            logger.error(f"Training failed: {e}")
        raise e
    
    finally:
        # Cleanup distributed training
        if dist.distributed:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()