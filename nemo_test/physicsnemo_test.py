# Copyright (c) 2025 NVIDIA Corporation. All Rights Reserved.
import torch
import hydra
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import LaunchLogger, PythonLogger, initialize_loggers

from physicsnemo_solver import PhysicsNeMoPINNSolver


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def test_model(cfg: DictConfig) -> None:
    """Test trained PhysicsNeMo PINN model"""
    
    # Initialize distributed training (for consistency)
    DistributedManager.initialize()
    dist = DistributedManager()
    
    # Initialize loggers
    initialize_loggers()
    logger = PythonLogger("test")
    
    if dist.rank == 0:
        logger.info("Starting PhysicsNeMo PINN testing")
    
    # Initialize solver
    solver = PhysicsNeMoPINNSolver(cfg)
    
    # Test different Reynolds numbers and checkpoints
    test_configs = [
        {"Re": 3000, "stage": "Stage 1", "epochs": [0, 100000, 200000, 300000, 400000, 500000]},
        {"Re": 5000, "stage": "Stage 1", "epochs": [0, 100000, 200000, 300000, 400000, 500000]},
        {"Re": 5000, "stage": "Stage 2", "epochs": [0, 100000, 200000, 300000, 400000, 500000]},
        {"Re": 5000, "stage": "Stage 3", "epochs": [0, 100000, 200000, 300000, 400000, 500000]},
    ]
    
    for test_config in test_configs:
        reynolds_number = test_config["Re"]
        stage = test_config["stage"]
        
        # Update solver for this Reynolds number
        solver.reynolds_number = reynolds_number
        solver.vis_t0 = 20.0 / reynolds_number
        
        # Load reference data
        try:
            ref_filename = f"./data/cavity_Re{reynolds_number}_256_Uniform.mat"
            solver.dataset.reynolds_number = reynolds_number
            solver.dataset.reference_data = solver.dataset._load_reference_data()
            
            if len(solver.dataset.reference_data) == 0:
                if dist.rank == 0:
                    logger.warning(f"No reference data found for Re={reynolds_number}")
                continue
                
        except Exception as e:
            if dist.rank == 0:
                logger.error(f"Failed to load reference data for Re={reynolds_number}: {e}")
            continue
        
        # Test different epochs
        for epoch in test_config["epochs"]:
            checkpoint_path = f"./checkpoints/model_{stage.replace(' ', '_')}_epoch_{epoch}.pth"
            
            try:
                # Load checkpoint
                checkpoint = torch.load(checkpoint_path, map_location=solver.dist.device)
                solver.model.load_state_dict(checkpoint)
                
                # Evaluate
                ref_data = solver.dataset.reference_data
                errors = solver.evaluate(
                    ref_data["x"], ref_data["y"],
                    ref_data["u"], ref_data["v"], ref_data["p"]
                )
                
                if dist.rank == 0:
                    logger.info(f"Re={reynolds_number}, {stage}, Epoch={epoch}")
                    logger.info(f"  Error U: {errors['error_u']:.3f}%")
                    logger.info(f"  Error V: {errors['error_v']:.3f}%") 
                    logger.info(f"  Error P: {errors['error_p']:.3f}%")
                    logger.info("-" * 40)
                    
            except FileNotFoundError:
                if dist.rank == 0:
                    logger.warning(f"Checkpoint not found: {checkpoint_path}")
            except Exception as e:
                if dist.rank == 0:
                    logger.error(f"Error testing {checkpoint_path}: {e}")
    
    if dist.rank == 0:
        logger.info("Testing completed!")


if __name__ == "__main__":
    test_model()