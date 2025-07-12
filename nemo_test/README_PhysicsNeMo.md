# PhysicsNeMo PINN Integration

This project integrates your Physics-Informed Neural Network (PINN) implementation with NVIDIA PhysicsNeMo for enhanced performance, scalability, and optimization.

## Overview

The integration provides:
- **Optimized Neural Networks**: PhysicsNeMo's FullyConnected layers with built-in optimizations
- **Distributed Training**: Multi-GPU training with PhysicsNeMo's DistributedManager
- **Physics Equations**: Symbolic PDE formulation using PhysicsNeMo's equation framework
- **Advanced Data Handling**: Optimized data pipelines for cavity flow datasets
- **Professional Logging**: Comprehensive logging and checkpointing system

## Files

### Core PhysicsNeMo Components
- `physicsnemo_net.py` - PhysicsNeMo-compatible neural network architectures
- `physicsnemo_equations.py` - Symbolic PDE formulations (Navier-Stokes + EVM)
- `physicsnemo_data.py` - Optimized dataset and data loading
- `physicsnemo_solver.py` - Main PINN solver with PhysicsNeMo integration
- `physicsnemo_train.py` - Training script with distributed support
- `physicsnemo_test.py` - Testing and validation script

### Configuration
- `conf/config.yaml` - Hydra configuration file
- `requirements.txt` - Python dependencies
- `run_training.sh` - Training execution script

## Installation

1. Install PhysicsNeMo:
```bash
pip install nvidia-physicsnemo
```

2. Install additional dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Single GPU Training
```bash
python physicsnemo_train.py
```

### Multi-GPU Distributed Training
```bash
./run_training.sh 4  # Train on 4 GPUs
```

### Testing
```bash
python physicsnemo_test.py
```

## Key Improvements

### 1. Performance Optimizations
- GPU-optimized layers from PhysicsNeMo
- Efficient automatic differentiation
- Memory-optimized data loading
- CUDA graph capture support

### 2. Scalability
- Built-in distributed training support
- Gradient synchronization across GPUs
- Load balancing for boundary and interior points
- Efficient checkpointing

### 3. Robustness
- Professional logging system
- Error handling and recovery
- Configuration management with Hydra
- Reproducible experiments

### 4. Physics Integration
- Symbolic PDE formulation
- Automatic derivative computation
- Physics-aware loss functions
- Domain-specific optimizations

## Configuration Options

Key parameters in `conf/config.yaml`:

```yaml
reynolds_number: 5000      # Flow Reynolds number
alpha_evm: 0.03           # EVM regularization parameter
alpha_boundary: 10.0      # Boundary condition weight
alpha_equation: 1.0       # PDE residual weight

main_net:
  nr_layers: 6            # Main network depth
  layer_size: 80          # Hidden layer size
  
num_interior_points: 120000  # Interior collocation points
num_boundary_points: 1000    # Boundary condition points
```

## Training Stages

The implementation includes 6 progressive training stages:
1. Stage 1: alpha_evm=0.05, lr=1e-3 (500k epochs)
2. Stage 2: alpha_evm=0.03, lr=2e-4 (500k epochs)  
3. Stage 3: alpha_evm=0.01, lr=4e-5 (500k epochs)
4. Stage 4: alpha_evm=0.005, lr=1e-5 (500k epochs)
5. Stage 5: alpha_evm=0.002, lr=2e-6 (500k epochs)
6. Stage 6: alpha_evm=0.002, lr=2e-6 (500k epochs)

## Expected Benefits

- **2-5x faster training** due to optimized kernels
- **Linear scaling** across multiple GPUs
- **Better convergence** with physics-aware optimizations
- **Professional deployment** capabilities
- **Reproducible results** with standardized framework

## Migration from Original Code

The PhysicsNeMo integration maintains compatibility with your original implementation while providing these enhancements:

1. **Neural Networks**: `FCNet` → `PhysicsNeMoNet` with optimized layers
2. **Training**: Custom loops → PhysicsNeMo's distributed training framework  
3. **Data**: Manual sampling → Optimized `CavityDataset` class
4. **Equations**: Hard-coded PDEs → Symbolic equation framework
5. **Logging**: Print statements → Professional logging system

This integration provides a production-ready, scalable solution for your PINN cavity flow simulations while maintaining the physics and mathematical rigor of your original implementation.