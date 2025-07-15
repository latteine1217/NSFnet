
# AGENTS.md

This file guides AI agents in understanding and interacting with the **ev-NSFnet** project.

## Project Overview

**ev-NSFnet** is a research project that implements a Physics-Informed Neural Network (PINN) to solve the Navier-Stokes equations for a 2D lid-driven cavity flow problem. The project uses PyTorch for the neural network implementation and supports distributed training. Detail can be found in the [paper](~/Documents/coding/ldc_pinns/NSFnet/paper.pdf).

## Build/Test Commands

**Training:**
- Single GPU: `python train.py`
- Distributed: `torchrun --nproc_per_node=NUM_GPUS train.py`
- Testing: `python test.py` (modify checkpoint paths in script)
- Syntax check: `python -m py_compile <filename>.py`
- Run single test: No test framework - modify `test.py` directly for specific evaluations

## Code Style Guidelines

**Imports:** Standard library first, then third-party (torch, numpy, scipy), then local modules
**Formatting:** PEP 8 compliant, 4-space indentation, line length ~80-100 chars
**Types:** Use type hints for function parameters (see `pinn_solver.py:24`)
**Naming:** `snake_case` for variables/functions, `PascalCase` for classes, `UPPER_CASE` for constants
**Error Handling:** Use try-except blocks with specific exception types, print informative error messages
**Comments:** GPL license headers required, docstrings for classes/functions, inline comments for complex logic
**File Structure:** Modular design - separate data loading, network definition, and training logic

## Architecture

The project follows a modular architecture:

1.  **Data Loading (`cavity_data.py`):** Loads and preprocesses the data.
2.  **Network Definition (`net.py`):** Defines the neural network architecture.
3.  **PINN Solver (`pinn_solver.py`):** Implements the core PINN logic, including the physics-informed loss function.
4.  **Training and Testing (`train.py`, `test.py`):** High-level scripts for training and evaluating the model.

## How to Contribute

1.  **Bug Fixes:** Identify and fix bugs in the existing codebase. Please provide a detailed description of the bug and the fix.
2.  **Feature Enhancements:** Propose and implement new features, such as support for different physical problems or network architectures.
3.  **Performance Improvements:** Optimize the code for better performance, especially the training loop and data loading process.
4.  **Documentation:** Improve the documentation, including this `AGENT.md` file, to make the project more accessible to new contributors.
