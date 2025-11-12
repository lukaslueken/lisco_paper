# Self-Supervised Learning of Iterative Solvers for Constrained Optimization

This repository contains the implementation code for the paper "Self-Supervised Learning of Iterative Solvers for Constrained Optimization" by Lukas Lüken and Sergio Lucia (currently under review).

## Overview

This work presents **LISCO** (Learning-based Iterative Solver for Constrained Optimization), a novel approach for solving parametric nonlinear optimization problems with applications in model predictive control (MPC) and other real-time optimization scenarios.

### Key Features

- **Two-stage architecture**: Neural network predictor for initial primal-dual estimates + learned iterative solver for refinement
- **Self-supervised training**: No pre-sampled optimizer solutions required - training based on KKT optimality conditions
- **Theoretical guarantees**: Loss function proven to have minima exclusively at KKT points
- **Convexification procedure**: Enables application to nonconvex problems while preserving theoretical properties
- **Significant speedups**: Up to 10× faster than state-of-the-art solvers (IPOPT) while achieving orders of magnitude higher accuracy than competing learning-based approaches (DC3, PDL)

### Applications

The method is demonstrated on:
1. Nonlinear MPC of a double integrator system
2. Nonconvex parametric optimization problems (100 decision variables, 50 parameters)

## Citation

If you use this code in your research, please cite:
```
[Citation information will be added upon publication]
```

## Code Availability

Code for reproducing the paper results: https://github.com/lukaslueken/lisco-paper

## Repository Structure

### Core Source Files (`src/`)

#### `src/models.py`
Contains the core neural network architectures, solver/predictor classes, and training utilities:

**Neural Network Architectures:**
- `FeedforwardNN`: Configurable feedforward neural network with customizable depth, width, and activation functions

**Main Classes:**
- `Predictor`: Warm-start predictor that maps parameters to initial primal-dual solutions
- `Solver`: Learned iterative solver that refines solutions step-by-step
- `ApproxMPC`: Approximate MPC controller using direct neural network mapping (baseline)

**Utility Classes:**
- `TrainLogger`: Training metrics logger with GPU-buffered logging

**Utility Functions:**
- `export_jit_cpu()`, `export_jit_gpu()`: Model export with JIT compilation for CPU/GPU
- `count_params()`: Count trainable parameters
- `get_activation_layer()`: Activation layer selection
- `generate_experiment_dir()`: Automatic experiment directory generation

#### `src/mpl_config.py`
Matplotlib configuration for consistent visualization across all experiments

### Parametric Optimization Problem Example (`src/examples/parametric_OP/`)

This example demonstrates LISCO on general parametric nonlinear optimization problems.

#### Core Implementation
- **`parametric_OP.py`**: Main implementation with two classes:
  - `NLP`: PyTorch-based parametric optimization problem with nonconvex objectives (based on the formulation in DC3 paper)
  - `NLPCasadi`: CasADi-based implementation for benchmarking with IPOPT

#### Training Scripts
- **`training.py`**: Main training orchestration script supporting sweep configurations
  - Loads configuration files for sweeps
  - Functions: `train_predictor()`, `train_solver()`, `train_sweep()`
  - Trains predictor and/or solver models
- **`training_predictor.py`**: Standalone script for training warm-start predictor
- **`training_solver.py`**: Standalone script for training iterative solver
- **`make_configs.py`**: Configuration generator for sweeps

#### Data, Evaluation and Visualization
- **`generate_OPs.py`**: Script to generate random optimization problem instances
- **`data_sampling.py`**: Samples optimal solutions using IPOPT for testing
- **`evaluation.py`**: Comprehensive evaluation framework for both predictor and solver models
- **`visualization_paramOP.py`**: Generates aggregated figures and tables across multiple problem instances for parametric OP example evaluation

### Nonlinear Double Integrator Example (`src/examples/nonlinear_double_integrator/`)

This example demonstrates LISCO on a nonlinear model predictive control (NMPC) problem.

#### Core Implementation
- **`nonlinear_double_integrator.py`**: Main implementation with classes:
  - `NonlinearDoubleIntegrator`: PyTorch-based system dynamics (discrete-time nonlinear double integrator)
  - `NLP`: Optimal control problem formulation (multiple shooting)
  - `NonlinearDoubleIntegratorCasadi`: CasADi-based system model
  - `NLPCasadi`: CasADi-based OCP for IPOPT benchmarking
  - `NMPCCasadi`: Closed-loop NMPC controller for data generation

#### Training Scripts
- **`training.py`**: Main training orchestration script supporting sweep configurations
  - Loads configuration files for sweeps
  - Functions: `training_approxMPC()`, `train_predictor()`, `train_solver()`, `train_sweep()`
  - Trains predictor and/or solver models
- **`training_approxMPC.py`**: Standalone script for approximate MPC baseline
- **`training_predictor.py`**: Standalone script for training warm-start predictor
- **`training_solver.py`**: Standalone script for training iterative solver
- **`make_configs.py`**: Configuration generator for sweeps

#### Data, Evaluation and Visualization
- **`data_sampling.py`**: Generates closed-loop NMPC trajectories using IPOPT
- **`evaluation.py`**: Comprehensive evaluation framework
- **`visualization.py`**: Generates figures and tables for NMPC example evaluation

## Requirements
The project dependencies can be installed using either traditional pip/venv or the modern uv package manager (https://docs.astral.sh/uv/).

**Using pip with virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Using uv (recommended for faster, reproducible installations):**
```bash
uv sync
```

The `requirements.txt` file is used for pip-based installations, while `pyproject.toml` and `uv.lock` files are used by uv to ensure fully reproducible dependency resolution.

**Note on `torch.compile()`**: This project uses `torch.compile()` for performance optimization. Installing the required Triton compiler on Windows can be challenging and may require manual fixes (see [related GitHub issue](https://github.com/pytorch/pytorch/issues/162430)). 
On **Linux or macOS**, installation should work without issues. However, is using Linux or macOS, "triton" instead of "triton-windows" should be installed, either by changing the `pyproject.toml` or by changing the `requirements.txt` accordingly.

If you encounter problems with Triton/`torch.compile()`, you can disable compilation by setting `torch_compile=False` in the configuration files or training scripts. Note that this will increase computation times but does not affect the reproducibility of results.

## Reproducing Results
To reproduce the results from the paper, please perform the following steps:
1. Install the required dependencies using either pip/venv or uv as described above.
2. Navigate to src directory
3. Run experiments for nonlinear double integrator
    - the datasets for testing are already included in the repository, also the training datasets for the approximate MPC baseline are included
    - the configuration files for training all models are already included in the repository (`approxMPC_cfgs.json`, `predictor_cfgs.json`, `solver_cfgs.json`)
    1. run `training.py` in `src/examples/nonlinear_double_integrator/` to train predictor, solver and approximate MPC models
    2. run `evaluation.py` in `src/examples/nonlinear_double_integrator/` to evaluate all trained models
    3. run `visualization.py` in `src/examples/nonlinear_double_integrator/` to generate figures and tables
4. Run experiments for parametric optimization problems
    - the datasets for testing are already included in the repository
    - the configuration files for training all models are already included in the repository (`predictor_cfgs.json`, `solver_cfgs.json`)
    1. run `training.py` in `src/examples/parametric_OP/` to train predictor and solver models
    2. run `evaluation.py` in `src/examples/parametric_OP/` to evaluate all trained models
    3. run `visualization_paramOP.py` in `src/examples/parametric_OP/` to generate figures and tables
