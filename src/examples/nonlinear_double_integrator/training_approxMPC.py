# %% Imports
import sys
# setting path
sys.path.append('../../')

import torch
from pathlib import Path
from nonlinear_double_integrator import NonlinearDoubleIntegrator
from models import generate_experiment_dir
from training import train_approxMPC, print_training_end, DTYPE, DEVICE, SEED

# Meta Configuration
FILE_PTH = Path(__file__).parent.resolve()
torch.set_default_dtype(DTYPE)
torch.set_default_device(DEVICE)
torch.autograd.set_detect_anomaly(False)
torch.manual_seed(SEED)

# %% Configuration
RUN_FOLDER = "results"  # where the results should be stored

system = NonlinearDoubleIntegrator(device=DEVICE, dtype=DTYPE)

model_cfg = {
    "n_in": system.x_dim,
    "n_out": system.u_dim,
    "n_hidden_layers": 4,
    "n_neurons": 512,
    "act_fn": "gelu",
    "output_act_fn": "linear",
}

approxMPC_cfg = {
    "scale_data": True,
    "bounds": {
        "x_lb": system.x_lb,
        "x_ub": system.x_ub,
        "u_lb": system.u_lb,
        "u_ub": system.u_ub
    },
    "torch_compiled": False,
}

train_cfg = {
    "N_epochs": 50000,
    "batch_size": 4096,
    "weight_decay": 0.0,
    "use_amsgrad": True,
    "lr": 1e-3,
    "train_data_pth": str(Path("data", "closed_loop_N_25000_Nsim_5_0", "nmpc_data.npz")),
    "val_data_pth": str(Path("data", "closed_loop_N_2500_Nsim_5_0", "nmpc_data.npz")),
    "use_lr_scheduler": False,
    "stop_at_min_lr": False,
    "lr_scheduler_patience": 500,
    "lr_scheduler_cooldown": 500,
    "lr_reduce_factor": 0.1,
    "min_lr": 1e-9,
}

config = {
    "mode": "approxMPC",
    "model_cfg": model_cfg,
    "approxMPC_cfg": approxMPC_cfg,
    "train_cfg": train_cfg,
}

# %% Main
mode = "approxMPC"
exp_pth = generate_experiment_dir(log_dirs=[RUN_FOLDER, mode], save_dir=FILE_PTH)
train_logger, approx_mpc, cfg = train_approxMPC(config, exp_pth)
print_training_end(cfg)