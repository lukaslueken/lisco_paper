# %% Imports
import sys
# setting path
sys.path.append('../../')

import torch
from pathlib import Path
from nonlinear_double_integrator import NonlinearDoubleIntegrator, NLP
from models import generate_experiment_dir
from training import train_predictor, print_training_end, DTYPE, DEVICE, SEED

# Meta Configuration
FILE_PTH = Path(__file__).parent.resolve()
torch.set_default_dtype(DTYPE)
torch.set_default_device(DEVICE)
torch.autograd.set_detect_anomaly(False)
torch.manual_seed(SEED)

# %%
# Configs
RUN_FOLDER = "results"  # where the results should be stored

system = NonlinearDoubleIntegrator(device=DEVICE, dtype=DTYPE)
nlp_cfg = {
    "N": 10,
    "eps": 1e-16,
    "sigma": 1e-2,
}
nlp = NLP(system, **nlp_cfg)

model_cfg = {
    "n_in": nlp.n_p,
    "n_out": nlp.n_z,
    "n_hidden_layers": 4,
    "n_neurons": 512,
    "act_fn": "gelu",
    "output_act_fn": "linear",
}

train_cfg = {
    "N_epochs": 50,
    "batch_size": 4096,
    "weight_decay": 0.0,
    "lr": 1e-3,
    "use_amsgrad": True,
}

predictor_cfg = {
    "convexification": True,
    "torch_compiled": False,
}

config = {
    "mode": "predictor",
    "model_cfg": model_cfg,
    "train_cfg": train_cfg,
    "nlp_cfg": nlp_cfg,
    "predictor_cfg": predictor_cfg,
}

# %%
# Main
mode = "predictor"
exp_pth = generate_experiment_dir(log_dirs=[RUN_FOLDER, mode], save_dir=FILE_PTH)
train_logger, predictor, cfg = train_predictor(config, exp_pth)
print_training_end(cfg)
