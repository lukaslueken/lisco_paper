# %% Imports
import sys
# setting path
sys.path.append('../../')

import torch
import numpy as np
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from parametric_OP import NLP
from models import FeedforwardNN, Solver, generate_experiment_dir, Predictor, count_params, ApproxMPC

# %% Meta Configuration
SEED = 42
DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FILE_PTH = Path(__file__).parent.resolve()

torch.set_default_dtype(DTYPE)
torch.set_default_device(DEVICE)
torch.autograd.set_detect_anomaly(False)
torch.manual_seed(SEED)

# %% Functions
def convert_tensors_to_lists(d):
    """Recursively convert tensors in dict to lists for JSON serialization."""
    if isinstance(d, dict):
        return {k: convert_tensors_to_lists(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_tensors_to_lists(i) for i in d]
    elif torch.is_tensor(d):
        return d.tolist()
    else:
        return d

def train_predictor(config, exp_pth):
    # Configs
    model_cfg = config["model_cfg"]
    train_cfg = config["train_cfg"]
    nlp_cfg = config["nlp_cfg"]
    predictor_cfg = config["predictor_cfg"]

    # Setup NLP
    nlp = setup_nlp(nlp_cfg)

    # Setup NN
    model = FeedforwardNN(**model_cfg)

    # Setup Predictor
    predictor = Predictor(model, **predictor_cfg)

    # Train Predictor
    train_logger = predictor.train(nlp, train_cfg, log_values=True)

    # Postprocessing
    train_logger.post_calculate_ema_resample_metrics(batch_size=train_cfg["batch_size"], ema_beta=0.99)

    # Visualizations
    _ = train_logger.visualize_history("loss", exp_pth=exp_pth, save_fig=True)

    if "p_final_batch" in train_logger.history.keys():
        p_final_batch = np.array(train_logger.history["p_final_batch"])
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(p_final_batch[:, 0], p_final_batch[:, 1], "x")
        ax.set_title("p_final_batch")
        ax.set_xlabel("p0")
        ax.set_ylabel("p1")
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        fig.savefig(exp_pth.joinpath("p_final_batch.png"), bbox_inches="tight")

    plt.close("all")

    # Save predictor
    train_logger.save_history(exp_pth=exp_pth, file_name="history", as_json=True)
    predictor.save_weights(exp_pth, file_name="predictor_weights.pt")

    # Extend Config
    cfg = config.copy()
    op_hparams = nlp.export_hparams()
    cfg.update({"op_hparams": op_hparams})

    cfg["trained"] = True
    cfg["evaluated"] = False
    cfg["train_time"] = train_logger.history["train_time"]
    cfg["N_epochs_trained"] = train_logger.history["epoch"][-1] # + 1
    cfg["n_params"] = count_params(model)
    cfg["exp_pth"] = str(exp_pth)
    cfg["dtype"] = str(DTYPE)
    cfg["device"] = str(DEVICE)

    with open(exp_pth.joinpath("config.json"), "w") as f:
        json.dump(cfg, f, indent=4)
    
    return train_logger, predictor, cfg

def train_solver(config, exp_pth):
    """Train solver for a single optimization problem.
    
    Args:
        config (dict): Configuration dictionary containing:
            - model_cfg: Neural network architecture config
            - train_cfg: Training hyperparameters
            - nlp_cfg: NLP problem configuration
            - solver_cfg: Solver-specific config
        exp_pth (Path): Experiment directory path
        
    Returns:
        train_logger: Training history logger
        solver: Trained solver instance
        cfg: Extended configuration with training results
    """
    # Configs
    model_cfg = config["model_cfg"]
    train_cfg = config["train_cfg"]
    nlp_cfg = config["nlp_cfg"]
    solver_cfg = config["solver_cfg"]

    # Setup NLP
    nlp = setup_nlp(nlp_cfg)

    # Setup NN
    model = FeedforwardNN(**model_cfg)

    # Load predictor if needed
    predictor, pred_cfg = load_predictor(train_cfg["predictor_pth"])

    # Setup Solver
    solver = Solver(model, nlp, predictor=predictor, **solver_cfg)

    # Train Solver
    train_logger = solver.train(train_cfg, log_values=True)

    # Postprocessing
    train_logger.post_calculate_ema_resample_metrics(batch_size=train_cfg["batch_size"], ema_beta=0.99)

    # Visualizations
    _ = train_logger.visualize_history("loss", exp_pth=exp_pth, save_fig=True)
    _ = train_logger.visualize_history("conv_frac_samples", exp_pth=exp_pth, save_fig=True)
    _ = train_logger.visualize_history("conv_frac_batch", exp_pth=exp_pth, save_fig=True)
    _ = train_logger.visualize_history("n_resample", exp_pth=exp_pth, save_fig=True)
    _ = train_logger.visualize_history("n_resample_tol", exp_pth=exp_pth, save_fig=True)
    _ = train_logger.visualize_history("n_resample_iter", exp_pth=exp_pth, save_fig=True)

    if "p_final_batch" in train_logger.history.keys():
        p_final_batch = np.array(train_logger.history["p_final_batch"])
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(p_final_batch[:, 0], p_final_batch[:, 1], "x")
        ax.set_title("p_final_batch")
        ax.set_xlabel("p0")
        ax.set_ylabel("p1")
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        fig.savefig(exp_pth.joinpath("p_final_batch.png"), bbox_inches="tight")

    plt.close("all")

    # Save solver
    train_logger.save_history(exp_pth=exp_pth, file_name="history", as_json=True)
    solver.save_weights(exp_pth, file_name="solver_weights.pt")

    # Extend Config
    cfg = config.copy()

    if predictor is not None:
        cfg.update({"predictor_cfg": pred_cfg})

    cfg["trained"] = True
    cfg["evaluated"] = False
    cfg["train_time"] = train_logger.history["train_time"]
    cfg["N_epochs_trained"] = train_logger.history["epoch"][-1] # + 1
    cfg["n_params"] = count_params(model)
    cfg["exp_pth"] = str(exp_pth)
    cfg["dtype"] = str(DTYPE)
    cfg["device"] = str(DEVICE)

    with open(exp_pth.joinpath("config.json"), "w") as f:
        json.dump(cfg, f, indent=4)
    
    return train_logger, solver, cfg

def setup_nlp(nlp_cfg):
    op_pth = FILE_PTH.joinpath(nlp_cfg["op_pth"])
    nlp = NLP.from_json(op_pth, "op_cfg")
    nlp.set_device(DEVICE)
    nlp.set_dtype(DTYPE)
    if "eps" in nlp_cfg:
        nlp.eps = nlp_cfg["eps"]
        nlp._setup_functions()  # re-setup functions to account for new eps
    if "sigma" in nlp_cfg:
        nlp.sigma = nlp_cfg["sigma"]
        nlp._setup_functions()  # re-setup functions to account for new sigma
    return nlp

def load_predictor(pth):
    if pth is None:
        return None, None
    else:
        pth = FILE_PTH.joinpath(pth)
        with open(pth.joinpath("config.json"),"r") as fp:
            prd_cfg = json.load(fp)
        model_cfg = prd_cfg["model_cfg"]
        model = FeedforwardNN(**model_cfg)
        # predictor_cfg = prd_cfg.get("predictor_cfg",{})
        # predictor = Predictor(model,**predictor_cfg)
        predictor = Predictor(model,convexification=False,torch_compiled=False)
        predictor.load_weights(pth,file_name="predictor_weights.pt")
        return predictor, prd_cfg

def load_config(config_pth):
    with open(config_pth,"r") as fp:
        cfg = json.load(fp)
    return cfg

def train_single(config,run_folder):
    mode = config["mode"]
    op_name = config["nlp_cfg"].get("op_name")
    exp_pth = generate_experiment_dir(log_dirs=[run_folder,op_name,mode],save_dir=FILE_PTH)
    
    if mode == "predictor":
        train_logger, predictor, cfg = train_predictor(config,exp_pth)
        print_training_end(cfg)
    
    elif mode == "solver":
        train_logger, solver, cfg = train_solver(config,exp_pth)
        print_training_end(cfg)

    else:
        raise ValueError(f"Mode {mode} not recognized.")
    
    return cfg

def train_sweep(sweep_config,sweep_config_pth,run_folder):
    sweep_cfg = sweep_config.copy()
    cfgs = []
    for i,config in enumerate(sweep_cfg):
        trained = config.get("trained",False)
        if trained:
            print("Skipping config as it is already trained.")
            continue
        cfg = train_single(config,run_folder)
        cfgs.append(cfg)

        # update configs
        sweep_cfg[i] = cfg

        # save and update configs after every training run
        with open(sweep_config_pth,"w") as fp:
            json.dump(sweep_cfg,fp,indent=4)
    return cfgs

def print_training_end(cfg):
    print("-" * 20)
    print(f"Training of {cfg['mode']} completed after {cfg['train_time']}s.")
    print(f"Results stored in {cfg['exp_pth']}")
    print("\n")

# %%
# RUN TRAINING
if __name__ == "__main__":
    # CONFIG
    run_folder = "results" # where the results should be stored

    sweep_config_pth = FILE_PTH.joinpath("predictor_cfgs.json")
    sweep_config = load_config(sweep_config_pth)
    cfgs = train_sweep(sweep_config,sweep_config_pth,run_folder)
    
    sweep_config_pth = FILE_PTH.joinpath("solver_cfgs.json")
    sweep_config = load_config(sweep_config_pth)
    cfgs = train_sweep(sweep_config,sweep_config_pth,run_folder)