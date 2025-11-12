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
from nonlinear_double_integrator import NonlinearDoubleIntegrator, NLP
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
def load_data(pth):
    """Load data from npz file and convert to torch tensors."""
    data = np.load(pth)
    data = {key: data[key] for key in data.files}
    # Convert to torch
    for key in data.keys():
        if key == "status":
            data[key] = data[key].astype(str)
        else:
            data[key] = torch.tensor(data[key], dtype=DTYPE, device=DEVICE)
    return data

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

def train_approxMPC(config, exp_pth):
    """Train ApproxMPC model.
    
    Args:
        config (dict): Configuration dictionary with keys:
            - model_cfg: Neural network architecture config
            - approxMPC_cfg: ApproxMPC-specific config (scaling, bounds, etc.)
            - train_cfg: Training hyperparameters
            - train_data_pth: Path to training data
            - val_data_pth: Path to validation data (optional)
        exp_pth (Path): Experiment directory path
        
    Returns:
        train_logger: Training history logger
        approx_mpc: Trained ApproxMPC instance
        cfg: Extended configuration with training results
    """
    # Extract configs
    model_cfg = config["model_cfg"]
    approxMPC_cfg = config["approxMPC_cfg"]
    train_cfg = config["train_cfg"]
    train_data_pth = FILE_PTH.joinpath(train_cfg["train_data_pth"])
    val_data_pth = FILE_PTH.joinpath(train_cfg.get("val_data_pth")) if "val_data_pth" in train_cfg else None

    # Data loading
    print("Loading data...")
    train_data = load_data(train_data_pth)
    val_data = load_data(val_data_pth) if val_data_pth is not None else None
    print("Data loaded.")

    # Create model
    model = FeedforwardNN(**model_cfg)
    
    # Create ApproxMPC
    approx_mpc = ApproxMPC(model, **approxMPC_cfg)

    # Setup datasets
    if val_data is not None:
        train_dataset, val_dataset = approx_mpc.setup_datasets(
            X_train=train_data["x0"],
            Y_train=train_data["u0"],
            X_val=val_data["x0"],
            Y_val=val_data["u0"],
            batch_size=train_cfg["batch_size"]
        )
    else:
        train_dataset, val_dataset = approx_mpc.setup_datasets(
            X_train=train_data["x0"],
            Y_train=train_data["u0"],
            batch_size=train_cfg["batch_size"]
        )

    # Training
    train_logger = approx_mpc.train(
        train_cfg,
        train_dataset,
        val_dataset,
        log_values=True
    )

    # Visualizations
    _ = train_logger.visualize_history("train_loss", exp_pth=exp_pth, save_fig=True,log_scaling=True)
    if val_dataset is not None:
        _ = train_logger.visualize_history("val_loss", exp_pth=exp_pth, save_fig=True,log_scaling=True)
    _ = train_logger.visualize_history("lr", exp_pth=exp_pth, save_fig=True,log_scaling=True)
    plt.close("all")

    # Save results
    train_logger.save_history(exp_pth=exp_pth, file_name="history", as_json=True)
    approx_mpc.save_weights(exp_pth, file_name="approx_mpc_weights.pt")

    # Extend Config
    cfg = config.copy()
    system = NonlinearDoubleIntegrator(dtype=DTYPE, device=DEVICE)
    system_cfg = system.export_hparams()
    cfg.update({"system_cfg": system_cfg})
    
    cfg["trained"] = True
    cfg["evaluated"] = False
    cfg["train_time"] = train_logger.history["train_time"]
    cfg["N_epochs_trained"] = train_logger.history["epoch"][-1] # + 1
    cfg["n_params"] = count_params(model)
    cfg["exp_pth"] = str(exp_pth)
    cfg["dtype"] = str(DTYPE)
    cfg["device"] = str(DEVICE)

    cfg = convert_tensors_to_lists(cfg)

    with open(exp_pth.joinpath("config.json"), "w") as f:
        json.dump(cfg, f, indent=4)

    return train_logger, approx_mpc, cfg

def setup_nlp(nlp_cfg):
    system = NonlinearDoubleIntegrator(dtype=DTYPE,device=DEVICE)
    nlp = NLP(system,**nlp_cfg)
    return nlp

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

    # Visualizations
    _ = train_logger.visualize_history("loss", exp_pth=exp_pth, save_fig=True)

    if "p_final_batch" in train_logger.history.keys():
        p_final_batch = np.array(train_logger.history["p_final_batch"])
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(p_final_batch[:, 0], p_final_batch[:, 1], "x")
        ax.set_title("p_final_batch")
        ax.set_xlabel("x0")
        ax.set_ylabel("x1")
        ax.set_xlim([-12, 12])
        ax.set_ylim([-12, 12])
        fig.savefig(exp_pth.joinpath("p_final_batch.png"), bbox_inches="tight")

    plt.close("all")

    # Save predictor
    train_logger.save_history(exp_pth=exp_pth, file_name="history", as_json=True)
    predictor.save_weights(exp_pth, file_name="predictor_weights.pt")

    # Extend Config
    cfg = config.copy()
    system_cfg = nlp.system.export_hparams()
    cfg.update({"system_cfg": system_cfg})

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

def load_predictor(pth):
    if pth is None:
        return None, None
    else:
        # pth = Path(pth)
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

def train_solver(config,exp_pth):
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
    predictor, pred_cfg = load_predictor(train_cfg["predictor_pth"]) # is none if not needed

    # Setup Solver
    solver = Solver(model,nlp,predictor=predictor,**solver_cfg)

    # Train Solver
    train_logger = solver.train(train_cfg,log_values=True)

    # Postprocessing
    train_logger.post_calculate_ema_resample_metrics(batch_size=train_cfg["batch_size"],ema_beta=0.99)

    # Visualizations
    _ = train_logger.visualize_history("loss",exp_pth=exp_pth,save_fig=True)
    _ = train_logger.visualize_history("conv_frac_samples",exp_pth=exp_pth,save_fig=True)
    _ = train_logger.visualize_history("conv_frac_batch",exp_pth=exp_pth,save_fig=True)
    _ = train_logger.visualize_history("n_resample",exp_pth=exp_pth,save_fig=True)
    _ = train_logger.visualize_history("n_resample_tol",exp_pth=exp_pth,save_fig=True)
    _ = train_logger.visualize_history("n_resample_iter",exp_pth=exp_pth,save_fig=True)

    if "p_final_batch" in train_logger.history.keys():
            p_final_batch = np.array(train_logger.history["p_final_batch"])
            fig,ax = plt.subplots(1,1,figsize=(10,5))
            ax.plot(p_final_batch[:,0],p_final_batch[:,1],"x")
            ax.set_title("p_final_batch")
            ax.set_xlabel("x0")
            ax.set_ylabel("x1")
            ax.set_xlim([-12,12])
            ax.set_ylim([-12,12])
            fig.savefig(exp_pth.joinpath("p_final_batch.png"),bbox_inches="tight")

    plt.close("all")

    # Save solver
    train_logger.save_history(exp_pth=exp_pth,file_name="history",as_json=True)
    solver.save_weights(exp_pth,file_name="solver_weights.pt")

    # Extend Config
    cfg = config.copy()
    system_cfg = nlp.system.export_hparams()
    cfg.update({"system_cfg": system_cfg})

    if predictor is not None:
        cfg.update({"predictor_cfg": pred_cfg})

    cfg["trained"] = True
    cfg["evaluated"] = False
    cfg["train_time"] = train_logger.history["train_time"]
    cfg["N_epochs_trained"] = train_logger.history["epoch"][-1] #+ 1
    cfg["n_params"] = count_params(model)
    cfg["exp_pth"] = str(exp_pth)
    cfg["dtype"] = str(DTYPE)
    cfg["device"] = str(DEVICE)

    with open(exp_pth.joinpath("config.json"), "w") as f:
        json.dump(cfg, f,indent=4)
    
    return train_logger, solver, cfg

def load_config(config_pth):
    with open(config_pth,"r") as fp:
        cfg = json.load(fp)
    return cfg

def train_single(config,run_folder):
    mode = config["mode"]
    exp_pth = generate_experiment_dir(log_dirs=[run_folder,mode],save_dir=FILE_PTH)
    
    if mode == "approxMPC":
        train_logger, approx_mpc, cfg = train_approxMPC(config,exp_pth)
        print_training_end(cfg)

    elif mode == "predictor":
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
    """Print training completion message."""
    print("-" * 20)
    print(f"Training of {cfg['mode']} completed after {cfg['train_time']}s.")
    print(f"Results stored in {cfg['exp_pth']}")
    print("\n")


# %%
# RUN TRAINING
if __name__ == "__main__":
    # CONFIG
    run_folder = "results" # where the results should be stored

    sweep_config_pth = FILE_PTH.joinpath("approxMPC_cfgs.json")
    sweep_config = load_config(sweep_config_pth)
    cfgs = train_sweep(sweep_config,sweep_config_pth,run_folder)
    
    sweep_config_pth = FILE_PTH.joinpath("predictor_cfgs.json")
    sweep_config = load_config(sweep_config_pth)
    cfgs = train_sweep(sweep_config,sweep_config_pth,run_folder)
    
    sweep_config_pth = FILE_PTH.joinpath("solver_cfgs.json")
    sweep_config = load_config(sweep_config_pth)
    cfgs = train_sweep(sweep_config,sweep_config_pth,run_folder)