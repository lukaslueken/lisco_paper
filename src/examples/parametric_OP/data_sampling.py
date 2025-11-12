# %% - Imports
import sys
# setting path
sys.path.append('../../')

import numpy as np
from pathlib import Path
import json
import torch
from time import perf_counter
from parametric_OP import NLP, NLPCasadi
import matplotlib.pyplot as plt

# %% - CONFIGURATION
data_folder = "data"
TOL = 1e-6
save_data = True
seed = 42
offset = 0.0  # Offset on parameter sampling space
N_samples = 10  # Number of samples to generate
filter_mode = "successful"  # Only save successful solves

# Set paths
file_pth = Path(__file__).parent.resolve()
DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# List of problem names to process
problem_names = [
    "nonconvex_100x50x50_0", 
    "nonconvex_100x50x50_1", 
    "nonconvex_100x50x50_2", 
    "nonconvex_100x50x50_3", 
    "nonconvex_100x50x50_4", 
]

# %% - FUNCTIONS
def check_KKT_conditions(z_opt, p, KKT_lim=1e-5):
    """Check if solution satisfies KKT conditions up to tolerance."""
    z_opt = torch.tensor(z_opt, dtype=DTYPE, device=DEVICE)
    p = torch.tensor(p, dtype=DTYPE, device=DEVICE)
    KKT_val = nlp.KKT_func(z_opt, p)
    KKT_inf_norm = torch.linalg.norm(KKT_val, ord=float("inf")).item()
    return KKT_inf_norm <= KKT_lim

def visualize_data(data, meta_data):
    """Create visualization of sampled data."""
    figures = []
    
    # Plot parameter distribution
    fig, ax = plt.subplots()
    ax.hist(data["p"].reshape(-1), bins=50)
    ax.set_title("Parameter Distribution")
    ax.set_xlabel("Parameter Value")
    ax.set_ylabel("Count")
    figures.append(fig)
    
    # Plot solve times
    fig, ax = plt.subplots()
    ax.hist(data["solve_time"], bins=50)
    ax.set_title("Solve Time Distribution")
    ax.set_xlabel("Solve Time [s]")
    ax.set_ylabel("Count")
    ax.axvline(meta_data["mean_optimal_solve_time"], color='r', linestyle='--', 
               label=f'Mean: {meta_data["mean_optimal_solve_time"]:.3f}s')
    ax.legend()
    figures.append(fig)
    
    return figures

# %% - SAMPLE DATA
for problem_name in problem_names:
    print(f"\nProcessing problem: {problem_name}")
    
    # Load the optimization problem
    problem_path = file_pth.joinpath(data_folder, f"{problem_name}","op_cfg.json")
    if not problem_path.exists():
        print(f"Problem file not found: {problem_path}")
        continue
        
    # Load and initialize the problem
    nlp = NLP.from_json(problem_path.parent, problem_path.stem)
    nlp.set_dtype(DTYPE)
    nlp_casadi = NLPCasadi(op_dict=nlp.op_dict, **nlp.export_hparams())
    nlp_casadi.opts = {"ipopt.tol":TOL}
    
    N_optimal = 0
    N_total = 0
    N_visited = 0

    p_data = []
    status_data = []
    z_opt_data = []
    solve_time_data = []

    start_time = perf_counter()

    while N_optimal < N_samples:
        # Generate random parameter
        p = nlp.batch_gen_p(1).squeeze().cpu().numpy()
        
        # Solve optimization problem
        w_opt, nu_opt, lam_opt, status, solve_time = nlp_casadi.solve(p)
        N_visited += 1
        
        # Check optimality
        z_opt = nlp_casadi.stack_primal_dual(w_opt, nu_opt, lam_opt)
        optimal = check_KKT_conditions(z_opt, p,KKT_lim=TOL)
        
        if filter_mode == "successful" and optimal and status == "Solve_Succeeded":
            p_data.append(p)
            status_data.append(status)
            z_opt_data.append(z_opt)
            solve_time_data.append(solve_time)
            N_optimal += 1
            N_total += 1
            
            if N_optimal % 1000 == 0:
                print(f"Generated {N_optimal}/{N_samples} optimal solutions")

    end_time = perf_counter()
    sampling_time = end_time - start_time

    # Convert to arrays
    p_data = np.array(p_data)
    status_data = np.array(status_data)
    z_opt_data = np.array(z_opt_data)
    solve_time_data = np.array(solve_time_data)

    # Calculate statistics
    solve_time_mean = np.mean(solve_time_data)
    solve_time_std = np.std(solve_time_data)
    solve_time_max = np.max(solve_time_data)

    # Package data
    data = {
        "p": p_data,
        "status": status_data,
        "z_opt": z_opt_data,
        "solve_time": solve_time_data
    }

    meta_data = {
        "N_visited": N_visited,
        "N_total": N_total,
        "N_optimal": N_optimal,
        "sampling_time": sampling_time,
        "mean_optimal_solve_time": solve_time_mean,
        "std_optimal_solve_time": solve_time_std,
        "max_optimal_solve_time": solve_time_max
    }

    # Print statistics
    print(f"\nSampling Results for {problem_name}:")
    for key in ["N_visited", "N_total", "N_optimal", "sampling_time", 
                "mean_optimal_solve_time", "max_optimal_solve_time"]:
        print(f"{key}: {meta_data[key]}")

    # Visualize data
    figures = visualize_data(data, meta_data)

    # Save data
    if save_data:
        # Create configuration dictionary
        config = {
            "problem_name": problem_name,
            "tolerance": TOL,
            "offset": offset,
            "N_samples": N_samples,
            "filter_mode": filter_mode,
            "seed": seed
        }
        config.update(meta_data)
        
        # Create save folder
        save_folder = f"N_{meta_data['N_optimal']}"
        
        j = 0
        while j < 100:
            folder_path = problem_path.parent.joinpath(save_folder + f"_{j}")
            if not folder_path.exists():
                folder_path.mkdir(parents=True)
                break
            j += 1
        
        # Save figures
        for idx, fig in enumerate(figures):
            fig.savefig(folder_path.joinpath(f"figure_{idx}.png"))
            plt.close(fig)
        
        # Save data using np.savez
        np.savez(folder_path.joinpath("op_data.npz"), **data)
        
        # Save config using json
        with open(folder_path.joinpath("config.json"), "w") as f:
            json.dump(config, f, indent=4)
        
        print(f"\nData saved to: {folder_path}")