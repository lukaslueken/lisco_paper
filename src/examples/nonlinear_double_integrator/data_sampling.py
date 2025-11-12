# %% - Imports
import sys
# setting path
sys.path.append('../../')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import torch
from time import perf_counter
from nonlinear_double_integrator import NonlinearDoubleIntegratorCasadi, NMPCCasadi, NonlinearDoubleIntegrator, NLP
import mpl_config

DTYPE = torch.float64
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

# %% - CONFIGURATION
data_folder = "data"
TOL = 1e-6
save_data = True
seed = 42
N_control = 10
offset = 0.0 # Offset on initial state sampling space (fraction of state space)
noise = 0.0 # Noise has no effect for N_sim=1
N_sim = 5 # N_sim=1: open-loop sampling, N_sim>1: closed-loop sampling
N_trajectories = 5 # Number of successful(!) trajectories to sample
# filter_mode = "all" # saves all datapoints (successful and unsuccessful): N_total=N_trajectories*(individual_length) = N_visited >= N_optimal
filter_mode = "successful_trajectories" # saves only successful trajectories: N_total=N_optimal = N_trajectories*N_sim <= N_visited

# Setup System and NMPC
system_casadi = NonlinearDoubleIntegratorCasadi()
system = NonlinearDoubleIntegrator(dtype=DTYPE, device=DEVICE)
nlp = NLP(system, N=N_control)
nmpc_casadi = NMPCCasadi(system_casadi, N=N_control)
nmpc_casadi.opts = {"ipopt.tol": TOL} # Set ipopt options


# %% - FUNCTIONS

def check_trajectory_KKT(trajectory,KKT_lim=1e-6):
    """
    Check if all values in trajectory are optimal up to tolerance.
    """
    for val in trajectory:
        z_opt = val["z_opt"].squeeze()
        p = val["x"].squeeze()
        z_opt = torch.tensor(z_opt, dtype=DTYPE, device=DEVICE)
        p = torch.tensor(p, dtype=DTYPE, device=DEVICE)
        # Check KKT conditions
        KKT_val = nlp.KKT_func(z_opt,p)

        KKT_inf_norm = torch.linalg.norm(KKT_val, ord=float("inf")).item()
        if KKT_inf_norm >= KKT_lim:
            print(f"Trajectory not optimal: KKT_2_norm = {KKT_inf_norm:.2e} > {KKT_lim:.2e}")
            return False
        
        # KKT_2_norm = torch.linalg.norm(KKT_val, ord=2).item()
        # if KKT_2_norm >= KKT_lim:
            # print(f"Trajectory not optimal: KKT_2_norm = {KKT_2_norm:.2e} > {KKT_lim:.2e}")
            # return False

    return True

def visualize_nmpc_casadi_data(data,nmpc_casadi):
    x_data = data["x0"]
    u_data = data["u0"]
    status_data = data["status"]
    z_opt_data = data["z_opt"]

    # Visualization
    # 1. Plot x_data distribution x1 over x2
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    ax.set_title("Distribution of sampled data")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    x1_data = []
    x2_data = []
    u0_data = []
    x1_data_inf = []
    x2_data_inf = []
    for idx,x in enumerate(x_data):
        if status_data[idx] == "Solve_Succeeded":
            x1_data.append(x[0])
            x2_data.append(x[1])
            u0_data.append(u_data[idx][0])
        else:
            x1_data_inf.append(x[0])
            x2_data_inf.append(x[1])
    ax.scatter(x1_data,x2_data,label="Succeeded",color="blue",marker="x")
    ax.scatter(x1_data_inf,x2_data_inf,label="Failed",color="red",marker="x")
    ax.set_ylim(nmpc_casadi.system.x_lb[1,:].toarray(),nmpc_casadi.system.x_ub[1,:].toarray())
    ax.set_xlim(nmpc_casadi.system.x_lb[0,:].toarray(),nmpc_casadi.system.x_ub[0,:].toarray())
    ax.legend()

    fig2, ax2 = plt.subplots(1,1,figsize=(10,10))
    ax2.hist(u0_data,bins=20)
    ax2.set_title("Distribution of sampled data")
    ax2.set_xlabel("u0")
    ax2.set_ylabel("Frequency")

    # w
    idx_start = 0
    idx_end = nmpc_casadi.nlp.n_w

    fig_w, ax_w = plt.subplots(1,1,figsize=(10,10))
    val = z_opt_data[:,idx_start:idx_end]
    ax_w.plot(np.max(val,axis=0),"x",label="max")
    ax_w.plot(np.quantile(val,0.90,axis=0),"x",label="90%")
    # ax_w.plot(np.quantile(val,0.75,axis=0),"x",label="75%")
    ax_w.plot(np.median(val,axis=0),"x",label="median")
    # ax_w.plot(np.mean(val,axis=0),"x")
    # ax_w.plot(np.quantile(val,0.25,axis=0),"x",label="25%")
    ax_w.plot(np.quantile(val,0.10,axis=0),"x",label="10%")
    ax_w.plot(np.min(val,axis=0),"x",label="min")
    ax_w.legend()
    ax_w.set_title("w")

    # nu
    idx_start = nmpc_casadi.nlp.n_w
    idx_end = nmpc_casadi.nlp.n_w+nmpc_casadi.nlp.n_nu

    fig_nu, ax_nu = plt.subplots(1,1,figsize=(10,10))
    val = z_opt_data[:,idx_start:idx_end]
    val = np.abs(val)
    ax_nu.plot(np.max(val,axis=0),"x",label="max")
    ax_nu.plot(np.quantile(val,0.90,axis=0),"x",label="90%")
    ax_nu.plot(np.quantile(val,0.75,axis=0),"x",label="75%")
    ax_nu.plot(np.median(val,axis=0),"x",label="median")
    # ax_nu.plot(np.mean(val,axis=0),"x")
    ax_nu.plot(np.min(val,axis=0),"x",label="min")
    ax_nu.legend()
    ax_nu.set_yscale("log")
    ax_nu.set_title("log-abs nu")

    # lam
    idx_start = nmpc_casadi.nlp.n_w+nmpc_casadi.nlp.n_nu
    idx_end = nmpc_casadi.nlp.n_w+nmpc_casadi.nlp.n_nu+nmpc_casadi.nlp.n_lam

    fig_lam, ax_lam = plt.subplots(1,1,figsize=(10,10))
    val = z_opt_data[:,idx_start:idx_end]
    val = np.abs(val)
    ax_lam.plot(np.max(val,axis=0),"x",label="max")
    ax_lam.plot(np.quantile(val,0.90,axis=0),"x",label="90%")
    ax_lam.plot(np.quantile(val,0.75,axis=0),"x",label="75%")
    ax_lam.plot(np.median(val,axis=0),"x",label="median")
    # ax_lam.plot(np.mean(val,axis=0),"x")
    ax_lam.plot(np.min(val,axis=0),"x",label="min")
    ax_lam.legend()
    ax_lam.set_yscale("log")
    ax_lam.set_title("log-abs lam")

    # plt.show()
    plt.close('all')

    return [fig,fig2,fig_w,fig_nu,fig_lam]

# %% - SAMPLE DATA
N_optimal = 0
N_total = 0
N_visited = 0
k = 0

x_data = []
u_data = []
status_data = []
z_opt_data = []
solve_time_data = []

start_time = perf_counter()

while k < N_trajectories:
    x0 = nmpc_casadi.nlp.sample_x0(offset=offset)
    trajectory, trajectory_status = nmpc_casadi.run_nmpc_closed_loop(x0,N_sim,noise=noise)

    # check if trajectory is optimal
    optimal = check_trajectory_KKT(trajectory,KKT_lim=TOL)
    if not optimal:
        trajectory_status = False

    if filter_mode == "all":
        for val in trajectory:
            x_data.append(val["x"])
            u_data.append(val["u"])
            status_data.append(val["status"])
            z_opt_data.append(val["z_opt"])
            solve_time_data.append(val["solve_time"])
            if val["status"] == "Solve_Succeeded":
                N_optimal += 1 # count all optimal samples
            N_total += 1 # count all samples
            N_visited += 1 # count all visited samples
        k += 1 # count all trajectories
    
    elif filter_mode == "successful_trajectories": # only consider successful trajectories (i.e. all samples in the trajectory are successful)
        if trajectory_status:
            for val in trajectory:
                x_data.append(val["x"])
                u_data.append(val["u"])
                status_data.append(val["status"])
                z_opt_data.append(val["z_opt"])
                solve_time_data.append(val["solve_time"])
                N_total += 1 # count all samples
                if val["status"] == "Solve_Succeeded":
                    N_optimal += 1 # count all optimal samples
            k += 1 # count successful trajectories
        N_visited += len(trajectory) # count all visited samples    
    else:
        raise ValueError("Invalid filter_mode")

end_time = perf_counter()
sampling_time = end_time-start_time
print(f"Elapsed time: {sampling_time:.2f} s")

x_data = np.array(x_data).squeeze()
u_data = np.array(u_data)
status_data = np.array(status_data)
solve_time_data = np.array(solve_time_data)
z_opt_data = np.array(z_opt_data)
data = {"x0":x_data,"u0":u_data,"status":status_data,"z_opt":z_opt_data,"solve_time":solve_time_data}

# per optimal sample time
idx_successful_samples = np.where(data["status"] == "Solve_Succeeded")
solve_time_mean = np.mean(data["solve_time"][idx_successful_samples])
solve_time_std = np.std(data["solve_time"][idx_successful_samples])
solve_time_max = np.max(data["solve_time"][idx_successful_samples])

meta_data = {"N_visited":N_visited,"N_total":N_total,"N_optimal":N_optimal,"sampling_time":sampling_time,
                "mean_optimal_solve_time":solve_time_mean,"std_optimal_solve_time":solve_time_std,"max_optimal_solve_time":solve_time_max}

for key in ["N_visited","N_total","N_optimal","sampling_time","mean_optimal_solve_time","max_optimal_solve_time"]:
    print(f"{key}: {meta_data[key]}")

nmpc_data = data

# %% - VISUALIZE DATA
figures = visualize_nmpc_casadi_data(nmpc_data,nmpc_casadi)

# %% - POSTPROCESSING
config = {"tolerance":TOL,"N_control": N_control,"offset": offset,"noise": noise,"N_sim": N_sim,"N_trajectories": N_trajectories,"filter_mode": filter_mode,"seed": seed}
config.update(meta_data)
config.update(nmpc_casadi.opts)

if save_data:
    save_pth = Path(__file__).parent.resolve().joinpath(data_folder)
    # Add folder to save path based on configuration
    if filter_mode == "successful_trajectories":
        if N_sim==1:
            save_folder = f"open_loop_N_{meta_data['N_optimal']}"
        else:
            save_folder = f"closed_loop_N_{meta_data['N_optimal']}_Nsim_{N_sim}"
    else:
        raise ValueError("Only successful_trajectories mode is supported for now.")
    
    j = 0
    while j < 100:
        if not save_pth.joinpath(save_folder+f"_{j}").exists():
            save_pth = save_pth.joinpath(save_folder+f"_{j}")
            save_pth.mkdir(parents=True)
            break
        j += 1

    # Save Figures
    for idx, fig in enumerate(figures):
        fig.savefig(save_pth.joinpath(f"figure_{idx}.png"))

    # Save Data using np.savez
    np.savez(save_pth.joinpath("nmpc_data.npz"), **nmpc_data)

    # Save Config using json
    with open(save_pth.joinpath("config.json"), "w") as f:
        json.dump(config, f,indent=4)