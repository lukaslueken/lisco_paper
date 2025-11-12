# Imports
import sys
# setting path
sys.path.append('../../')

import time
import numpy as np
import torch
import json
import copy
import gc
from pathlib import Path
from parametric_OP import NLP, NLPCasadi
from models import FeedforwardNN, Predictor, Solver, export_jit_gpu, export_jit_cpu

import matplotlib.pyplot as plt
import mpl_config

# Meta Configuration
SEED = 42
DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
FILE_PTH = Path(__file__).parent.resolve()
torch.set_default_dtype(DTYPE)
torch.set_default_device(DEVICE)
torch.autograd.set_detect_anomaly(False)
torch.manual_seed(SEED)


# Functions
NUM_ITERATIONS = 100
SOLVER_EVAL_ITERATIONS = [0,1,10,20,100,200,500]
TOL = 1e-6
UPDATE_MODE = "line_search" # ignore training settings and apply this update mode for evaluation
# UPDATE_MODE = "full"

def convert_arrays_to_lists(d):
    if isinstance(d, dict):
        return {k: convert_arrays_to_lists(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_arrays_to_lists(i) for i in d]
    elif isinstance(d,np.ndarray):
        return d.tolist()
    else:
        return d

def get_fraction_iteration(results_dict,key="KKT_lim_frac",total=1.0):
    fracs = [val[key] for val in results_dict["trajectory_evaluation"]]
    for idx, frac in enumerate(fracs):
        if frac>=total:
            return idx
    return None
    
def load_config(config_pth):
    with open(config_pth,"r") as fp:
        cfg = json.load(fp)
    return cfg

def get_all_configs(pth):
    # configs
    config_list = []
    # op_names = [f for f in os.listdir(pth) if os.path.isdir(pth.joinpath(f))]
    for i in range(300):
        local_pth = Path(pth,f"exp_{i}")
        if local_pth.exists():
            config_pth = local_pth.joinpath("config.json")
            if config_pth.exists():
                config = load_config(config_pth)
                config_list.append(config)
            else:
                continue
    return config_list

def save_results(results_dict,cfg):
    results_dict = convert_arrays_to_lists(results_dict)  
    with open(Path(cfg["exp_pth"]).joinpath("eval.json"), 'w') as f:
        json.dump(results_dict,f,indent=4)
    config = cfg.copy()
    config["evaluated"] = True
    with open(Path(config["exp_pth"]).joinpath("config.json"), 'w') as f:
        json.dump(config,f,indent=4)

def setup_nlp(nlp_cfg):
    op_pth = Path(nlp_cfg["op_pth"])
    nlp = NLP.from_json(op_pth,"op_cfg")
    nlp.set_device(DEVICE)
    nlp.set_dtype(DTYPE)
    return nlp

def load_predictor(pth):
    if pth is None:
        return None, None
    else:
        pth = Path(pth)
        with open(pth.joinpath("config.json"),"r") as fp:
            prd_cfg = json.load(fp)
        model_cfg = prd_cfg["model_cfg"]
        model = FeedforwardNN(**model_cfg)
        # model.to(device=DEVICE,dtype=DTYPE)
        predictor = Predictor(model, convexification=False, torch_compiled=False)
        predictor.load_weights(pth,file_name="predictor_weights.pt")
        return predictor, prd_cfg

def load_solver(pth):
    slv_cfg = load_config(pth.joinpath("config.json"))
    nlp_cfg = slv_cfg["nlp_cfg"]
    model_cfg = slv_cfg["model_cfg"]
    train_cfg = slv_cfg["train_cfg"]
    solver_cfg = slv_cfg["solver_cfg"]
    gamma = solver_cfg.get("gamma", 0.01)
    nlp = setup_nlp(nlp_cfg)
    model = FeedforwardNN(**model_cfg)
    # model.to(device=DEVICE,dtype=DTYPE)
    if train_cfg["predictor_pth"] is not None:
        predictor_pth = Path(train_cfg["predictor_pth"])
        predictor, prd_cfg = load_predictor(predictor_pth)  
    else:
        predictor = None
    solver = Solver(model,nlp,predictor=predictor,gamma=gamma,convexification=False,torch_compiled=False)
    # load weights
    solver.load_weights(pth,file_name="solver_weights.pt")
    # solver.model = solver.model.to(device=DEVICE,dtype=DTYPE)
    return solver, slv_cfg

def load_data(pth):
    data = np.load(pth)
    data = {key: data[key] for key in data.files}
    # to torch
    for key in data.keys():
        if key == "status":
            data[key] = data[key].astype(str)
        else:
            data[key] = torch.tensor(data[key],dtype=DTYPE,device=DEVICE)
    return data

def check_data_integrity(data,nlp,KKT_inf_tol=1e-6):
    # filter data
    nlp = copy.deepcopy(nlp)
    z_opt = data["z_opt"]
    p = data["p"]
    KKT_batch = nlp.KKT_batch_func(z_opt,p)
    KKT_inf_norm = torch.norm(KKT_batch,p=float("inf"),dim=1)
    idx = KKT_inf_norm < KKT_inf_tol
    # check if all data is valid
    data_validity = torch.all(idx).item()
    return data_validity

def get_metrics(torch_tensor,var_name=None):
    # assert torch_tensor.dim() == 1
    # mean, min, 1%, 5%, 10%, 50%, 90%, 95%, 99%, max, std
    metric_dict = {
    "mean":torch.mean(torch_tensor).item(),
    "min":torch.min(torch_tensor).item(),
    "01":torch.quantile(torch_tensor,0.01).item(),
    "05":torch.quantile(torch_tensor,0.05).item(),
    "10":torch.quantile(torch_tensor,0.10).item(),
    "med":torch.median(torch_tensor).item(),
    "90":torch.quantile(torch_tensor,0.90).item(),
    "95":torch.quantile(torch_tensor,0.95).item(),
    "99":torch.quantile(torch_tensor,0.99).item(),
    "max":torch.max(torch_tensor).item(),
    "std":torch.std(torch_tensor).item(),
    }
    if var_name is not None:
        # rename all keys with var_name in front
        metric_dict = {f"{var_name}_{k}":v for k,v in metric_dict.items()}
    return metric_dict

def evaluate_predictions(z_pred_batch,test_data,nlp):
    # 1. load data and data metrics
    p_batch = test_data["p"].to(DTYPE)
    z_opt_batch = test_data["z_opt"].to(DTYPE)
    w_opt_batch, nu_opt_batch, lam_opt_batch = nlp.extract_primal_dual_batch(z_opt_batch)
 
    w_pred_batch, nu_pred_batch, lam_pred_batch = nlp.extract_primal_dual_batch(z_pred_batch)
       
    # 3. evaluate
    # 3.1 optimality conditions
    # 3.1.1 KKT residual
    KKT_batch = nlp.KKT_batch_func(z_pred_batch,p_batch)
    KKT_inf_norm = torch.linalg.vector_norm(KKT_batch,ord=torch.inf,dim=1)
    KKT_inf_results = get_metrics(KKT_inf_norm,var_name="KKT_inf")
    KKT_2_norm = torch.linalg.vector_norm(KKT_batch,ord=2,dim=1)
    KKT_2_results = get_metrics(KKT_2_norm,var_name="KKT_2")

    # 3.1.2 Loss function values (Tk)
    Tk_batch = nlp.Tk_batch_func(z_pred_batch,p_batch)
    Tk_results = get_metrics(Tk_batch,var_name="Tk")

    # 3.1.3 stack results
    optimality_dict = {**KKT_inf_results,**KKT_2_results,**Tk_results}

    # 3.2 distance to data
    z_diff_batch = z_pred_batch - z_opt_batch
    w_diff_batch = w_pred_batch - w_opt_batch
    nu_diff_batch = nu_pred_batch - nu_opt_batch
    lam_diff_batch = lam_pred_batch - lam_opt_batch    
    z_diff_norm_batch = torch.norm(z_diff_batch,dim=1)
    w_diff_norm_batch = torch.norm(w_diff_batch,dim=1)
    nu_diff_norm_batch = torch.norm(nu_diff_batch,dim=1)
    lam_diff_norm_batch = torch.norm(lam_diff_batch,dim=1)
    z_diff_norm_results = get_metrics(z_diff_norm_batch,var_name="z_diff_norm")
    w_diff_norm_results = get_metrics(w_diff_norm_batch,var_name="w_diff_norm")
    nu_diff_norm_results = get_metrics(nu_diff_norm_batch,var_name="nu_diff_norm")
    lam_diff_norm_results = get_metrics(lam_diff_norm_batch,var_name="lam_diff_norm")

    distance_dict = {**z_diff_norm_results,**w_diff_norm_results,**nu_diff_norm_results,**lam_diff_norm_results}

    # 3.3 constraint violations
    h_val_batch = nlp.h_batch_func(w_pred_batch,p_batch)
    g_val_batch = nlp.g_batch_func(w_pred_batch,p_batch)        
    viol_h_batch = torch.abs(h_val_batch)
    viol_g_batch = torch.relu(g_val_batch)
    viol_lam_batch = torch.relu(-lam_pred_batch)
    viol_h_results = get_metrics(viol_h_batch,var_name="viol_h")
    viol_g_results = get_metrics(viol_g_batch,var_name="viol_g")
    viol_lam_results = get_metrics(viol_lam_batch,var_name="viol_lam")

    constraints_dict = {**viol_h_results,**viol_g_results,**viol_lam_results}

    # 3.4 optimality gap
    f_pred_batch = nlp.f_batch_func(w_pred_batch,p_batch)
    f_opt_batch = nlp.f_batch_func(w_opt_batch,p_batch)
    opt_gap_abs_batch = torch.abs(f_pred_batch - f_opt_batch)
    opt_gap_rel_batch = opt_gap_abs_batch/(torch.abs(f_opt_batch)+1e-16)
    
    f_pred_results = get_metrics(f_pred_batch,var_name="f_pred")
    opt_gap_abs_results = get_metrics(opt_gap_abs_batch,var_name="opt_gap_abs")
    opt_gap_rel_results = get_metrics(opt_gap_rel_batch,var_name="opt_gap_rel")

    opt_gap_dict = {**f_pred_results,**opt_gap_abs_results,**opt_gap_rel_results}

    results_dict = {**optimality_dict, **distance_dict, **constraints_dict, **opt_gap_dict}

    results_dict.update({"w_diff_norm": w_diff_norm_batch.cpu().numpy().squeeze()})
    results_dict.update({"nu_diff_norm": nu_diff_norm_batch.cpu().numpy().squeeze()})
    results_dict.update({"lam_diff_norm": lam_diff_norm_batch.cpu().numpy().squeeze()})

    return results_dict

def evaluate_predictor(predictor,nlp,test_data,speed_eval=False):
    # 1. load data and data metrics
    p_batch = test_data["p"].to(DTYPE)

    # 2. predict
    z_pred_batch = predictor.predict(p_batch).to(DTYPE)

    # 3. evaluate
    results_dict = evaluate_predictions(z_pred_batch,test_data,nlp)

    # 4. inference speed
    if speed_eval:
        model_jit_gpu = export_jit_gpu(predictor.model)
        model_jit_cpu = export_jit_cpu(predictor.model)
        # 4.1 batch prediction GPU
        inference_times_batch_gpu = measure_inference_time_predictor(model_jit_gpu, p_batch, use_cuda=True)
        # 4.2 single datapoint jit prediction GPU
        p_i_gpu = p_batch[0,:]
        inference_times_single_gpu = measure_inference_time_predictor(model_jit_gpu, p_i_gpu, use_cuda=True)
        # 4.3 single datapoint jit prediction CPU
        p_i_cpu = p_batch[0,:]
        p_i_cpu = p_i_cpu.cpu()
        inference_times_single_cpu = measure_inference_time_predictor(model_jit_cpu, p_i_cpu, use_cuda=False)    

        # 4.4 stack results
        # add info to keys corresponding to 4.1 to 4.3
        inference_times_batch_gpu = {f"gpu_batch_{k}": v for k,v in inference_times_batch_gpu.items()}
        inference_times_single_gpu = {f"gpu_single_{k}": v for k,v in inference_times_single_gpu.items()}
        inference_times_single_cpu = {f"cpu_single_{k}": v for k,v in inference_times_single_cpu.items()}
    else:
        inference_times_batch_gpu = {}
        inference_times_single_gpu = {}
        inference_times_single_cpu = {}

    # 6. stack results
    results_dict = results_dict.copy()
    results_dict.update(inference_times_batch_gpu)
    results_dict.update(inference_times_single_gpu)
    results_dict.update(inference_times_single_cpu)

    return results_dict

def evaluate_solver(solver,test_data,solver_cfg,speed_eval=False):
    # 1. load data and data metrics
    p_batch = test_data["p"].to(DTYPE)
    z_opt_batch = test_data["z_opt"].to(DTYPE)

    # 2. solve batch
    max_iter = SOLVER_EVAL_ITERATIONS[-1]

    # Get alpha from solver_cfg if available
    alpha = solver_cfg["train_cfg"].get("alpha", 1.0)

    # warmup GPU if speed eval
    if speed_eval:
        z_warmup, _, _ = solver.solve_batch(
            p_batch,
            max_iter=10,
            alpha=alpha,
            predictor=solver.predictor,
            update_mode=UPDATE_MODE,
            return_trajectory=True,
            compile_tk_func=True,
            compile_step_func=True
        )
        del z_warmup
        gc.collect()
        torch.cuda.empty_cache()

    # Solve batch with the actual parameters supported by solve_batch
    z_hat_batch, zk_traj, step_time_list = solver.solve_batch(
        p_batch,
        max_iter=max_iter,
        alpha=alpha,
        predictor=solver.predictor,
        update_mode=UPDATE_MODE,
        return_trajectory=True,
        compile_tk_func=True,
        compile_step_func=True
    )

    # 3. evaluate
    all_results_dict = {}

    # 3.1 evaluate final solution
    results_dict = evaluate_predictions(z_hat_batch,test_data,solver.nlp)
    all_results_dict.update(results_dict)

    # 3.2 trajectory results (convergence and success rate)
    trajectory_evaluation_list = evaluate_trajectory(zk_traj,p_batch,nlp,torch_compiled=True)
    all_results_dict["trajectory_evaluation"] = trajectory_evaluation_list

    # 3.3 solver results after different number of iterations
    for i in SOLVER_EVAL_ITERATIONS:
        z_pred_batch = zk_traj[i]
        results_dict = evaluate_predictions(z_pred_batch,test_data,solver.nlp)
        all_results_dict[f"solver_{i}"] = results_dict

    # 4. inference speed
    if speed_eval:
        # casadi nlp for fast cpu evaluation
        nlp_casadi = NLPCasadi(**solver.nlp.export_hparams())
        nlp_casadi.update_op_dict(solver.nlp.op_dict)
         
        # solve_step_cpu = solver.get_casadi_solver_step_funcs(nlp_casadi,jit=True)
        solve_step_cpu,Tk_func_cpu = solver.get_casadi_solver_step_funcs(nlp_casadi,jit=True)

        # single steps on cpu
        z_i = z_opt_batch[0,:]
        p_i = p_batch[0,:]
        cpu_step_times = measure_step_time_solver_cpu(solve_step_cpu, Tk_func_cpu, z_i, p_i) # TODO: fix

        # GPU inference time
        gpu_step_times = get_metrics(torch.tensor(step_time_list),var_name="gpu_step_time")

        # full solve time compared to IPOPT
        cpu_solver_times_results = measure_full_time_solver_cpu(test_data,solve_step_cpu,Tk_func_cpu,solver,UPDATE_MODE,alpha) # TODO: fix
    else:
        cpu_step_times = {}
        gpu_step_times = {}
        cpu_solver_times_results = {}

    all_results_dict.update(cpu_step_times)
    all_results_dict.update(gpu_step_times)
    all_results_dict.update(cpu_solver_times_results)

    return all_results_dict

def evaluate_trajectory(zk_traj,p_batch,nlp,torch_compiled=False):
    if torch_compiled:
        Tk_batch_func = torch.compile(copy.deepcopy(nlp.Tk_batch_func),mode="max-autotune",fullgraph=True)
        KKT_batch_func = torch.compile(copy.deepcopy(nlp.KKT_batch_func),mode="max-autotune",fullgraph=True)
    else:
        Tk_batch_func = nlp.Tk_batch_func
        KKT_batch_func = nlp.KKT_batch_func
    
    n_data = p_batch.shape[0]
    traj_eval_list = []
    Tk_list = []
    for idx, zk_i in enumerate(zk_traj):
        Tk_i = Tk_batch_func(zk_i,p_batch)
        Tk_i = Tk_i.detach().clone()
        Tk_list.append(Tk_i)
        # mean, min, 1%, 5%, 10%, 50%, 90%, 95%, 99%, max, std
        Tk_metrics_dict = get_metrics(Tk_i,var_name="Tk")

        KKT_i = KKT_batch_func(zk_i,p_batch)
        KKT_i = KKT_i.detach().clone()

        KKT_inf_norm_i = torch.linalg.vector_norm(KKT_i,ord=torch.inf,dim=1)
        KKT_inf_metrics_dict = get_metrics(KKT_inf_norm_i,var_name="KKT_inf")

        KKT_2_norm_i = torch.linalg.vector_norm(KKT_i,ord=2,dim=1)
        KKT_2_metrics_dict = get_metrics(KKT_2_norm_i,var_name="KKT_2")

        # IMPORTANT: Here, the fraction of data points satisfying the KKT conditions within the tolerance TOL is computed (INFINITY NORM!)
        KKT_lim_frac = torch.sum(KKT_inf_norm_i <= TOL).item()/n_data
        # KKT_lim_frac = torch.sum(KKT_2_norm_i <= TOL).item()/n_data

        traj_eval = {}
        traj_eval["KKT_lim_frac"] = KKT_lim_frac
        traj_eval.update(KKT_2_metrics_dict)
        traj_eval.update(KKT_inf_metrics_dict)
        # Tk_lim_frac = torch.sum(Tk_i <= TOL).item()/n_data
        # traj_eval["Tk_lim_frac"] = Tk_lim_frac
        traj_eval.update(Tk_metrics_dict)
        traj_eval_list.append(traj_eval)
    
    return traj_eval_list

# inference time
def measure_inference_time_predictor(model, input_data, use_cuda=True):
    # Set the model to evaluation mode
    model.eval()
    
    # Move model and input data to GPU if necessary
    if use_cuda and torch.cuda.is_available():
        model = model.cuda()
        input_data = input_data.cuda()
    else:
        model = model.cpu()
        input_data = input_data.cpu()
    
    # Tensor to store inference times
    inference_times = torch.zeros(NUM_ITERATIONS)

    # Measure inference time
    with torch.no_grad():
        # Perform warm-up iterations (not measured)
        for _ in range(NUM_ITERATIONS):
            _ = model(input_data)
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()  # Ensure all CUDA operations are complete

        for i in range(NUM_ITERATIONS):
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()  # Ensure all CUDA operations are complete
            
            start_time = time.perf_counter()
            _ = model(input_data)
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()  # Ensure all CUDA operations are complete
            end_time = time.perf_counter()            
            inference_times[i] = end_time - start_time
    
    # Calculate statistics
    results = get_metrics(inference_times,var_name="inference_time")

    return results

def measure_step_time_solver_cpu(solver_step_func,Tk_func_cpu, zk, p):    
    zk = zk.cpu().numpy()
    p = p.cpu().numpy()
    
    # Tensor to store inference times
    inference_times = torch.zeros(NUM_ITERATIONS)

    # Measure inference time
    with torch.no_grad():
        # Perform warm-up iterations (not measured)
        for _ in range(100):
            _ = solver_step_func(zk,p)
            _ = Tk_func_cpu(zk,p)

        for i in range(NUM_ITERATIONS):
            start_time = time.perf_counter()
            _ = solver_step_func(zk,p)
            _ = Tk_func_cpu(zk,p)
            end_time = time.perf_counter()            
            inference_times[i] = end_time - start_time
    # Calculate statistics
    results = get_metrics(inference_times,var_name="cpu_step_time") 
    return results

def measure_full_time_solver_cpu(test_data,solve_step_cpu,Tk_func_cpu,solver,update_mode="line_search",alpha=0.01):
    # 1. load data
    p_batch = test_data["p"].to(DTYPE)
    p_batch = p_batch.cpu().numpy()
    n_data = p_batch.shape[0]

    predictor_cpu = solver.predictor
    if predictor_cpu is not None:
        predictor_cpu = export_jit_cpu(predictor_cpu.model)

    full_solve_times = []
    step_times = []
    success_list = []
    n_iter_list = []
    speedup_factor_list = []
    for i in range(n_data):
        p_i = p_batch[i,:]

        start_time = time.perf_counter()
        z_pred_i,n_iter_i,success_i = solver.solve_fast_single_cpu(solve_step_cpu,Tk_func_cpu,p_i,predictor=predictor_cpu,alpha=alpha,max_iter=SOLVER_EVAL_ITERATIONS[-1],tol=TOL,update_mode=update_mode)
        end_time = time.perf_counter()

        full_solve_time = end_time - start_time
        step_time = full_solve_time/n_iter_i
        success_list.append(success_i)
        ipopt_solve_time = test_data["solve_time"][i].item()
        speedup_factor = ipopt_solve_time/full_solve_time
        if success_i:
            n_iter_list.append(float(n_iter_i))
            full_solve_times.append(full_solve_time)
            step_times.append(step_time)
            speedup_factor_list.append(speedup_factor)

    # summarize results
    # success rate
    success_rate = np.mean(success_list)
    
    # if successrate is zero, return zeros and print warning
    if success_rate == 0.0:
        n_iter_list = [0.0]
        full_solve_times = [0.0]
        speedup_factor_list = [0.0]
        step_times = [0.0]
        print("Warning: No successful solves in CPU evaluation.")
    
    # iterations necessary
    n_iter_results = get_metrics(torch.tensor(n_iter_list),var_name="cpu_n_iter")

    # full solve time
    full_solve_time_results = get_metrics(torch.tensor(full_solve_times),var_name="cpu_full_solve_time")

    # # step time
    # step_time_results = get_metrics(torch.tensor(step_times),var_name="step_time")

    # speedup factor
    speedup_factor_results = get_metrics(torch.tensor(speedup_factor_list),var_name="cpu_speedup_factor")

    cpu_solver_times_results = {}
    cpu_solver_times_results["success_rate"] = success_rate
    cpu_solver_times_results.update(n_iter_results)
    cpu_solver_times_results.update({"cpu_n_iter_list": n_iter_list})
    cpu_solver_times_results.update(full_solve_time_results)
    cpu_solver_times_results.update({"cpu_full_solve_time_list": full_solve_times})
    cpu_solver_times_results.update(speedup_factor_results)
    cpu_solver_times_results.update({"cpu_speedup_factor_list": speedup_factor_list})
    return cpu_solver_times_results

def plot_histogram_diff(diff_vals,var_name=None,ax=None,label=None,log_scale=False):
    # settings
    n_bins = 100
    alpha_trans = 0.3

    diff_vals = diff_vals.copy()
    if log_scale:
        diff_vals = np.log10(diff_vals)
    
    diff_hist = np.histogram(diff_vals,bins=n_bins)

    # figure
    if ax is None:
        fig, ax = plt.subplots(1,1)
    else:
        fig = ax.figure
    # plot histogram
    ax.plot(diff_hist[1][:-1],diff_hist[0])

    if label is not None:
        # ax.hist(log_diff,bins=n_bins,alpha=alpha_trans,label=label)
        ax.hist(diff_vals,bins=n_bins,alpha=alpha_trans,label=label)
    else:
        # ax.hist(log_diff,bins=n_bins,alpha=alpha_trans)
        ax.hist(diff_vals,bins=n_bins,alpha=alpha_trans)

    # X-AXIS
    # if log_scale:
    #     # change xticks to 10^x
    #     xticks = ax.get_xticks()
    #     xticks_labels = [f"$10^{{{int(x)}}}$" for x in xticks]
    #     ax.set_xticks(xticks)
    #     ax.set_xticklabels(xticks_labels)

    if var_name is not None:
        ax.set_xlabel(f"Error measured as {var_name}")

    # Y-AXIS
    ax.set_ylabel("Frequency")

    # LEGEND
    ax.legend()
    # tighten legend box
    leg = ax.get_legend()
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_alpha(0.0)

    # TITLE
    ax.set_title("Distribution of prediction errors")

    return fig, ax

def plot_kkt_inf_convergence(results_dict,ax=None):
    traj_eval_list = results_dict["trajectory_evaluation"]
    n_iterations = len(traj_eval_list)
    iterations = np.arange(n_iterations)+1

    if ax is None:
        fig, ax = plt.subplots(1,1)
    else:
        fig = ax.figure

    ax.plot(iterations,[val["KKT_inf_max"] for val in traj_eval_list],label="max")
    ax.plot(iterations,[val["KKT_inf_99"] for val in traj_eval_list],label="99th perc.")
    ax.plot(iterations,[val["KKT_inf_95"] for val in traj_eval_list],label="95th perc.")
    ax.plot(iterations,[val["KKT_inf_90"] for val in traj_eval_list],label="90th perc.")
    ax.plot(iterations,[val["KKT_inf_med"] for val in traj_eval_list],label="50th perc.")

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$||KKT||_{\infty}$")
    ax.set_title(r"$\infty$-Norm of KKT conditions over iterations")
    ax.legend()
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim([1e-16,1e2])
    ax.set_xlim([1,n_iterations])

    ax.grid(True,which="both",linestyle="--",linewidth=0.3)

    return fig, ax

def plot_kkt_2_convergence(results_dict,ax=None):
    traj_eval_list = results_dict["trajectory_evaluation"]
    n_iterations = len(traj_eval_list)
    iterations = np.arange(n_iterations)+1

    if ax is None:
        fig, ax = plt.subplots(1,1)
    else:
        fig = ax.figure

    ax.plot(iterations,[val["KKT_2_max"] for val in traj_eval_list],label="max")
    ax.plot(iterations,[val["KKT_2_99"] for val in traj_eval_list],label="99th perc.")
    ax.plot(iterations,[val["KKT_2_95"] for val in traj_eval_list],label="95th perc.")
    ax.plot(iterations,[val["KKT_2_90"] for val in traj_eval_list],label="90th perc.")
    ax.plot(iterations,[val["KKT_2_med"] for val in traj_eval_list],label="50th perc.")

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$||KKT||_{2}$")
    ax.set_title(r"2-Norm of KKT conditions over iterations")
    ax.legend()
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim([1e-16,1e2])
    ax.set_xlim([1,n_iterations])

    ax.grid(True,which="both",linestyle="--",linewidth=0.3)

    return fig, ax

def plot_tk_convergence(results_dict,ax=None):
    traj_eval_list = results_dict["trajectory_evaluation"]
    n_iterations = len(traj_eval_list)
    iterations = np.arange(n_iterations)+1

    if ax is None:
        fig, ax = plt.subplots(1,1)
    else:
        fig = ax.figure

    ax.plot(iterations,[val["Tk_max"] for val in traj_eval_list],label="max")
    ax.plot(iterations,[val["Tk_99"] for val in traj_eval_list],label="99th perc.")
    ax.plot(iterations,[val["Tk_95"] for val in traj_eval_list],label="95th perc.")
    ax.plot(iterations,[val["Tk_90"] for val in traj_eval_list],label="90th perc.")
    ax.plot(iterations,[val["Tk_med"] for val in traj_eval_list],label="50th perc.")

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$||F||_{2}^{2}$")
    ax.set_title(r"Loss function values over iterations")
    ax.legend()
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim([1e-32,1e2])
    ax.set_xlim([1,n_iterations])

    ax.grid(True,which="both",linestyle="--",linewidth=0.3)

    return fig, ax

def plot_success_rate(results_dict,ax=None):
    traj_eval_list = results_dict["trajectory_evaluation"]
    n_iterations = len(traj_eval_list)
    iterations = np.arange(n_iterations)

    # get idx of iteration where all KKT_inf_norm <= TOL
    idx_90 = get_fraction_iteration(results_dict,key="KKT_lim_frac",total=0.9)
    idx_99 = get_fraction_iteration(results_dict,key="KKT_lim_frac",total=0.99)
    idx_100 = get_fraction_iteration(results_dict,key="KKT_lim_frac",total=1.0)

    if ax is None:
        fig, ax = plt.subplots(1,1)
    else:
        fig = ax.figure

    ax.plot(iterations,[val["KKT_lim_frac"] for val in traj_eval_list],label="KKT inf. norm <= TOL")

    if idx_90 is not None:
        ax.plot(idx_90,0.9,"rx",label=r"90% success rate")
    if idx_99 is not None:
        ax.plot(idx_99,0.99,"rx",label=r"99% success rate")
    if idx_100 is not None:
        ax.plot(idx_100,1.0,"rx",label=r"100% success rate")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fraction of data points")
    ax.set_title("Success rate over iterations")
    ax.legend()

    ax.grid(True,which="both",linestyle="--",linewidth=0.3)

    return fig, ax

def visualize_predictor(results_dict,pth=None):
    # Extract error norms for visualization
    w_diff_norm = results_dict["w_diff_norm"]
    nu_diff_norm = results_dict["nu_diff_norm"]
    lam_diff_norm = results_dict["lam_diff_norm"]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1,3,figsize=(15,5))
    
    # Plot histogram for each variable type
    plot_histogram_diff(w_diff_norm,var_name=r"$||w - w^*||_2$",ax=axes[0],label="Primal variables",log_scale=True)
    plot_histogram_diff(nu_diff_norm,var_name=r"$||\nu - \nu^*||_2$",ax=axes[1],label="Equality multipliers",log_scale=True)
    plot_histogram_diff(lam_diff_norm,var_name=r"$||\lambda - \lambda^*||_2$",ax=axes[2],label="Inequality multipliers",log_scale=True)
    
    fig.tight_layout()

    # save figure
    if pth is not None:
        fig.savefig(pth.joinpath("predictor_histogram.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return [fig]

def visualize_solver(results_dict,pth=None):
    # 1. Histogram of solver over all iterations which are considered explicitly
    fig1, axes1 = plt.subplots(1,3,figsize=(15,5))
    
    for i in SOLVER_EVAL_ITERATIONS:
        w_diff_norm = results_dict[f"solver_{i}"]["w_diff_norm"]
        nu_diff_norm = results_dict[f"solver_{i}"]["nu_diff_norm"]
        lam_diff_norm = results_dict[f"solver_{i}"]["lam_diff_norm"]
        
        # Filter very small values
        w_diff_norm = w_diff_norm[w_diff_norm > 1e-12]
        nu_diff_norm = nu_diff_norm[nu_diff_norm > 1e-12]
        lam_diff_norm = lam_diff_norm[lam_diff_norm > 1e-12]
        
        plot_histogram_diff(w_diff_norm,var_name=r"$||w - w^*||_2$",ax=axes1[0],label=f"iter {i}",log_scale=True)
        plot_histogram_diff(nu_diff_norm,var_name=r"$||\nu - \nu^*||_2$",ax=axes1[1],label=f"iter {i}",log_scale=True)
        plot_histogram_diff(lam_diff_norm,var_name=r"$||\lambda - \lambda^*||_2$",ax=axes1[2],label=f"iter {i}",log_scale=True)
    
    fig1.tight_layout()

    # 2. convergence behavior of solver (KKT infinity norm metrics over iterations)
    fig2, ax2 = plot_kkt_inf_convergence(results_dict,ax=None)

    # 3. success rate of solver (fraction of KKT infinity norm <= TOL over iterations)
    fig3, ax3 = plot_success_rate(results_dict,ax=None)

    # 4. loss function convergence of solver (Tk metrics over iterations)
    fig4, ax4 = plot_tk_convergence(results_dict,ax=None)

    # 5. 2-norm on KKT conditions
    fig5, ax5 = plot_kkt_2_convergence(results_dict,ax=None)

    # save figures
    if pth is not None:
        fig1.savefig(pth.joinpath("solver_histogram.png"), dpi=300, bbox_inches='tight')
        fig2.savefig(pth.joinpath("solver_kkt_inf_convergence.png"), dpi=300, bbox_inches='tight')
        fig3.savefig(pth.joinpath("solver_success_rate.png"), dpi=300, bbox_inches='tight')
        fig4.savefig(pth.joinpath("solver_tk_convergence.png"), dpi=300, bbox_inches='tight')
        fig5.savefig(pth.joinpath("solver_kkt_2_convergence.png"), dpi=300, bbox_inches='tight')
        plt.close("all")

    return [fig1,fig2,fig3,fig4]


# %% 
# RUN EVALUATION
if __name__ == "__main__":
    # Config
    overwrite = True # overwrite existing results
    speed_eval = True # expensive to run for solver
    visualize_results = True

    results_folder = "results"
    problem_folder = "data"
    test_data_name = "N_1000_0"

    modes = ["predictor","solver"]

    op_names = [
        "nonconvex_100x50x50_0", 
        "nonconvex_100x50x50_1", 
        "nonconvex_100x50x50_2", 
        "nonconvex_100x50x50_3", 
        "nonconvex_100x50x50_4", 
    ]

    # Build config list
    cfg_list = []
    
    for mode in modes:
        # Build config list from all op_names and mode
        for op_name in op_names:
            pth = FILE_PTH.joinpath(results_folder,op_name,mode)
            if pth.exists():
                configs = get_all_configs(pth)
                cfg_list.extend(configs)
        
        if len(cfg_list) == 0:
            raise ValueError(f"No configs found.")
    
    for cfg in cfg_list:        
        if not cfg["evaluated"] or overwrite:
            
            # setup NLP
            nlp = setup_nlp(cfg["nlp_cfg"])
            test_data_pth = Path(cfg["nlp_cfg"]["op_pth"]).joinpath(test_data_name,"op_data.npz")

            # load test data
            test_data = load_data(test_data_pth)
            assert check_data_integrity(test_data,nlp,1e-5), "Data integrity check failed!"

            mode = cfg["mode"]

            if mode == "predictor":
                predictor_pth = Path(cfg["exp_pth"])
                predictor, predictor_cfg = load_predictor(predictor_pth)
                # evaluate
                results_dict = evaluate_predictor(predictor,nlp,test_data,speed_eval=speed_eval)
                # visualize results
                if visualize_results:
                    figs = visualize_predictor(results_dict,pth=predictor_pth)

            elif mode == "solver":
                solver_pth = Path(cfg["exp_pth"])
                solver, solver_cfg = load_solver(solver_pth)
                # evaluate
                results_dict = evaluate_solver(solver,test_data,solver_cfg,speed_eval=speed_eval)
                # visualize results
                if visualize_results:
                    figs = visualize_solver(results_dict,pth=solver_pth)

            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            # add name of test data to results dict
            results_dict["test_data"] = test_data_pth.parent.name

            # save results
            save_results(results_dict,cfg)
        else:
            print("Results already evaluated. Set overwrite=True to re-evaluate.")

        # clear GPU memory after each evaluation        
        torch.cuda.empty_cache()
        gc.collect()
