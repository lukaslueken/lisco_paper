"""
Visualization Script for Nonlinear Double Integrator Case Study

This script loads and compares results from three different MPC approaches:
1. Approximate MPC (baseline neural network approach)
2. Predictor (warm-start initialization)
3. Solver with Predictor (learned iterative solver)

The script generates:
- KKT convergence plots
- Performance comparison table with control error metrics
- CPU solver performance metrics (speedup factors, iterations, solve times)
- Inference time comparisons (CPU and GPU)
- Training time comparison table
"""

# %% Imports
import sys
# setting path
sys.path.append('../../')

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from pathlib import Path

plt.style.use(['science','ieee','no-latex'])

# %% Configuration
RUN_FOLDER = "results"
SAVE_FOLDER = "visualization"
TOL = 1e-6

# Specification of results to load and visualize
results_to_visualize = {
    "approxMPC": "exp_0",
    "predictor": "exp_0", 
    "solver": "exp_0",
    "solver_no_pred": "exp_1"  # Solver without predictor for comparison
}

FILE_PTH = Path(__file__).parent.resolve()
RESULTS_PATH = FILE_PTH.joinpath(RUN_FOLDER)

# %% Utility Functions

def load_results(mode, exp_id):
    """Load evaluation and configuration results for a specific experiment."""
    results_folder = RESULTS_PATH.joinpath(mode, exp_id)
    eval_file = results_folder.joinpath("eval.json")
    config_file = results_folder.joinpath("config.json")
    
    try:
        with open(eval_file, "r") as f:
            eval_dict = json.load(f)
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        return {"eval": eval_dict, "config": config_dict}
    except FileNotFoundError:
        print(f"No results file found for {mode} with exp_id {exp_id}")
        return None

def as_string_scientific(x):
    """Convert float to scientific notation string."""
    if x == 0.0:
        return "0.0"
    else:
        return f"{x:.2e}"

# %% Plotting Functions

def plot_kkt_convergence(results_dict, ax=None, tol=None):
    """Plot KKT convergence over solver iterations."""
    traj_eval_list = results_dict["trajectory_evaluation"]
    n_iterations = len(traj_eval_list)
    iterations = np.arange(n_iterations) + 1

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure

    # Plot different percentiles of KKT norm
    ax.plot(iterations, [val["KKT_inf_max"] for val in traj_eval_list], label="max")
    ax.plot(iterations, [val["KKT_inf_99"] for val in traj_eval_list], label="99th perc.")
    ax.plot(iterations, [val["KKT_inf_95"] for val in traj_eval_list], label="95th perc.")
    ax.plot(iterations, [val["KKT_inf_90"] for val in traj_eval_list], label="90th perc.")
    ax.plot(iterations, [val["KKT_inf_med"] for val in traj_eval_list], label="50th perc.")

    # Add tolerance line if specified
    if tol is not None:
        ax.axhline(tol, color='red', linestyle='--', linewidth=0.3, 
                  label=f'Tolerance: {tol:.1e}')

    # Configure plot
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$||KKT||_{\infty}$")
    ax.set_title(r"$\infty$-Norm of KKT conditions over iterations")
    ax.legend(loc='upper right', fontsize='small')
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim([1e-12, 1e2])
    ax.set_xlim([1, n_iterations])
    ax.grid(True, which="major", linestyle="--", linewidth=0.3, axis='y')
    ax.grid(True, which="both", linestyle="--", linewidth=0.3, axis='x')
    # ax.grid(True, which="both", linestyle="--", linewidth=0.3)
    # ax.grid(True, which="major", linestyle="--", linewidth=0.3)

    return fig, ax

# %% Table Generation Functions

def extract_solver_cpu_metrics(res_solver):
    """Extract CPU solver performance metrics (speedup, iterations, solve times)."""
    if res_solver is None:
        return {}
    
    eval_data = res_solver["eval"]
    suffixes = ["max", "99", "95", "med", "min"]
    metrics_data = {}
    
    # Extract speedup factors
    speedup_row = {}
    for suffix in suffixes:
        key = f"cpu_speedup_factor_{suffix}"
        speedup_row[suffix] = eval_data.get(key, np.nan)
    metrics_data["Speedup over IPOPT"] = speedup_row
    
    # Extract iteration counts
    iter_row = {}
    for suffix in suffixes:
        key = f"cpu_n_iter_{suffix}"
        iter_row[suffix] = eval_data.get(key, np.nan)
    metrics_data["N Iterations"] = iter_row
    
    # Extract solve times
    solve_time_row = {}
    for suffix in suffixes:
        key = f"cpu_full_solve_time_{suffix}"
        solve_time_row[suffix] = eval_data.get(key, np.nan)
    metrics_data["Full Solve Time [s]"] = solve_time_row
    
    # Extract success rate (only one value, so we repeat it for all statistics)
    success_rate = eval_data.get("success_rate", np.nan)
    success_rate_row = {}
    for suffix in suffixes:
        success_rate_row[suffix] = success_rate
    metrics_data["Success Rate"] = success_rate_row
    
    return metrics_data

def extract_combined_inference_metrics(res_approxmpc, res_predictor, res_solver):
    """Extract CPU and GPU inference times for all methods."""
    methods_data = []
    
    if res_approxmpc is not None:
        eval_data = res_approxmpc["eval"]
        metrics = {
            "Method": "Approx. MPC",
            "CPU Inference Time [s]": eval_data.get("cpu_single_inference_time_mean", np.nan),
            "GPU Batch Time [s]": eval_data.get("gpu_batch_inference_time_mean", np.nan)
        }
        methods_data.append(metrics)
    
    if res_predictor is not None:
        eval_data = res_predictor["eval"]
        metrics = {
            "Method": "Predictor",
            "CPU Inference Time [s]": eval_data.get("cpu_single_inference_time_mean", np.nan),
            "GPU Batch Time [s]": eval_data.get("gpu_batch_inference_time_mean", np.nan)
        }
        methods_data.append(metrics)
    
    if res_solver is not None:
        eval_data = res_solver["eval"]
        metrics = {
            "Method": "Solver Single Step",
            "CPU Inference Time [s]": eval_data.get("cpu_step_time_mean", np.nan),
            "GPU Batch Time [s]": eval_data.get("gpu_step_time_mean", np.nan)
        }
        methods_data.append(metrics)
    
    return methods_data

def extract_training_times(res_approxmpc, res_predictor, res_solver):
    """Extract training times for all methods."""
    training_data = []
      # Handle Approx. MPC with additional sampling time
    if res_approxmpc is not None:
        train_time = res_approxmpc["config"].get("train_time", np.nan)
        # Convert seconds to minutes
        train_time_min = train_time / 60 if not pd.isna(train_time) else np.nan
        training_data.append({
            "Method": "Approx. MPC",
            "Training Time [min]": train_time_min
        })
        
        # Load training data config to get sampling time
        try:
            train_data_config_path = res_approxmpc["config"]["train_cfg"].get("train_data_pth")
            if train_data_config_path:
                train_data_config_file = FILE_PTH.joinpath(train_data_config_path).parent / "config.json"
                with open(train_data_config_file, "r") as f:
                    train_data_config = json.load(f)
                sampling_time = train_data_config.get("sampling_time", 0)
                total_time = train_time + sampling_time if not pd.isna(train_time) else np.nan
                total_time_min = total_time / 60 if not pd.isna(total_time) else np.nan
                training_data.append({
                    "Method": "Approx. MPC (incl. sampling)",
                    "Training Time [min]": total_time_min
                })
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            # If we can't load the sampling time, just add NaN
            training_data.append({
                "Method": "Approx. MPC (incl. sampling)",
                "Training Time [min]": np.nan
            })
      # Handle other methods
    if res_predictor is not None:
        predictor_train_time = res_predictor["config"].get("train_time", np.nan)
        predictor_train_time_min = predictor_train_time / 60 if not pd.isna(predictor_train_time) else np.nan
        training_data.append({
            "Method": "Predictor",
            "Training Time [min]": predictor_train_time_min
        })
    else:
        predictor_train_time = np.nan
    
    if res_solver is not None:
        solver_train_time = res_solver["config"].get("train_time", np.nan)
        solver_train_time_min = solver_train_time / 60 if not pd.isna(solver_train_time) else np.nan
        training_data.append({
            "Method": "Solver w. Predictor",
            "Training Time [min]": solver_train_time_min
        })
        
        # Add combined training time (solver + predictor)
        if not pd.isna(solver_train_time) and not pd.isna(predictor_train_time):
            combined_time = solver_train_time + predictor_train_time
            combined_time_min = combined_time / 60
        else:
            combined_time_min = np.nan
        training_data.append({
            "Method": "Solver w. Predictor (incl. predictor)",
            "Training Time [min]": combined_time_min
        })
    
    return training_data

def format_table_for_display(df, table_type):
    """Format tables for display with appropriate number formatting."""
    if df.empty:
        return df
    
    # Create a copy and convert to object dtype to allow mixed types
    formatted_df = df.copy().astype(object)
    
    if table_type == "solver":
        # Format solver performance table
        for idx in formatted_df.index:
            for col in formatted_df.columns:
                value = formatted_df.loc[idx, col]
                if pd.isna(value):
                    formatted_df.loc[idx, col] = "N/A"
                elif "Speedup over IPOPT" in idx:
                    formatted_df.loc[idx, col] = f"{value:.2f}"
                elif "N Iterations" in idx:
                    formatted_df.loc[idx, col] = f"{int(value)}"
                elif "Time" in idx:
                    formatted_df.loc[idx, col] = as_string_scientific(value)
                else:
                    formatted_df.loc[idx, col] = f"{value:.3f}" if isinstance(value, (int, float)) else str(value)
    else:
        # Format other tables
        for col in formatted_df.columns:
            if col == "Method":
                continue
            elif "Time" in col:
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: as_string_scientific(x) if pd.notna(x) else "N/A"
                )
            elif "Speedup over IPOPT" in col or "Success Rate" in col:
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
                )
            elif "N Iterations" in col:
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{int(x)}" if pd.notna(x) else "N/A"
                )
            else:
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{x:.3f}" if pd.notna(x) and isinstance(x, (int, float)) else str(x) if pd.notna(x) else "N/A"
                )
    
    return formatted_df


# %% Load Results

res_solver = load_results("solver", results_to_visualize["solver"])
res_predictor = load_results("predictor", results_to_visualize["predictor"])
res_approxmpc = load_results("approxMPC", results_to_visualize["approxMPC"])
res_solver_no_pred = load_results("solver", results_to_visualize["solver_no_pred"])
assert res_solver_no_pred["config"]["train_cfg"]["predictor_pth"] is None

# %% Validate Results Consistency

# Ensure all methods used the same test data
assert res_solver["eval"]["test_data"] == res_predictor["eval"]["test_data"], \
    "Test data mismatch between solver and predictor"
assert res_solver["eval"]["test_data"] == res_approxmpc["eval"]["test_data"], \
    "Test data mismatch between solver and approxMPC"
assert res_solver["eval"]["test_data"] == res_solver_no_pred["eval"]["test_data"], \
    "Test data mismatch between solver and solver_no_pred"

# Verify predictor matches solver step 0
for k, v in res_solver["eval"]["trajectory_evaluation"][0].items():
    if k in res_predictor["eval"].keys():
        assert np.allclose(res_predictor["eval"][k],v), f"{k} does not match"

# %% Generate KKT Convergence Plot

fig_1, ax_1 = plot_kkt_convergence(res_solver["eval"], tol=TOL)
plt.close(fig_1)

save_path = RESULTS_PATH.joinpath(SAVE_FOLDER)
save_path.mkdir(parents=True, exist_ok=True)

# Save as both PNG and PDF
convergence_plot_path_png = save_path.joinpath("kkt_convergence_solver.png")
convergence_plot_path_pdf = save_path.joinpath("kkt_convergence_solver.pdf")
fig_1.savefig(convergence_plot_path_png, bbox_inches='tight', dpi=300)
fig_1.savefig(convergence_plot_path_pdf, bbox_inches='tight', dpi=300)
print(f"Saved convergence plot to {convergence_plot_path_png}")
print(f"Saved convergence plot to {convergence_plot_path_pdf}")        

# %% Generate Performance Tables

# Table 1: CPU Solver Performance Metrics (speedup, iterations, solve times)
print("\nTable 1: CPU Solver Performance")
print("=" * 80)

solver_cpu_metrics = extract_solver_cpu_metrics(res_solver)
solver_no_pred_cpu_metrics = extract_solver_cpu_metrics(res_solver_no_pred)

if solver_cpu_metrics:
    # Create multi-level index structure
    metric_names = ["Speedup over IPOPT", "N Iterations", "Full Solve Time [s]"]
    statistic_names = ["max", "med", "min"]
    
    # Create multi-index for rows
    row_tuples = []
    for metric in metric_names:
        for stat in statistic_names:
            row_tuples.append((metric, stat))
    
    # Add Success Rate as a single row without statistics
    row_tuples.append(("Success Rate", ""))
    
    multi_index = pd.MultiIndex.from_tuples(row_tuples, names=['Metric', 'Statistic'])
    
    # Create data for both solver types
    solver_types = []
    solver_data = []
    
    if solver_cpu_metrics:
        solver_types.append("w. predictor")
        data_col = []
        for metric in metric_names:
            for stat in statistic_names:
                value = solver_cpu_metrics[metric][stat]
                data_col.append(value)
        # Add success rate as a single value
        data_col.append(solver_cpu_metrics["Success Rate"]["max"])  # Use any stat since they're all the same
        solver_data.append(data_col)
    
    if solver_no_pred_cpu_metrics:
        solver_types.append("w/o. predictor")
        data_col = []
        for metric in metric_names:
            for stat in statistic_names:
                value = solver_no_pred_cpu_metrics[metric][stat]
                data_col.append(value)
        # Add success rate as a single value
        data_col.append(solver_no_pred_cpu_metrics["Success Rate"]["max"])  # Use any stat since they're all the same
        solver_data.append(data_col)
    
    # Create DataFrame with multi-level index and solver types as columns
    table1_df = pd.DataFrame(
        data=np.array(solver_data).T,
        index=multi_index,
        columns=solver_types
    )
    
    # Format the table for display
    table1_formatted = table1_df.copy().astype(object)
    for col in table1_formatted.columns:
        for idx in table1_formatted.index:
            metric, stat = idx
            value = table1_formatted.loc[idx, col]
            if pd.isna(value):
                table1_formatted.loc[idx, col] = "N/A"
            elif "Speedup over IPOPT" in metric:
                table1_formatted.loc[idx, col] = f"{value:.2f}"
            elif "N Iterations" in metric:
                table1_formatted.loc[idx, col] = f"{int(value)}"
            elif "Time" in metric:
                table1_formatted.loc[idx, col] = as_string_scientific(value)
            elif "Success Rate" in metric:
                table1_formatted.loc[idx, col] = f"{value:.4f}"
            else:
                table1_formatted.loc[idx, col] = f"{value:.3f}" if isinstance(value, (int, float)) else str(value)
    
    print(table1_formatted.to_string())
    
    latex_path1 = RESULTS_PATH.joinpath(SAVE_FOLDER, "table1_cpu_solver_performance.tex")
    latex_string1 = table1_formatted.to_latex(
        escape=False,
        column_format='l' + 'c' * len(table1_formatted.columns) + 'c'
    )
    with open(latex_path1, 'w') as f:
        f.write(latex_string1)
    print(f"Saved Table 1 to {latex_path1}")
else:
    print("No solver CPU metrics available")

# Table 2: Inference Times (CPU and GPU)
print("\n\nTable 2: Inference Times (CPU and GPU)")
print("=" * 80)

combined_inference_data = extract_combined_inference_metrics(res_approxmpc, res_predictor, res_solver)
if combined_inference_data:
    combined_table_df = pd.DataFrame(combined_inference_data).set_index("Method")
    combined_table_formatted = combined_table_df.copy()
    
    for col in combined_table_formatted.columns:
        if "Time" in col:
            combined_table_formatted[col] = combined_table_formatted[col].apply(
                lambda x: as_string_scientific(x) if pd.notna(x) else "N/A"
            )
    
    print(combined_table_formatted.to_string())
    
    latex_path_combined = RESULTS_PATH.joinpath(SAVE_FOLDER, "table2_combined_inference_times.tex")
    latex_string_combined = combined_table_formatted.to_latex(
        escape=False,
        column_format='l' + 'c' * len(combined_table_formatted.columns)
    )
    with open(latex_path_combined, 'w') as f:
        f.write(latex_string_combined)
    print(f"Saved Table 2 to {latex_path_combined}")
else:
    print("No inference metrics available")

# Table 3: Training Time Comparison
print("\n\nTable 3: Training Time Comparison")
print("=" * 80)

training_time_data = extract_training_times(res_approxmpc, res_predictor, res_solver)
if training_time_data:
    training_time_df = pd.DataFrame(training_time_data).set_index("Method")
    training_time_formatted = training_time_df.copy()
    training_time_formatted["Training Time [min]"] = training_time_formatted["Training Time [min]"].apply(
        lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
    )
    print(training_time_formatted.to_string())
    
    latex_path3 = RESULTS_PATH.joinpath(SAVE_FOLDER, "table3_training_times.tex")
    latex_string3 = training_time_formatted.to_latex(
        escape=False,
        column_format='l' + 'c' * len(training_time_formatted.columns)
    )
    with open(latex_path3, 'w') as f:
        f.write(latex_string3)
    print(f"Saved Table 3 to {latex_path3}")
else:
    print("No training time data available")

# Table 4: Hyperparameters + train time
print("\n\nTable 4: Hyperparameters + Train Time")
print("=" * 80)

def extract_hyperparameters(res_approxmpc, res_predictor, res_solver):
    """Extract hyperparameters for all methods."""
    hyperparams_data = {}
    
    # Extract Approx. MPC hyperparameters
    if res_approxmpc is not None:
        config = res_approxmpc["config"]
        train_cfg = config.get("train_cfg", {})
        model_cfg = config.get("model_cfg", {})
        
        # Calculate batches per epoch for approx. MPC
        # Load training dataset size from the training data config
        # train_data_size = np.nan
        try:
            train_data_pth = train_cfg.get("train_data_pth")
            if train_data_pth:
                train_data_config_file = FILE_PTH.joinpath(train_data_pth).parent / "config.json"
                with open(train_data_config_file, "r") as f:
                    train_data_config = json.load(f)
                train_data_size = train_data_config.get("N_total", np.nan)
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            # If we can't load the training data size, use NaN
            train_data_size = np.nan
        
        batch_size = train_cfg.get("batch_size", np.nan)
        if not pd.isna(train_data_size) and not pd.isna(batch_size) and batch_size > 0:
            batches_per_epoch = int(np.ceil(train_data_size / batch_size))
        else:
            batches_per_epoch = np.nan
        
        batch_size = train_cfg.get("batch_size", np.nan)
        if not pd.isna(batch_size) and not pd.isna(batches_per_epoch):
            batch_size_str = f"{int(batch_size)} ({int(batches_per_epoch)})"
        elif not pd.isna(batch_size):
            batch_size_str = f"{int(batch_size)} (N/A)"
        else:
            batch_size_str = np.nan
        
        # Calculate training time in minutes
        train_time = config.get("train_time", np.nan)
        train_time_min = train_time / 60 if not pd.isna(train_time) else np.nan
        
        # Get sampling time from training data config
        sampling_time = 0
        try:
            train_data_pth = train_cfg.get("train_data_pth")
            if train_data_pth:
                train_data_config_file = FILE_PTH.joinpath(train_data_pth).parent / "config.json"
                with open(train_data_config_file, "r") as f:
                    train_data_config = json.load(f)
                sampling_time = train_data_config.get("sampling_time", 0)
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            sampling_time = 0
        
        # Format training time with sampling time in parentheses
        if not pd.isna(train_time_min):
            total_time_min = (train_time + sampling_time) / 60
            train_time_str = f"{train_time_min:.2f} ({total_time_min:.2f})"
        else:
            train_time_str = "N/A"
        
        hyperparams_data["Approx. MPC"] = {
            "Hidden Layers": model_cfg.get("n_hidden_layers", np.nan),
            "Neurons per Layer": model_cfg.get("n_neurons", np.nan),
            "Training Dataset Size": train_data_size,
            "Batch Size (Batches per Epoch)": batch_size_str,
            "Epochs": train_cfg.get("N_epochs", np.nan),
            "Learning Rate": train_cfg.get("lr", np.nan),
            "Training Time [min]": train_time_str
        }
    
    # Extract Predictor hyperparameters
    if res_predictor is not None:
        config = res_predictor["config"]
        train_cfg = config.get("train_cfg", {})
        model_cfg = config.get("model_cfg", {})
        
        batch_size = train_cfg.get("batch_size", np.nan)
        if not pd.isna(batch_size):
            batch_size_str = f"{int(batch_size)} (1)"
        else:
            batch_size_str = "N/A"
        
        # Calculate training time in minutes
        train_time = config.get("train_time", np.nan)
        train_time_min = train_time / 60 if not pd.isna(train_time) else np.nan
        train_time_str = f"{train_time_min:.2f}" if not pd.isna(train_time_min) else "N/A"
        
        hyperparams_data["Predictor"] = {
            "Hidden Layers": model_cfg.get("n_hidden_layers", np.nan),
            "Neurons per Layer": model_cfg.get("n_neurons", np.nan),
            "Training Dataset Size": "-",
            "Batch Size (Batches per Epoch)": batch_size_str,
            "Epochs": train_cfg.get("N_epochs", np.nan),
            "Learning Rate": train_cfg.get("lr", np.nan),
            "Training Time [min]": train_time_str
        }
    
    # Extract Solver hyperparameters
    if res_solver is not None:
        config = res_solver["config"]
        train_cfg = config.get("train_cfg", {})
        model_cfg = config.get("model_cfg", {})
        
        batch_size = train_cfg.get("batch_size", np.nan)
        if not pd.isna(batch_size):
            batch_size_str = f"{int(batch_size)} (1)"
        else:
            batch_size_str = "N/A"
        
        # Calculate training time in minutes
        train_time = config.get("train_time", np.nan)
        train_time_min = train_time / 60 if not pd.isna(train_time) else np.nan
        
        # Get predictor training time
        predictor_train_time = np.nan
        if res_predictor is not None:
            predictor_train_time = res_predictor["config"].get("train_time", np.nan)
        
        # Format training time with combined time in parentheses
        if not pd.isna(train_time_min):
            if not pd.isna(predictor_train_time):
                combined_time_min = (train_time + predictor_train_time) / 60
                train_time_str = f"{train_time_min:.2f} ({combined_time_min:.2f})"
            else:
                train_time_str = f"{train_time_min:.2f} (N/A)"
        else:
            train_time_str = "N/A"
        
        hyperparams_data["Solver w. Predictor"] = {
            "Hidden Layers": model_cfg.get("n_hidden_layers", np.nan),
            "Neurons per Layer": model_cfg.get("n_neurons", np.nan),
            "Training Dataset Size": "-",
            "Batch Size (Batches per Epoch)": batch_size_str,
            "Epochs": train_cfg.get("N_epochs", np.nan),
            "Learning Rate": train_cfg.get("lr", np.nan),
            "Training Time [min]": train_time_str
        }
    
    return hyperparams_data

hyperparams_data = extract_hyperparameters(res_approxmpc, res_predictor, res_solver)
if hyperparams_data:
    # Create DataFrame with methods as columns and hyperparameters as rows
    hyperparams_df = pd.DataFrame(hyperparams_data)
    
    # Format the table for display
    hyperparams_formatted = hyperparams_df.copy().astype(object)
    
    for col in hyperparams_formatted.columns:
        for idx in hyperparams_formatted.index:
            value = hyperparams_formatted.loc[idx, col]
            if pd.isna(value):
                hyperparams_formatted.loc[idx, col] = "N/A"
            elif idx == "Learning Rate":
                hyperparams_formatted.loc[idx, col] = f"{value:.0e}" if not pd.isna(value) else "N/A"
            elif idx == "Epochs":
                if not pd.isna(value):
                    if value >= 1000:
                        hyperparams_formatted.loc[idx, col] = f"{int(value/1000)}k"
                    else:
                        hyperparams_formatted.loc[idx, col] = f"{int(value)}"
                else:
                    hyperparams_formatted.loc[idx, col] = "N/A"
            elif idx in ["Hidden Layers", "Neurons per Layer", "Training Dataset Size"]:
                if idx == "Training Dataset Size" and value == "-":
                    hyperparams_formatted.loc[idx, col] = "-"
                else:
                    hyperparams_formatted.loc[idx, col] = f"{int(value)}" if not pd.isna(value) else "N/A"
            else:
                hyperparams_formatted.loc[idx, col] = str(value)
    
    print(hyperparams_formatted.to_string())
    
    latex_path4 = RESULTS_PATH.joinpath(SAVE_FOLDER, "table4_hyperparameters.tex")
    latex_string4 = hyperparams_formatted.to_latex(
        escape=False,
        column_format='l' + 'c' * len(hyperparams_formatted.columns)
    )
    
    # Add midrule before Training Time row in LaTeX
    latex_lines = latex_string4.split('\n')
    midrule_inserted = False
    for i, line in enumerate(latex_lines):
        if 'Training Time [min]' in line and not midrule_inserted:
            latex_lines.insert(i, '\\midrule')
            midrule_inserted = True
            break
    latex_string4 = '\n'.join(latex_lines)
    
    with open(latex_path4, 'w') as f:
        f.write(latex_string4)
    print(f"Saved Table 4 to {latex_path4}")
else:
    print("No hyperparameter data available")


# Table 5: Infeasibility Comparison
print("\n\nTable 5: Infeasibility Comparison")
print("=" * 80)

# Create infeasibility comparison table
# n_iter_max = res_solver["eval"]["cpu_n_iter_max"]
n_iter_max = res_solver["eval"]["gpu_n_iter_max"]
steps_to_consider = [1, 5, 20, int(n_iter_max)+1]

infeasibility_columns_dict = {
    "frac_u0_infeasible": r"Frac. $u_0$ infeasible",
    "frac_x1_infeasible": r"Frac. $x_1$ infeasible"
}

# Build comparison data for infeasibility
infeasibility_index_tuples = []
infeasibility_data_rows = []

# Add baseline methods
baseline_methods = [
    (res_approxmpc, "approx. MPC", ""),
    (res_predictor, "predictor", "")
]

for res, method_name, step_label in baseline_methods:
    infeasibility_index_tuples.append((method_name, step_label))
    row = []
    for col in infeasibility_columns_dict.keys():
        if col in res["eval"]:
            row.append(res["eval"][col])
        else:
            row.append(np.nan)
    infeasibility_data_rows.append(row)

# Add solver results for different iterations
for step in steps_to_consider:
    step_label = f"k={step}" if step != n_iter_max+1 else f"k={n_iter_max} (max)"
    infeasibility_index_tuples.append(("solver w. predictor", step_label))
    
    if step <= len(res_solver["eval"]["trajectory_evaluation"]):
        eval_dict = res_solver["eval"]["trajectory_evaluation"][step]
    else:
        eval_dict = res_solver["eval"]["trajectory_evaluation"][-1]
    
    row = []
    for col in infeasibility_columns_dict.keys():
        row.append(eval_dict.get(col, np.nan))
    infeasibility_data_rows.append(row)

# Create and format infeasibility comparison table
infeasibility_multi_index = pd.MultiIndex.from_tuples(infeasibility_index_tuples, names=['Method', 'Iter.'])
infeasibility_comparison_df = pd.DataFrame(infeasibility_data_rows, index=infeasibility_multi_index, columns=infeasibility_columns_dict.values())

# Format as percentages with 2 decimal places
infeasibility_comparison_df_formatted = infeasibility_comparison_df.map(
    lambda x: as_string_scientific(x) if pd.notna(x) else 'N/A'
)

print(infeasibility_comparison_df_formatted.to_string())

# Export to LaTeX
infeasibility_latex_string = infeasibility_comparison_df_formatted.to_latex(
    escape=False,
    multirow=True,
    column_format='ll' + 'c' * len(infeasibility_columns_dict.values())
)

infeasibility_latex_file_path = RESULTS_PATH.joinpath(SAVE_FOLDER, "table5_infeasibility_comparison.tex")
with open(infeasibility_latex_file_path, 'w') as f:
    f.write(infeasibility_latex_string)
print(f"Saved Table 5 to {infeasibility_latex_file_path}")

# Table 6: Performance Comparison
print("\n\nTable 6: Performance Comparison")
print("=" * 80)

# n_iter_max = res_solver["eval"]["cpu_n_iter_max"]
n_iter_max = res_solver["eval"]["gpu_n_iter_max"]
steps_to_consider = [1, 5, 20, int(n_iter_max)]

columns_dict = {
    "KKT_inf_max": r"$||KKT||_{\infty}$ (max)",
    "u0_abs_diff_med": r"$|u_0 - u_0^{*}|$ (median)",
    "u0_abs_diff_95": r"$|u_0 - u_0^{*}|$ (95th perc.)",
    "u0_abs_diff_max": r"$|u_0 - u_0^{*}|$ (max)"
}

# Build comparison data
index_tuples = []
data_rows = []

# Add baseline methods
methods = [
    (res_approxmpc, "approx. MPC", ""),
    (res_predictor, "predictor", "")
]

for res, method_name, step_label in methods:
    index_tuples.append((method_name, step_label))
    row = []
    for col in columns_dict.keys():
        if col in res["eval"]:
            row.append(res["eval"][col])
        else:
            row.append(np.nan)
    data_rows.append(row)

# Add solver results for different iterations
for step in steps_to_consider:
    step_label = f"k={step}" if step != n_iter_max else f"k={n_iter_max} (max)"
    index_tuples.append(("solver w. predictor", step_label))
    
    if step <= len(res_solver["eval"]["trajectory_evaluation"]):
        eval_dict = res_solver["eval"]["trajectory_evaluation"][step]
    else:
        eval_dict = res_solver["eval"]["trajectory_evaluation"][-1]
    
    row = []
    for col in columns_dict.keys():
        row.append(eval_dict.get(col, np.nan))
    data_rows.append(row)

# Create and format comparison table
multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['Method', 'Iter.'])
comparison_df = pd.DataFrame(data_rows, index=multi_index, columns=columns_dict.values())
comparison_df_formatted = comparison_df.map(lambda x: as_string_scientific(x) if pd.notna(x) else 'N/A')

print(comparison_df_formatted.to_string())

# Export to LaTeX
latex_string = comparison_df_formatted.to_latex(
    escape=False,
    float_format="{:.2e}".format,
    multirow=True,
    column_format='ll' + 'c' * len(columns_dict.values())
)

latex_file_path = RESULTS_PATH.joinpath(SAVE_FOLDER, "table6_performance_comparison.tex")
with open(latex_file_path, 'w') as f:
    f.write(latex_string)
print(f"Saved Table 6 to {latex_file_path}")

# Table 7: Combined Performance and Infeasibility Comparison
print("\n\nTable 7: Combined Performance and Infeasibility Comparison")
print("=" * 80)

# Combine columns from both tables with multi-level column headers
combined_columns_dict = {
    "KKT_inf_max": (r"$||KKT||_{\infty}$", "max"),
    "u0_abs_diff_med": (r"$|u_0 - u_0^{*}|$", "median"),
    "u0_abs_diff_95": (r"$|u_0 - u_0^{*}|$", "95th perc."),
    "u0_abs_diff_max": (r"$|u_0 - u_0^{*}|$", "max"),
    "frac_u0_infeasible": (r"$u_0$ viol.", "fraction"),
    "frac_x1_infeasible": (r"$x_1$ viol.", "fraction")
}

# Build combined comparison data
combined_index_tuples = []
combined_data_rows = []

# Add baseline methods
combined_methods = [
    (res_approxmpc, "approx. MPC", ""),
    (res_predictor, "predictor", "")
]

for res, method_name, step_label in combined_methods:
    combined_index_tuples.append((method_name, step_label))
    row = []
    for col in combined_columns_dict.keys():
        if col in res["eval"]:
            row.append(res["eval"][col])
        else:
            row.append(np.nan)
    combined_data_rows.append(row)

# Add solver results for different iterations
for step in steps_to_consider:
    step_label = f"k={step}" if step != n_iter_max+1 else f"k={n_iter_max} (max)"
    combined_index_tuples.append(("solver w. predictor", step_label))
    
    if step <= len(res_solver["eval"]["trajectory_evaluation"]):
        eval_dict = res_solver["eval"]["trajectory_evaluation"][step]
    else:
        eval_dict = res_solver["eval"]["trajectory_evaluation"][-1]
    
    row = []
    for col in combined_columns_dict.keys():
        row.append(eval_dict.get(col, np.nan))
    combined_data_rows.append(row)

# Create multi-level column index
combined_column_tuples = list(combined_columns_dict.values())
combined_multi_columns = pd.MultiIndex.from_tuples(combined_column_tuples, names=['Metric', 'Statistic'])

# Create and format combined comparison table
combined_multi_index = pd.MultiIndex.from_tuples(combined_index_tuples, names=['Method', 'Iter.'])
combined_comparison_df = pd.DataFrame(combined_data_rows, index=combined_multi_index, columns=combined_multi_columns)
combined_comparison_df_formatted = combined_comparison_df.map(lambda x: as_string_scientific(x) if pd.notna(x) else 'N/A')

print(combined_comparison_df_formatted.to_string())

# Export to LaTeX with custom column formatting for centered headers
combined_latex_string = combined_comparison_df_formatted.to_latex(
    escape=False,
    float_format="{:.2e}".format,
    multirow=True,
    column_format='ll' + 'c' * len(combined_columns_dict.values())
)

# Modify the LaTeX to add \multicolumn commands for centering top-level headers
latex_lines = combined_latex_string.split('\n')
for i, line in enumerate(latex_lines):
    # Find the header line with the metric names
    if '$||KKT||_{\\infty}$' in line and '$|u_0 - u_0^{*}|$' in line:
        # Replace the repeated $|u_0 - u_0^{*}|$ entries with a single multicolumn entry
        parts = line.split(' & ')
        new_parts = parts[:2]  # Keep Method and Iter. columns
        new_parts.append('$||KKT||_{\\infty}$')  # KKT column
        new_parts.append('\\multicolumn{3}{c}{$|u_0 - u_0^{*}|$}')  # Centered u0 header over 3 columns
        new_parts.append('$u_0$ viol.')  # u0 violation column
        new_parts.append('$x_1$ viol.')  # x1 violation column
        latex_lines[i] = ' & '.join(new_parts) + ' \\\\'
        break

combined_latex_string = '\n'.join(latex_lines)

combined_latex_file_path = RESULTS_PATH.joinpath(SAVE_FOLDER, "table7_combined_performance_infeasibility.tex")
with open(combined_latex_file_path, 'w') as f:
    f.write(combined_latex_string)
print(f"Saved Table 7 to {combined_latex_file_path}")


# %% Generate Speedup Histogram

def plot_speedup_histogram(speedup_factors, ax=None):
    """Plot histogram of speedup factors."""
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure
    
    ax.hist(speedup_factors, bins=50, color='blue', alpha=0.7)
    # ax.axvline(x=1, color='red', linestyle=':', linewidth=1, label='Baseline (IPOPT)')
    ax.axvline(x=1, color='black', linestyle=':', linewidth=1)
    ax.set_xlabel("Speedup Factor (IPOPT time / LISCO time)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Speedup Factors vs. IPOPT")
    ax.grid(True, linestyle="--", linewidth=0.3)
    
    # Set x-axis ticks starting at 0 with increments of 1
    max_speedup = max(speedup_factors)
    ax.set_xticks(np.arange(0, int(max_speedup) + 2, 1))
    
    # ax.legend(loc='center left')
    
    # Optional: Add text annotation showing percentage of cases with speedup > 1
    n_speedup = sum(1 for s in speedup_factors if s > 1)
    pct_speedup = 100 * n_speedup / len(speedup_factors)
    ax.text(0.98, 0.98, f'{pct_speedup:.1f}% faster than IPOPT', 
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return fig, ax

# Collect speedup factors from solver results
print("\nGenerating Speedup Histogram")

if res_solver is not None and "cpu_speedup_factor_list" in res_solver["eval"]:
    speedup_factors = res_solver["eval"]["cpu_speedup_factor_list"]
    
    if len(speedup_factors) > 0:
        fig_speedup, ax_speedup = plot_speedup_histogram(speedup_factors)
        
        save_path = RESULTS_PATH.joinpath(SAVE_FOLDER)
        save_path.mkdir(parents=True, exist_ok=True)
        # Save as both PNG and PDF
        speedup_plot_path_png = save_path.joinpath("speedup_histogram_nmpc.png")
        speedup_plot_path_pdf = save_path.joinpath("speedup_histogram_nmpc.pdf")
        fig_speedup.savefig(speedup_plot_path_png, bbox_inches='tight', dpi=300)
        fig_speedup.savefig(speedup_plot_path_pdf, bbox_inches='tight', dpi=300)
        print(f"Saved speedup histogram to {speedup_plot_path_png}")
        print(f"Saved speedup histogram to {speedup_plot_path_pdf}")
        
        plt.close(fig_speedup)
        print("Generated speedup histogram")
    else:
        print("No speedup factors available for histogram")
else:
    print("No speedup factor list found in solver results")

# %% Generating speedup histogram which compares the results of solver with predictor and solver without predictor (two different histograms in one plot)
print("\nGenerating Speedup Histogram Comparison")
if res_solver is not None and res_solver_no_pred is not None:
    speedup_factors_with_pred = res_solver["eval"].get("cpu_speedup_factor_list", [])
    speedup_factors_without_pred = res_solver_no_pred["eval"].get("cpu_speedup_factor_list", [])
    
    if len(speedup_factors_with_pred) > 0 and len(speedup_factors_without_pred) > 0:
        fig, ax = plt.subplots(1, 1)
        
        ax.hist(speedup_factors_with_pred, bins=50, alpha=0.7, color="blue", label='with predictor')
        ax.hist(speedup_factors_without_pred, bins=50, alpha=0.7, color="orange", label='without predictor')
        ax.set_ylim(0,600)

        ax.axvline(x=1, color='black', linestyle=':', linewidth=1)
        ax.set_xlabel("Speedup Factor (IPOPT time / LISCO time)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Speedup Factors vs. IPOPT")
        ax.grid(True, linestyle="--", linewidth=0.3)

        # Calculate percentage of cases with speedup > 1 for both methods
        n_speedup_with_pred = sum(1 for s in speedup_factors_with_pred if s > 1)
        pct_speedup_with_pred = 100 * n_speedup_with_pred / len(speedup_factors_with_pred)

        n_speedup_without_pred = sum(1 for s in speedup_factors_without_pred if s > 1)
        pct_speedup_without_pred = 100 * n_speedup_without_pred / len(speedup_factors_without_pred)

        # Display percentages in two separate text boxes with matching colors
        ax.text(0.98, 0.98, f'with predictor: {pct_speedup_with_pred:.1f}% faster than IPOPT', 
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3, edgecolor='blue'))
        
        ax.text(0.98, 0.88, f'without predictor: {pct_speedup_without_pred:.1f}% faster than IPOPT', 
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3, edgecolor='orange'))
        
        save_path = RESULTS_PATH.joinpath(SAVE_FOLDER)
        save_path.mkdir(parents=True, exist_ok=True)
        # Save as both PNG and PDF
        speedup_comparison_plot_path_png = save_path.joinpath("speedup_histogram_comparison.png")
        speedup_comparison_plot_path_pdf = save_path.joinpath("speedup_histogram_comparison.pdf")
        fig.savefig(speedup_comparison_plot_path_png, bbox_inches='tight', dpi=300)
        fig.savefig(speedup_comparison_plot_path_pdf, bbox_inches='tight', dpi=300)
        print(f"Saved speedup histogram comparison to {speedup_comparison_plot_path_png}")
        print(f"Saved speedup histogram comparison to {speedup_comparison_plot_path_pdf}")
        
        plt.close(fig)
        print("Generated speedup histogram comparison")
    else:
        print("Insufficient speedup factors for comparison histogram")
