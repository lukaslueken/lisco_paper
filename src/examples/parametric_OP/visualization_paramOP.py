"""
Visualization Script for Parametric Optimization Case Study

This script loads and aggregates results from multiple runs of the nonconvex QP case study
to produce performance comparison tables for publication.

The script:
1. Loads results from multiple experiment runs for each method (predictor, solver)
2. Aggregates metrics across runs (mean and std)
3. Generates two formatted tables:
   - Table 1: Constraints violations and optimality gaps
   - Table 2: Solve times, iteration counts, and speedup factors
4. Exports results to LaTeX format for publication
"""

# %% Imports
import sys
# setting path
sys.path.append('../../')

from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from typing import Dict, List, Optional

plt.style.use(['science', 'ieee','no-latex'])

# %% Configuration
RUN_FOLDER = "results"
SAVE_FOLDER = "visualization"
TOL = 1e-6

# Specification of results to load and visualize
results_to_visualize = {
    "nonconvex_100x50x50_0": {"predictor": "exp_0", "solver": "exp_0"},
    "nonconvex_100x50x50_1": {"predictor": "exp_0", "solver": "exp_0"},
    "nonconvex_100x50x50_2": {"predictor": "exp_0", "solver": "exp_0"},
    "nonconvex_100x50x50_3": {"predictor": "exp_0", "solver": "exp_0"},
    "nonconvex_100x50x50_4": {"predictor": "exp_0", "solver": "exp_0"}
}

FILE_PTH = Path(__file__).parent.resolve()
RESULTS_PATH = FILE_PTH.joinpath(RUN_FOLDER)

# Table column specifications
TABLE_1_COLUMNS = ["opt_gap_rel_max", "opt_gap_rel_mean", "viol_h_max", "viol_g_max", "viol_h_mean", "viol_g_mean"]
TABLE_2_COLUMNS = ["cpu_full_solve_time_mean", "cpu_full_solve_time_max", "cpu_n_iter_mean", "cpu_n_iter_max", 
                   "cpu_speedup_factor_min", "cpu_speedup_factor_mean", "cpu_speedup_factor_max", "success_rate"]

# Key transformation dictionary for column/index renaming
TRANSFORMATION_DICT = {
    'predictor': 'Predictor',    
    'solver': 'Solver',
    'opt_gap_rel_mean': r'Opt. Gap(\%)',
    'opt_gap_rel_max': r'Max. Opt. Gap(\%)',
    'viol_h_max': 'Max. Eq.',
    'viol_g_max': 'Max. Ineq.',
    'viol_h_mean': 'Mean Eq.',
    'viol_g_mean': 'Mean Ineq.',
    'cpu_full_solve_time_mean': 'Mean Solve Time (s)',
    'cpu_full_solve_time_max': 'Max Solve Time (s)',
    'cpu_n_iter_mean': 'Mean Iter. Count',
    'cpu_n_iter_max': 'Max Iter. Count',
    'cpu_speedup_factor_min': 'Min. Speedup',
    'cpu_speedup_factor_max': 'Max. Speedup',
    'cpu_speedup_factor_mean': 'Mean Speedup',
    'success_rate': 'Success Rate',
}

# %% Utility Functions

def load_results(case_name: str, method: str, exp_id: str) -> Optional[Dict]:
    """Load evaluation and configuration results for a specific experiment.
    
    Args:
        case_name: Name of the case study
        method: Method type ('predictor' or 'solver')
        exp_id: Experiment ID
        
    Returns:
        Dictionary containing eval and config data, or None if not found
    """
    results_folder = RESULTS_PATH.joinpath(case_name, method, exp_id)
    eval_file = results_folder.joinpath("eval.json")
    config_file = results_folder.joinpath("config.json")
    
    try:
        with open(eval_file, "r") as f:
            eval_dict = json.load(f)
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        
        return {"eval": eval_dict, "config": config_dict}
    except FileNotFoundError:
        print(f"No results file found for {case_name}/{method} with exp_id {exp_id}")
        return None

def as_string_decimals(x: float, decimals: int = 3) -> str:
    """Format float as string with specified decimal places."""
    if pd.isna(x) or x is None:
        return "N/A"
    return f"{x:.{decimals}f}"

def format_significant_digits(x: float, sig_digits: int = 5) -> str:
    """Format float with up to specified significant digits."""
    if pd.isna(x) or x is None:
        return "N/A"
    if x == 0:
        return "0"
    
    # Format with significant digits, removing trailing zeros
    return f"{x:.{sig_digits-1}g}"

def apply_key_transformation(df: pd.DataFrame) -> pd.DataFrame:
    """Apply key transformations to DataFrame index and columns."""
    df_transformed = df.copy()
    
    # Transform column names
    df_transformed.columns = [TRANSFORMATION_DICT.get(col, col) for col in df_transformed.columns]
    
    # Transform index names
    if hasattr(df_transformed.index, 'map'):
        df_transformed.index = df_transformed.index.map(lambda x: TRANSFORMATION_DICT.get(x, x))
    
    return df_transformed

def aggregate_results_across_runs(method_data: Dict) -> Dict:
    """Aggregate results across multiple runs for a method.
    
    Uses different aggregation strategies for different metrics:
    - Table 1 metrics: mean and std
    - Solve times/iterations: appropriate min/max/mean for each metric
    - Success rate: minimum (worst case) and mean
    """
    all_metrics = {}
    
    # Collect all metrics across runs
    for run_name, run_data in method_data.items():
        if run_data is not None:
            eval_data = run_data["eval"]
            for key in TABLE_1_COLUMNS + TABLE_2_COLUMNS:
                if key in eval_data:
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(eval_data[key])
    
    # Calculate aggregations with appropriate strategies
    aggregated = {}
    for key, values in all_metrics.items():
        if not values:
            aggregated[key] = np.nan
            aggregated[f"{key}_std"] = np.nan
            continue
            
        # Apply specific aggregation strategy based on metric type
        if key in ["cpu_full_solve_time_mean", "cpu_n_iter_mean", "cpu_speedup_factor_mean"]:
            # For mean metrics: calculate mean across all runs
            aggregated[key] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)
        elif key in ["cpu_full_solve_time_max", "cpu_n_iter_max", "cpu_speedup_factor_max"]:
            # For max metrics: take maximum across all runs
            aggregated[key] = np.max(values)
            aggregated[f"{key}_std"] = np.std(values)
        elif key == "cpu_speedup_factor_min":
            # For min speedup: take minimum across all runs
            aggregated[key] = np.min(values)
            aggregated[f"{key}_std"] = np.std(values)
        elif key == "success_rate":
            # For success rate: take minimum and store mean separately
            aggregated[key] = np.min(values)
            aggregated[f"{key}_mean"] = np.mean(values)
        else:
            # For Table 1 metrics: use mean and std
            aggregated[key] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)
    
    return aggregated

def create_combined_table(aggregated_data: Dict, include_std: bool = False) -> pd.DataFrame:
    """Create combined table with properly formatted values."""
    methods = ["predictor", "solver"]
    
    # Determine all available columns
    all_columns = {k for method_data in aggregated_data.values() 
                   for k in method_data.keys()}
    all_columns = sorted(list(all_columns))
    
    # Create DataFrame
    data_rows = []
    for method in methods:
        if method in aggregated_data:
            row = {}
            for col in all_columns:
                mean_val = aggregated_data[method].get(col, np.nan)
                std_val = aggregated_data[method].get(f"{col}_std", np.nan)
                
                # Handle percentage conversion for optimality gaps
                if "opt_gap_rel" in col and not pd.isna(mean_val):
                    mean_val *= 100
                    if not pd.isna(std_val):
                        std_val *= 100
                
                # Format value with std in parentheses if requested
                if include_std and not pd.isna(std_val):
                    row[col] = f"{as_string_decimals(mean_val)} ({as_string_decimals(std_val)})" if not pd.isna(mean_val) else "N/A"
                else:
                    row[col] = as_string_decimals(mean_val) if not pd.isna(mean_val) else "N/A"
            
            data_rows.append(row)
        else:
            # Fill with N/A if method not available
            row = {col: "N/A" for col in all_columns}
            data_rows.append(row)
    
    return pd.DataFrame(data_rows, index=methods)

def save_latex_table(df: pd.DataFrame, filename: str, custom_header: Optional[List[str]] = None) -> None:
    """Save DataFrame as LaTeX table with optional custom header."""
    save_path = RESULTS_PATH.joinpath(SAVE_FOLDER)
    save_path.mkdir(parents=True, exist_ok=True)
    
    if custom_header:
        # Create custom LaTeX table
        latex_lines = custom_header
    else:
        # Use pandas to_latex and extract tabular environment
        latex_string = df.to_latex(
            escape=False,
            column_format='l' + 'c' * len(df.columns),
            caption=None,
            label=None
        )
        
        latex_lines = latex_string.split('\n')
        tabular_start = next((i for i, line in enumerate(latex_lines) if '\\begin{tabular}' in line), None)
        tabular_end = next((i for i, line in enumerate(latex_lines) if '\\end{tabular}' in line), None)
        
        if tabular_start is not None and tabular_end is not None:
            latex_lines = latex_lines[tabular_start:tabular_end + 1]
    
    latex_file_path = save_path.joinpath(filename)
    with open(latex_file_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    print(f"Saved LaTeX table to {latex_file_path}")

# %% Load and Aggregate Results

print("Loading and aggregating results for nonconvex QP case study...")

# Load all experiment results
all_results = {}
for case_name, case_experiments in results_to_visualize.items():
    print(f"Loading results for {case_name}...")
    case_results = {}
    
    for method, exp_id in case_experiments.items():
        result = load_results(case_name, method, exp_id)
        if result is not None:
            case_results[method] = result
            print(f"  Loaded {method}: {exp_id}")
        else:
            print(f"  Failed to load {method}: {exp_id}")
    
    all_results[case_name] = case_results

# Aggregate results across all cases for each method
aggregated_data = {}
for method in ["predictor", "solver"]:
    method_results = {case_name: case_results[method] 
                     for case_name, case_results in all_results.items() 
                     if method in case_results}
    
    if method_results:
        aggregated_data[method] = aggregate_results_across_runs(method_results)
        print(f"Aggregated {len(method_results)} results for {method}")
    else:
        print(f"No results found for {method}")

# %% Generate Table 1: Constraints and Optimality Gap

print("\nGenerating Table 1: Constraints and Optimality Gap")

# Extract only Table 1 columns
table_1_data = {method: {k: v for k, v in data.items() 
                        if any(col in k for col in TABLE_1_COLUMNS)}
                for method, data in aggregated_data.items()}

df_table_1 = create_combined_table(table_1_data, include_std=True)

# Select only relevant columns and apply transformations
available_cols = [col for col in TABLE_1_COLUMNS if col in df_table_1.columns]
df_table_1 = apply_key_transformation(df_table_1[available_cols])

# Reorder columns to match the requested format: Method, Max. Opt. Gap(%), Opt. Gap(%), Max. Eq., Max. Ineq., Mean Eq., Mean Ineq.
desired_column_order = [
    r'Max. Opt. Gap(\%)',
    r'Opt. Gap(\%)',
    'Max. Eq.',
    'Max. Ineq.',
    'Mean Eq.',
    'Mean Ineq.'
]
available_ordered_cols = [col for col in desired_column_order if col in df_table_1.columns]
df_table_1 = df_table_1[available_ordered_cols]

print("\nTable 1 - Constraints and Optimality Gap:")
print(df_table_1.to_string())

# Save to LaTeX
save_latex_table(df_table_1, "table_1_nonconvex_qp_results.tex")

# %% Generate Table 2: Solve Times and Iteration Counts

print("\nGenerating Table 2: Solve Times and Iteration Counts")

# Extract only Table 2 columns for solver method only
if "solver" in aggregated_data:
    table_2_data = {k: v for k, v in aggregated_data["solver"].items() 
                   if any(col in k for col in TABLE_2_COLUMNS)}
    
    # Create table with only solver data
    solver_data = {}
    for col in TABLE_2_COLUMNS:
        if col in table_2_data:
            mean_val = table_2_data[col]
            
            # Special formatting for success rate: show min (mean)
            if col == "success_rate":
                mean_success = table_2_data.get("success_rate_mean", np.nan)
                if not pd.isna(mean_val) and not pd.isna(mean_success):
                    min_formatted = format_significant_digits(mean_val, 5)
                    mean_formatted = format_significant_digits(mean_success, 5)
                    solver_data[col] = f"{min_formatted} ({mean_formatted})"
                else:
                    solver_data[col] = "N/A"
            # Format iteration counts as integers
            elif "iter" in col:
                solver_data[col] = f"{int(round(mean_val))}" if not pd.isna(mean_val) else "N/A"
            else:
                # Format other columns with more precision
                solver_data[col] = f"{mean_val:.4f}" if not pd.isna(mean_val) else "N/A"
    
    # Create DataFrame and apply transformations
    df_table_2 = pd.DataFrame([solver_data], index=["solver"])
    df_table_2 = apply_key_transformation(df_table_2)
    
    print("\nTable 2 - Solve Times and Iteration Counts:")
    print(df_table_2.to_string())
    
    # Save to LaTeX with custom multi-line headers
    solver_row = df_table_2.loc["Solver"]
    data_values = [
        solver_row["Mean Solve Time (s)"],
        solver_row["Max Solve Time (s)"],
        solver_row["Mean Iter. Count"],
        solver_row["Max Iter. Count"],
        solver_row["Min. Speedup"],
        solver_row["Mean Speedup"],
        solver_row["Max. Speedup"],
        solver_row["Success Rate"]
    ]
    
    custom_header = [
        "\\begin{tabular}{l|cc|cc|ccc|c}",
        "\\toprule",
        "& \\multicolumn{2}{c|}{Solve Time (s)} & \\multicolumn{2}{c|}{Iter. Count} & \\multicolumn{3}{c|}{Speedup} & Success Rate \\\\",
        "& Mean & Max & Mean & Max & Min. & Mean & Max. & \\\\",
        "\\midrule",
        "Solver & " + " & ".join(data_values) + " \\\\",
        "\\bottomrule",
        "\\end{tabular}"
    ]
    
    save_latex_table(df_table_2, "table_2_solve_times_iter_counts.tex", custom_header)
else:
    print("No solver data found for Table 2")

# %% Generate Table 2 Transposed: Solve Times and Iteration Counts (Multi-Index)

print("\nGenerating Table 2 Transposed: Solve Times and Iteration Counts (Multi-Index)")

if "solver" in aggregated_data:
    table_2_data = {k: v for k, v in aggregated_data["solver"].items() 
                   if any(col in k for col in TABLE_2_COLUMNS)}
    
    # Create multi-index structure like in visualization.py
    row_tuples = []
    data_values = []
    
    # Solve Time values
    mean_solve_time = table_2_data.get("cpu_full_solve_time_mean", np.nan)
    max_solve_time = table_2_data.get("cpu_full_solve_time_max", np.nan)
    row_tuples.extend([
        ("Solve Time (s)", "mean"),
        ("Solve Time (s)", "max")
    ])
    data_values.extend([
        f"{mean_solve_time:.4f}" if not pd.isna(mean_solve_time) else "N/A",
        f"{max_solve_time:.4f}" if not pd.isna(max_solve_time) else "N/A"
    ])
    
    # Iter Count values
    mean_iter = table_2_data.get("cpu_n_iter_mean", np.nan)
    max_iter = table_2_data.get("cpu_n_iter_max", np.nan)
    row_tuples.extend([
        ("Iter. Count", "mean"),
        ("Iter. Count", "max")
    ])
    data_values.extend([
        f"{int(round(mean_iter))}" if not pd.isna(mean_iter) else "N/A",
        f"{int(round(max_iter))}" if not pd.isna(max_iter) else "N/A"
    ])
    
    # Speedup values
    min_speedup = table_2_data.get("cpu_speedup_factor_min", np.nan)
    mean_speedup = table_2_data.get("cpu_speedup_factor_mean", np.nan)
    max_speedup = table_2_data.get("cpu_speedup_factor_max", np.nan)
    row_tuples.extend([
        ("Speedup", "min"),
        ("Speedup", "mean"),
        ("Speedup", "max")
    ])
    data_values.extend([
        f"{min_speedup:.4f}" if not pd.isna(min_speedup) else "N/A",
        f"{mean_speedup:.4f}" if not pd.isna(mean_speedup) else "N/A",
        f"{max_speedup:.4f}" if not pd.isna(max_speedup) else "N/A"
    ])
    
    # Success Rate value (formatted as min (mean))
    success_rate = table_2_data.get("success_rate", np.nan)
    mean_success = table_2_data.get("success_rate_mean", np.nan)
    if not pd.isna(success_rate) and not pd.isna(mean_success):
        min_formatted = format_significant_digits(success_rate, 5)
        mean_formatted = format_significant_digits(mean_success, 5)
        success_formatted = f"{min_formatted} ({mean_formatted})"
    else:
        success_formatted = "N/A"
    row_tuples.append(("Success Rate", ""))
    data_values.append(success_formatted)
    
    # Create DataFrame with multi-level index using pandas MultiIndex
    multi_index = pd.MultiIndex.from_tuples(row_tuples, names=['Metric', 'Statistic'])
    
    df_table_2_transposed = pd.DataFrame(
        data=data_values,
        index=multi_index,
        columns=["Solver"]
    )
    
    print("\nTable 2 Transposed - Solve Times and Iteration Counts:")
    print(df_table_2_transposed.to_string())
    
    # Save to LaTeX using pandas built-in multi-index support
    save_path = RESULTS_PATH.joinpath(SAVE_FOLDER)
    save_path.mkdir(parents=True, exist_ok=True)
    
    latex_string = df_table_2_transposed.to_latex(
        escape=False,
        multirow=True,
        column_format='ll' + 'c' * len(df_table_2_transposed.columns)
    )
    
    latex_file_path = save_path.joinpath("table_2_transposed_solve_times_iter_counts.tex")
    with open(latex_file_path, 'w') as f:
        f.write(latex_string)
    print(f"Saved LaTeX table to {latex_file_path}")
else:
    print("No solver data found for Table 2 Transposed")

# %% Generate Table 3: Solver with Predictor Performance

print("\nGenerating Table 3: Solver with Predictor Performance")

if "solver" in aggregated_data:
    # Instead of using pre-aggregated data, collect all raw individual values from all 5 experimental runs
    all_speedup_values = []
    all_iter_values = []
    all_solve_time_values = []
    all_success_rates = []
    
    # Collect raw values from all solver runs
    for case_name, case_results in all_results.items():
        if "solver" in case_results and case_results["solver"] is not None:
            solver_eval = case_results["solver"]["eval"]
            
            # Collect speedup factors (use the list if available, otherwise the aggregated value)
            if "cpu_speedup_factor_list" in solver_eval:
                all_speedup_values.extend(solver_eval["cpu_speedup_factor_list"])
            else:
                raise ValueError("Expected data")
            # elif "cpu_speedup_factor_mean" in solver_eval:
            #     all_speedup_values.append(solver_eval["cpu_speedup_factor_mean"])
            
            # Collect iteration counts
            if "cpu_n_iter_list" in solver_eval:
                all_iter_values.extend(solver_eval["cpu_n_iter_list"])
            else:
                raise ValueError("Expected data")
            # elif "cpu_n_iter_mean" in solver_eval:
            #     all_iter_values.append(solver_eval["cpu_n_iter_mean"])
            
            # Collect solve times
            if "cpu_full_solve_time_list" in solver_eval:
                all_solve_time_values.extend(solver_eval["cpu_full_solve_time_list"])
            else:
                raise ValueError("Expected data")
            # elif "cpu_full_solve_time_mean" in solver_eval:
            #     all_solve_time_values.append(solver_eval["cpu_full_solve_time_mean"])
            
            # Collect success rate
            if "success_rate" in solver_eval:
                all_success_rates.append(solver_eval["success_rate"])
            else:
                raise ValueError("Expected success_rate data")
    
    # Convert to numpy arrays for easier calculation
    all_speedup_values = np.array(all_speedup_values)
    all_iter_values = np.array(all_iter_values)
    all_solve_time_values = np.array(all_solve_time_values)
    all_success_rates = np.array(all_success_rates)
    
    # Build the table data - only showing solver results (w. predictor)
    table_rows = []
    
    # Speedup over IPOPT
    if len(all_speedup_values) > 0:
        speedup_max = np.max(all_speedup_values)
        speedup_median = np.median(all_speedup_values)
        speedup_min = np.min(all_speedup_values)
        
        table_rows.extend([
            ["Speedup over IPOPT", "max", f"{speedup_max:.2f}"],
            ["Speedup over IPOPT", "med", f"{speedup_median:.2f}"],
            ["Speedup over IPOPT", "min", f"{speedup_min:.2f}"]
        ])
    else:
        table_rows.extend([
            ["Speedup over IPOPT", "max", "N/A"],
            ["Speedup over IPOPT", "med", "N/A"],
            ["Speedup over IPOPT", "min", "N/A"]
        ])
    
    # N Iterations
    if len(all_iter_values) > 0:
        iter_max = np.max(all_iter_values)
        iter_median = np.median(all_iter_values)
        iter_min = np.min(all_iter_values)
        
        table_rows.extend([
            ["N Iterations", "max", f"{int(iter_max)}"],
            ["N Iterations", "med", f"{int(iter_median)}"],
            ["N Iterations", "min", f"{int(iter_min)}"]
        ])
    else:
        table_rows.extend([
            ["N Iterations", "max", "N/A"],
            ["N Iterations", "med", "N/A"],
            ["N Iterations", "min", "N/A"]
        ])
    
    # Full Solve Time [s]
    if len(all_solve_time_values) > 0:
        time_max = np.max(all_solve_time_values)
        time_median = np.median(all_solve_time_values)
        time_min = np.min(all_solve_time_values)
        
        table_rows.extend([
            ["Full Solve Time [s]", "max", f"{time_max:.2e}"],
            ["Full Solve Time [s]", "med", f"{time_median:.2e}"],
            ["Full Solve Time [s]", "min", f"{time_min:.2e}"]
        ])
    else:
        table_rows.extend([
            ["Full Solve Time [s]", "max", "N/A"],
            ["Full Solve Time [s]", "med", "N/A"],
            ["Full Solve Time [s]", "min", "N/A"]
        ])
    
    # Success Rate
    if len(all_success_rates) > 0:
        success_mean = np.mean(all_success_rates)
        
        table_rows.append([
            "Success Rate", " ", f"{success_mean:.4f}"
        ])
    else:
        table_rows.append([
            "Success Rate", "mean", "N/A"
        ])
    
    # Create DataFrame for display
    df_table_3 = pd.DataFrame(table_rows, columns=["Metric", "Statistic", "w. predictor"])
    
    print("\nTable 3 - Solver with Predictor Performance:")
    print(df_table_3.to_string(index=False))
    
    # Generate custom LaTeX table matching the template format
    latex_lines = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        " &  & w. predictor \\\\",
        "Metric & Statistic &  \\\\",
        "\\midrule"
    ]
    
    current_metric = None
    for i, (metric, stat, value) in enumerate(table_rows):
        if metric != current_metric:
            # New metric group - use multirow
            current_metric = metric
            # Count how many statistics this metric has
            metric_count = sum(1 for row in table_rows if row[0] == metric)
            latex_lines.append(f"\\multirow[t]{{{metric_count}}}{{*}}{{{metric}}} & {stat} & {value} \\\\")
        else:
            # Continuation of same metric
            latex_lines.append(f" & {stat} & {value} \\\\")
        
        # Add cline after each metric group (except the last one)
        if i < len(table_rows) - 1 and table_rows[i+1][0] != metric:
            latex_lines.append("\\cline{1-3}")
    
    latex_lines.extend([
        "\\cline{1-3}",
        "\\bottomrule",
        "\\end{tabular}"
    ])
    
    save_latex_table(df_table_3, "table_3_solver_with_predictor.tex", latex_lines)
else:
    print("No solver data found for Table 3")

# %% Generate Table 4: Solver Iteration Analysis (Similar to MPC Case Study)

print("\nGenerating Table 4: Solver Iteration Analysis")

if "predictor" in aggregated_data and "solver" in aggregated_data:
    # First, we need to collect solver iteration data for solver_1 and solver_10
    solver_iteration_data = {}
    
    # Collect solver iteration results across all runs
    for case_name, case_results in all_results.items():
        if "solver" in case_results and case_results["solver"] is not None:
            solver_eval = case_results["solver"]["eval"]
            
            # Check for solver_1 and solver_10 data
            for iter_key in ["solver_1", "solver_10"]:
                if iter_key in solver_eval:
                    if iter_key not in solver_iteration_data:
                        solver_iteration_data[iter_key] = []
                    solver_iteration_data[iter_key].append(solver_eval[iter_key])
    
    # Aggregate solver iteration data
    aggregated_solver_iterations = {}
    for iter_key, iter_data_list in solver_iteration_data.items():
        if iter_data_list:
            # Aggregate metrics across runs for this iteration
            all_metrics = {}
            for iter_data in iter_data_list:
                for metric in TABLE_1_COLUMNS:
                    if metric in iter_data:
                        if metric not in all_metrics:
                            all_metrics[metric] = []
                        all_metrics[metric].append(iter_data[metric])
            
            # Calculate mean and std values
            aggregated_iter = {}
            for metric, values in all_metrics.items():
                if values:
                    aggregated_iter[metric] = np.mean(values)
                    aggregated_iter[f"{metric}_std"] = np.std(values)
                else:
                    aggregated_iter[metric] = np.nan
                    aggregated_iter[f"{metric}_std"] = np.nan
            
            aggregated_solver_iterations[iter_key] = aggregated_iter
    
    # Define the methods and iterations to analyze
    methods_iterations = [
        ("predictor", None, " "),
        ("solver w. predictor", "solver_1", "k=1"),
        ("solver w. predictor", "solver_10", "k=10"),
        ("solver w. predictor", None, "converged / max iter")
    ]
    
    table_4_rows = []
    
    for method_name, solver_key, display_name in methods_iterations:
        row_data = {"Method": method_name, "Iteration": display_name}
        
        if method_name == "predictor":
            # Use predictor data
            source_data = aggregated_data["predictor"]
        else:
            # Use solver data
            if solver_key is not None:
                # Get data for specific iteration (e.g., solver_1, solver_10)
                source_data = aggregated_solver_iterations.get(solver_key, {})
            else:
                # Use final converged results (default solver data)
                source_data = aggregated_data["solver"]
        
        # Extract Table 1 metrics
        for col in TABLE_1_COLUMNS:
            if col in source_data:
                value = source_data[col]
                std_value = source_data.get(f"{col}_std", np.nan)
                
                # Handle percentage conversion for optimality gaps
                if "opt_gap_rel" in col and not pd.isna(value):
                    value *= 100
                    if not pd.isna(std_value):
                        std_value *= 100
                
                # Format value with std in parentheses, similar to Table 1
                if not pd.isna(value):
                    if not pd.isna(std_value):
                        row_data[col] = f"{value:.3f} ({std_value:.3f})"
                    else:
                        row_data[col] = f"{value:.3f}"
                else:
                    row_data[col] = "N/A"
            else:
                row_data[col] = "N/A"
        
        table_4_rows.append(row_data)
    
    # Create DataFrame
    df_table_4 = pd.DataFrame(table_4_rows)
    
    # Apply transformations to column names
    column_mapping = {col: TRANSFORMATION_DICT.get(col, col) for col in df_table_4.columns}
    df_table_4 = df_table_4.rename(columns=column_mapping)
    
    # Reorder columns to match Table 1 format
    base_columns = ["Method", "Iteration"]
    table_1_transformed = [TRANSFORMATION_DICT.get(col, col) for col in TABLE_1_COLUMNS]
    available_table_1_cols = [col for col in table_1_transformed if col in df_table_4.columns]
    
    # Reorder to match desired format: Method, Iteration, Max Opt Gap, Mean Opt Gap, Max Eq, Max Ineq, Mean Eq, Mean Ineq
    desired_order = base_columns + [
        r'Max. Opt. Gap(\%)',
        r'Opt. Gap(\%)', 
        'Max. Eq.',
        'Max. Ineq.',
        'Mean Eq.',
        'Mean Ineq.'
    ]
    
    final_columns = [col for col in desired_order if col in df_table_4.columns]
    df_table_4 = df_table_4[final_columns]
    
    print("\nTable 4 - Solver Iteration Analysis:")
    print(df_table_4.to_string(index=False))
    
    # Generate custom LaTeX table with single-row headers matching Table 1
    latex_lines = [
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        "Method &  & Max. Opt. Gap(\\%) & Opt. Gap(\\%) & Max. Eq. & Max. Ineq. & Mean Eq. & Mean Ineq. \\\\",
        "\\midrule"
    ]
    
    current_method = None
    for i, row in df_table_4.iterrows():
        method = row["Method"]
        iteration = row["Iteration"]
        
        # Get the data values (skip Method and Iteration columns)
        data_cols = [col for col in df_table_4.columns if col not in ["Method", "Iteration"]]
        values = [str(row[col]) for col in data_cols]
        
        if method != current_method:
            # New method group - use multirow
            current_method = method
            # Count how many iterations this method has
            method_count = sum(1 for _, r in df_table_4.iterrows() if r["Method"] == method)
            
            if method_count > 1:
                latex_lines.append(f"\\multirow[t]{{{method_count}}}{{*}}{{{method}}} & {iteration} & {' & '.join(values)} \\\\")
            else:
                latex_lines.append(f"{method} & {iteration} & {' & '.join(values)} \\\\")
        else:
            # Continuation of same method
            latex_lines.append(f" & {iteration} & {' & '.join(values)} \\\\")
        
        # Add cline after each method group (except the last one)
        if i < len(df_table_4) - 1 and df_table_4.iloc[i+1]["Method"] != method:
            latex_lines.append("\\cline{1-8}")
    
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}"
    ])
    
    save_latex_table(df_table_4, "table_4_solver_iteration_analysis.tex", latex_lines)
        
else:
    print("Missing predictor or solver data for Table 4")

# %% Generate Convergence Plot

print("\nGenerating Convergence Plot")

def plot_kkt_convergence_aggregated(all_results, ax=None, tol=None):
    """Plot aggregated KKT convergence over solver iterations with shaded areas."""
    # Collect trajectory evaluations from all solver runs
    all_trajectory_evals = []
    success_rates = []
    for case_name, case_results in all_results.items():
        if "solver" in case_results and case_results["solver"] is not None:
            solver_eval = case_results["solver"]["eval"]
            if "trajectory_evaluation" in solver_eval:
                all_trajectory_evals.append(solver_eval["trajectory_evaluation"])
                success_rates.append(solver_eval.get("success_rate"))
    success_rate_overall = np.mean(success_rates) if success_rates else None
    print(f"Overall solver success rate across runs: {success_rate_overall:.5}" if success_rate_overall is not None else "No success rate data available")
    if not all_trajectory_evals:
        print("No trajectory evaluation data found for convergence plot")
        return None, None
    
    # Find the maximum number of iterations across all runs
    max_iterations = max(len(traj_eval) for traj_eval in all_trajectory_evals)
    
    # Initialize arrays to store aggregated metrics
    metrics_to_plot = ["KKT_inf_max", "KKT_inf_99", "KKT_inf_95", "KKT_inf_90", "KKT_inf_med"]
    # metrics_to_plot = ["Tk_max", "Tk_99", "Tk_95", "Tk_90", "Tk_med"]
    metric_labels = ["max", "99th perc.", "95th perc.", "90th perc.", "50th perc."]
    
    aggregated_metrics = {}
    for metric in metrics_to_plot:
        aggregated_metrics[metric] = {
            "median": [],
            "min": [],
            "max": []
        }
    
    # Aggregate metrics across runs for each iteration
    for iteration in range(max_iterations):
        for metric in metrics_to_plot:
            values_at_iteration = []
            
            # Collect values from all runs at this iteration
            for traj_eval in all_trajectory_evals:
                if iteration < len(traj_eval) and metric in traj_eval[iteration]:
                    values_at_iteration.append(traj_eval[iteration][metric])
            
            if values_at_iteration:
                # aggregated_metrics[metric]["mean"].append(np.mean(values_at_iteration))
                aggregated_metrics[metric]["median"].append(np.median(values_at_iteration))
                aggregated_metrics[metric]["min"].append(np.min(values_at_iteration))
                aggregated_metrics[metric]["max"].append(np.max(values_at_iteration))
            else:
                # Fill with NaN if no data available
                aggregated_metrics[metric]["median"].append(np.nan)
                aggregated_metrics[metric]["min"].append(np.nan)
                aggregated_metrics[metric]["max"].append(np.nan)
    
    # Create plot
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure
    
    iterations = np.arange(max_iterations) + 1
    
    # Plot median lines with shaded areas for min/max
    # colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        median_values = np.array(aggregated_metrics[metric]["median"])
        min_values = np.array(aggregated_metrics[metric]["min"])
        max_values = np.array(aggregated_metrics[metric]["max"])
        
        # Filter out NaN values for plotting
        valid_indices = ~np.isnan(median_values)
        if np.any(valid_indices):
            valid_iterations = iterations[valid_indices]
            valid_median = median_values[valid_indices]
            valid_min = min_values[valid_indices]
            valid_max = max_values[valid_indices]
            
            # Plot median line
            ax.plot(valid_iterations, valid_median, 
                #    color=colors[i % len(colors)], 
                   label=label)
            
            # Add shaded area for min/max
            ax.fill_between(valid_iterations, valid_min, valid_max, 
                        #    color=colors[i % len(colors)], 
                           alpha=0.2)
    
    # Add tolerance line if specified
    if tol is not None:
        ax.axhline(tol, color='red', linestyle='--', linewidth=0.3, 
                  label=f'Tolerance: {tol:.1e}')
    
    # Configure plot
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$||KKT||_{\infty}$")
    ax.set_title(r"$\infty$-Norm of KKT conditions over iterations")
    # ax.set_title(r"$\infty$-Norm of KKT conditions over iterations (Parametric QP)")
    ax.legend()
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim([1e-12, 1e2])
    # ax.set_ylim([1e-12, 1e2])
    ax.set_xlim([1, max_iterations])
    ax.grid(True, which="major", linestyle="--", linewidth=0.3, axis='y')
    ax.grid(True, which="both", linestyle="--", linewidth=0.3, axis='x')
    # ax.grid(True, which="both", linestyle="--", linewidth=0.3)
    # ax.grid(True, which="major", linestyle="--", linewidth=0.3)
    
    # Add success rate text box in top right corner
    if success_rate_overall is not None:
        success_pct = success_rate_overall * 100
        success_text = f'{success_pct:.2f}% solved to tolerance'
        ax.text(0.98, 0.98, success_text, 
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    return fig, ax

# Generate convergence plot
if all_results:
    fig_convergence, ax_convergence = plot_kkt_convergence_aggregated(all_results, tol=TOL)
    
    if fig_convergence is not None:
        save_path = RESULTS_PATH.joinpath(SAVE_FOLDER)
        save_path.mkdir(parents=True, exist_ok=True)
        # Save as both PNG and PDF
        convergence_plot_path_png = save_path.joinpath("kkt_convergence_nonconvex_qp.png")
        convergence_plot_path_pdf = save_path.joinpath("kkt_convergence_nonconvex_qp.pdf")
        fig_convergence.savefig(convergence_plot_path_png, bbox_inches='tight', dpi=300)
        fig_convergence.savefig(convergence_plot_path_pdf, bbox_inches='tight', dpi=300)
        print(f"Saved convergence plot to {convergence_plot_path_png}")
        print(f"Saved convergence plot to {convergence_plot_path_pdf}")
        
        plt.close(fig_convergence)
        print("Generated convergence plot with aggregated data from 5 problem instances")
    else:
        print("Failed to generate convergence plot - no trajectory data found")
else:
    print("No results available for convergence plot")

# %% Save Raw Data

save_path = RESULTS_PATH.joinpath(SAVE_FOLDER)
json_path = save_path.joinpath("aggregated_results.json")

# Convert numpy types to native Python types for JSON serialization
json_data = {method: {k: float(v) if not pd.isna(v) else None 
                        for k, v in data.items()}
            for method, data in aggregated_data.items()}

with open(json_path, 'w') as f:
    json.dump(json_data, f, indent=2)
print(f"Saved aggregated data to {json_path}")

# %%

# aggregate all the speedup factors in a list (see above, table 3) and return a histogram plot
def plot_speedup_histogram(speedup_factors: List[float], ax=None) -> None:
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
    ax.set_ylim(0,500)
    
    # Set x-axis ticks starting at 0 with increments of 1
    max_speedup = max(speedup_factors)
    ax.set_xticks(np.arange(0, int(max_speedup) + 2, 1))
    
    ax.legend(loc="center left")
    
    # Optional: Add text annotation showing percentage of cases with speedup > 1
    n_speedup = sum(1 for s in speedup_factors if s > 1)
    pct_speedup = 100 * n_speedup / len(speedup_factors)
    # ax.text(0.98, 0.98, f'{pct_speedup:.1f}% faster than IPOPT', 
    #         transform=ax.transAxes, ha='right', va='top',
    #         bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))
    ax.text(0.98, 0.98, f'{pct_speedup:.1f}% faster than IPOPT', 
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return fig, ax

# Generate speedup histogram
print("\nGenerating Speedup Histogram")

if len(all_speedup_values) > 0:
    fig_speedup, ax_speedup = plot_speedup_histogram(all_speedup_values)
    
    save_path = RESULTS_PATH.joinpath(SAVE_FOLDER)
    save_path.mkdir(parents=True, exist_ok=True)
    # Save as both PNG and PDF
    speedup_plot_path_png = save_path.joinpath("speedup_histogram_nonconvex_qp.png")
    speedup_plot_path_pdf = save_path.joinpath("speedup_histogram_nonconvex_qp.pdf")
    fig_speedup.savefig(speedup_plot_path_png, bbox_inches='tight', dpi=300)
    fig_speedup.savefig(speedup_plot_path_pdf, bbox_inches='tight', dpi=300)
    print(f"Saved speedup histogram to {speedup_plot_path_png}")
    print(f"Saved speedup histogram to {speedup_plot_path_pdf}")
    
    plt.close(fig_speedup)
    print("Generated speedup histogram")

# Table with Hyperparameters + train time
print("\n\nTable 5: Hyperparameters + Train Time")
print("=" * 80)

def extract_hyperparameters_aggregated(all_results):
    """Extract and aggregate hyperparameters for all methods across runs."""
    hyperparams_data = {}
    
    # Aggregate Predictor hyperparameters
    predictor_configs = []
    predictor_train_times = []
    
    for case_name, case_results in all_results.items():
        if "predictor" in case_results and case_results["predictor"] is not None:
            config = case_results["predictor"]["config"]
            predictor_configs.append(config)
            train_time = config.get("train_time", np.nan)
            if not pd.isna(train_time):
                predictor_train_times.append(train_time)
    
    if predictor_configs:
        # Check that all hyperparameters are the same (except training time)
        first_config = predictor_configs[0]
        train_cfg = first_config.get("train_cfg", {})
        model_cfg = first_config.get("model_cfg", {})
        
        # Verify consistency across runs
        for config in predictor_configs[1:]:
            config_train_cfg = config.get("train_cfg", {})
            config_model_cfg = config.get("model_cfg", {})
            
            # Check if key hyperparameters are consistent
            for key in ["n_hidden_layers", "n_neurons"]:
                if model_cfg.get(key) != config_model_cfg.get(key):
                    raise ValueError(f"Inconsistent {key} in predictor configs")
            
            for key in ["batch_size", "N_epochs", "lr"]:
                if train_cfg.get(key) != config_train_cfg.get(key):
                    raise ValueError(f"Inconsistent {key} in predictor configs")
        
        # Calculate mean training time
        mean_train_time_min = np.mean(predictor_train_times) / 60 if predictor_train_times else np.nan
        train_time_str = f"{mean_train_time_min:.2f}" if not pd.isna(mean_train_time_min) else "N/A"
        
        batch_size = train_cfg.get("batch_size", np.nan)
        batch_size_str = f"{int(batch_size)}" if not pd.isna(batch_size) else "N/A"
        # batch_size_str = f"{int(batch_size)} (1)" if not pd.isna(batch_size) else "N/A"
        
        hyperparams_data["Predictor"] = {
            "Hidden Layers": model_cfg.get("n_hidden_layers", np.nan),
            "Neurons per Layer": model_cfg.get("n_neurons", np.nan),
            # "Training Dataset Size": "-",
            "Batch Size": batch_size_str,
            # "Batch Size (Batches per Epoch)": batch_size_str,
            "Epochs": train_cfg.get("N_epochs", np.nan),
            "Learning Rate": train_cfg.get("lr", np.nan),
            "Avg. Training Time [min]": train_time_str,
            # "Training Time [min]": train_time_str
        }
    
    # Aggregate Solver hyperparameters
    solver_configs = []
    solver_train_times = []
    
    for case_name, case_results in all_results.items():
        if "solver" in case_results and case_results["solver"] is not None:
            config = case_results["solver"]["config"]
            solver_configs.append(config)
            train_time = config.get("train_time", np.nan)
            if not pd.isna(train_time):
                solver_train_times.append(train_time)
    
    if solver_configs:
        # Check that all hyperparameters are the same (except training time)
        first_config = solver_configs[0]
        train_cfg = first_config.get("train_cfg", {})
        model_cfg = first_config.get("model_cfg", {})
        
        # Verify consistency across runs
        for config in solver_configs[1:]:
            config_train_cfg = config.get("train_cfg", {})
            config_model_cfg = config.get("model_cfg", {})
            
            # Check if key hyperparameters are consistent
            for key in ["n_hidden_layers", "n_neurons"]:
                if model_cfg.get(key) != config_model_cfg.get(key):
                    raise ValueError(f"Inconsistent {key} in solver configs")
            
            for key in ["batch_size", "N_epochs", "lr"]:
                if train_cfg.get(key) != config_train_cfg.get(key):
                    raise ValueError(f"Inconsistent {key} in solver configs")
        
        # Calculate mean training times
        mean_solver_train_time_min = np.mean(solver_train_times) / 60 if solver_train_times else np.nan
        mean_predictor_train_time_min = np.mean(predictor_train_times) / 60 if predictor_train_times else np.nan
        
        # Format training time with combined time in parentheses
        if not pd.isna(mean_solver_train_time_min):
            if not pd.isna(mean_predictor_train_time_min):
                combined_time_min = mean_solver_train_time_min + mean_predictor_train_time_min
                train_time_str = f"{mean_solver_train_time_min:.2f} ({combined_time_min:.2f})"
            else:
                train_time_str = f"{mean_solver_train_time_min:.2f} (N/A)"
        else:
            train_time_str = "N/A"
        
        batch_size = train_cfg.get("batch_size", np.nan)
        batch_size_str = f"{int(batch_size)}" if not pd.isna(batch_size) else "N/A"
        # batch_size_str = f"{int(batch_size)} (1)" if not pd.isna(batch_size) else "N/A"
        
        hyperparams_data["Solver w. Predictor"] = {
            "Hidden Layers": model_cfg.get("n_hidden_layers", np.nan),
            "Neurons per Layer": model_cfg.get("n_neurons", np.nan),
            # "Training Dataset Size": "-",
            "Batch Size": batch_size_str,
            # "Batch Size (Batches per Epoch)": batch_size_str,
            "Epochs": train_cfg.get("N_epochs", np.nan),
            "Learning Rate": train_cfg.get("lr", np.nan),
            "Avg. Training Time [min]": train_time_str
        }
    
    return hyperparams_data

hyperparams_data = extract_hyperparameters_aggregated(all_results)
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
    
    # Save to LaTeX
    save_latex_table(hyperparams_formatted, "table_5_hyperparameters.tex")
else:
    print("No hyperparameter data available")
