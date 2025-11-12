# %%
# Init
import json
import numpy as np
from pathlib import Path
from nonlinear_double_integrator import NonlinearDoubleIntegrator, NLP
FILE_PTH = Path(__file__).parent.resolve()
append_cfg = False

# %% - Functions (for Sweep)
def number_of_cfgs(model_cfg,train_cfg,nlp_cfg,mode_cfg):
    # check all entries of model_cfg and train_cfg and check whether they are lists
    # if they are lists, return the number of elements, else 1, and store in a list
    n_model_cfgs = []
    for key in model_cfg.keys():
        if isinstance(model_cfg[key],list):
            n_model_cfgs.append(len(model_cfg[key]))
        else:
            n_model_cfgs.append(1)
    n_train_cfgs = []
    for key in train_cfg.keys():
        if isinstance(train_cfg[key],list):
            n_train_cfgs.append(len(train_cfg[key]))
        else:
            n_train_cfgs.append(1)
    n_nlp_cfgs = []
    for key in nlp_cfg.keys():
        if isinstance(nlp_cfg[key],list):
            n_nlp_cfgs.append(len(nlp_cfg[key]))
        else:
            n_nlp_cfgs.append(1)
    n_mode_cfgs = []
    for key in mode_cfg.keys():
        if isinstance(mode_cfg[key],list):
            n_mode_cfgs.append(len(mode_cfg[key]))
        else:
            n_mode_cfgs.append(1)
    # multiply all values in the list
    n_model_cfgs = np.prod(n_model_cfgs)
    n_train_cfgs = np.prod(n_train_cfgs)
    n_nlp_cfgs = np.prod(n_nlp_cfgs)
    n_mode_cfgs = np.prod(n_mode_cfgs)
    return n_model_cfgs*n_train_cfgs*n_nlp_cfgs*n_mode_cfgs

def expand_dict_combinations(input_dict):
    """
    Generate all possible combinations of dictionaries from an input dictionary
    where some values are lists.
    
    For list values, each item in the list will generate a new dictionary.
    For non-list values, the value remains the same across all combinations.
    
    Args:
        input_dict (dict): Dictionary with string and list values
        
    Returns:
        list: List of dictionaries with all possible combinations
    """
    # Initialize with an empty dictionary
    result = [{}]
    
    # Iterate through each key-value pair in the input dictionary
    for key, value in input_dict.items():
        # If the value is a list, create combinations
        if isinstance(value, list):
            new_result = []
            # For each existing partial result
            for partial_dict in result:
                # For each value in the list
                for item in value:
                    # Create a new dictionary
                    new_dict = partial_dict.copy()
                    # Add the current key with the specific list item
                    new_dict[key] = item
                    new_result.append(new_dict)
            # Replace the current result with the new combinations
            result = new_result
        else:
            # For non-list values, just add to all existing dictionaries
            for partial_dict in result:
                partial_dict[key] = value
    
    return result

def combine_dict_lists(model_cfgs, train_cfgs, nlp_cfgs, mode_cfgs, mode):
    """
    Combines four lists of dictionaries to create all possible dictionary pairs.
    
    Args:
        model_cfgs (list): First list of dictionaries
        train_cfgs (list): Second list of dictionaries
        nlp_cfgs (list): Third list of dictionaries
        mode_cfgs (list): Fourth list of dictionaries (predictor_cfg or solver_cfg)
        mode (str): Mode identifier ("predictor" or "solver")
        
    Returns:
        list: List of dictionaries with all possible combinations
    """
    # Initialize an empty result list
    cfg_combinations = []
    
    # Create all possible combinations
    for dict1 in model_cfgs:
        for dict2 in train_cfgs:
            for dict3 in nlp_cfgs:
                for dict4 in mode_cfgs:
                    result = {}
                    result["mode"] = mode
                    result["model_cfg"] = dict1
                    result["train_cfg"] = dict2
                    if mode == "predictor":
                        result["nlp_cfg"] = dict3
                        result["predictor_cfg"] = dict4
                    elif mode == "solver":
                        result["nlp_cfg"] = dict3
                        result["solver_cfg"] = dict4
                    elif mode == "approxMPC":
                        result["approxMPC_cfg"] = dict4
                    cfg_combinations.append(result)
    return cfg_combinations

def generate_cfgs(model_cfg,train_cfg,nlp_cfg,mode_cfg,mode):
    n_total_cfgs = number_of_cfgs(model_cfg,train_cfg,nlp_cfg,mode_cfg)
    print("-"*20)
    assert n_total_cfgs < 20, "Too many configurations to generate.. Are you sure?"
    print(f"Generating {n_total_cfgs} configurations..")
    print("-"*20)
    model_cfgs = expand_dict_combinations(model_cfg)
    train_cfgs = expand_dict_combinations(train_cfg)
    nlp_cfgs = expand_dict_combinations(nlp_cfg)
    mode_cfgs = expand_dict_combinations(mode_cfg)
    cfgs = combine_dict_lists(model_cfgs,train_cfgs,nlp_cfgs,mode_cfgs,mode)
    return cfgs

def remove_duplicate_dicts(dict_list):
    """
    Removes duplicate dictionaries from a list of dictionaries.
    
    Args:
        dict_list (list): List of dictionaries to process
        
    Returns:
        list: List with duplicate dictionaries removed
    """
    # Use a set to track unique dictionaries
    seen = set()
    result = []
    
    for d in dict_list:
        # Convert dictionary to a hashable representation
        # Sort the items to ensure consistent ordering
        hashable_dict = tuple(sorted((k, str(v)) for k, v in d.items()))
        
        # If we haven't seen this dictionary before, add it to the result
        if hashable_dict not in seen:
            seen.add(hashable_dict)
            result.append(d)

    print("-"*20)
    print(f"Removed {len(dict_list) - len(result)} duplicate configurations.")
    print(f"Kept {len(result)} unique configurations.")
    print("-"*20)
    
    return result

def save_cfgs(cfgs,cfg_name,append_cfg=False):
    cfg_file_pth = FILE_PTH.joinpath(cfg_name)
    pth_exists = cfg_file_pth.exists()

    if not pth_exists:
        cfgs = remove_duplicate_dicts(cfgs) # remove duplicates
        with open(cfg_file_pth, 'w') as f:
            json.dump(cfgs, f, indent=4)
        print(f"Saved {cfg_name}.")
        return cfgs
    else:
        if append_cfg:
            with open(cfg_file_pth, 'r') as f:
                prev_configs = json.load(f)
            extended_cfgs = prev_configs + cfgs
            print(f"Appended to {cfg_name}.")
            extended_cfgs = remove_duplicate_dicts(extended_cfgs) # remove duplicates
            with open(cfg_file_pth, 'w') as f:
                json.dump(extended_cfgs, f, indent=4)
            return extended_cfgs
        else:
            raise FileExistsError(f"{cfg_name} already exists. Set append_cfg to True to append to existing config.")

# %% - NLP Config
system = NonlinearDoubleIntegrator()
nlp_cfg = {
    "N": 10,
    "eps": 1e-16,
    "sigma": 1e-2,
}
nlp = NLP(system, **nlp_cfg)

# %% - Predictor Configs
model_cfg = {
    "n_in": nlp.n_p,
    "n_out": nlp.n_z,
    "n_hidden_layers": 4,
    "n_neurons": 512,
    "act_fn": "gelu",
    "output_act_fn": "linear",
}

train_cfg = {
    "N_epochs": 50000,
    "batch_size": 4096,
    "weight_decay": 0.0,
    "lr": 1e-3,
    "use_amsgrad": True,
}

predictor_cfg = {
    "convexification": True,
    "torch_compiled": True,
}

predictor_cfgs = generate_cfgs(model_cfg,train_cfg,nlp_cfg,predictor_cfg,"predictor")
cfg_name = "predictor_cfgs.json"
cfgs = save_cfgs(predictor_cfgs,cfg_name,append_cfg)

# %% - Solver Configs (with and without Predictor)
predictor_pth = str(Path("results","predictor","exp_0"))

model_cfg = {
    "n_in": nlp.n_p+nlp.n_z+1,
    "n_out": nlp.n_z,
    "n_hidden_layers": 4,
    "n_neurons": 512,
    "act_fn": "gelu",
    "output_act_fn": "linear",
}

train_cfg = {
    "N_epochs": 500000,
    "batch_size": 4096,
    "weight_decay": 0.0,
    "lr": 1e-3,
    "use_amsgrad": True,
    "n_steps_max_total": 2000,
    "n_steps_max_no_improvement": 5,
    "Tk_lim": 1e-16,
    "predictor_pth": [predictor_pth, None],
}

solver_cfg = {
    "convexification":True,
    "gamma":0.1,
    "torch_compiled":True,
}

solver_cfgs = generate_cfgs(model_cfg,train_cfg,nlp_cfg,solver_cfg,"solver")
cfg_name = "solver_cfgs.json"
cfgs = save_cfgs(solver_cfgs,cfg_name,append_cfg)

# %% Approx MPC configs
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
        "x_lb": system.x_lb.numpy().tolist(),
        "x_ub": system.x_ub.numpy().tolist(),
        "u_lb": system.u_lb.numpy().tolist(),
        "u_ub": system.u_ub.numpy().tolist()
    },
    "torch_compiled": True,
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

approxMPC_cfgs = generate_cfgs(model_cfg,train_cfg,nlp_cfg,approxMPC_cfg,"approxMPC")
cfg_name = "approxMPC_cfgs.json"
cfgs = save_cfgs(approxMPC_cfgs,cfg_name,append_cfg)