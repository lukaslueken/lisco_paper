import torch
from pathlib import Path
from parametric_OP import NLP

def generate_problem(op_dict, op_pth, overwrite=False):
    """Generate a parametric optimization problem based on provided specification.
    
    Args:
        op_dict (dict): Dictionary containing problem specifications
        op_pth (Path): Base path to save the problem
        overwrite (bool, optional): Whether to overwrite existing problems. Defaults to False.
        
    Returns:
        tuple: (success flag, problem path)
    """
    idx = op_dict["idx"]
    # Setup OP to populate parameters Q,p,A,G,h
    op = NLP(
        obj_type=op_dict["obj_type"],
        n_vars=op_dict["n_vars"],
        n_eq=op_dict["n_eq"],
        n_ineq=op_dict["n_ineq"]
    )

    op_name = f'{op.obj_type}_{op.n_vars}x{op.n_eq}x{op.n_ineq}_{idx}'
    folder_pth = op_pth.joinpath(op_name)
    
    if folder_pth.exists() and not overwrite:
        print(f"Parametric OP {op_name} already exists. Skipping...")
        return False, folder_pth
    
    print(f"Generating {op_name}...")
    folder_pth.mkdir(parents=True, exist_ok=True)
    
    # Save problem configuration
    op.save_config(folder_path=folder_pth, file_name="op_cfg")
    
    print(f"Parametric OP {op_name} saved successfully.")
    return True, folder_pth


if __name__ == "__main__":
    # Configuration
    n_problems = 5 # number of different problem instances to generate
    obj_type = "nonconvex"
    if obj_type != "nonconvex":
        raise NotImplementedError("Only 'nonconvex' objective type is implemented.")
    op_dims = [{"n_vars": 100, "n_eq": 50, "n_ineq": 50}]
    folder_name = "data"
    overwrite_existing = False

    # Set default dtype and device
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    
    print(f"Using device: {device}")

    # Setup paths
    file_pth = Path(__file__).parent.resolve()
    op_pth = file_pth.joinpath(folder_name)
    op_pth.mkdir(parents=True, exist_ok=True)
    
    # Generate problem specifications
    op_dicts = []
    for idx in range(n_problems):
        for op_dim in op_dims:
            op_dicts.append({"obj_type": obj_type, **op_dim, "idx": idx})
    
    print(f"Will generate {len(op_dicts)} parametric optimization problems")
    
    # Generate optimization problems
    generated_problems = []
    for i, op_dict in enumerate(op_dicts):
        print(f"Problem {i+1}/{len(op_dicts)}")
        success, folder_pth = generate_problem(op_dict, op_pth, overwrite=overwrite_existing)
        if success or overwrite_existing:
            generated_problems.append(op_dict)
        print("------------------------------------------")
    print(f"Process completed. Generated {len(generated_problems)} problems.")
