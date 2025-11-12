# %% Imports
import torch
import time
import copy
import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


# %% Functions
def get_activation_layer(act_fn):
    if act_fn == 'relu':
        return torch.nn.ReLU()
    elif act_fn == 'tanh':
        return torch.nn.Tanh()
    elif act_fn == 'leaky_relu':
        return torch.nn.LeakyReLU()
    elif act_fn == 'sigmoid':
        return torch.nn.Sigmoid()
    elif act_fn == "gelu":
        return torch.nn.GELU()
    else:
        raise ValueError("Activation function not implemented.")
    
def generate_experiment_dir(log_dirs=["runs"],save_dir=None):
    # log_dirs: where to save experiments (e.g. "runs", "predictor")
    # save_dir: where to save experiments (top-level) / if none, save in current directory
    for i in range(1000):
        local_pth = Path(*log_dirs,f"exp_{i}")
        if not local_pth.exists():
            break
        if i == 999:
            raise ValueError("Experiment directory could not be generated.")
    if save_dir is None:
        exp_pth = Path(__file__).parent.absolute()
    else:
        exp_pth = Path(save_dir)
    exp_pth = exp_pth.joinpath(local_pth)
    exp_pth.mkdir(parents=True,exist_ok=True)
    return exp_pth

@torch.no_grad()
def count_params(net):
    n_params = sum([param.numel() for param in net.parameters()])
    return n_params

def export_jit_cpu(model):
    mod_cpu = copy.deepcopy(model).cpu()
    mod_cpu = torch.jit.script(mod_cpu)
    mod_cpu = torch.jit.optimize_for_inference(mod_cpu)
    return mod_cpu

def export_jit_gpu(model):
    mod_gpu = copy.deepcopy(model).cuda()
    mod_gpu = torch.jit.script(mod_gpu)
    mod_gpu = torch.jit.optimize_for_inference(mod_gpu)
    return mod_gpu


# %% Classes

# FeedforwardNN
class FeedforwardNN(torch.nn.Module):
    """Feedforward Neural Network model.

    Args:
        n_in (int): Number of input features.
        n_out (int): Number of output features.
        n_hidden_layers (int): Number of hidden layers.
        n_neurons (int): Number of neurons in each hidden layer.
        act_fn (str): Activation function.
        output_act_fn (str): Output activation function.
    """
    def __init__(self, n_in, n_out, n_hidden_layers=2, n_neurons=500, act_fn='relu', output_act_fn='linear',nn_dtype=torch.float32):
        super().__init__()
        assert n_hidden_layers > 0, "Number of hidden layers must be > 0."
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden_layers = n_hidden_layers
        self.n_layers = n_hidden_layers+1
        self.n_neurons = n_neurons
        self.nn_dtype = nn_dtype
        self.act_fn = act_fn
        self.output_act_fn = output_act_fn
        self.layers = torch.nn.ModuleList()

        # add hidden layers
        for i in range(self.n_hidden_layers):
            if i == 0:
                self.layers.append(torch.nn.Linear(n_in, n_neurons,dtype=self.nn_dtype))
                self.layers.append(get_activation_layer(act_fn))
            else:
                self.layers.append(torch.nn.Linear(n_neurons, n_neurons,dtype=self.nn_dtype))
                self.layers.append(get_activation_layer(act_fn))

        # add output layer
        self.layers.append(torch.nn.Linear(n_neurons, n_out,dtype=self.nn_dtype))
        if output_act_fn != 'linear':
            self.layers.append(get_activation_layer(output_act_fn))

    def forward(self, x):
        x = x.to(self.nn_dtype)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

# Solver
class Solver(torch.nn.Module):
    def __init__(self,model,nlp,predictor=None,convexification=False,gamma=0.1,torch_compiled=True):
        super().__init__()
        self.model = model #step model
        self.nlp = nlp
        self.predictor = predictor
        self.torch_compiled = torch_compiled # solver_cfg
        self.gamma = gamma # solver_cfg
        self.convexification = convexification # solver_cfg
        self.offset = 1e-16
        self.initialize_functions()
        print("----------------------------------")
        print(self)
        print("----------------------------------")

    def initialize_functions(self):
        # Select Tk function based on convexification flag
        if self.convexification:
            print("Using convexified Tk function for solver.\n")
            Tk_batch_func_train = self.nlp.Tk_conv_batch_func
        else:
            Tk_batch_func_train = self.nlp.Tk_batch_func
        Tk_batch_func_eval = self.nlp.Tk_batch_func
        
        # Torch compile functions if enabled
        if self.torch_compiled:
            torch.compiler.reset()
            print("Torch compiler reset.\n")
            self.forward = torch.compile(self._forward,mode="max-autotune",fullgraph=True)
            self.Tk_batch_func_train = torch.compile(copy.deepcopy(Tk_batch_func_train),mode="max-autotune",fullgraph=True)
            self.Tk_batch_func_eval = torch.compile(copy.deepcopy(Tk_batch_func_eval),mode="max-autotune",fullgraph=True)
            self.training_step = torch.compile(self._training_step, mode="max-autotune")
            print("Solver functions torch-compiled.\n")
        else:
            self.forward = self._forward
            self.Tk_batch_func_train = Tk_batch_func_train
            self.Tk_batch_func_eval = Tk_batch_func_eval
            self.training_step = self._training_step
        
        print("Solver functions initialized.")

    def _forward(self,zk,p):
        """Predict step direction for given batch of primal-dual iterates and parameters."""
        # full KKT conditions for solver framework
        # KKT_batch = self.nlp.KKT_batch_func(zk,p)
        KKT_batch = self.nlp.Fk_batch_func(zk,p)

        # 2-norm of modified KKT conditions and log-scaled 2-norm (norm_KKT_batch, norm_KKT_batch_log)
        norm_KKT_batch = (torch.linalg.vector_norm(KKT_batch,ord=2,dim=1)+self.offset).unsqueeze(-1)
        norm_KKT_batch_log = torch.log(norm_KKT_batch)

        # Normalize KKT_batch (KKT_batch_normalized)
        KKT_batch_normalized = torch.divide(KKT_batch,norm_KKT_batch)

        # Stack NN inputs: parameters of nlp, normalized KKT_batch, log-scaled 2-norm of KKT_batch
        nn_inputs_batch = torch.hstack((p,KKT_batch_normalized,norm_KKT_batch_log))
        
        # clip norm_KKT_batch to avoid nan gradients
        norm_KKT_batch = torch.clamp(norm_KKT_batch,max=1.0)

        # Predict step direction
        dzk = self.model(nn_inputs_batch)*norm_KKT_batch
        
        return dzk*self.gamma

    def _training_step(self, zk_batch, p_batch, optim):
        # Zero Grads
        optim.zero_grad(set_to_none=True)
        
        # Predict step
        dzk_batch = self.forward(zk_batch, p_batch)

        # Compute new iterates
        zk_new = zk_batch + dzk_batch

        # Evaluate per-sample loss
        Tk_batch = self.Tk_batch_func_train(zk_new, p_batch)

        # Compute loss
        loss = torch.mean(torch.log(Tk_batch + self.offset**2))
        # loss = torch.mean(torch.log1p(Tk_batch))

        # Backprop
        loss.backward()

        # Step
        optim.step()
        
        return loss, zk_new, Tk_batch
    
    def sample_z_p(self,n_samples):
        p_batch = self.nlp.batch_gen_p(n_samples,offset=0.0)
        if self.predictor is None:
            zk_batch = self.nlp.batch_gen_z(n_samples)
        else:
            zk_batch = self.predictor.predict(p_batch).to(self.nlp.dtype)
            # raise NotImplementedError("Resampling with predictor not implemented yet.")
        return zk_batch, p_batch

    def train(self,train_config,print_frequency=100,log_values=True):
        # Reset torch compiler
        # if self.torch_compiled:
        #     torch.compiler.reset()
        #     print("Torch compiler reset.\n")
        # CONFIG    
        N_epochs = train_config.get("N_epochs",1000)
        batch_size = train_config.get("batch_size",None)
        weight_decay = train_config.get("weight_decay",0.0)
        lr = train_config.get("lr",1e-4)
        use_amsgrad = train_config.get("use_amsgrad",True)
        # update_mode = train_config.get("update_mode","full")
        resampling_strategy = train_config.get("resampling_strategy","all")
        Tk_lim = train_config.get("Tk_lim",1e-6)
        n_steps_max_total = train_config.get("n_steps_max_total",None)
        n_steps_max_no_improvement = train_config.get("n_steps_max_no_improvement",None)

        # Train logger
        train_logger = TrainLogger(device=self.nlp.device)

        # Optimizer initialization
        optim = torch.optim.AdamW(self.parameters(),lr=lr,weight_decay=weight_decay,amsgrad=use_amsgrad)

        # Start train timer
        train_tic = time.perf_counter()

        # Initial batch generation
        zk_batch, p_batch = self.sample_z_p(batch_size)
        
        # Initial evaluations
        Tk_best = self.Tk_batch_func_eval(zk_batch,p_batch)
        Tk_best = Tk_best.detach().clone()
        # T0_batch = T0_batch.detach().clone()
        n_steps_batch = torch.zeros(batch_size)
        n_steps_no_improv = torch.zeros(batch_size)

        for epoch in range(N_epochs):                
            # Training step
            loss, zk_batch, Tk_batch = self.training_step(zk_batch, p_batch, optim)
            zk_batch = zk_batch.detach().clone()

            with torch.no_grad():
                # update n_steps_batch and n_steps_no_improv
                n_steps_batch += 1
                # update Tk_best
                mask_best = Tk_batch < Tk_best
                Tk_best[mask_best] = Tk_batch[mask_best]
                n_steps_no_improv[~mask_best] += 1
                n_steps_no_improv[mask_best] = 0

                # Create boolean mask for resampling conditions
                mask_tol = Tk_batch < Tk_lim
                
                # Build max iteration mask based on strategy
                if n_steps_max_total is not None:
                    mask_max_total = n_steps_batch >= n_steps_max_total
                else:
                    mask_max_total = torch.zeros_like(mask_tol, dtype=torch.bool)
                
                if n_steps_max_no_improvement is not None:
                    mask_max_no_improv = n_steps_no_improv >= n_steps_max_no_improvement
                else:
                    mask_max_no_improv = torch.zeros_like(mask_tol, dtype=torch.bool)
                
                # Combine all masks with logical OR
                mask_resample = mask_tol | mask_max_total | mask_max_no_improv
                
                # Resample where necessary
                n_resample = torch.sum(mask_resample)
                n_resample_tol = torch.sum(mask_tol)
                n_resample_iter = n_resample - n_resample_tol
                if n_resample > 0:
                    if resampling_strategy == "all":
                        z_new, p_new = self.sample_z_p(n_resample)
                        zk_batch[mask_resample] = z_new
                        p_batch[mask_resample] = p_new
                        Tk_best[mask_resample] = float('inf')
                        n_steps_batch[mask_resample] = 0
                        n_steps_no_improv[mask_resample] = 0
                    else:
                        raise NotImplementedError("Resampling strategy not implemented yet.")

                # Print
                if (epoch+1) % print_frequency == 0:
                    print("Epoch {0}: Loss = {1:.3e}".format(epoch+1, loss))
                    print("-------------------------------")
                
                # Log values
                if log_values:
                    train_logger.log_value_buffered(epoch+1, "epoch")
                    train_logger.log_value_buffered(loss, "loss")
                    train_logger.log_value_buffered(n_resample, "n_resample")
                    train_logger.log_value_buffered(n_resample_tol, "n_resample_tol")
                    train_logger.log_value_buffered(n_resample_iter, "n_resample_iter")

        train_toc = time.perf_counter()
        train_time = train_toc - train_tic
        if log_values:
            train_logger.log_data(train_time, "train_time")
            train_logger.log_array(p_batch.detach().cpu(), "p_final_batch")
            train_logger.flush_all_buffers()

        print("#----------------------------------#")
        print("Training complete.")
        print("Total number of epochs: {0}".format(epoch+1))
        print("Total training time: {0:.2f} sec".format(train_time))
        print("#----------------------------------#")

        return train_logger
    
    @torch.no_grad()
    def solve_batch(self,p_batch,max_iter=100,alpha=1.0,predictor=None,update_mode="full",return_trajectory=False,compile_tk_func=False,compile_step_func=False):
        if compile_tk_func or compile_step_func:
            print("Compiling solver functions for evaluation...")
            torch.compiler.reset()
        # Compile Tk_batch_func if requested
        if compile_tk_func:
            Tk_batch_func = torch.compile(copy.deepcopy(self.nlp.Tk_batch_func), mode="max-autotune", fullgraph=True)
        else:
            Tk_batch_func = self.nlp.Tk_batch_func
        
        # Compile step function if requested
        if compile_step_func:
            step_func = torch.compile(self.forward, mode="max-autotune", fullgraph=True)
        else:
            step_func = self.forward
        
        alpha_batch = torch.ones(p_batch.shape[0],device=p_batch.device)*alpha
        # initial batch
        n_batch = p_batch.shape[0]
        step_time_list = []
        if predictor is None:
            zk_batch = self.nlp.batch_gen_z(n_batch)
        else:
            pred_tic = time.perf_counter()
            zk_batch = predictor.predict(p_batch).to(p_batch.dtype)
            pred_toc = time.perf_counter()
            pred_time = pred_toc - pred_tic
            step_time_list.append(pred_time)

        if update_mode == "line_search":
            T0_batch = Tk_batch_func(zk_batch,p_batch)
            zk_best = zk_batch.detach().clone()
            Tk_best = T0_batch.detach().clone()

        # Pre-allocate trajectory buffer if needed (GPU-friendly)
        if return_trajectory:
            zk_traj_buffer = torch.empty((max_iter + 1, n_batch, zk_batch.shape[1]), 
                                         dtype=zk_batch.dtype, device=zk_batch.device)
            zk_traj_buffer[0] = zk_batch.detach().clone()
        
        # LOOP
        for i in range(max_iter):
            print(f"Solve iteration {i+1}/{max_iter}")
            step_tic = time.perf_counter()

            dzk_batch = step_func(zk_batch,p_batch)

            if update_mode == "full":
                zk_batch = zk_batch + alpha_batch[:,None]*dzk_batch

            elif update_mode == "line_search":
                # update
                zk_batch = zk_batch + alpha_batch[:,None]*dzk_batch
                Tk_batch = Tk_batch_func(zk_batch,p_batch)
                # update best iterates
                mask_best = Tk_batch < Tk_best
                zk_best[mask_best] = zk_batch[mask_best].detach().clone()
                Tk_best[mask_best] = Tk_batch[mask_best].detach().clone()
                # diverging
                mask_diverged = Tk_batch > 10*Tk_best
                # reset to best + reduce alpha
                zk_batch[mask_diverged] = zk_best[mask_diverged].detach().clone()
                alpha_batch[mask_diverged] = alpha_batch[mask_diverged]*0.95
                alpha_batch[mask_best] = torch.clip(alpha_batch[mask_best]/0.95,0.0,alpha)

            else:
                raise ValueError("Update mode not implemented.")
            
            step_toc = time.perf_counter()
            step_time = step_toc - step_tic
            step_time_list.append(step_time)
            
            if return_trajectory:
                if update_mode == "line_search":
                    zk_traj_buffer[i+1] = zk_best.detach().clone()
                else:
                    zk_traj_buffer[i+1] = zk_batch.detach().clone()

        # END OF SOLVE
        if update_mode == "line_search":
            zk_batch = zk_best.detach().clone()
        
        if return_trajectory:
            # Convert buffer to list only after loop completes
            zk_traj = [zk_traj_buffer[j] for j in range(max_iter + 1)]
            return zk_batch, zk_traj, step_time_list
        else:
            return zk_batch, step_time_list
        
    def get_casadi_solver_step_funcs(self,nlp_casadi,jit=True):
        model = copy.deepcopy(self.model)
        model.cpu()

        if jit:
            model = torch.jit.script(model)
            model = torch.jit.optimize_for_inference(model)

        step_func = nlp_casadi._build_solver_step(model)
        Tk_func = nlp_casadi.Tk_func
        # return nlp_casadi._build_solver_step(model)
        return step_func, Tk_func

    @torch.no_grad()
    def solve_fast_single_cpu(self,solve_step_func,Tk_func,p,predictor=None,alpha=1.0,max_iter=100,tol=1e-6,update_mode="line_search"):
        success = False
        gamma = self.gamma
        device = "cpu"

        if predictor is None:
            zk = self.nlp.batch_gen_z(1).squeeze().to(device).numpy()
        else:
            zk = predictor(torch.from_numpy(p)).squeeze().to(device).numpy()
        
        # Initializaton for "line_search"
        z_best = zk
        T_best = Tk_func(zk,p)
 
        if update_mode == "line_search":
            alpha_k = alpha
            for i in range(max_iter):
                dzk, Fk_2_norm = solve_step_func(zk,p)
                
                if Fk_2_norm < tol:
                    success = True
                    break
                
                zk = zk + dzk*alpha_k*gamma
                Tk = Tk_func(zk,p)

                if Tk < T_best:
                    # update best iterate and error
                    z_best = zk
                    T_best = Tk
                    # increase alpha
                    alpha_k = np.clip(alpha_k/0.95,0.0,alpha) # two-way backtracking line search, increase alpha again if Tk is smaller than Tk_best (not larger than initial alpha)

                if Tk > 10*T_best:
                    zk = z_best
                    # decrease alpha
                    alpha_k = alpha_k*0.95

        else:
            raise ValueError("Update mode not implemented.")  
        
        return torch.from_numpy(zk),i+1,success        
    
    ### Save/Load Methods
    def save_weights(self,save_dir, file_name = "weights.pt"):
        save_pth = Path(save_dir,file_name)
        torch.save(self.state_dict(),save_pth)
        print("Weights (state dict) saved to: ", save_pth)

    def load_weights(self,load_dir, file_name = "weights.pt"):
        load_pth = Path(load_dir,file_name)
        weights = torch.load(load_pth,weights_only=True)
        self.load_state_dict(weights)
        print("Weights (state dict) loaded from: ", load_pth)

    def save_model(self,save_dir,file_name = "model.pt"):
        save_pth = Path(save_dir,file_name)
        torch.save(self.model,save_pth)
        print("Model saved to: ", save_pth)

    @staticmethod
    def load_model(load_dir,file_name = "model.pt"):
        load_pth = Path(load_dir,file_name)
        model = torch.load(load_pth)
        print("Model loaded from: ", load_pth)
        return model    

# Predictor
class Predictor(torch.nn.Module):
    """Predictor class for Neural Network model that predicts primal-dual solutions.

    Args:
        model (torch.nn.Module): Neural Network model
        convexification (bool): Whether to use convexified loss function
        torch_compiled (bool): Whether to use torch.compile for forward pass
    """

    def __init__(self, model, convexification=False, torch_compiled=False):
        super().__init__()
        self.model = model
        self.convexification = convexification
        self.torch_compiled = torch_compiled
        self.offset = 1e-16
        print("----------------------------------")
        print(self)
        print("----------------------------------")

    def initialize_functions(self, nlp):
        """Initialize and optionally compile functions after NLP is available."""
        # Select Tk function based on convexification flag
        if self.convexification:
            print("Using convexified Tk function for predictor.\n")
            Tk_batch_func = nlp.Tk_conv_batch_func
        else:
            Tk_batch_func = nlp.Tk_batch_func
        
        # Torch compile functions if enabled
        if self.torch_compiled:
            torch.compiler.reset()
            print("Torch compiler reset.\n")
            self.forward = torch.compile(self._forward, mode="max-autotune", fullgraph=True)
            self.Tk_batch_func = torch.compile(copy.deepcopy(Tk_batch_func), mode="max-autotune", fullgraph=True)
            self.training_step = torch.compile(self._training_step, mode="max-autotune")
            print("Predictor functions torch-compiled.\n")
        else:
            self.forward = self._forward
            self.Tk_batch_func = Tk_batch_func
            self.training_step = self._training_step
        
        print("Predictor functions initialized.")

    def _forward(self, p):
        return self.model(p)

    # Predict
    @torch.no_grad()
    def predict(self, p_batch):
        return self.model(p_batch)

    def _training_step(self, p_batch, optim):
        """Single training step with loss computation."""
        # Zero gradients
        optim.zero_grad(set_to_none=True)

        # Forward pass
        z_hat_batch = self.forward(p_batch).to(p_batch.dtype)

        # Compute loss
        Tk_batch = self.Tk_batch_func(z_hat_batch, p_batch)

        assert not torch.isnan(Tk_batch).any(), "Tk_batch contains nan"

        # Log-scaled loss
        loss = torch.mean(torch.log(Tk_batch + self.offset**2))

        # Backward pass
        loss.backward()

        # Step
        optim.step()

        return loss

    # Train
    def train(self, nlp, train_config, print_frequency=100, log_values=True):
        """Train the predictor model.
        
        Args:
            nlp: NLP problem instance
            train_config (dict): Training configuration with keys:
                - N_epochs (int): Number of training epochs
                - batch_size (int): Batch size for training
                - weight_decay (float): Weight decay for optimizer
                - lr (float): Learning rate
                - use_amsgrad (bool, optional): Whether to use AMSGrad variant of Adam
                - max_grad_norm (float, optional): Maximum gradient norm for clipping
            print_frequency (int): How often to print progress
            
        Returns:
            TrainLogger: Training history logger
        """
        # Initialize functions if not already done
        if not hasattr(self, 'Tk_batch_func'):
            self.initialize_functions(nlp)
        
        # Reset torch compiler
        # if self.torch_compiled:
        #     torch.compiler.reset()
        #     print("Torch compiler reset.\n")

        # Extract config
        N_epochs = train_config.get("N_epochs", 10000)
        batch_size = train_config.get("batch_size", 1024)
        weight_decay = train_config.get("weight_decay", 0.0)
        lr = train_config.get("lr", 1e-4)
        use_amsgrad = train_config.get("use_amsgrad", True)

        # Setup logger
        train_logger = TrainLogger()

        # Setup optimizer
        optim = torch.optim.AdamW(self.parameters(),lr=lr,weight_decay=weight_decay,amsgrad=use_amsgrad)

        # Training loop
        train_tic = time.perf_counter()
        
        for epoch in range(N_epochs):

            # Generate batch (uniform sampling only)
            p_batch = nlp.batch_gen_p(batch_size,offset=0.0)

            # Training step
            loss = self.training_step(p_batch, optim)

            # Log values
            if log_values:
                train_logger.log_value_buffered(epoch+1, "epoch")
                train_logger.log_value_buffered(loss, "loss")
        
            # Print progress
            if (epoch + 1) % print_frequency == 0:
                print("Epoch {0}: Loss = {1:.3e}".format(epoch + 1, loss))
                print("-------------------------------")

        # End of training
        train_toc = time.perf_counter()
        train_time = train_toc - train_tic
        if log_values:
            train_logger.log_data(train_time, "train_time")
            train_logger.log_array(p_batch.detach().cpu(), "p_final_batch")
            train_logger.flush_all_buffers()

        print("#----------------------------------#")
        print("Training complete.")
        print(f"Total number of epochs: {epoch + 1}")
        print(f"Total training time: {train_time:.2f} sec")
        print("#----------------------------------#")

        return train_logger
    
    # Save and Load    
    def save_weights(self,save_dir, file_name = "weights.pt"):
        save_pth = Path(save_dir,file_name)
        torch.save(self.state_dict(),save_pth)
        print("Weights (state dict) saved to: ", save_pth)

    def load_weights(self,load_dir, file_name = "weights.pt"):
        load_pth = Path(load_dir,file_name)
        weights = torch.load(load_pth,weights_only=True)
        self.load_state_dict(weights)
        print("Weights (state dict) loaded from: ", load_pth)

    def save_model(self,save_dir,file_name = "model.pt"):
        save_pth = Path(save_dir,file_name)
        torch.save(self,save_pth)
        print("Model saved to: ", save_pth)

    @staticmethod
    def load_model(load_dir,file_name = "model.pt"):
        load_pth = Path(load_dir,file_name)
        model = torch.load(load_pth)
        print("Model loaded from: ", load_pth)
        return model

# ApproxMPC
class ApproxMPC(torch.nn.Module):
    """Approximate MPC class for Neural Network model.
    
    Args:
        model (torch.nn.Module): Neural Network model
        scale_data (bool): Whether to scale input/output data
        bounds (dict, optional): Dictionary with keys 'x_lb', 'x_ub', 'u_lb', 'u_ub' for scaling
        torch_compiled (bool): Whether to compile forward/training functions
    """
    def __init__(self, model, scale_data, bounds=None, torch_compiled=False):
        super().__init__()
        self.model = model
        self.scale_data = scale_data
        self.torch_compiled = torch_compiled
        
        if self.scale_data:
            assert bounds is not None, "Bounds must be provided if scale_data is True."
            # check if bounds are tensors or lists/arrays if not none
            bounds = bounds.copy()
            for key in bounds.keys():
                if not torch.is_tensor(bounds[key]):
                    bounds[key] = torch.tensor(bounds[key],dtype=torch.float32)
            self.bounds = bounds
            self.set_scaling_from_bounds()
        
        self.initialize_functions()
        
        print("----------------------------------")
        print(self)
        print("----------------------------------")
    
    def initialize_functions(self):
        """Initialize and optionally compile functions."""
        if self.torch_compiled:
            torch.compiler.reset()
            print("Torch compiler reset.\n")
            self.forward = torch.compile(self._forward, mode="max-autotune", fullgraph=True)
            self.training_step = torch.compile(self._training_step, mode="max-autotune")
            print("ApproxMPC functions torch-compiled.\n")
        else:
            self.forward = self._forward
            self.training_step = self._training_step
        
        print("ApproxMPC functions initialized.")
    
    def _forward(self, x):
        return self.model(x)
    
    def set_scaling_from_bounds(self):
        """Set min-max scaling parameters from bounds."""
        self.x_lb = self.bounds["x_lb"]
        self.x_ub = self.bounds["x_ub"]
        self.u_lb = self.bounds["u_lb"]
        self.u_ub = self.bounds["u_ub"]

        self.x_range = self.x_ub - self.x_lb
        self.x_shift = self.x_lb
        self.y_range = self.u_ub - self.u_lb
        self.y_shift = self.u_lb

        print("Scaling factors set from bounds")
        print("x_range: ", self.x_range)
        print("x_shift: ", self.x_shift)
        print("y_range: ", self.y_range)
        print("y_shift: ", self.y_shift)

    def scale_inputs(self, x_data):
        x_scaled = (x_data - self.x_shift) / self.x_range
        return x_scaled

    def scale_outputs(self, y_data):
        y_scaled = (y_data - self.y_shift) / self.y_range
        return y_scaled

    def rescale_inputs(self, x_scaled):
        x_data = x_scaled * self.x_range + self.x_shift
        return x_data

    def rescale_outputs(self, y_scaled):
        y_data = y_scaled * self.y_range + self.y_shift
        return y_data

    def scale_dataset(self, x_data, y_data):
        x_scaled = self.scale_inputs(x_data)
        y_scaled = self.scale_outputs(y_data)
        return x_scaled, y_scaled

    @torch.no_grad()
    def make_step(self, x, scale_inputs=True, rescale_outputs=True, clip_outputs=True):
        """Apply the approximate MPC to compute control input."""
        if scale_inputs:
            x_scaled = self.scale_inputs(x)
        else:
            x_scaled = x
        
        # Prediction
        y_scaled = self.forward(x_scaled)
        
        if rescale_outputs:
            y = self.rescale_outputs(y_scaled)
        else:
            y = y_scaled
        
        # Clip outputs to satisfy input constraints
        if clip_outputs:
            y = torch.clamp(y, self.u_lb, self.u_ub)
        
        return y

    def _training_step(self, x_batch, y_batch, optim):
        """Single training step with loss computation."""
        optim.zero_grad(set_to_none=True)
        y_pred = self.forward(x_batch).to(y_batch.dtype)
        loss = torch.nn.functional.mse_loss(y_pred, y_batch)
        loss.backward()
        optim.step()
        return loss

    def setup_datasets(self, X_train, Y_train, X_val=None, Y_val=None, batch_size=1):
        """Setup training and validation datasets with scaling."""
        n_data_train = X_train.shape[0]
        n_steps = n_data_train // batch_size
        effective_batch_size = min(batch_size, n_data_train)

        print("Number of training data points: ", n_data_train)
        print("Number of training steps per epoch: ", n_steps)
        print("Effective batch size: ", effective_batch_size)
        
        X_train_scaled, Y_train_scaled = self.scale_dataset(X_train, Y_train)
        train_dataset = (X_train_scaled, Y_train_scaled)

        if X_val is not None:
            X_val_scaled, Y_val_scaled = self.scale_dataset(X_val, Y_val)
            val_dataset = (X_val_scaled, Y_val_scaled)
            return train_dataset, val_dataset
        else:
            return train_dataset, None

    def train(self, train_config, train_dataset, val_dataset=None, print_frequency=100, log_values=True):
        """Train the approximate MPC model.
        
        Args:
            train_config (dict): Training configuration with keys:
                - N_epochs (int): Number of training epochs
                - batch_size (int): Batch size for training
                - weight_decay (float): Weight decay for optimizer
                - lr (float): Learning rate
                - use_lr_scheduler (bool): Whether to use learning rate scheduler
                - use_amsgrad (bool, optional): Whether to use AMSGrad variant
                - stop_at_min_lr (bool): Whether to stop when minimum LR is reached
                Additional scheduler params if use_lr_scheduler=True:
                    - lr_scheduler_patience (int)
                    - lr_scheduler_cooldown (int)
                    - lr_reduce_factor (float)
                    - min_lr (float)
            train_dataset (tuple): (X_train, Y_train) tensors
            val_dataset (tuple, optional): (X_val, Y_val) tensors
            print_frequency (int): How often to print progress
            log_values (bool): Whether to log training metrics
            
        Returns:
            TrainLogger: Training history logger
        """
        # Reset torch compiler
        # if self.torch_compiled:
        #     torch.compiler.reset()
        #     print("Torch compiler reset.\n")
        
        # Extract config
        N_epochs = train_config["N_epochs"]
        batch_size = train_config["batch_size"]
        weight_decay = train_config["weight_decay"]
        lr = train_config["lr"]
        use_lr_scheduler = train_config["use_lr_scheduler"]
        stop_at_min_lr = train_config["stop_at_min_lr"]
        use_amsgrad = train_config.get("use_amsgrad", True)

        # Unpack datasets
        X_train, Y_train = train_dataset
        n_data = X_train.shape[0]
        n_steps = n_data // batch_size

        # Setup logger
        train_logger = TrainLogger(device=X_train.device)

        # Setup optimizer
        optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            amsgrad=use_amsgrad
        )
        
        # LR Scheduler
        if use_lr_scheduler:
            lr_scheduler_patience = train_config["lr_scheduler_patience"]
            lr_scheduler_cooldown = train_config["lr_scheduler_cooldown"]
            lr_reduce_factor = train_config["lr_reduce_factor"]
            min_lr = train_config["min_lr"]
            
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, mode='min', factor=lr_reduce_factor,
                patience=lr_scheduler_patience,
                threshold=1e-5, threshold_mode='rel',
                cooldown=lr_scheduler_cooldown,
                min_lr=min_lr, eps=0.0
            )

        # Training loop
        train_tic = time.perf_counter()
        
        for epoch in range(N_epochs):

            # Shuffle data
            idx_shuffle = torch.randperm(n_data)
            
            train_loss = 0.0
            # Training steps
            for idx_train_batch in range(n_steps):
                idx_batch = idx_shuffle[idx_train_batch*batch_size:(idx_train_batch+1)*batch_size]
                x_batch = X_train[idx_batch]
                y_batch = Y_train[idx_batch]
                
                loss = self.training_step(x_batch, y_batch, optim)
                train_loss += loss.item()
            
            train_loss = train_loss / n_steps
            
            # Validation
            if val_dataset is not None:
                with torch.no_grad():
                    X_val, Y_val = val_dataset
                    y_val_pred = self.forward(X_val)
                    val_loss = torch.nn.functional.mse_loss(y_val_pred, Y_val).item()
            
            # LR Scheduler step
            if use_lr_scheduler:
                lr_scheduler.step(train_loss)
            
            
            # Log values
            if log_values:
                train_logger.log_value_buffered(epoch, "epoch")
                train_logger.log_value_buffered(train_loss, "train_loss")
                if val_dataset is not None:
                    train_logger.log_value_buffered(val_loss, "val_loss")
                train_logger.log_value_buffered(optim.param_groups[0]["lr"], "lr")
            
            # Print progress
            if (epoch + 1) % print_frequency == 0:
                print("Epoch {0}: Train Loss = {1:.3e}".format(epoch + 1, train_loss))
                if val_dataset is not None:
                    print("         Val Loss = {0:.3e}".format(val_loss))
                print("-------------------------------")
            
            # Early stopping
            if stop_at_min_lr and use_lr_scheduler:
                if optim.param_groups[0]["lr"] <= min_lr:
                    break
        
        # End of training
        train_toc = time.perf_counter()
        train_time = train_toc - train_tic
        
        if log_values:
            train_logger.log_data(train_time, "train_time")
            train_logger.flush_all_buffers()
        
        print("#----------------------------------#")
        print("Training complete.")
        print(f"Total number of epochs: {epoch + 1}")
        print(f"Total training time: {train_time:.2f} sec")
        print("#----------------------------------#")
        
        return train_logger
    
    # Save and Load
    def save_weights(self, save_dir, file_name="weights.pt"):
        save_pth = Path(save_dir, file_name)
        torch.save(self.state_dict(), save_pth)
        print("Weights (state dict) saved to: ", save_pth)

    def load_weights(self, load_dir, file_name="weights.pt"):
        load_pth = Path(load_dir, file_name)
        weights = torch.load(load_pth, weights_only=True)
        self.load_state_dict(weights)
        print("Weights (state dict) loaded from: ", load_pth)

    def save_model(self, save_dir, file_name="model.pt"):
        save_pth = Path(save_dir, file_name)
        torch.save(self, save_pth)
        print("Model saved to: ", save_pth)

    @staticmethod
    def load_model(load_dir, file_name="model.pt"):
        load_pth = Path(load_dir, file_name)
        model = torch.load(load_pth)
        print("Model loaded from: ", load_pth)
        return model
    
    # Export
    def export_jit_gpu(self):
        mod_gpu = copy.deepcopy(self.model).cuda()
        mod_gpu = torch.jit.script(mod_gpu)
        mod_gpu = torch.jit.optimize_for_inference(mod_gpu)
        return mod_gpu
    
    def export_jit_cpu(self):
        mod_cpu = copy.deepcopy(self.model).cpu()
        mod_cpu = torch.jit.script(mod_cpu)
        mod_cpu = torch.jit.optimize_for_inference(mod_cpu)
        return mod_cpu

class TrainLogger:
    def __init__(self, buffer_size=100,device=torch.device('cuda')):
        """
        Initialize TrainLogger with GPU buffering capability.
        
        Args:
            buffer_size (int): Number of entries to accumulate in GPU buffer before flushing to CPU
            device (str): Device to store buffers on ('cuda' or 'cpu')
        """
        self.history = {}
        self.buffer_size = buffer_size
        self.device = device if torch.cuda.is_available() else torch.device('cpu')
        
        # Buffers: dict of {key: tensor}
        self.buffers = {}
        self.buffer_counts = {}  # Track how many entries in each buffer

    def _init_buffer(self, key):
        if key not in self.buffers:
            dev = self.device
            self.buffers[key] = torch.empty(self.buffer_size, device=dev, dtype=torch.float32)
            self.buffer_counts[key] = 0

    def _flush_buffer(self, key):
        """Flush buffer contents to CPU history."""
        if key not in self.buffers or self.buffer_counts[key] == 0:
            return
        
        # Get valid entries from buffer
        count = self.buffer_counts[key]
        buffer_data = self.buffers[key][:count].detach().cpu().tolist()
        
        # Add to history
        if key not in self.history:
            self.history[key] = []
        self.history[key].extend(buffer_data)
        
        # Reset buffer count
        self.buffer_counts[key] = 0

    def flush_all_buffers(self):
        """Flush all buffers to CPU history."""
        for key in list(self.buffers.keys()):
            self._flush_buffer(key)

    def log_value_buffered(self, val, key):
        # normalize to float32 tensor on target device
        if torch.is_tensor(val):
            t = val.detach()
            if t.device != self.device:
                t = t.to(self.device, non_blocking=True)
            # always cast to float32
            if t.dtype != torch.float32:
                t = t.float()
        else:
            # python numbers become float32
            t = torch.tensor(val, device=self.device, dtype=torch.float32)

        # initialize buffer if needed
        if key not in self.buffers:
            self._init_buffer(key)

        # ensure scalar
        if t.numel() != 1:
            raise ValueError(f"log_value_buffered('{key}'): expected scalar, got shape {tuple(t.shape)}")
            
        idx = self.buffer_counts[key]
        self.buffers[key][idx] = t
        self.buffer_counts[key] += 1

        if self.buffer_counts[key] >= self.buffer_size:
            self._flush_buffer(key)

    def log_values_buffered(self, kv: dict, allow_reduce_to_mean=False):
        # e.g., {"loss": loss_scalar_tensor, "acc": acc_scalar_tensor}
        for k, v in kv.items():
            self.log_value_buffered(v, k, allow_reduce_to_mean=allow_reduce_to_mean)

    def log_value(self, val, key):
        """Original immediate logging (for backward compatibility)."""
        if torch.is_tensor(val):
            val = val.detach().cpu().item()        
        assert isinstance(val, (int, float)), "Value must be a scalar."
        if key not in self.history.keys():
            self.history[key] = []
        self.history[key].append(val)

    def log_data(self, val, key):
        """Log data that overwrites previous value."""
        if torch.is_tensor(val):
            val = val.detach().cpu().item()
        self.history[key] = val

    def log_array(self, val, key):
        """Log array/tensor as list."""
        if torch.is_tensor(val):
            val = val.detach().cpu().numpy()
        val = val.tolist()
        self.history[key] = val
    
    def calculate_metrics(self, val: torch.Tensor):
        assert torch.is_tensor(val), "Value must be a tensor."
        # promote to float for quantiles if needed
        if not val.is_floating_point():
            val = val.float()
        qs = torch.tensor([0.99, 0.95, 0.50, 0.01], device=val.device)
        q_vals = torch.quantile(val, qs)  # shape [4]
        val_mean = val.mean()
        val_max  = val.max()
        val_min  = val.min()
        # unpack
        val_99, val_95, val_50, val_1 = q_vals.unbind()
        return val_mean, val_max, val_min, val_99, val_95, val_50, val_1
    
    def log_metrics(self, val, key, use_buffer=False):
        # WARNING: This function leads to non-negligible overhead if used frequently during training!
        """
        Log distribution metrics for a tensor.
        
        Args:
            val: Input tensor
            key: Base key for metrics
            use_buffer: If True, use buffered logging
        """
        val_mean, val_max, val_min, val_99, val_95, val_50, val_1 = self.calculate_metrics(val)
        
        log_fn = self.log_value_buffered if use_buffer else self.log_value
        
        log_fn(val_mean, key + "_mean")
        log_fn(val_max, key + "_max")
        log_fn(val_min, key + "_min")
        log_fn(val_99, key + "_99")
        log_fn(val_95, key + "_95")
        log_fn(val_50, key + "_50")
        log_fn(val_1, key + "_1")
            
    def save_history(self, exp_pth, file_name="history", as_json=True):
        """Save history to disk. Flushes all buffers first."""
        # Ensure all buffers are flushed before saving
        self.flush_all_buffers()
        
        exp_pth = Path(exp_pth)
        exp_pth.mkdir(parents=True, exist_ok=True)
        
        # Save as .pt
        torch.save(self.history, exp_pth.joinpath(file_name + ".pt"))
        
        # Save as JSON
        if as_json:
            with open(exp_pth.joinpath(file_name + ".json"), 'w') as f:
                json.dump(self.history, f, indent=4)
        
        print("History saved to: ", exp_pth.joinpath(file_name))

    def visualize_history(self, key, log_scaling=False, exp_pth=None, save_fig=False):
        """Visualize history for a key."""
        # Flush buffers to ensure all data is available
        self.flush_all_buffers()
        
        fig, ax = plt.subplots()
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key)
        ax.set_title(key)
        if log_scaling:
            ax.set_yscale('log')
        
        if key in self.history:
            ax.plot(self.history[key], label=key)
        else:
            print(f"Key '{key}' not found in history.")
                
        ax.legend()
        plt.show()
        if save_fig:
            assert exp_pth is not None, "exp_pth must be provided."
            plt.savefig(Path(exp_pth).joinpath(key + ".png"))
        return fig, ax

    def post_calculate_ema_resample_metrics(self, batch_size, ema_beta=0.95):
        """Calculate EMA-based resampling metrics and convergence fractions."""
        # Flush buffers to ensure all data is available
        self.flush_all_buffers()
        
        n_resample_values = self.history.get('n_resample', [])
        n_resample_tol_values = self.history.get('n_resample_tol', [])
        n_resample_iter_values = self.history.get('n_resample_iter', [])
        
        n_resample_ema_values = []
        n_resample_tol_ema_values = []
        n_resample_iter_ema_values = []
        conv_frac_samples_values = []
        conv_frac_batch_values = []
        
        n_resample_ema = 1.0
        n_resample_tol_ema = 0.0
        n_resample_iter_ema = 0.0
        
        for i, (n_resample, n_resample_tol, n_resample_iter) in enumerate(zip(n_resample_values, n_resample_tol_values, n_resample_iter_values)):
            # Update EMAs
            n_resample_ema = ema_beta * n_resample_ema + (1 - ema_beta) * n_resample
            n_resample_tol_ema = ema_beta * n_resample_tol_ema + (1 - ema_beta) * n_resample_tol
            n_resample_iter_ema = ema_beta * n_resample_iter_ema + (1 - ema_beta) * n_resample_iter
            
            # Calculate convergence fractions
            conv_frac_samples = n_resample_tol_ema / (n_resample_ema + 1e-16)
            conv_frac_batch = n_resample_tol_ema / batch_size
            
            # Store values
            n_resample_ema_values.append(n_resample_ema)
            n_resample_tol_ema_values.append(n_resample_tol_ema)
            n_resample_iter_ema_values.append(n_resample_iter_ema)
            conv_frac_samples_values.append(conv_frac_samples)
            conv_frac_batch_values.append(conv_frac_batch)
        
        # Update history
        self.history['n_resample_ema'] = n_resample_ema_values
        self.history['n_resample_tol_ema'] = n_resample_tol_ema_values
        self.history['n_resample_iter_ema'] = n_resample_iter_ema_values
        self.history['conv_frac_samples'] = conv_frac_samples_values
        self.history['conv_frac_batch'] = conv_frac_batch_values


