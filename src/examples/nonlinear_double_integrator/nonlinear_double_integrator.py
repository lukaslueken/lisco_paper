"""
Model modified from:
https://doi.org/10.1016/j.sysconle.2007.06.013 Lazar et al. 2008
"""
# TODO: update docstrings

import torch
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

# Notes
# - Difference to ECC Version: no R-Term


class NonlinearDoubleIntegrator:
    """A class representing a nonlinear double integrator system.
    This class implements a discrete-time nonlinear double integrator system with
    state dynamics: x[k+1] = Ax[k] + Bu[k] + F||x[k]||^2
    The system has:
    - 2 states: position and velocity 
    - 1 control input
    - Nonlinear quadratic state term
    Attributes:
        dtype (torch.dtype): Data type for torch tensors (default: torch.float32)
        device (str): Device to run computations on (default: 'cpu')
        x_dim (int): Dimension of state space (2)
        u_dim (int): Dimension of control space (1)
        A (torch.Tensor): System matrix (2x2)
        B (torch.Tensor): Input matrix (2x1) 
        F (torch.Tensor): Nonlinear term coefficient (2x1)
        Q (torch.Tensor): State cost matrix (2x2)
        R (torch.Tensor): Control cost matrix (1x1)
    Example:
        >>> system = NonlinearDoubleIntegrator()
        >>> x = torch.tensor([1.0, 0.0])  # Initial state
        >>> u = torch.tensor([0.5])       # Control input
        >>> x_next = system.step(x, u)    # Compute next state
    """

    def __init__(self, dtype=torch.float32, device='cpu'):
        self.dtype = dtype
        self.device = device

        self.x_dim = 2
        self.u_dim = 1
        self.x_lb = torch.tensor([-10.0, -10.0], dtype=dtype, device=device)
        self.x_ub = torch.tensor([10.0, 10.0], dtype=dtype, device=device)
        self.u_lb = torch.tensor([-2.0], dtype=dtype, device=device)
        self.u_ub = torch.tensor([2.0], dtype=dtype, device=device)

        self.A = torch.tensor([[1.0, 1.0],
                                [0.0, 1.0]], dtype=dtype, device=device)
        
        self.B = torch.tensor([[0.5],
                                [1.0]], dtype=dtype, device=device)
        
        self.F = torch.tensor([0.025,0.025], dtype=dtype, device=device)

        self.Q = torch.tensor([[0.8, 0.0],
                                 [0.0, 0.8]], dtype=dtype, device=device)
        
        self.R = torch.tensor([[0.1]], dtype=dtype, device=device)

    # difference equation evaluation xk+1 = Axk + Buk + F||xk||^2
    def step(self, x, u):
        """Compute next state for a single state-action pair."""
        assert x.dim() == 1
        assert u.dim() == 1
        # Expects x of shape (2,) and u of shape (1,)
        quadratic_term = torch.sum(x * x)
        return self.A @ x + self.B @ u + self.F * quadratic_term

    # Parallelized difference equation evaluation
    def parallel_step(self, x, u):
        # Expects x of shape (batch_size, 2) and u of shape (batch_size, 1)
        # batch_size = x.shape[0]
    
        # Compute quadratic term
        quadratic_term = torch.sum(x * x, dim=1, keepdim=False)
        
        # Extract components to avoid matmul
        x1, x2 = x[:, 0], x[:, 1]
        u1 = u[:, 0]
        
        # Calculate first output component
        result1 = self.A[0, 0] * x1 + self.A[0, 1] * x2 + self.B[0, 0] * u1 + self.F[0] * quadratic_term
        
        # Calculate second output component
        result2 = self.A[1, 0] * x1 + self.A[1, 1] * x2 + self.B[1, 0] * u1 + self.F[1] * quadratic_term
        
        # Combine results
        return torch.stack([result1, result2], dim=1)

    # def parallel_step(self, x, u):
    #     # x: (batch, 2), u: (batch, 1); all tensors already on the same CUDA device and dtype
    #     # Preconditions: self.A (2,2), self.B (2,1), self.F (2,) or (1,2) are on the same device/dtype.

    #     # Keep the quadratic sum as (batch, 1) for broadcast
    #     quadratic_term = (x * x).sum(dim=1, keepdim=True)  # (batch,1)

    #     # Compute: x @ A^T + u * B^T + quadratic_term * F^T   -> (batch, 2)
    #     # Make shapes explicit for broadcast correctness and kernel fusion
    #     Ax = x.matmul(self.A.T)                              # (batch,2)
    #     uB = u.matmul(self.B.T)                              # (batch,2)
    #     F  = self.F.view(1, 2)                               # (1,2)
    #     out = Ax + uB + quadratic_term * F                   # (batch,2)
    #     return out


    # Cost functions
    def stage_cost(self,x,u):        
        return x@self.Q@x + u@self.R@u
    
    def parallel_stage_cost(self,x,u):
        return (x @ self.Q * x).sum(dim=1) + (u @ self.R * u).sum(dim=1)

    def terminal_cost(self, x):
        """Compute terminal cost for a single state vector."""
        return x @ self.Q @ x  # This is more direct than the current implementation

    def parallel_terminal_cost(self, x):
        """Compute terminal cost for a batch of state vectors."""
        return (x @ self.Q * x).sum(dim=1)

        
    ### Functions for simulation and cost evaluation ###
    def simulate(self,x0,U_seq):
        N_seq = U_seq.shape[0]
        X_seq = torch.zeros(N_seq+1, self.x_dim, dtype=self.dtype, device=self.device)
        X_seq[0,:] = x0
        for k in range(N_seq):
            X_seq[k+1,:] = self.step(X_seq[k,:],U_seq[k,:])
        return X_seq
    
    def simulate_batch(self,x0_batch,U_seq_batch):
        N_batch, N_seq, _ = U_seq_batch.shape
        X_seq_batch = torch.zeros(N_batch, N_seq+1, self.x_dim, dtype=self.dtype, device=self.device)
        X_seq_batch[:,0,:] = x0_batch
        for k in range(N_seq):
            X_seq_batch[:,k+1,:] = self.parallel_step(X_seq_batch[:,k,:],U_seq_batch[:,k,:])
        return X_seq_batch
        
    def total_cost_seq(self,X_seq,U_seq):
        N_seq, _ = U_seq.shape
        cost = torch.zeros(1, dtype=self.dtype, device=self.device)
        for k in range(N_seq):
            cost += self.stage_cost(X_seq[k,:],U_seq[k,:])
        cost += self.terminal_cost(X_seq[-1,:])
        return cost
    
    def total_cost_seq_batch(self, X_seq_batch, U_seq_batch):
        N_batch, N_seq, _ = U_seq_batch.shape
        cost = torch.zeros(N_batch, dtype=self.dtype, device=self.device)
        for k in range(N_seq):
            cost += self.parallel_stage_cost(X_seq_batch[:,k,:], U_seq_batch[:,k,:])
        cost += self.parallel_terminal_cost(X_seq_batch[:,-1,:])
        return cost
    
    # def check_constraints(self, x, u):
    #     """Check if state and control inputs are within bounds."""
    #     eps = 1e-5 # tolerance for constraints
    #     x_ok = torch.all(x >= self.x_lb-eps) and torch.all(x <= self.x_ub+eps)
    #     u_ok = torch.all(u >= self.u_lb-eps) and torch.all(u <= self.u_ub+eps)
    #     return x_ok, u_ok
    
    # def check_constraints_batch(self, x_batch, u_batch):
    #     N_batch = x_batch.shape[0]
    #     x_ok_batch = []
    #     u_ok_batch = []
    #     for i in range(N_batch):
    #         x_ok, u_ok = self.check_constraints(x_batch[i], u_batch[i])
    #         x_ok_batch.append(x_ok)
    #         u_ok_batch.append(u_ok)
    #     x_ok_batch = torch.tensor(x_ok_batch, dtype=torch.bool, device=self.device)
    #     u_ok_batch = torch.tensor(u_ok_batch, dtype=torch.bool, device=self.device)
    #     return x_ok_batch, u_ok_batch

    def check_constraints_batch(self, x_batch, u_batch):
        """
        Memory-efficient vectorized constraint checking.
        Returns boolean tensors indicating feasibility.
        """
        eps = 1e-6
        # For states
        if x_batch.dim() == 3:  # [batch, N, n_states]
            # Check lower and upper bounds separately, then combine
            x_ok = torch.all(
                (x_batch >= self.x_lb-eps) & (x_batch <= self.x_ub+eps),
                dim=-1  # Check across state dimensions
            )  # Result: [batch, N]
        else:  # [batch, n_states]
            x_ok = torch.all(
                (x_batch >= self.x_lb-eps) & (x_batch <= self.x_ub+eps),
                dim=-1
            )  # Result: [batch]
        
        # For controls
        u_ok = torch.all(
            (u_batch >= self.u_lb-eps) & (u_batch <= self.u_ub+eps),
            dim=-1
        )  # Result: [batch]
        
        return x_ok, u_ok

    
    def export_hparams(self):
        hparams = {
            "dtype": str(self.dtype),
            "device": self.device.type,
            "x_dim": self.x_dim,
            "u_dim": self.u_dim,
            "A": self.A.cpu().numpy(),
            "B": self.B.cpu().numpy(),
            "F": self.F.cpu().numpy(),
            "Q": self.Q.cpu().numpy(),
            "R": self.R.cpu().numpy(),
            "x_lb": self.x_lb.cpu().numpy(),
            "x_ub": self.x_ub.cpu().numpy(),
            "u_lb": self.u_lb.cpu().numpy(),
            "u_ub": self.u_ub.cpu().numpy()
        }
        for key, value in hparams.items():
            if isinstance(value,torch.Tensor):
                hparams[key] = value.tolist()
            elif isinstance(value,np.ndarray):
                hparams[key] = value.tolist()
        return hparams

class NonlinearDoubleIntegratorCasadi:
    """A class representing a nonlinear double integrator system using CasADi.
    
    This class implements a discrete-time nonlinear double integrator system with
    state dynamics: x[k+1] = Ax[k] + Bu[k] + F||x[k]||^2
    
    The system has:
    - 2 states: position and velocity 
    - 1 control input
    - Nonlinear quadratic state term
    
    Attributes:
        x_dim (int): Dimension of state space (2)
        u_dim (int): Dimension of control space (1)
        A (ca.DM): System matrix (2x2)
        B (ca.DM): Input matrix (2x1) 
        F (ca.DM): Nonlinear term coefficient (2x1)
        Q (ca.DM): State cost matrix (2x2)
        R (ca.DM): Control cost matrix (1x1)
        x_lb (ca.DM): Lower bounds on states
        x_ub (ca.DM): Upper bounds on states
        u_lb (ca.DM): Lower bounds on controls
        u_ub (ca.DM): Upper bounds on controls
    """
    
    def __init__(self):
        self.x_dim = 2
        self.u_dim = 1
        
        # System matrices
        self.A = ca.DM([[1.0, 1.0],
                       [0.0, 1.0]])
        
        self.B = ca.DM([[0.5],
                       [1.0]])
        
        self.F = ca.DM([0.025, 0.025])
        
        # Cost matrices
        self.Q = ca.DM([[0.8, 0.0],
                       [0.0, 0.8]])
        
        self.R = ca.DM([[0.1]])
        
        # Bounds
        self.x_lb = ca.DM([-10.0, -10.0])
        self.x_ub = ca.DM([10.0, 10.0])
        self.u_lb = ca.DM([-2.0])
        self.u_ub = ca.DM([2.0])
        
        # Create symbolic variables for the dynamics
        self._x_sym = ca.SX.sym('x', self.x_dim)
        self._u_sym = ca.SX.sym('u', self.u_dim)
        
        # Build symbolic expressions for dynamics and costs
        self._build_dynamics()
        self._build_costs()
    
    def _build_dynamics(self):
        """Builds the symbolic expression for system dynamics."""
        quadratic_term = ca.dot(self._x_sym, self._x_sym)
        next_state = ca.mtimes(self.A, self._x_sym) + \
                    ca.mtimes(self.B, self._u_sym) + \
                    self.F * quadratic_term
        
        self.dynamic_sym = next_state
                    
        self._dynamics_fn = ca.Function('dynamics',
                                      [self._x_sym, self._u_sym],
                                      [next_state],
                                      ['x', 'u'],
                                      ['next_x'])
    
    def _build_costs(self):
        """Builds the symbolic expressions for cost functions."""
        stage_cost = ca.mtimes([self._x_sym.T, self.Q, self._x_sym]) + \
                    ca.mtimes([self._u_sym.T, self.R, self._u_sym])
        
        terminal_cost = ca.mtimes([self._x_sym.T, self.Q, self._x_sym])

        self.stage_cost_sym = stage_cost
        self.terminal_cost_sym = terminal_cost
        
        self._stage_cost_fn = ca.Function('stage_cost',
                                        [self._x_sym, self._u_sym],
                                        [stage_cost],
                                        ['x', 'u'],
                                        ['cost'])
        
        self._terminal_cost_fn = ca.Function('terminal_cost',
                                           [self._x_sym],
                                           [terminal_cost],
                                           ['x'],
                                           ['cost'])
    
    def step(self, x, u):
        """Compute next state given current state and input.
        
        Args:
            x (ca.DM or numpy.ndarray): Current state (2,)
            u (ca.DM or numpy.ndarray): Current input (1,)
            
        Returns:
            ca.DM: Next state (2,)
        """
        if isinstance(x, np.ndarray):
            x = ca.DM(x)
        if isinstance(u, np.ndarray):
            u = ca.DM(u)
            
        return self._dynamics_fn(x=x, u=u)['next_x']
    
    def simulate(self, x0, U_seq):
        """Simulate system trajectory given initial state and input sequence.
        
        Args:
            x0 (ca.DM or numpy.ndarray): Initial state (2,)
            U_seq (ca.DM or numpy.ndarray): Sequence of inputs (N, 1)
            
        Returns:
            ca.DM: Sequence of states (N+1, 2)
        """
        if isinstance(x0, np.ndarray):
            x0 = ca.DM(x0)
        if isinstance(U_seq, np.ndarray):
            U_seq = ca.DM(U_seq)
            
        N = U_seq.shape[0]
        X_seq = ca.DM.zeros(N + 1, self.x_dim)
        X_seq[0, :] = x0
        
        for k in range(N):
            X_seq[k + 1, :] = self.step(X_seq[k, :], U_seq[k])
            
        return X_seq
    
    def stage_cost(self, x, u):
        """Compute stage cost for state-input pair.
        
        Args:
            x (ca.DM or numpy.ndarray): State (2,)
            u (ca.DM or numpy.ndarray): Input (1,)
            
        Returns:
            ca.DM: Stage cost value (scalar)
        """
        if isinstance(x, np.ndarray):
            x = ca.DM(x)
        if isinstance(u, np.ndarray):
            u = ca.DM(u)
            
        return self._stage_cost_fn(x=x, u=u)['cost']
    
    def terminal_cost(self, x):
        """Compute terminal cost for state.
        
        Args:
            x (ca.DM or numpy.ndarray): State (2,)
            
        Returns:
            ca.DM: Terminal cost value (scalar)
        """
        if isinstance(x, np.ndarray):
            x = ca.DM(x)
            
        return self._terminal_cost_fn(x=x)['cost']
    
    def total_cost_seq(self, X_seq, U_seq):
        """Compute total cost for a trajectory.
        
        Args:
            X_seq (ca.DM or numpy.ndarray): State sequence (N+1, 2)
            U_seq (ca.DM or numpy.ndarray): Input sequence (N, 1)
            
        Returns:
            ca.DM: Total cost value (scalar)
        """
        if isinstance(X_seq, np.ndarray):
            X_seq = ca.DM(X_seq)
        if isinstance(U_seq, np.ndarray):
            U_seq = ca.DM(U_seq)
            
        N = U_seq.shape[0]
        total_cost = 0
        
        for k in range(N):
            total_cost += self.stage_cost(X_seq[k, :], U_seq[k])
        
        total_cost += self.terminal_cost(X_seq[-1, :])
        return total_cost
    
class NLP:
    """A class which handles the transformation between as system model to the NLP representing the optimal control problem.        
    Conventions:
    - parameters: p
        - initial state: x_0
    - decision variables: w
        - states: x
        - controls: u
    - lagrange multipliers: nu, lam
        - equality constraints: nu
        - inequality constraints: lam
    - objective function: f
        - integrated stage cost
        - terminal cost
    - equality constraints: h
        - initial value embedding
        - dynamics (multiple shooting)
    - inequality constraints: g
    - NLP:  | min f(w,p)       |
            | s.t. h(w,p) = 0  |
            |      g(w,p) <= 0 |

    Unrolling states and controls to decision variable vector:
    - (x_0_0, x_0_1), x_1_0, x_1_1 .... x_N_0, x_N_1, u_0_0, u_0_1, u_1_0, u_1_1 .... u_N-1_0, u_N-1_1

    Dimensions of decision variables:
    - N(+1) times x_dim for states: x_0 to x_N (or x_1 to x_N if no initial value embedding)
    - N times u_dim for controls: u_0 to u_N-1

    Structure of decision variables:
    w = [x_0_0, x_0_1, x_1_0, x_1_1 .... x_N_0, x_N_1, u_0_0, u_0_1, u_1_0, u_1_1 .... u_N-1_0, u_N-1_1]    
    # if no initial value embedding: starting at x_1_0, x_1_1, ....
    # if initial value embedding: starting at x_0_0, x_0_1, ....

    indices: 
    x_k_i = x_{time step}_{state dimension}
    u_k_i = u_{time step}_{control dimension}

    Attributes
    ----------
    system : System model
        System model for the NMPC problem
    N : int
        Prediction/Control horizon
    OCP_mode : str, optional
        MS: Multiple Shooting, SS: Single Shooting, by default "MS"
    initial_value_embedding : bool, optional
        If True, initial value is embedded in the decision variables, by default False

    """

    def __init__(self,system,N,eps=1e-16,sigma=0.0):
        self.system = system # System model
        self.N = N # Prediction/Control horizon     
        self.device = system.device
        self.dtype = system.dtype
        self.eps = eps
        self.sigma = sigma

        self._sanity_checks()
        self._calc_dims()
        self._setup_functions()
    
    def clip_z(self,z):
        # clip z, such that lam >= 0 and x,u within bounds
        w,nu,lam = self.extract_primal_dual(z)
        X,U = self.extract_states_controls(w)
        X = torch.clip(X,self.system.x_lb,self.system.x_ub)
        U = torch.clip(U,self.system.u_lb,self.system.u_ub)
        lam = torch.clip(lam,0.0,float("inf"))
        return self.stack_primal_dual(self.stack_states_controls(X,U),nu,lam)

    def _sanity_checks(self):
        assert hasattr(self.system,"x_dim"), "System model must have attribute x_dim"
        assert hasattr(self.system,"u_dim"), "System model must have attribute u_dim"
        assert hasattr(self.system,"x_lb"), "System model must have attribute x_lb"
        assert hasattr(self.system,"x_ub"), "System model must have attribute x_ub"
        assert hasattr(self.system,"u_lb"), "System model must have attribute u_lb"
        assert hasattr(self.system,"u_ub"), "System model must have attribute u_ub"
        assert hasattr(self.system,"parallel_stage_cost"), "System model must have method parallel_stage_cost"
        assert hasattr(self.system,"terminal_cost"), "System model must have method terminal_cost"
        assert hasattr(self.system,"parallel_step"), "System model must have method parallel_step"

    def _calc_dims(self):
        # if self.initial_value_embedding:
        self.n_w = self.N * (self.system.x_dim + self.system.u_dim) + self.system.x_dim # Number of decision variables
        self.n_nu = (self.N+1)*self.system.x_dim # Number of equality constraints (initial value embedding + N-times system dynamics)
        self.n_lam = 2*self.N * (self.system.x_dim + self.system.u_dim) # Number of inequality constraints (N-times state bounds)
        # auxilliary
        self.n_p = self.system.x_dim
        self.n_wx = (self.N+1) * self.system.x_dim
        self.n_wu = self.N * self.system.u_dim
        self.n_z = self.n_w + self.n_nu + self.n_lam # Number of primal and dual variables

    def _setup_functions(self):
        # Derivatives
        self.dLdw_func = torch.func.grad(self.L_func,argnums=0)
        self.dLdw_conv_func = torch.func.grad(self.L_conv_func,argnums=0)
        self.dfdw_func = torch.func.grad(self.f_func,argnums=0)      
        self.dhdw_func = torch.func.jacrev(self.h_func,argnums=0) # TODO: change to JVP formulation.
        self.dgdw_func = torch.func.jacfwd(self.g_func,argnums=0) # TODO: change to JVP formulation.

        # Batch Functions
        self.f_batch_func = torch.vmap(self.f_func)
        self.h_batch_func = torch.vmap(self.h_func)
        self.g_batch_func = torch.vmap(self.g_func)
        self.L_batch_func = torch.vmap(self.L_func)
        self.dLdw_batch_func = torch.vmap(self.dLdw_func)
        self.dfdw_batch_func = torch.vmap(self.dfdw_func)
        # self.dhdw_batch_func = torch.vmap(self.dhdw_func)
        # self.dgdw_batch_func = torch.vmap(self.dgdw_func)
        self.KKT_batch_func = torch.vmap(self.KKT_func)
        self.Fk_batch_func = torch.vmap(self.Fk_func)
        self.Tk_batch_func = torch.vmap(self.Tk_func)
        self.Fk_conv_batch_func = torch.vmap(self.Fk_conv_func)
        self.Tk_conv_batch_func = torch.vmap(self.Tk_conv_func)
        # self.Vk_batch_func = torch.vmap(self.Vk_func)
        self.gamma_batch_func = torch.vmap(self.gamma_func)
        self.gamma_conv_batch_func = torch.vmap(self.gamma_conv_func)
        self.DFk_Dz_batch_func = torch.vmap(self.DFk_Dz_func)
        self.cond_num_batch_func = torch.vmap(self.cond_num_func,chunk_size=1)

        # extraction and stacking functions
        self.extract_states_controls_batch = torch.vmap(self.extract_states_controls)
        self.stack_states_controls_batch = torch.vmap(self.stack_states_controls)
        self.stack_primal_dual_batch = torch.vmap(self.stack_primal_dual)
        self.extract_primal_dual_batch = torch.vmap(self.extract_primal_dual)
        self.clip_z_batch = torch.vmap(self.clip_z)

        # self._prepare_dhdw()
        # self.dnuh_dw_func = torch.func.grad(self.nu_h_func,argnums=0)

        # self.dLdw_func = self.dLdw_fast
        # self.scale_lagmul_pred_batch_func = torch.vmap(self.scale_lagmul_pred_func)
        # self.scale_lagmul_solver_batch_func = torch.vmap(self.scale_lagmul_solver_func)

        # Norm Functions
        # self.DTk_Dz_func = torch.func.grad(self.Tk_func,argnums=0)
        # self.DTk_Dz_batch_func = torch.vmap(self.DTk_Dz_func)
        # self.Rk_batch_func = torch.vmap(self.Rk_func)
        # self.dfdw_norm_func = lambda w,p: torch.norm(self.dfdw_func(w,p),p=2)
        # self.dfdw_norm_batch_func = torch.vmap(self.dfdw_norm_func)   
        # self.dhdw_norm_func = lambda w,p: torch.norm(self.dhdw_func(w,p),p=2,dim=1)
        # self.dhdw_norm_batch_func = torch.vmap(self.dhdw_norm_func)
        # self.dgdw_norm_func = lambda w,p: torch.norm(self.dgdw_func(w,p),p=2,dim=1)
        # self.dgdw_norm_batch_func = torch.vmap(self.dgdw_norm_func)
    
    def move_to_device(self,device,dtype=None):
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = self.system.dtype
        self.system = NonlinearDoubleIntegrator(dtype=self.dtype,device=device)
        self._sanity_checks()
        self._calc_dims()
        self._setup_functions()

    def extract_states_controls(self,w):
        """Extract states and controls from decision variable vector w.
        
        Parameters
        ----------
        w : torch.Tensor
            Decision variable vector
            Dimensions: n_w
        Returns
        -------
        torch.Tensor, torch.Tensor
            States, Controls
            Dimensions: (N+1) x x_dim, N x u_dim
        """
        X = w[:self.n_wx].view(self.N+1,self.system.x_dim)
        U = w[self.n_wx:].view(self.N,self.system.u_dim)
        # X = w[:self.n_wx].reshape(self.N+1,self.system.x_dim)
        # U = w[self.n_wx:].reshape(self.N,self.system.u_dim)
        return X,U
    
    def stack_states_controls(self,X,U):
        """Stack states and controls to decision variable vector w.
        
        Parameters
        ----------
        X : torch.Tensor
            States
            Dimensions: (N+1) x x_dim
        U : torch.Tensor
            Controls
            Dimensions: N x u_dim
        
        Returns
        -------
        torch.Tensor
            Decision variable vector
            Dimensions: n_w
        """
        X = X.view(-1)
        U = U.view(-1)
        # X = X.reshape(-1)
        # U = U.reshape(-1)
        w = torch.hstack([X,U])
        return w

    def stack_primal_dual(self,w,nu,lam):
        return torch.hstack([w,nu,lam])
    
    def extract_primal_dual(self,z):
        w = z[:self.n_w]
        nu = z[self.n_w:self.n_w+self.n_nu]
        lam = z[self.n_w+self.n_nu:]
        return w,nu,lam
    
    def f_func(self,w,p):
        """Objective function for the NMPC problem.
        
        Parameters
        ----------
        w : torch.Tensor
            Decision variable vector
            Dimensions: n_w
        p : torch.Tensor
            Parameters
            Dimensions: n_p
        
        Returns
        -------
        torch.Tensor
            Objective function value
            Dimensions: 1
        """
        X,U = self.extract_states_controls(w)
        # X = w[:self.n_wx].view(self.N+1,self.system.x_dim)
        # U = w[self.n_wx:].view(self.N,self.system.u_dim)
        # X = w[:self.n_wx].reshape(self.N+1,self.system.x_dim)
        # U = w[self.n_wx:].reshape(self.N,self.system.u_dim)
        stage_cost_val = self.system.parallel_stage_cost(X[:-1],U).sum()
        terminal_cost_val = self.system.terminal_cost(X[-1])
        return (stage_cost_val + terminal_cost_val)*0.1         
   
    def h_func(self,w,p):
        """Equality constraints for the NMPC problem.
        
        Parameters
        ----------
        w : torch.Tensor
            Decision variable vector
            Dimensions: n_w
        p : torch.Tensor
            Parameters
            Dimensions: n_p
        
        Returns
        -------
        torch.Tensor
            Equality constraints value
            Dimensions: n_nu
        """
        x0 = p
        X,U = self.extract_states_controls(w)
        # X = w[:self.n_wx].view(self.N+1,self.system.x_dim)
        # U = w[self.n_wx:].view(self.N,self.system.u_dim)

        # Initial value embedding
        h0 = X[0,:] - x0
        # h0 = h0.view(-1)
        # h0 = h0.reshape(-1)

        # Multiple shooting
        # h = self.system.parallel_step(X[:-1,:],U) - X[1:,:]
        h  = (self.system.parallel_step(X[:-1], U) - X[1:]).flatten()
        # h = h.view(-1)
        # h = h.reshape(-1)

        return torch.hstack([h0,h])

    # def h_func(self, w, p):
    #     """
    #     Equality constraints for the NMPC problem.
    #     Shapes:
    #     X: (N+1, x_dim), U: (N, u_dim), p/x0: (x_dim,)
    #     A: (x_dim, x_dim), B: (x_dim, u_dim), F: (x_dim,)
    #     Returns:
    #     h: (x_dim + N*x_dim,)
    #     """
    #     # Unpack decision vector into states/controls
    #     x0 = p
    #     X, U = self.extract_states_controls(w)   # X: (N+1, x_dim), U: (N, u_dim)

    #     # Keep memory contiguous (helps GEMM)
    #     X = X.contiguous()
    #     U = U.contiguous()

    #     # Optionally cache these transposes once in __init__ as buffers
    #     A_T = self.system.A.transpose(0, 1).contiguous()     # (x_dim, x_dim)
    #     B_T = self.system.B.transpose(0, 1).contiguous()     # (u_dim, x_dim)
    #     Fv  = self.system.F.view(1, -1)                      # (1, x_dim)

    #     # Initial-condition residual
    #     h0 = (X[0] - x0).view(-1)                     # (x_dim,)

    #     # Vectorized multiple-shooting residuals (no loops, no scalars)
    #     Xk   = X[:-1]                                  # (N, x_dim)
    #     Uk   = U                                       # (N, u_dim)
    #     quad = (Xk * Xk).sum(dim=1, keepdim=True)      # (N, 1)

    #     # Two GEMMs + one broadcasted mul/add
    #     Ax   = Xk.matmul(A_T)                          # (N, x_dim)
    #     Bu   = Uk.matmul(B_T)                          # (N, x_dim)
    #     Xkp1_pred = Ax + Bu + quad * Fv                # (N, x_dim)

    #     h_dyn = (Xkp1_pred - X[1:]).reshape(-1)        # (N*x_dim,)
    #     return torch.cat([h0, h_dyn], dim=0)

    
    def nu_h_func(self,w,p,nu):
        return nu@self.h_func(w,p)
        
    def g_func(self,w,p):
        """Inequality constraints for the NMPC problem.
        
        Parameters
        ----------
        w : torch.Tensor
            Decision variable vector
            Dimensions: n_w
        p : torch.Tensor
            Parameters
            Dimensions: n_p
        
        Returns
        -------
        torch.Tensor
            Inequality constraints value
            Dimensions: n_lam
        """
        X,U = self.extract_states_controls(w)
        # X = w[:self.n_wx].view(self.N+1,self.system.x_dim)
        # U = w[self.n_wx:].view(self.N,self.system.u_dim)

        # # State bounds (for x1,...,xN, not for x0)
        # g_lbx = self.system.x_lb.repeat(self.N,1) - X[1:,:]
        g_lbx = self.system.x_lb.unsqueeze(0) - X[1:]
        g_lbx = g_lbx.view(-1)
        # g_lbx = g_lbx.reshape(-1)
        # g_ubx = X[1:,:] - self.system.x_ub.repeat(self.N,1)
        g_ubx = X[1:] - self.system.x_ub.unsqueeze(0)
        g_ubx = g_ubx.view(-1)
        # g_ubx = g_ubx.reshape(-1)

        # # Control bounds
        # g_lbu = self.system.u_lb.repeat(self.N,1) - U
        g_lbu = self.system.u_lb.unsqueeze(0) - U
        g_lbu = g_lbu.view(-1)
        # g_lbu = g_lbu.reshape(-1)
        # g_ubu = U - self.system.u_ub.repeat(self.N,1)
        g_ubu = U - self.system.u_ub.unsqueeze(0)
        g_ubu = g_ubu.view(-1)
        # g_ubu = g_ubu.reshape(-1)

        return torch.hstack([g_lbx,g_ubx,g_lbu,g_ubu])
    
    def L_func(self,w,nu,lam,p):
        """Lagrangian function for the NMPC problem.

        Parameters
        ----------
        w : torch.Tensor
            Decision variable vector
            Dimensions: n_w
        nu : torch.Tensor
            Lagrange multipliers for equality constraints
            Dimensions: n_nu
        lam : torch.Tensor
            Lagrange multipliers for inequality constraints
            Dimensions: n_lam
        p : torch.Tensor
            Parameters
            Dimensions: n_p

        Returns
        -------
        torch.Tensor
            Lagrangian function value
            Dimensions: 1
        """
        
        f_val = self.f_func(w,p) #*self.df
        h_val = self.h_func(w,p) #*self.dh
        g_val = self.g_func(w,p) #*self.dg

        # return f_val + torch.sum(nu*h_val) + torch.sum(lam*g_val)
        # return f_val + nu.detach()@h_val + lam.detach()@g_val
        return f_val + nu@h_val + lam@g_val
    
    # def L_conv_func(self,w,nu,lam,p):
    #     w_bar = w.detach().clone()
    #     f_val = self.f_func(w,p)
    #     h_val = self.h_func(w_bar,p) + self.dhdw_func(w_bar,p)@(w - w_bar)
    #     g_val = self.g_func(w,p)
    #     return f_val + nu@h_val + lam@g_val

    def L_conv_func(self, w, nu, lam, p):
        # ensure all tensors share device/dtype before compile
        # (helps Dynamo/Inductor avoid hidden transfers)
        w_bar = w.detach().clone()

        f_val = self.f_func(w, p)
        # Compute J_h(w_bar) @ (w - w_bar) via JVP (no full Jacobian)
        v = w - w_bar
        # jvp returns (h(w_bar),  J_h(w_bar) @ v)
        h_base, h_lin = torch.func.jvp(self.h_func, (w_bar, p), (v, torch.zeros_like(p)))
        h_val = h_base + h_lin

        g_val = self.g_func(w, p)
        return f_val + nu @ h_val + lam @ g_val

    ### Sampling Functions ###
    def batch_gen_x0(self,N_batch,offset=0):
        """Generate random parameters for N_batch samples.
        
        Parameters
        ----------
        N_batch : int
            Number of samples
        offset : float, optional
            Offset for the parameters, by default 0
            Provides percentage of the range which should exceed the bounds, e.g. 0.1 for 10% of the range to add to the bounds        
        Returns
        -------
        torch.Tensor
            Parameters
            Dimensions: N_batch x n_p
        """
        # Sample initial states x0 uniformly from the state space (see bounds)
        lb = self.system.x_lb*(1+offset)
        ub = self.system.x_ub*(1+offset)

        x0 = torch.rand(N_batch,self.n_p,device=self.device)
        x0 = lb + x0*(ub-lb)
        
        return x0
    
    def batch_gen_p(self,N_batch,offset=0):
        return self.batch_gen_x0(N_batch,offset)
    
    def batch_gen_w(self,N_batch,offset=0):
        """Generate random decision variable vectors for N_batch samples.

        Parameters
        ----------
        N_batch : int
            Number of samples
        offset : float, optional

        Returns
        -------
        torch.Tensor
            Decision variable vectors
            Dimensions: N_batch x n_w
        """
        # Sample decision variables w uniformly from the state and control space (see bounds)
        lbu = self.system.u_lb*(1+offset)
        ubu = self.system.u_ub*(1+offset)
        lbx = self.system.x_lb*(1+offset)
        ubx = self.system.x_ub*(1+offset)

        # sample control actions
        U = torch.rand(N_batch,self.N,self.system.u_dim,device=self.device)
        U = lbu + U*(ubu-lbu)
        # sample states
        X = torch.rand(N_batch,self.N+1,self.system.x_dim,device=self.device)
        X = lbx + X*(ubx-lbx)
        w = self.stack_states_controls_batch(X,U)
        return w
    
    def batch_gen_z(self,N_batch,offset=0):
        """Generate random primal and dual variables for N_batch samples.

        Parameters
        ----------
        N_batch : int
            Number of samples
        offset : float, optional

        Returns
        -------
        torch.Tensor
            Primal and dual variables
            Dimensions: N_batch x n_z
        """
        w = self.batch_gen_w(N_batch,offset)
        nu = torch.randn(N_batch,self.n_nu,device=self.device)
        lam = torch.rand(N_batch,self.n_lam,device=self.device) # >= 0
        z = self.stack_primal_dual_batch(w,nu,lam)
        return z

    def KKT_func(self,z,p):
        w,nu,lam = self.extract_primal_dual(z)
        dLdw_val = self.dLdw_func(w,nu,lam,p)
        h_val = self.h_func(w,p)
        g_val = self.g_func(w,p)
        g_val_plus = torch.relu(g_val)
        dual_feas_minus = torch.relu(-lam)
        comp_slack = lam*g_val
        KKT_val = torch.hstack([dLdw_val,h_val,g_val_plus,dual_feas_minus,comp_slack])
        return KKT_val
    
    # KKT conditions
    def Fk_func(self,z,p):
        w,nu,lam = self.extract_primal_dual(z)
        nu_bar = nu.detach().clone()
        lam_bar = lam.detach().clone()
        dLdw_val = self.dLdw_func(w,nu,lam,p)
        h_val = self.h_func(w,p) - self.sigma *(nu - nu_bar)
        g_val = self.g_func(w,p) - self.sigma *(lam - lam_bar)
        
        # g_val_plus = torch.relu(g_val)
        # dual_feas_minus = torch.relu(-lam)
        # # comp_slack = -torch.relu(lam)*torch.relu(-g_val)

        # comp_slack = -torch.relu(lam)*torch.relu(-g_val)
        # KKT_val = torch.hstack([dLdw_val, h_val, g_val_plus + dual_feas_minus + comp_slack])
        
        # comp_slack = lam*g_val
        # KKT_val = torch.hstack([dLdw_val, h_val, g_val_plus + dual_feas_minus, comp_slack])

        # comp_slack = lam*g_val
        # KKT_val = torch.hstack([dLdw_val,h_val,g_val_plus,dual_feas_minus,comp_slack])

        # fb_cond = lam - g_val - torch.sqrt(lam**2 + g_val**2 + 1e-12)

        # fb_cond = lam - g_val - torch.sqrt((lam+g_val)**2 + 1e-12)
        # fb_cond = 0.8*(lam - g_val - torch.sqrt((lam+g_val)**2)) + 0.2*(torch.relu(lam)*torch.relu(-g_val))
        # fb_cond = lam - g_val - torch.sqrt((lam+g_val)**2 + 4*1e-8)

        # KKT_val = torch.hstack([dLdw_val, h_val, -fb_cond - 1e-6*(lam-lam.detach())])

        # fb_cond = 0.8*(lam - g_val - torch.sqrt(lam**2 + g_val**2)) + 0.2*(torch.relu(lam)*torch.relu(-g_val))
        fb_cond = 0.8*(lam - g_val - torch.sqrt(lam**2 + g_val**2 + self.eps)) + 0.2*(torch.relu(lam)*torch.relu(-g_val))
        KKT_val = torch.hstack([dLdw_val, h_val, fb_cond])        
         
        # g_val_plus = torch.relu(g_val)
        # dual_feas_minus = torch.relu(-lam)
        # comp_slack = torch.relu(lam)*torch.relu(-g_val)
        # KKT_val = torch.hstack([dLdw_val, h_val, g_val_plus, dual_feas_minus, comp_slack])

        return KKT_val
        
    def Tk_func(self,z,p):
        Fk = self.Fk_func(z,p)
        Tk = 0.5*torch.dot(Fk,Fk)
        # Tk = torch.norm(Fk,p=2)
        return Tk
        
    def Fk_conv_func(self,z,p):
        w,nu,lam = self.extract_primal_dual(z)
        nu_bar = nu.detach().clone()
        lam_bar = lam.detach().clone()
        # sigma = 1e-2
        # dLdw_val = self.dLdw_conv_func(w,nu,lam,p,w_bar)
        dLdw_val = self.dLdw_conv_func(w,nu,lam,p)
        h_val = self.h_func(w,p) - self.sigma *(nu - nu_bar)
        g_val = self.g_func(w,p) - self.sigma *(lam - lam_bar)
        # fb_cond = 0.8*(lam - g_val - torch.sqrt(lam**2 + g_val**2)) + 0.2*(torch.relu(lam)*torch.relu(-g_val))
        fb_cond = 0.8*(lam - g_val - torch.sqrt(lam**2 + g_val**2 + self.eps)) + 0.2*(torch.relu(lam)*torch.relu(-g_val))
        KKT_val = torch.hstack([dLdw_val, h_val, fb_cond])     
        return KKT_val
        
    def Tk_conv_func(self,z,p):
        Fk_conv = self.Fk_conv_func(z,p)
        Tk_conv = 0.5*torch.dot(Fk_conv,Fk_conv)
        # Tk_conv = torch.norm(Fk_conv,p=2)
        return Tk_conv
        
    def gamma_func(self,z,p,dz):
        """Function to compute the step size."""
        Fk, jvpFk = torch.func.jvp(self.Fk_func, (z,p), (dz,torch.zeros_like(p)))
        # Fk, jvpFk = torch.func.jvp(self.KKT_func, (z,p), (dz,torch.zeros_like(p)))
        gamma = - Fk@jvpFk/(jvpFk@jvpFk)
        return gamma
        
    def gamma_conv_func(self,z,p,dz):
        """Function to compute the step size."""
        Fk, jvpFk = torch.func.jvp(self.Fk_conv_func, (z,p), (dz,torch.zeros_like(p)))
        # Fk, jvpFk = torch.func.jvp(self.Fk_func, (z,p), (dz,torch.zeros_like(p)))
        # Fk, jvpFk = torch.func.jvp(self.KKT_func, (z,p), (dz,torch.zeros_like(p)))
        gamma = - Fk@jvpFk/(jvpFk@jvpFk)
        return gamma

    # KKT Matrix
    def DFk_Dz_func(self,z,p):
        DFk_Dz_val = torch.func.jacfwd(self.Fk_func, argnums=0)(z,p)
        return DFk_Dz_val
    
    def cond_num_func(self,z,p):
        # calculate KKT matrix
        DFk_Dz = self.DFk_Dz_func(z,p)
        # calculate condition number
        cond_num = torch.linalg.cond(DFk_Dz)
        return cond_num
    
    # def Vk_func(self,z,p,dz):
    #     """Newton type loss as in Lüken L. and Lucia S. (2024)"""
    #     Fk, jvpFk = torch.func.jvp(self.Fk_func, (z,p), (dz,torch.zeros_like(p)))
    #     dFk = Fk+jvpFk
    #     Vk = 0.5*torch.dot(dFk,dFk)
    #     return Vk
    
    # # Predictor Lagrange Multiplier Scaling
    # def scale_lagmul_pred_func(self,z): # TODO: check whether to differentiate w.r.t torch.abs(val) or detach
    #     # core idea: take predicted lagrange multipliers nu and lambda and scale with its own absolute value
    #     # example: nu = max(torch.abs(nu_pred),1) * nu_pred
    #     w,nu,lam = self.extract_primal_dual(z)
    #     nu = torch.max(torch.abs(nu),torch.ones_like(nu))*nu
    #     lam = torch.max(torch.abs(lam),torch.ones_like(lam))*lam
    #     z = self.stack_primal_dual(w,nu,lam)
    #     return z
    
    # def scale_lagmul_solver_func(self,dz,z):
    #     # core idea: solver predicts only scaled step dz, which needs to be unscaled
    #     # dnu = max(torch.abs(nu),1) * dnu
    #     # dlam = max(torch.abs(lam),1) * dlam
    #     w,nu,lam = self.extract_primal_dual(z)
    #     dw,dnu,dlam = self.extract_primal_dual(dz)
    #     dnu = torch.maximum(torch.abs(nu.detach()),torch.ones_like(nu.detach()))*dnu
    #     dlam = torch.maximum(torch.abs(lam.detach()),torch.ones_like(lam.detach()))*dlam
    #     dz = self.stack_primal_dual(dw,dnu,dlam)
    #     return dz


    # extra
    def Rk_func(self,z,p):
        DTk_Dz = self.DTk_Dz_func(z,p)
        return torch.norm(DTk_Dz,p=2)    
    
    # export hparams
    def export_hparams(self):
        hparams = {
            "N": self.N,
            # "rho": self.rho,
        }
        return hparams

class NLPCasadi:    
    def __init__(self, system, N):
        """Initialize the NMPC problem with CasADi.
        
        Args:
            system (NonlinearDoubleIntegratorCasadi): System model
            N (int): Prediction/Control horizon
        """
        self.system = system
        self.N = N
        
        # Calculate dimensions
        self._calc_dims()
        
        # Create symbolic variables
        self._setup_variables()
        
        # Build optimization problem
        self._build_nlp()
        self._build_standard_nlp() # for standard NLP formulation for Fk_func etc.
        
    def _calc_dims(self):
        """Calculate dimensions for the optimization problem."""
        self.n_w = self.N * (self.system.x_dim + self.system.u_dim) + self.system.x_dim
        self.n_p = self.system.x_dim
        self.n_nu = (self.N + 1) * self.system.x_dim
        self.n_lam = 2 * self.N * (self.system.x_dim + self.system.u_dim)
        self.n_z = self.n_w + self.n_nu + self.n_lam
        
        # Auxiliary dimensions
        self.n_wx = (self.N + 1) * self.system.x_dim
        self.n_wu = self.N * self.system.u_dim
        
    def _setup_variables(self):
        """Create symbolic variables for the optimization problem."""
        # States
        self.X = ca.SX.sym('X', self.N + 1, self.system.x_dim)
        
        # Controls
        self.U = ca.SX.sym('U', self.N, self.system.u_dim)
        
        # Parameters (initial state)
        self.P = ca.SX.sym('P', self.system.x_dim)
        
        # Create decision variable vector
        x = ca.reshape(self.X.T, -1, 1)
        u = ca.reshape(self.U.T, -1, 1)
        self.w = ca.vertcat(x, u)

    def _build_objective(self):        
        # Objective function - f
        obj = 0
        for k in range(self.N):
            stage_cost = self.system._stage_cost_fn(self.X[k, :], self.U[k, :])
            obj += stage_cost
        obj += self.system._terminal_cost_fn(self.X[-1, :])
        return obj*0.1 # CRITICAL: scaling factor hardcoded here (same as in torch model)

    def _build_equality_constraints(self):
        # Equality constraints - h
        h = []
        
        # initial value embedding
        # h.append(self.X[0, :].T - self.P)
        h.append((self.X[0, :].T - self.P).T)

        # Dynamics
        for k in range(self.N):
            x_next = self.system._dynamics_fn(self.X[k, :], self.U[k, :])
            # h.append(x_next - self.X[k + 1, :].T)
            h.append((x_next - self.X[k + 1, :].T).T)
        
        # reshape h to a vector
        h = ca.horzcat(*h)
        h = ca.reshape(h.T, -1, 1)
        return h
    
    def _build_inequality_constraints(self):
        # Inequality constraints - g
        # State bounds
        g_lbx = self.system.x_lb - self.X[1:, :].T
        g_lbx = ca.reshape(g_lbx, -1, 1)
        g_ubx = self.X[1:, :].T - self.system.x_ub
        g_ubx = ca.reshape(g_ubx, -1, 1)

        # Control bounds
        g_lbu = self.system.u_lb - self.U.T
        g_lbu = ca.reshape(g_lbu, -1, 1)
        g_ubu = self.U.T - self.system.u_ub
        g_ubu = ca.reshape(g_ubu, -1, 1)

        g = ca.vertcat(g_lbx, g_ubx, g_lbu, g_ubu)
        return g   

    def _build_nlp(self):
        """Build the nonlinear programming problem."""
        
        # Objective function - f
        obj = self._build_objective()

        # Equality constraints - h
        h = self._build_equality_constraints()

        # Inequality constraints - g
        g = self._build_inequality_constraints()

        # bounds for constraints
        lower_bounds_h = np.zeros(self.n_nu)
        upper_bounds_h = np.zeros(self.n_nu)
        lower_bounds_g = -np.inf*np.ones(self.n_lam)
        upper_bounds_g = np.zeros(self.n_lam)

        # Merge constraints (for casadi solver)
        g = ca.vertcat(h, g)
        lower_bounds = np.hstack([lower_bounds_h, lower_bounds_g])
        upper_bounds = np.hstack([upper_bounds_h, upper_bounds_g])

        # Create NLP
        self.nlp_bounds = {
            # 'lbx': -np.inf * np.ones_like(self.w),
            # 'ubx': np.inf * np.ones_like(self.w),
            'lbg': lower_bounds,
            'ubg': upper_bounds
        }

        self.nlp = {
            'x': self.w,
            'f': obj,
            'g': g,
            'p': self.P
        }

    def _build_standard_nlp(self):
        # eps_fb = 0.0
        w_sym = self.w
        f_sym = self._build_objective()
        h_sym = self._build_equality_constraints()
        g_sym = self._build_inequality_constraints()
        p_sym = self.P
        
        nu_sym = ca.SX.sym('nu', self.n_nu)
        lam_sym = ca.SX.sym('lam', self.n_lam)
        z_sym = ca.vertcat(w_sym, nu_sym, lam_sym)

        L_sym = f_sym + ca.dot(nu_sym, h_sym) + ca.dot(lam_sym, g_sym)
        dLdw_sym = ca.jacobian(L_sym, w_sym).T
        dfdw_sym = ca.jacobian(f_sym, w_sym).T
        dhdw_sym = ca.jacobian(h_sym, w_sym)
        dgdw_sym = ca.jacobian(g_sym, w_sym)

        # Build KKT conditions
        g_plus_sym = ca.if_else(g_sym > 0, g_sym, 0)
        g_minus_sym = ca.if_else(g_sym < 0, -g_sym, 0)
        lam_plus_sym = ca.if_else(lam_sym > 0, lam_sym, 0)
        lam_minus_sym = ca.if_else(lam_sym < 0, -lam_sym, 0)

        comp_slack_sym = lam_sym * g_sym
        KKT_sym = ca.vertcat(dLdw_sym, h_sym, g_plus_sym, lam_minus_sym, comp_slack_sym)
        
        fb_sym = 0.8*(lam_sym - g_sym - ca.sqrt(lam_sym**2 + g_sym**2)) + 0.2*(lam_plus_sym*g_minus_sym)
        
        Fk_sym = ca.vertcat(dLdw_sym,h_sym,fb_sym)
        # Tk_sym = ca.norm_2(Fk_sym)
        Tk_sym = 0.5*ca.dot(Fk_sym, Fk_sym)

        f_func = ca.Function('f', [w_sym, p_sym], [f_sym])
        h_func = ca.Function('h', [w_sym, p_sym], [h_sym])
        g_func = ca.Function('g', [w_sym, p_sym], [g_sym])

        L_func = ca.Function('L', [w_sym, nu_sym, lam_sym, p_sym], [L_sym])
        dLdw_func = ca.Function('dLdw', [w_sym, nu_sym, lam_sym, p_sym], [dLdw_sym])
        dfdw_func = ca.Function('dfdw', [w_sym, p_sym], [dfdw_sym])
        dhdw_func = ca.Function('dhdw', [w_sym, p_sym], [dhdw_sym])
        dgdw_func = ca.Function('dgdw', [w_sym, p_sym], [dgdw_sym])
        Fk_func = ca.Function('Fk', [z_sym,p_sym], [Fk_sym])
        KKT_func = ca.Function('Fk', [z_sym,p_sym], [KKT_sym])
        Tk_func = ca.Function('Tk', [z_sym,p_sym], [Tk_sym])

        # gamma func
        dw_sym = ca.SX.sym("dw_sym",self.n_w) # Decision Variable
        dnu_sym = ca.SX.sym("dnu_sym",self.n_nu) # Lagrange Multiplier
        dlam_sym = ca.SX.sym("dlam_sym",self.n_lam) # Lagrange Multiplier
        dz_sym = ca.vertcat(dw_sym, dnu_sym, dlam_sym) # Step
        # dz_sym = ca.SX.sym("dz_sym",self.n_z) # Solver Step
        jvp_sym = ca.jtimes(Fk_sym,z_sym,dz_sym,False)
        gamma_sym = - (Fk_sym.T@jvp_sym) / (jvp_sym.T@ jvp_sym + 1e-16)
        gamma_func = ca.Function('gamma', [z_sym, p_sym, dz_sym], [gamma_sym])

        # gamma convexified func
        # dLdw_conv_sym = dfdw_sym + dhdw_sym@nu_sym + dgdw_sym@lam_sym
        # KKT_conv_sym = ca.vertcat(dfdw_sym, h_sym, g_plus_sym, lam_minus_sym, comp_slack_sym)
        Fk_conv_sym = ca.vertcat(dfdw_sym, h_sym, fb_sym)
        jvp_conv_w_sym = ca.jtimes(Fk_conv_sym,w_sym,dw_sym,False)
        jvp_conv_nu_sym = ca.jtimes(Fk_sym,nu_sym,dnu_sym,False)
        jvp_conv_lam_sym = ca.jtimes(Fk_sym,lam_sym,dlam_sym,False)
        jvp_conv_sym = jvp_conv_w_sym+jvp_conv_nu_sym+jvp_conv_lam_sym
        # jvp_conv_sym = ca.jtimes(KKT_conv_sym,z_sym,dz_sym,False)
        gamma_conv_sym = - (Fk_sym.T@jvp_conv_sym) / (jvp_conv_sym.T@ jvp_conv_sym + 1e-16)
        gamma_conv_func = ca.Function('gamma_conv', [z_sym, p_sym, dz_sym], [gamma_conv_sym])

        self.standard_nlp = {
            "w": self.w,
            "f": self._build_objective(),
            "g": self._build_equality_constraints(),
            "h": self._build_inequality_constraints(),
            "p": self.P,
            "nu": nu_sym,
            "lam": lam_sym,
            "L": L_sym,
            "dLdw": dLdw_sym
        }

        self.f_func = f_func
        self.h_func = h_func
        self.g_func = g_func
        self.L_func = L_func
        self.dLdw_func = dLdw_func
        self.dfdw_func = dfdw_func
        self.dhdw_func = dhdw_func
        self.dgdw_func = dgdw_func
        self.KKT_func = KKT_func
        self.Fk_func = Fk_func
        self.Tk_func = Tk_func

        self.gamma_func = gamma_func
        self.gamma_conv_func = gamma_conv_func

    def _build_solver_step(self,nn_model):
        @torch.no_grad()
        def solve_step(zk,p):
            Fk = self.Fk_func(zk,p)
            norm_Fk = ca.norm_2(Fk) + 1e-16
            norm_Fk_log = ca.log(norm_Fk)
            KKT_normalized = Fk/norm_Fk
            nn_inputs = np.hstack((p,KKT_normalized.toarray().squeeze(),norm_Fk_log.toarray().squeeze()))
            nn_inputs = torch.Tensor(nn_inputs).to("cpu")
            norm_Fk = torch.Tensor(norm_Fk.toarray()).to("cpu")
            # clip norm_Fk to max 1.0
            norm_Fk = torch.clamp(norm_Fk, max=1.0)
            dzk = nn_model(nn_inputs)*norm_Fk

            dzk = dzk.numpy()

            # infinity-norm KKT error at k (output side)
            # KKT_inf = ca.norm_inf(KKT) # TODO: adapt evaluation code as well as other examples to match this
            return np.array(dzk).squeeze(),np.array(norm_Fk).squeeze()
        
        # @torch.no_grad()
        # def solve_step(zk,p):
        #     # Fk = self.Fk_func(zk,p)
        #     KKT = self.KKT_func(zk,p)
        #     norm_KKT = ca.norm_2(KKT) + 1e-16
        #     norm_KKT_log = ca.log(norm_KKT)
        #     KKT_normalized = KKT/norm_KKT
        #     nn_inputs = np.hstack((p,KKT_normalized.toarray().squeeze(),norm_KKT_log.toarray().squeeze()))
        #     nn_inputs = torch.Tensor(nn_inputs).to("cpu")
        #     norm_KKT = torch.Tensor(norm_KKT.toarray()).to("cpu")
        #     dzk = nn_model(nn_inputs)*norm_KKT

        #     dzk = dzk.numpy()

        #     # infinity-norm KKT error at k (output side)
        #     KKT_inf = ca.norm_inf(KKT) # TODO: adapt evaluation code as well as other examples to match this
        #     return np.array(dzk).squeeze(),np.array(KKT_inf).squeeze()
        # def solve_step(zk,p):
        #     # Fk = self.Fk_func(zk,p)
        #     KKT = self.KKT_func(zk,p)
        #     KKT = torch.Tensor(KKT.toarray()).to("cpu")
        #     norm_KKT = (torch.linalg.vector_norm(KKT,ord=2) + 1e-16)#.to("cpu")
        #     norm_KKT_log = torch.log(norm_KKT)
        #     KKT_normalized = KKT/norm_KKT
        #     nn_inputs = torch.hstack((torch.tensor(p).to("cpu"),KKT_normalized.squeeze(),norm_KKT_log.squeeze()))
        #     dzk = nn_model(nn_inputs)*norm_KKT

        #     dzk = dzk.numpy()

        #     # infinity-norm KKT error at k (output side)
        #     KKT_inf = torch.linalg.vector_norm(KKT,ord=torch.inf) # TODO: adapt evaluation code as well as other examples to match this
        #     return dzk.squeeze(),KKT_inf.squeeze()
        # def solve_step(zk,p):
        #     # Fk = self.Fk_func(zk,p)
        #     KKT = self.KKT_func(zk,p)
        #     KKT = np.float32(np.array(KKT))  # Convert to numpy array for processing
        #     norm_KKT = np.float32(np.linalg.norm(KKT,ord=2) + 1e-16)
        #     norm_KKT_log = np.log(norm_KKT)
        #     KKT_normalized = KKT/norm_KKT
        #     nn_inputs = np.hstack((p,KKT_normalized.squeeze(),norm_KKT_log.squeeze()))
        #     nn_inputs = torch.Tensor(nn_inputs).to("cpu")
        #     norm_KKT = torch.tensor(norm_KKT).to("cpu")
        #     dzk = nn_model(nn_inputs)*norm_KKT

        #     dzk = dzk.numpy()

        #     # infinity-norm KKT error at k (output side)
        #     KKT_inf = np.float32(np.linalg.norm(KKT,ord=np.inf)) # TODO: adapt evaluation code as well as other examples to match this
        #     return dzk.squeeze(),KKT_inf.squeeze()
        return solve_step

    def solve(self, x0, opts=None):
        if opts is None:
            opts = {}
        
        solver = ca.nlpsol('solver', 'ipopt', self.nlp, opts)

        # Solve the NLP
        start = perf_counter()
        res = solver(p=x0, **self.nlp_bounds)
        end = perf_counter()
        w_opt = res['x'].full().flatten()
        status = solver.stats()['return_status']
        solve_time = end - start
        lam_g = res['lam_g'].full().flatten()
        nu_opt = lam_g[:self.n_nu]
        lam_opt = lam_g[self.n_nu:]
        return w_opt, nu_opt, lam_opt, status, solve_time
    
    def extract_states_controls(self, w):
        """Extract states and controls from decision variable vector.
        
        Args:
            w (np.ndarray): Decision variable vector
            
        Returns:
            tuple: States array, Controls array
        """
        w = w.flatten()
        X = np.reshape(w[:self.n_wx], (self.N + 1, self.system.x_dim))
        U = np.reshape(w[self.n_wx:], (self.N, self.system.u_dim))
        return X, U

    def stack_primal_dual(self, w, nu, lam):
        return np.hstack([w, nu, lam])
    
    def sample_x0(self,offset=0.0): # TODO: change to formulation using offset on range instead of absolute value
        """Generate random initial state."""
        # Sample initial states x0 uniformly from the state space (see bounds)
        lb = self.system.x_lb*(1+offset)
        ub = self.system.x_ub*(1+offset)
        x0 = np.random.uniform(lb, ub)
        return x0
    
class NMPCCasadi:
    def __init__(self,system_casadi,N):
        self.system = system_casadi
        self.N = N
        self.nlp = NLPCasadi(system_casadi,N)
        self.opts = None

    def solve_step(self,x0):
        # Solve NLP
        w_opt, nu_opt, lam_opt, status, solve_time = self.nlp.solve(x0,opts=self.opts)
        z_opt = np.hstack([w_opt, nu_opt, lam_opt])
        X_opt, U_opt = self.nlp.extract_states_controls(w_opt)
        u0 = U_opt[0,:]
        return u0, z_opt, status, solve_time
    
    def get_max_violation_step(self,x0,u0):
        violation_x_lb = np.maximum(0.0,self.system.x_lb - x0)
        violation_x_ub = np.maximum(0.0,x0 - self.system.x_ub)
        violation_u_lb = np.maximum(0.0,self.system.u_lb - u0)
        violation_u_ub = np.maximum(0.0,u0 - self.system.u_ub)
        violation = np.vstack([violation_x_lb,violation_x_ub,violation_u_lb,violation_u_ub])
        max_violation = np.max(violation)
        return max_violation
    
    def get_stage_cost(self,x0,u0):
        return self.system.stage_cost(x0,u0)
    
    def run_nmpc_closed_loop(self,x0,N_sim,noise=0.0):
        trajectory = []
        for k in range(N_sim):
            u0, z_opt, status, solve_time = self.solve_step(x0)

            stage_cost = self.get_stage_cost(x0,u0)
            stage_cost = stage_cost.toarray().squeeze()
            max_con_viol = self.get_max_violation_step(x0,u0)
            trajectory.append({"x":x0,"u":u0,"z_opt":z_opt,"status":status,"solve_time":solve_time, "max_con_viol":max_con_viol,"stage_cost":stage_cost})

            # step
            x1 = self.system.step(x0,u0)
            if noise != 0.0:
                x1 += noise*np.random.randn(*x1.shape) # additive white noise on state transition
            x0 = x1.toarray()

            # check optimizer status
            if status != "Solve_Succeeded":
                print(f"Failed at step {k}")
                return trajectory, False
    
        return trajectory, True

    def sample_data(self,N_trajectories,N_sim,offset=0.0,noise=0.0,filter_mode="all"):
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
            x0 = self.nlp.sample_x0(offset=offset)
            trajectory, trajectory_status = self.run_nmpc_closed_loop(x0,N_sim,noise=noise)

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
            
            # elif filter_mode == "successful_samples": # only consider successful samples
            #     for val in trajectory:
            #         if val["status"] == "Solve_Succeeded":
            #             x_data.append(val["x"])
            #             u_data.append(val["u"])
            #             status_data.append(val["status"])
            #             z_opt_data.append(val["z_opt"])
            #             solve_time_data.append(val["solve_time"])
            #             N_total += 1 # count all samples
            #             N_optimal += 1 # count all optimal samples
            #     if trajectory[0]["status"] == "Solve_Succeeded":
            #         k += 1 # count all trajectories for which first step is successful, useful for closed-loop sampling with noise
            #     N_visited += 1 # count all visited samples  
            
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

        return data, meta_data

# Helper functions for plotting
def visualize_trajectory(trajectory):
    fig1, ax1 = plt.subplots(3,1)
    ax1[0].plot([x['x'][0] for x in trajectory],label='x1')
    ax1[1].plot([x['x'][1] for x in trajectory],label='x2')
    ax1[2].plot([u['u'] for u in trajectory],label='u')
    plt.show()

    fig2, ax2 = plt.subplots()
    ax2.plot([x['x'][0] for x in trajectory],[x['x'][1] for x in trajectory],"x-")

    fig3, ax3 = plt.subplots()
    ax3.plot([traj["max_con_viol"] for traj in trajectory])

    fig4, ax4 = plt.subplots()
    ax4.plot([traj["stage_cost"] for traj in trajectory])
    # add second plot which integrates the stage cost on right axis
    integrated_stage_cost = np.cumsum([traj["stage_cost"] for traj in trajectory])
    ax4_twin = ax4.twinx()
    ax4_twin.plot(integrated_stage_cost, 'r-')
    
    return fig1, fig2, fig3, fig4

def plot_states(X_seq, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    
    X_seq_np = np.array(X_seq)
    ax.plot(X_seq_np[:,0], label='x1')
    ax.plot(X_seq_np[:,1], label='x2')
    ax.set_xlabel("time")
    ax.legend()
    return ax

def plot_controls(U_seq, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    
    U_seq_np = np.array(U_seq)
    ax.plot(U_seq_np, label='u')
    ax.set_xlabel("time")
    ax.legend()
    return ax

def plot_states_batch(X_seq_batch,axs=None):
    N_batch, N_seq, x_dim = X_seq_batch.shape
    if axs is None:
        fig, axs = plt.subplots(2,1)
    for k in range(N_batch):
        axs[0].plot(X_seq_batch[k,:,0],label=f"{k}")
        axs[1].plot(X_seq_batch[k,:,1],label=f"{k}")
    axs[0].set_xlabel("time")
    axs[1].set_xlabel("time")
    axs[0].set_ylabel("x1")
    axs[1].set_ylabel("x2")
    axs[0].legend()
    axs[1].legend()
    return axs

def plot_controls_batch(U_seq_batch,ax=None):
    N_batch, N_seq = U_seq_batch.shape
    if ax is None:
        fig, ax = plt.subplots(1,1)
    for k in range(N_batch):
        ax.plot(U_seq_batch[k,:],label=f"{k}")
    ax.set_xlabel("time")
    ax.set_ylabel("u")
    ax.legend()
    return ax

if __name__ == "__main__":
    N_control = 10
    N_sim = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"Using device: {device}")
    torch.set_default_device(device)
    dtype = torch.float64
    torch.set_default_dtype(dtype)

    system = NonlinearDoubleIntegrator(device=device)
    system_casadi = NonlinearDoubleIntegratorCasadi()
    nlp = NLP(system, N=N_control)
    nlp_casadi = NLPCasadi(system_casadi, N=N_control)
    nmpc_casadi = NMPCCasadi(system_casadi, N=N_control)

    # Generate random initial state
    x0 = torch.tensor([1.0, 1.0], dtype=torch.float32)

    # Solve NLP
    w_opt, nu_opt, lam_opt, status, solve_time = nlp_casadi.solve(x0.cpu().numpy())
    w_opt +=10.0
    nu_opt -= 10.0
    lam_opt += 10.0


    w_ca, nu_ca, lam_ca = w_opt, nu_opt, lam_opt
    p_ca = x0.cpu().numpy().copy()
    z_ca = ca.vertcat(w_ca, nu_ca, lam_ca)

    w_opt = torch.tensor(w_opt, dtype=torch.float32)
    nu_opt = torch.tensor(nu_opt, dtype=torch.float32)
    lam_opt = torch.tensor(lam_opt, dtype=torch.float32)
    z_opt = torch.hstack([w_opt, nu_opt, lam_opt])
    p = x0

    f_val = nlp.f_func(w_opt, p)
    print(f"Objective function value: {f_val}")
    h_val = nlp.h_func(w_opt, p)
    print(f"Equality constraints value: {h_val}")
    g_val = nlp.g_func(w_opt, p)
    print(f"Inequality constraints value: {g_val}")


    nlp_casadi._build_standard_nlp()
    L_val = nlp_casadi.L_func(w_ca, nu_ca, lam_ca, p_ca)
    dLdw_val = nlp_casadi.dLdw_func(w_ca, nu_ca, lam_ca, p_ca)
    Fk_val = nlp_casadi.Fk_func(z_ca, p_ca)
    Tk_val = nlp_casadi.Tk_func(z_ca, p_ca)

    Fk_pt = nlp.Fk_func(z_opt,p)

    diff = torch.tensor(Fk_val.toarray(),dtype=torch.float32).squeeze() - Fk_pt

    # # check extraction
    # x_opt,u_opt = nlp_casadi.extract_states_controls(w_opt)
    # x,u = nlp.extract_states_controls(w_opt)
    # assert torch.allclose(x_opt, x), "States do not match"
    # assert torch.allclose(u_opt, u), "Controls do not match"


    # X0 = torch.zeros(4)
    # X1 = torch.ones(4)

    # X = torch.stack((X0,X1)).T
    # U = 3*torch.ones(3,1)

    # # RUN NMPC
    # x0 = nmpc_casadi.nlp.sample_x0(offset=-0.2)
    # trajectory = nmpc_casadi.run_nmpc_closed_loop(x0,N_sim)
    # figs = visualize_trajectory(trajectory)


    # # run through all attributes of nlp and check which data type they are
    # for attr in dir(nlp):
    #     if not attr.startswith("_"):
    #         print(f"{attr}: {type(getattr(nlp,attr))}")


    # Generate batch data by repeating z_opt and p
    N = 1024
    z_batch = z_opt.unsqueeze(0).repeat(N, 1)  # Shape: (1024, n_z)
    p_batch = p.unsqueeze(0).repeat(N, 1)      # Shape: (1024, n_p)

    print(f"z_batch shape: {z_batch.shape}")
    print(f"p_batch shape: {p_batch.shape}")