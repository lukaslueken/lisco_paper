# IMPORTS
import torch 
import casadi as ca
import numpy as np
from pathlib import Path
import json
import time

# Core PyTorch implementation
class NLP:
    """A class implementing a parametric optimization problem with PyTorch."""
    
    def __init__(self, n_vars=10, n_eq=5, n_ineq=5, obj_type="quad", op_dict=None, eps=1e-16, sigma=0.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.n_vars = n_vars
        self.n_eq = n_eq
        self.n_ineq = n_ineq
        self.obj_type = obj_type  # quad, nonconvex, rosenbrock, rosenbrock_modified, rastrigin
        self.eps = eps
        self.sigma = sigma
        # Only nonconvex is supported in this version
        if self.obj_type != "nonconvex":
            raise ValueError('Only obj_type="nonconvex" is implemented in this version. Other objective types are not implemented')

        self._calc_dims()

        if op_dict is None:
            self._setup_op()
        else:
            self.op_dict = op_dict
            self.Q = op_dict['Q']
            self.p = op_dict['p']
            self.A = op_dict['A']
            self.G = op_dict['G']
            self.h = op_dict['h']

        self._sanity_check()
        self._setup_functions()

    # SAVING and LOADING # -------------------------------------
    def save_config(self, folder_path=None, file_name="op_cfg"):
        cfg = {"n_vars": self.n_vars,
                "n_eq": self.n_eq,
                "n_ineq": self.n_ineq,
                "obj_type": self.obj_type}
        cfg["op_dict"] = self.op_dict_to_json(self.op_dict)
        if folder_path is None:
            save_pth = Path(file_name+".json")
        else:
            save_pth = Path(folder_path,file_name+".json")        
        with open(save_pth,"w") as f:
            json.dump(cfg,f,indent=4)
        print("settings saved to: ", save_pth)
    
    def op_dict_to_json(self,op_dict):
        op_dict_json = op_dict.copy()
        for key, value in op_dict_json.items():
            if isinstance(value,torch.Tensor):
                op_dict_json[key] = value.tolist()
        return op_dict_json
    
    @classmethod
    def json_to_op_dict(cls,op_dict_json,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        op_dict = op_dict_json.copy()
        for key, value in op_dict.items():
            if isinstance(value,list):
                op_dict[key] = torch.tensor(value,device=device)
        return op_dict

    @classmethod
    def from_dict(cls, cfg):
        cfg["op_dict"] = cls.json_to_op_dict(cfg["op_dict"])
        return cls(**cfg)
    
    @classmethod
    def from_json(cls, folder_pth=None, file_name="op_cfg"):
        if folder_pth is None:
            load_pth = Path(file_name+".json")
        else:
            load_pth = Path(folder_pth,file_name+".json")
        with open(load_pth,"r") as f:
            cfg = json.load(f)
        return cls.from_dict(cfg)    
    
    # MAIN # -------------------------------------
    def _calc_dims(self):
        # Dimension calculations similar to NLP class
        self.n_w = self.n_vars          # Number of decision variables (primal variables)
        self.n_nu = self.n_eq           # Number of equality constraints (dual variables)
        self.n_lam = self.n_ineq        # Number of inequality constraints (dual variables)
        self.n_z = self.n_w + self.n_nu + self.n_lam  # Total primal and dual variables
        self.n_p = self.n_eq            # Number of parameters (x in original formulation)

    def _setup_op(self):
        # Only nonconvex objective supported
        if self.obj_type == "nonconvex":
            self.Q = self.gen_Q()
            self.p = self.gen_p()
        else:
            raise ValueError('Only obj_type="nonconvex" is implemented in this version. Other objective types are not implemented.')
        self.A = self.gen_A()
        self.G = self.gen_G()
        self.h = self.gen_h()        
        op_dict = {
            'Q': self.Q,
            'p': self.p,
            'A': self.A,
            'G': self.G,
            'h': self.h
        }
        self.op_dict = op_dict
        print("Optimization Problem Matrices Generated.")

    def _sanity_check(self):
        # Nonconvex requires Q and p
        assert self.n_vars == self.op_dict['Q'].shape[0]
        assert self.n_vars == self.op_dict['Q'].shape[1]
        assert self.n_vars == self.op_dict['p'].shape[0]
        assert self.n_eq == self.op_dict['A'].shape[0]
        assert self.n_vars == self.op_dict['A'].shape[1]
        assert self.n_ineq == self.op_dict['G'].shape[0]
        assert self.n_vars == self.op_dict['G'].shape[1]
        assert self.n_ineq == self.op_dict['h'].shape[0]

    def _setup_functions(self):
        # Objective Function: only nonconvex
        self.f_func = self.f_nonconvex
        self.dfdw_func = self.f_nonconvex_grad
        self.dfdw_conv_func = self.f_nonconvex_grad_convexified

        # Batch Functions
        self.f_batch_func = torch.vmap(self.f_func)
        self.h_batch_func = torch.vmap(self.h_func)
        self.g_batch_func = torch.vmap(self.g_func)
        self.L_batch_func = torch.vmap(self.L_func)
        self.dLdw_batch_func = torch.vmap(self.dLdw_func)
        self.dfdw_batch_func = torch.vmap(self.dfdw_func)
        self.dhdw_batch_func = torch.vmap(self.dhdw_func)
        self.dgdw_batch_func = torch.vmap(self.dgdw_func)
        self.KKT_batch_func = torch.vmap(self.KKT_func)
        self.Fk_batch_func = torch.vmap(self.Fk_func)
        self.Tk_batch_func = torch.vmap(self.Tk_func)
        self.Fk_conv_batch_func = torch.vmap(self.Fk_conv_func)
        self.Tk_conv_batch_func = torch.vmap(self.Tk_conv_func)
        self.gamma_batch_func = torch.vmap(self.gamma_func)
        self.gamma_conv_batch_func = torch.vmap(self.gamma_conv_func)
        self.DFk_Dz_batch_func = torch.vmap(self.DFk_Dz_func)
        self.cond_num_batch_func = torch.vmap(self.cond_num_func, chunk_size=1)
        
        # Extraction and stacking functions (like NLP class)
        self.extract_primal_dual_batch = torch.vmap(self.extract_primal_dual)
        self.stack_primal_dual_batch = torch.vmap(self.stack_primal_dual)

    def set_device(self, device):
        """Function to move all attributes to new device."""
        self.device = device
        # Only nonconvex supported, always move Q and p
        self.Q = self.Q.to(device)
        self.p = self.p.to(device)
        self.A = self.A.to(device)
        self.G = self.G.to(device)
        self.h = self.h.to(device)
        # Re-setup functions to ensure they use the new device
        self._setup_functions()

    def set_dtype(self,dtype):
        """Function to change dtype of all attributes."""
        self.dtype = dtype
        # Only nonconvex supported, always convert Q and p
        self.Q = self.Q.to(dtype=self.dtype)
        self.p = self.p.to(dtype=self.dtype)
        self.A = self.A.to(dtype=self.dtype)
        self.G = self.G.to(dtype=self.dtype)
        self.h = self.h.to(dtype=self.dtype)
        # Re-setup functions to ensure they use the new dtype
        self._setup_functions()

    # Extraction and stacking functions like NLP class
    def extract_primal_dual(self, z):
        """Extract primal and dual variables from z vector."""
        w = z[:self.n_w]
        nu = z[self.n_w:self.n_w+self.n_nu]
        lam = z[self.n_w+self.n_nu:]
        return w, nu, lam
    
    def stack_primal_dual(self, w, nu, lam):
        """Stack primal and dual variables into z vector."""
        return torch.hstack([w, nu, lam])

    # Core objective functions
    def f_nonconvex(self, w, p=None):
        """Objective function: 0.5 * w^T Q w + p^T sin(w)"""
        f_val = 0.5 * w @ self.Q @ w + self.p @ torch.sin(w)
        return f_val
        
    # Constraint functions
    def h_func(self, w, p):
        """Equality constraints function: A*w - p"""
        h_val = self.A @ w - p
        return h_val
    
    def g_func(self, w, p=None):
        """Inequality constraints function: G*w - h"""
        g_val = self.G @ w - self.h
        return g_val
    
    # Gradient functions
    def f_nonconvex_grad(self, w, p=None):
        """Objective gradient function: Q*w + p*cos(w)"""
        f_grad_val = self.Q @ w + self.p * torch.cos(w)
        return f_grad_val
    
    def f_nonconvex_grad_convexified(self, w, p=None):
        """Objective gradient function: Q*w + p*cos(w) with convexification by removing the nonconvex part."""
        f_grad_val = self.Q @ w + self.p * torch.cos(w.detach())
        return f_grad_val
    
    def dLdw_func(self, w, nu, lam, p):
        dLdw_val = self.dfdw_func(w,p) + self.A.T @ nu + self.G.T @ lam
        return dLdw_val
    
    def dLdw_conv_func(self, w, nu, lam, p):
        dLdw_val = self.dfdw_conv_func(w,p) + self.A.T @ nu + self.G.T @ lam
        return dLdw_val
    
    def dhdw_func(self, w, p):
        dhdw_val = self.A
        return dhdw_val
    
    def dgdw_func(self, w, p):
        dgdw_val = self.G
        return dgdw_val

    # Matrix generation functions
    def gen_Q(self):
        """Generate random Q matrix (objective quadratic term)."""
        return torch.diag(torch.rand(self.n_vars, device=self.device, dtype=self.dtype))

    def gen_p(self):
        """Generate random p vector (objective linear term)."""
        return torch.rand(self.n_vars, device=self.device, dtype=self.dtype)

    def gen_A(self):
        """Generate random A matrix (equality constraints)."""
        return torch.randn(self.n_eq, self.n_vars, device=self.device, dtype=self.dtype)
    
    def gen_G(self):
        """Generate random G matrix (inequality constraints)."""
        return torch.randn(self.n_ineq, self.n_vars, device=self.device, dtype=self.dtype)

    def gen_h(self):
        """Generate h vector (inequality constraints)."""
        A_pinv = torch.pinverse(self.A)
        return torch.sum(torch.abs(self.G @ A_pinv), axis=1)

    # Sampling functions (matching NLP class)
    def batch_gen_p(self, N_batch, offset=0.0):
        """Generate random parameters for N_batch samples."""
        return (2.0*torch.rand(N_batch, self.n_p, device=self.device, dtype=self.dtype) - 1.0)*(1+offset)
    
    def batch_gen_w(self, N_batch):
        """Generate random decision variable vectors for N_batch samples."""        
        return torch.rand(N_batch, self.n_w, device=self.device, dtype=self.dtype)
    
    def batch_gen_z(self, N_batch):
        """Generate random primal and dual variables for N_batch samples."""
        w = self.batch_gen_w(N_batch)
        nu = torch.randn(N_batch, self.n_nu, device=self.device, dtype=self.dtype)
        lam = torch.rand(N_batch, self.n_lam, device=self.device, dtype=self.dtype) # >= 0
        z = self.stack_primal_dual_batch(w, nu, lam)
        return z

    # Lagrangian and KKT-related functions
    def L_func(self, w, nu, lam, p):
        """Lagrangian function for the optimization problem."""
        f_val = self.f_func(w, p)
        h_val = self.h_func(w, p)
        g_val = self.g_func(w, p)
        return f_val + nu @ h_val + lam @ g_val
    
    def KKT_func(self, z, p):
        """KKT conditions function."""
        w, nu, lam = self.extract_primal_dual(z)
        dLdw_val = self.dLdw_func(w, nu, lam, p)
        h_val = self.h_func(w, p)
        g_val = self.g_func(w, p)
        g_val_plus = torch.relu(g_val)
        dual_feas_minus = torch.relu(-lam)
        comp_slack = lam*g_val    
        KKT_val = torch.hstack([dLdw_val, h_val, g_val_plus, dual_feas_minus, comp_slack])
        return KKT_val
    
    def Fk_func(self, z, p):
        """Small modification to KKT conditions (same as KKT_func)."""
        w, nu, lam = self.extract_primal_dual(z)
        nu_bar = nu.detach().clone()
        lam_bar = lam.detach().clone()
        dLdw_val = self.dLdw_func(w, nu, lam, p)
        h_val = self.h_func(w, p) - self.sigma *(nu - nu_bar)
        g_val = self.g_func(w, p) - self.sigma *(lam - lam_bar)
        fb_cond = 0.8*(lam - g_val - torch.sqrt(lam**2 + g_val**2 + self.eps)) + 0.2*(torch.relu(lam)*torch.relu(-g_val))
        KKT_val = torch.hstack([dLdw_val, h_val, fb_cond])

        return KKT_val
    
    def Fk_conv_func(self, z, p):
        """Convexified KKT function."""
        w, nu, lam = self.extract_primal_dual(z)
        # delta_w = w-w.detach()
        nu_bar = nu.detach().clone()
        lam_bar = lam.detach().clone()
        dLdw_val = self.dLdw_conv_func(w, nu, lam, p)
        
        # h_val = self.A @ w_detach() - p + self.A @ delta_w - sigma *(nu - nu_bar) == following code line
        h_val = self.A @ w - p - self.sigma *(nu - nu_bar)

        # g_val = self.G @ w_detach() - self.h + self.G @ delta_w - sigma *(lam - lam_bar) == following code line
        g_val = self.G @ w - self.h - self.sigma *(lam - lam_bar)

        fb_cond = 0.8*(lam - g_val - torch.sqrt(lam**2 + g_val**2 + self.eps)) + 0.2*(torch.relu(lam)*torch.relu(-g_val))
        KKT_val = torch.hstack([dLdw_val, h_val, fb_cond])        

        return KKT_val
    
    def Tk_func(self, z, p):
        """L2 norm of Fk."""
        Fk = self.Fk_func(z, p)
        # Tk = torch.norm(Fk, p=2)
        Tk = 0.5*torch.dot(Fk, Fk)
        return Tk
    
    def Tk_conv_func(self, z, p):
        """L2 norm of convexified Fk."""
        Fk_conv = self.Fk_conv_func(z, p)
        # Tk_conv = torch.norm(Fk_conv, p=2)
        Tk_conv = 0.5*torch.dot(Fk_conv, Fk_conv)
        return Tk_conv
    
    def gamma_func(self, z, p, dz):
        """Function to compute the step size."""
        Fk, jvpFk = torch.func.jvp(self.Fk_func, (z, p), (dz, torch.zeros_like(p)))
        gamma = -Fk @ jvpFk / (jvpFk @ jvpFk)
        return gamma

    def gamma_conv_func(self, z, p, dz):
        """Function to compute the step size using linearized KKT conditions."""
        Fk, jvpFk = torch.func.jvp(self.Fk_conv_func, (z, p), (dz, torch.zeros_like(p)))
        gamma = -Fk @ jvpFk / (jvpFk @ jvpFk)
        return gamma

    # KKT Matrix functions
    def DFk_Dz_func(self, z, p):
        """Jacobian of Fk w.r.t. z."""
        DFk_Dz_val = torch.func.jacfwd(self.Fk_func, argnums=0)(z, p)
        return DFk_Dz_val
    
    def cond_num_func(self, z, p):
        """Calculate condition number of the KKT matrix."""
        DFk_Dz = self.DFk_Dz_func(z, p)
        cond_num = torch.linalg.cond(DFk_Dz)
        return cond_num
      
    # OSQP Solver (disabled in this version)
    def solve_osqp(self, p):
        raise ValueError('QP/OSQP solver is not available: only obj_type="nonconvex" is implemented in this version.')

    def sample_osqp_solutions(self, n_samples):
        raise ValueError('OSQP-based sampling is only for quadratic problems and is not implemented in this version.')

    # Export hyperparameters
    def export_hparams(self):
        """Export hyperparameters of the optimization problem."""
        hparams = {
            "n_vars": self.n_vars,
            "n_eq": self.n_eq,
            "n_ineq": self.n_ineq,
            "obj_type": self.obj_type,
        }
        return hparams

# CasADi implementation
class NLPCasadi:
    """A class implementing a parametric optimization problem with CasADi."""
    
    def __init__(self, n_vars=10, n_eq=5, n_ineq=5, obj_type="quad", op_dict=None):
        self.n_vars = n_vars
        self.n_eq = n_eq
        self.n_ineq = n_ineq
        self.obj_type = obj_type
        if self.obj_type != "nonconvex":
            raise ValueError('Only obj_type="nonconvex" is implemented in this version. Other objective types are not implemented.')

        # Create a PyTorch instance to generate matrices if needed
        self.torch_nlp = NLP(n_vars, n_eq, n_ineq, obj_type, op_dict)
        self.op_dict = self._convert_pytorch_to_casadi(self.torch_nlp.op_dict)

        self._calc_dims()
        self._setup_variables()
        self._build_nlp()
        self._build_standard_nlp()

    def update_op_dict(self,op_dict):
        """Update the optimization problem dictionary."""
        self.op_dict = self._convert_pytorch_to_casadi(op_dict)
        self.torch_nlp.op_dict = op_dict
        self._calc_dims()
        self._setup_variables()
        self._build_nlp()
        self._build_standard_nlp()

    def _convert_pytorch_to_casadi(self, op_dict):
        """Convert PyTorch tensors to CasADi matrices."""
        casadi_dict = {}
        for key, value in op_dict.items():
            if isinstance(value, torch.Tensor):
                casadi_dict[key] = ca.DM(value.cpu().numpy())
            else:
                casadi_dict[key] = value
        return casadi_dict

    def _calc_dims(self):
        """Calculate dimensions for the optimization problem."""
        self.n_w = self.n_vars
        self.n_nu = self.n_eq
        self.n_lam = self.n_ineq
        self.n_z = self.n_w + self.n_nu + self.n_lam
        self.n_p = self.n_eq

    def _setup_variables(self):
        """Create symbolic variables for the optimization problem."""
        self.w_sym = ca.SX.sym('w', self.n_vars)
        self.p_sym = ca.SX.sym('p', self.n_eq)
        self.nu_sym = ca.SX.sym('nu', self.n_nu)
        self.lam_sym = ca.SX.sym('lam', self.n_lam)
        self.z_sym = ca.vertcat(self.w_sym, self.nu_sym, self.lam_sym)

    def _build_nlp(self):
        """Build the optimization problem."""
        # Build objective function (only nonconvex)
        if self.obj_type == "nonconvex":
            f_sym = 0.5 * ca.mtimes([self.w_sym.T, self.op_dict['Q'], self.w_sym]) + \
                    ca.mtimes(self.op_dict['p'].T, ca.sin(self.w_sym))
        else:
            raise ValueError('Only obj_type="nonconvex" is implemented in this version. Other objective types are not implemented.')
        # Build constraints
        h_sym = ca.mtimes(self.op_dict['A'], self.w_sym) - self.p_sym
        g_sym = ca.mtimes(self.op_dict['G'], self.w_sym) - self.op_dict['h']
        
        # Create NLP
        self.nlp = {
            'x': self.w_sym,
            'p': self.p_sym,
            'f': f_sym,
            'g': ca.vertcat(h_sym, g_sym)
        }
        
        # Setup bounds
        self.lbg = ca.vertcat(np.zeros(self.n_eq), -np.inf*np.ones(self.n_ineq))
        self.ubg = ca.vertcat(np.zeros(self.n_eq), np.zeros(self.n_ineq))
        
        # Setup solver
        opts = {'ipopt.tol': 1e-8, 'ipopt.acceptable_tol': 1e-8}
        self.solver = ca.nlpsol('solver', 'ipopt', self.nlp, opts)

    def _build_standard_nlp(self):
        """Build the standard NLP formulation for KKT conditions."""
        # Create symbolic variables for optimization
        w_sym = self.w_sym
        p_sym = self.p_sym
        nu_sym = self.nu_sym
        lam_sym = self.lam_sym
        z_sym = self.z_sym

        # Build objective function (only nonconvex)
        if self.obj_type == "nonconvex":
            f_sym = 0.5 * ca.mtimes([w_sym.T, self.op_dict['Q'], w_sym]) + ca.mtimes(self.op_dict['p'].T, ca.sin(w_sym))
        else:
            raise ValueError('Only obj_type="nonconvex" is implemented in this version. Other objective types are not implemented.')

        # Build constraints
        h_sym = ca.mtimes(self.op_dict['A'], w_sym) - p_sym
        g_sym = ca.mtimes(self.op_dict['G'], w_sym) - self.op_dict['h']

        # Build Lagrangian
        L_sym = f_sym + ca.dot(nu_sym, h_sym) + ca.dot(lam_sym, g_sym)

        # Compute derivatives
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

        fb_sym = 0.8*(lam_sym - g_sym - ca.sqrt(lam_sym**2 + g_sym**2 + 1e-16)) + 0.2*(lam_plus_sym*g_minus_sym)

        Fk_sym = ca.vertcat(dLdw_sym,h_sym,fb_sym)
        # Tk_sym = ca.norm_2(Fk_sym)
        Tk_sym = 0.5 * ca.mtimes(Fk_sym.T, Fk_sym)

        # Create functions
        self.f_func = ca.Function('f', [w_sym, p_sym], [f_sym])
        self.h_func = ca.Function('h', [w_sym, p_sym], [h_sym])
        self.g_func = ca.Function('g', [w_sym, p_sym], [g_sym])
        self.L_func = ca.Function('L', [w_sym, nu_sym, lam_sym, p_sym], [L_sym])
        self.dLdw_func = ca.Function('dLdw', [w_sym, nu_sym, lam_sym, p_sym], [dLdw_sym])
        self.dfdw_func = ca.Function('dfdw', [w_sym, p_sym], [dfdw_sym])
        self.dhdw_func = ca.Function('dhdw', [w_sym, p_sym], [dhdw_sym])
        self.dgdw_func = ca.Function('dgdw', [w_sym, p_sym], [dgdw_sym])
        self.KKT_func = ca.Function('KKT', [z_sym, p_sym], [KKT_sym])
        self.Fk_func = ca.Function('Fk', [z_sym, p_sym], [Fk_sym])
        self.Tk_func = ca.Function('Tk', [z_sym, p_sym], [Tk_sym])

        # Add gamma functions for line search
        dw_sym = ca.SX.sym("dw_sym", self.n_w)
        dnu_sym = ca.SX.sym("dnu_sym", self.n_nu)
        dlam_sym = ca.SX.sym("dlam_sym", self.n_lam)
        dz_sym = ca.vertcat(dw_sym, dnu_sym, dlam_sym)

        # Standard gamma function
        jvp_sym = ca.jtimes(Fk_sym, z_sym, dz_sym, False)
        gamma_sym = -(Fk_sym.T @ jvp_sym) / (jvp_sym.T @ jvp_sym + 1e-16)
        self.gamma_func = ca.Function('gamma', [z_sym, p_sym, dz_sym], [gamma_sym])

        # Convexified gamma function
        if self.obj_type != "nonconvex":
            raise ValueError('Only obj_type="nonconvex" is implemented in this version. Other objective types are not implemented.')
        Fk_conv_sym = ca.vertcat(self.op_dict["Q"]@w_sym, h_sym, fb_sym) # only works for obj_type = "nonconvex"
        jvp_conv_w_sym = ca.jtimes(Fk_conv_sym, w_sym, dw_sym, False)
        jvp_conv_nu_sym = ca.jtimes(Fk_sym, nu_sym, dnu_sym, False)
        jvp_conv_lam_sym = ca.jtimes(Fk_sym, lam_sym, dlam_sym, False)
        jvp_conv_sym = jvp_conv_w_sym + jvp_conv_nu_sym + jvp_conv_lam_sym
        gamma_conv_sym = -(Fk_sym.T @ jvp_conv_sym) / (jvp_conv_sym.T @ jvp_conv_sym + 1e-16)
        self.gamma_conv_func = ca.Function('gamma_conv', [z_sym, p_sym, dz_sym], [gamma_conv_sym])

        # Store the standard NLP formulation
        self.standard_nlp = {
            "w": w_sym,
            "f": f_sym,
            "h": h_sym,
            "g": g_sym,
            "p": p_sym,
            "nu": nu_sym,
            "lam": lam_sym,
            "L": L_sym,
            "dLdw": dLdw_sym
        }

    def solve(self, p, w_init=None):
        if w_init is None:
            w_init = np.random.rand(self.n_vars)
        elif isinstance(w_init, torch.Tensor):
            w_init = w_init.cpu().numpy()
            
        if isinstance(p, torch.Tensor):
            p = p.cpu().numpy()

        start = time.perf_counter()            
        res = self.solver(x0=w_init, p=p, lbx=-np.inf, ubx=np.inf,
                         lbg=self.lbg, ubg=self.ubg)
        solve_time = time.perf_counter() - start

        w_opt = res['x'].full().flatten()
        lam_g = res['lam_g'].full().flatten()
        nu_opt = lam_g[:self.n_nu]
        lam_opt = lam_g[self.n_nu:]
        
        # return w_opt, nu_opt, lam_opt, self.solver.stats()['return_status'], self.solver.stats()['t_wall_total']
        return w_opt, nu_opt, lam_opt, self.solver.stats()['return_status'], solve_time

    def stack_primal_dual(self, w, nu, lam):
        return np.hstack([w, nu, lam])    
    
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

            return np.array(dzk).squeeze(),np.array(norm_Fk).squeeze()
        return solve_step
    

# For backward compatibility
class parametricOP(NLP):
    """Alias for backward compatibility."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Warning: parametricOP is deprecated, use NLP instead.")
        self._casadi_nlp = NLPCasadi(*args, **kwargs)