# DA system driver
# author: Quark
# ==================================================
import numpy as np
from scipy.optimize import minimize
# ==================================================
class ThreeDVar:
    def __init__(self, B, R, H, optimizer_func=None):
        self.B = B
        self.B_inv = np.linalg.inv(B)
        self.R = R
        self.R_inv = np.linalg.inv(R)
        self.H = H
        
        self.optimizer_func = optimizer_func or self._default_optimizer

    def set_optimizer(self, optimizer_func):
        """Allows swapping the optimization method dynamically."""
        self.optimizer_func = optimizer_func

    def cost_function(self, x, xb, y):
        """Computes the 3DVar cost function J(x)."""
        dx = x - xb
        dy = y - self.H @ x
        Jb = 0.5 * dx.T @ self.B_inv @ dx
        Jo = 0.5 * dy.T @ self.R_inv @ dy
        return Jb + Jo

    def gradient(self, x, xb, y):
        """Computes the Jacobian/Gradient of the cost function."""
        # ∇J(x) = B^(-1)(x - xb) - H^T R^(-1) (y - Hx)
        grad_b = self.B_inv @ (x - xb) 
        grad_o = self.H.T @ (self.R_inv @ (self.H @ x - y)) 
        return grad_b + grad_o

    def _default_optimizer(self, func, jac, x0, args):
        """Default optimizer using CG(scipy)."""
        res = minimize(func, x0, args=args, jac=jac, method='CG')
        return res.x

    def assimilate(self, xb, y):
        """Analysis step using 3DVar"""
        args = (xb, y)
        # The optimizer function takes the cost function, gradient, initial guess (xb), and arguments
        xa = self.optimizer_func(self.cost_function, self.gradient, xb, args)
        return xa
    
    
class Optimal_Interpolation:
    def __init__(self, B, R, H):
        self.B = B
        self.R = R
        self.H = H

    def assimilate(self, xb, y):
        """Analysis step using OI"""
        # OI formula: x_a = x_b + K(y - Hx_b)
        K = self.B @ self.H.T @ np.linalg.inv(self.H @ self.B @ self.H.T + self.R)
        xa = xb + K @ (y - self.H @ xb)
        return xa