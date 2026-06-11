# custom optimizer for cost function
# author: Quark
# ==================================================
import numpy as np
from scipy.optimize import line_search
# ==================================================
def adam_optimizer(func, jac, x0, args=(), lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, maxiter=100):
    """Adam optimization algorithm."""
    x = np.array(x0, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    
    for t in range(1, maxiter + 1):
        g = jac(x, *args)
        
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        x -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
        if np.linalg.norm(g) < 1e-5:
            break
            
    return x

def conjugate_gradient_optimizer(func, jac, x0, args=(), tol=1e-5, maxiter=100):
    """Conjugate Gradient method follow 3D-Var handout p.8,9"""
    x = np.array(x0, dtype=float)
    
    g = jac(x, *args)
    e = -g
    
    for _ in range(maxiter):
        if np.linalg.norm(g) < tol: break
            
        alpha = line_search(func, jac, x, e, g, args=args)[0]
        if alpha is None: alpha = 1e-4
            
        x_new = x + alpha * e
        g_new = jac(x_new, *args)
        
        # Polak-Ribiere
        beta_PR = max(0, np.dot(g_new, g_new - g) / np.dot(g, g))
        
        e = -g_new + beta_PR * e
        
        x = x_new
        g = g_new
        
    return x
