# custom optimizer for cost function
# author: Quark
# ==================================================
import numpy as np
from scipy.optimize import line_search
# ==================================================
maxiter = 50
tol = 1e-4

def adam_optimizer(func, jac, x0, args=(), lr=0.01, beta1=0.75, beta2=0.99, 
                   eps=1e-8, maxiter=maxiter, tol=tol):
    x = np.array(x0, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    x_best = x.copy()
    f_best = func(x, *args)
    
    for t in range(1, maxiter + 1):
        g = jac(x, *args)
        
        if np.linalg.norm(g) < tol:
            break
        
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        x -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
        f_x = func(x, *args)
        if f_x < f_best:
            f_best = f_x
            x_best = x.copy()
    
    print(f"Adam optimizer converged in {t} iterations.")
    return x_best

def conjugate_gradient_optimizer(func, jac, x0, args=(), 
                                 tol=tol, maxiter=maxiter):
    """Conjugate Gradient method with robust restarts"""
    x = np.array(x0, dtype=float)
    
    g = jac(x, *args)
    e = -g
    
    for t in range(maxiter):
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
    
    print(f"CG optimizer converged in {t+1} iterations.")
    return x

def steepest_gradient_descent_optimizer(func, jac, x0, args=(), 
                               tol=tol, maxiter=maxiter):
    """Steepest Gradient Descent (SGD)"""
    x = np.array(x0, dtype=float)
    
    for t in range(maxiter):
        g = jac(x, *args)
        e = -g

        if np.linalg.norm(g) < tol: break

        alpha = line_search(func, jac, x, e, g, args=args)[0]
        if alpha is None: alpha = 1e-4
        
        x = x + alpha * e
    
    print(f"SGD optimizer converged in {t+1} iterations.")
    return x