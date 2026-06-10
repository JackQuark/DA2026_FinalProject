import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

np.random.seed(0)

sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0

def lorenz_63(t, state):
    x, y, z = state
    return np.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ])

dt = 0.02
t_end = 10.0
time_steps = np.arange(0, t_end, dt)
n_steps = len(time_steps)

obs_freq = 10

# error covariance matrices
B = np.eye(3)
R = np.eye(3)
B_inv = np.linalg.inv(B)
R_inv = np.linalg.inv(R)

def cost_function(x, xb, yo, B_inv, R_inv):
    diff_b = xb - x
    diff_o = yo - x
    Jb = 0.5 * diff_b.T @ B_inv @ diff_b
    Jo = 0.5 * diff_o.T @ R_inv @ diff_o
    return Jb + Jo

def gradient(x, xb, yo, B_inv, R_inv):
    return B_inv @ (x - xb) + R_inv @ (x - yo)

print("INFO: Nature Run...")
x_true_0 = np.array([1.5, -1.5, 20.0])
nature_run = solve_ivp(lorenz_63, (0, t_end), x_true_0, t_eval=time_steps, method='RK45')
x_true = nature_run.y.T  # Shape: (n_steps, 3)
 
print("INFO: Simulating Observation...")
obs_time_indices = np.arange(obs_freq, n_steps, obs_freq)
observations = {}
for idx in obs_time_indices:
    observations[idx] = x_true[idx] + np.random.multivariate_normal(np.zeros(3), R)

print("INFO: Free Run...")
x_b_0 = x_true_0 + 0.1
free_run = solve_ivp(lorenz_63, (0, t_end), x_b_0, t_eval=time_steps, method='RK45')
x_free = free_run.y.T

print("INFO: DA (3D-Var)...")
x_da = np.zeros_like(x_true)
x_da[0] = x_b_0
x_current = x_b_0

for i in range(1, n_steps):
    t_span = (time_steps[i-1], time_steps[i])
    sol = solve_ivp(lorenz_63, t_span, x_current, method='RK45')
    xb = sol.y[:, -1]
    
    if i in observations:
        yo = observations[i]
        
        res = minimize(
            cost_function, 
            x0=xb,
            args=(xb, yo, B_inv, R_inv), 
            jac=gradient,
            method='CG'
        )
        x_current = res.x
    else:
        x_current = xb
        
    x_da[i] = x_current

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
variables = ['x', 'y', 'z']

obs_times = time_steps[obs_time_indices]
obs_vals = np.array([observations[i] for i in obs_time_indices])

for i, ax in enumerate(axs):
    ax.plot(time_steps, x_true[:, i], 'k-', label='Nature Run (True)', linewidth=1.5)
    ax.plot(time_steps, x_free[:, i], 'g--', label='Free Run (No DA)', alpha=0.6)
    ax.plot(time_steps, x_da[:, i], 'r--', label='3D-Var Assimilation', linewidth=1.5)
    ax.scatter(obs_times, obs_vals[:, i], c='b', marker='o', s=20, label='Observations', zorder=5)
    
    ax.set_ylabel(variables[i])
    ax.grid(True, linestyle=':', alpha=0.7)
    if i == 0:
        ax.legend(loc='upper right', ncol=4, fontsize='small')

axs[-1].set_xlabel('Time')
plt.suptitle('Lorenz 63 OSSE with 3D-Var (CG Method)', fontsize=14)
plt.tight_layout()
plt.show()

# ==================================================
print("INFO: RMSE Analysis...")
rmse_free_total = np.sqrt(np.mean((x_free - x_true)**2))
rmse_assim_total = np.sqrt(np.mean((x_da - x_true)**2))
print(f"RMSE (Free Run): {rmse_free_total:.4f}")
print(f"RMSE (3D-Var):   {rmse_assim_total:.4f}")
rmse_free_vars = np.sqrt(np.mean((x_free - x_true)**2, axis=0))
rmse_assim_vars = np.sqrt(np.mean((x_da - x_true)**2, axis=0))
print(f"RMSE (Free Run) [x, y, z]: {rmse_free_vars}")
print(f"RMSE (3D-Var)   [x, y, z]: {rmse_assim_vars}")

# L2 Norm
error_free = np.sqrt(np.sum((x_free - x_true)**2, axis=1))
error_da = np.sqrt(np.sum((x_da - x_true)**2, axis=1))

plt.figure(figsize=(10, 4))
plt.plot(time_steps, error_free, 'g--', label=f'Free Run Error', alpha=0.8)
plt.plot(time_steps, error_da, 'r-', label=f'3D-Var Error', linewidth=1.5)

obs_times = time_steps[obs_time_indices]
plt.vlines(obs_times, ymin=0, ymax=max(error_da), color='blue', alpha=0.3, linestyle=':', label='Assimilation Windows')

plt.axhline(0, color='k', linewidth=0.8)
plt.xlabel('Time')
plt.ylabel('Error Magnitude (L2 Norm)')
plt.title('Error Evolution over Time: Free Run vs 3D-Var')
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()