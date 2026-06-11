# main
# author: Quark
# ==================================================
import sys, os
import numpy as np
import matplotlib.pyplot as plt

from lorenz96 import lorenz96
from driver import Driver
from DASystem import ThreeDVar
from scipy.optimize import minimize

from optimizer import * # custom optimizers
# ==================================================
# OSSE Setup Parameters
t_per_day = 0.2

N = 40              # Number of variables in Lorenz-96
F = 8.0             # Forcing term
dt = 0.05 / 6       # Integration time step (1hr)
assim_window = 6    # Steps between analysis (6hrs)
total_cycles = 50   # Total number of assimilation cycles
total_steps = total_cycles * assim_window

model_driver = Driver(lorenz96, dt=dt, F=F)

np.random.seed(42)
# =========================================================
# >>> nature run >>>
# x_init = F * np.ones(N)
# x_init[19] += 0.01 # Initial perturbation
# # model spin-up
# x_spinup = model_driver.integrate(x_init, steps=int(100*t_per_day/dt)) # 100 days of spin-up
# x_true_0 = x_spinup[-1]
# # Generate the truth
# x_truth = model_driver.integrate(x_true_0, steps=total_steps)
# np.savez("nature_run.npz", truth_trajectory=x_truth, x_true_0=x_true_0)
# <<< nature run <<<

x_true_0 = np.load("nature_run.npz")["x_true_0"]
x_truth = np.load("nature_run.npz")["truth_trajectory"]

# =========================================================
# >>> simulating observations >>>
obs_variance = 0.1
obs_count = N
# H = np.zeros((obs_count, N))
# for i in range(obs_count):
#     H[i, 2*i] = 1.0

H = np.eye(N) 
R = obs_variance * np.eye(obs_count)

observations = []
for cycle in range(1, total_cycles + 1):
    t_idx = cycle * assim_window
    truth_state = x_truth[t_idx]
    obs = H @ truth_state \
        + np.random.normal(0, np.sqrt(obs_variance), obs_count)
    observations.append(obs)
# <<< simulating observations <<<
# =========================================================
# >>> background error covariance >>>
B_variance = 0.1
B = np.eye(N) * B_variance
# B = np.zeros((N, N))
# for i in range(N):
#     for j in range(N):
#         dist = min(abs(i - j), N - abs(i - j))
#         B[i, j] = B_variance * np.exp(-(dist**2) / 2)
# <<< background error covariance <<< 

# initial background state
x_b_0 = x_true_0 + np.random.normal(0, np.sqrt(obs_variance), N)

# Initialize DA System
da_system = ThreeDVar(B, R, H)

# INJECT OPTIMIZATION METHOD HERE
op_name = "adam"
# op_name = "CG"
# op_name = "GD"
match op_name:
    case "adam":
        da_system.set_optimizer(adam_optimizer)
    case "CG":
        da_system.set_optimizer(conjugate_gradient_optimizer)
    case "GD":
        da_system.set_optimizer(gradient_descent_optimizer)

x_b = np.copy(x_b_0)
x_a_list = [x_b]

obs_idx = 0
for step in range(1, total_steps + 1):
    
    # Forecast Step: Advance background using the model
    x_b = model_driver.step(x_b)

    # Analysis Step: Assimilate observations when available
    if step % assim_window == 0:
        print(f"da cycle {step // assim_window}", end='\r')
        y = observations[obs_idx]
        
        # 3DVar analysis (This uses your injected optimizer under the hood)
        x_a = da_system.assimilate(x_b, y)
        x_b = np.copy(x_a) # Update background with new analysis
        
        obs_idx += 1

    x_a_list.append(x_b)

x_a = np.array(x_a_list)

ofname = f"x_a_I_{op_name}.npz"
if not os.path.exists(ofname):
    np.savez(ofname, x_a=x_a)
else:
    print(f"{ofname} already exists")

rmse = np.sqrt(np.mean((x_truth - x_a)**2, axis=1))
print(f"Mean RMSE over simulation: {np.mean(rmse):.4f}")
