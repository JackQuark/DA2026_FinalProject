# main
# author: Quark
# ==================================================
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm

from lorenz96 import lorenz96
from driver import Driver
from DASystem import ThreeDVar, Optimal_Interpolation
from nmc_method import NMCEstimator
from optimizer import * # custom optimizers
# ==================================================
def set_HBR(exp: str):
    """set up H, B, R matrices"""
    obs_variance = 0.1
    B_variance = 0.1
    match exp:
        case 'I':
            H = np.eye(N)
            R = obs_variance * np.eye(N)
            B = B_variance * np.eye(N)
        case 'II':
            H = np.eye(N)
            R = obs_variance * np.eye(N)
            B = np.load("B_nmc.npz")["B_nmc"]
        case 'III':
            H = np.zeros((N//2, N))
            for i in range(N//2):
                H[i, 2*i] = 1.0
            R = obs_variance * np.eye(N//2)
            B = np.load("B_nmc.npz")["B_nmc"]
        case _:
            raise ValueError(f"Unknown experiment type: {exp}")
                    
    return H, B, R
# ==================================================
# OSSE Setup Parameters
np.random.seed(0)

t_per_day = 0.2

N = 40              # Number of variables in Lorenz-96
F = 8.0             # Forcing term
dt = 0.05 / 6       # Integration time step (1hr)
assim_window = 6    # Steps between analysis (6hrs)
total_cycles = 50   # Total number of assimilation cycles
total_steps = total_cycles * assim_window

model_driver = Driver(lorenz96, dt=dt, F=F)

da_method = "3DVar" # OI, 3DVar
exp = "II" # I, II, III
optimizer = "SGD" # adam, CG, SGD

H, B, R = set_HBR(exp)
# =========================================================
# nature run
def load_nature_run(overwrite=False):
    if not os.path.exists("nature_run.npz") or overwrite:
        x_init = F * np.ones(N)
        x_init[19] += 0.01 # Initial perturbation
        # model spin-up
        x_spinup = model_driver.integrate(x_init, steps=int(100*t_per_day/dt)) # 100 days of spin-up
        x_true_0 = x_spinup[-1]
        # Generate the truth
        x_truth = model_driver.integrate(x_true_0, steps=total_steps)
        np.savez("nature_run.npz", truth_trajectory=x_truth, x_true_0=x_true_0)
    else:
        x_true_0 = np.load("nature_run.npz")["x_true_0"]
        x_truth = np.load("nature_run.npz")["truth_trajectory"]
    return x_true_0, x_truth

x_true_0, x_truth = load_nature_run()
# =========================================================
# NMC
# nmc_traj = model_driver.integrate(x_true_0, steps=2000)
# nmc = NMCEstimator(
#     model_driver,
#     dt1_steps=12,      # short lead time
#     dt2_steps=24,      # long lead time
#     alpha=1.0,
#     n_grid=N
# )
# B_nmc = nmc.estimate_B(trajectory=nmc_traj)
# np.savez("B_nmc.npz", B_nmc=B_nmc)
# plt.pcolormesh(B_nmc, cmap='bwr', norm=CenteredNorm())
# plt.colorbar()
# plt.title("NMC Estimated Background Error Covariance")
# plt.show()
# =========================================================
# simulate observations
observations = []
obs_errors = []
for cycle in range(1, total_cycles + 1):
    t_idx = cycle * assim_window
    truth_state = x_truth[t_idx]
    obs_dim = H.shape[0]
    obs = H @ truth_state \
        + np.random.normal(0, 0.1, obs_dim)
    observations.append(obs)
    obs_errors.append(obs - H @ truth_state)
# plot observations error distribution
plt.hist(np.array(obs_errors).flatten(), bins=50, alpha=0.7, 
         color='k', edgecolor='w', density=True)

fit_x = np.linspace(-0.5, 0.5, 100)
from scipy.stats import norm
mu, std = norm.fit(np.array(obs_errors).flatten())
p = norm.pdf(fit_x, mu, std)
plt.plot(fit_x, p, 'r--', linewidth=2, label=f"Fit: $\mu$={mu:.2f}, $\sigma$={std:.2f}")

plt.xlabel("Error")
plt.ylabel("Density")
plt.title("Distribution of Observation Errors")
plt.legend()
plt.show()
sys.exit()
# =========================================================
# main
match da_method:
    case "OI":
        da_system = Optimal_Interpolation(B, R, H)
        optimizer = "OI" # just for naming the output file
    case "3DVar":
        da_system = ThreeDVar(B, R, H)
        match optimizer:
            case "adam":
                da_system.set_optimizer(adam_optimizer)
            case "CG":
                da_system.set_optimizer(conjugate_gradient_optimizer)
            case "SGD":
                da_system.set_optimizer(steepest_gradient_descent_optimizer)
    case _:
        raise ValueError(f"Unknown DA method: {da_method}")

x_b_0 = x_true_0 + np.random.normal(0, 1, N)
x_b = np.copy(x_b_0)
x_f = np.copy(x_b_0)

x_a_list = [x_b]
x_f_list = [x_f]

obs_idx = 0
for step in range(1, total_steps + 1):
    
    x_f = model_driver.step(x_f)
    x_f_list.append(x_f)

    x_b = model_driver.step(x_b)

    if step % assim_window == 0:
        print(f"da cycle {step // assim_window}", end='\r')
        y = observations[obs_idx]
        
        x_a = da_system.assimilate(x_b, y)
        x_b = np.copy(x_a)
        
        obs_idx += 1

    x_a_list.append(x_b)
    
x_a_arr = np.array(x_a_list)
x_f_arr = np.array(x_f_list)

plt.plot(np.sqrt(np.mean((x_truth - x_a_arr)**2, axis=1)), label="Analysis")
plt.plot(np.sqrt(np.mean((x_truth - x_f_arr)**2, axis=1)), label="Free Run", linestyle='--')
plt.xlabel("Time Step")
plt.ylabel("RMSE")
plt.yscale("log")
plt.title(f"RMSE (exp.{exp}, method:{da_method})")
plt.legend()
plt.show()

ofname = f"x_a_{exp}_{optimizer}.npz"
if not os.path.exists(ofname):
    np.savez(ofname, x_a=x_a_arr)
    print(f"Saved analysis trajectory to {ofname}")
else:
    print(f"{ofname} already exists")
    