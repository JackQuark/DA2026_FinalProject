# visualization
# author: Quark
# ==================================================
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
# ==================================================
dt = 0.05 / 6       # Integration time step (1hr)

def plot_hovmoller(x_a_files, x_t):
    N_subplots = len(x_a_files)
    total_steps, N = x_t.shape
    
    x_as = [np.load(fname)['x_a'] for fname in x_a_files]
    
    fig, axs = plt.subplots(1, N_subplots, figsize=(4*N_subplots, 6), sharex=True, sharey=True)
    axs: list[plt.Axes]

    _x = np.arange(N) + 1
    _t = np.arange(total_steps) * dt
    
    _cbar_kwargs = {"orientation": "vertical", "aspect": 30, "shrink": 0.6}

    max_error = max(np.max(np.abs(x_a - x_t)) for x_a in x_as)

    for i, x_a in enumerate(x_as):
        im = axs[i].pcolormesh(
            _x, _t, x_a - x_t, norm=CenteredNorm(vcenter=0, halfrange=max_error),
            cmap="RdBu_r"
        )
        axs[i].set_title(f"{os.path.splitext(x_a_files[i])[0].split('_')[-1]}")
        if i == 0: axs[i].set_ylabel("Time Unit")
        axs[i].set_xlabel("Variable Index")
        fig.colorbar(im, ax=axs[i], **_cbar_kwargs)

    return fig, axs
    
def plot_rmse(x_a_files, x_t):
    N_lines = len(x_a_files)
    total_steps = x_t.shape[0]
    
    x_as = [np.load(fname)['x_a'] for fname in x_a_files]
    
    _t = np.arange(total_steps) * dt
    _skip = slice(None, None, 1)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, x_a in enumerate(x_as):
        rmse = np.sqrt(np.mean((x_t - x_a)**2, axis=1))
        ax.plot(_t[_skip], rmse[_skip], label=f"{os.path.splitext(x_a_files[i])[0].split('_')[-1]}")
    
    ax.set_xlabel("Time Unit")
    ax.set_ylabel("RMSE")
    ax.legend()
    return fig, ax
    
# ==================================================
def main():
    exp = f'III'
    x_t = np.load("nature_run.npz")["truth_trajectory"]
    x_0 = np.load("nature_run.npz")["x_true_0"]
    x_a_files = [f for f in os.listdir() if f.startswith(f"x_a_{exp}_")]
    if not x_a_files:
        print(f"No analysis files found for exp.{exp}")
        return
    
    fig, ax = plot_rmse(x_a_files, x_t)
    ax.set_title(f"RMSE of Analysis Trajectories (exp.{exp})")
    # fig, axs = plot_hovmoller(x_a_files, x_t)
    # fig.suptitle(f"Hovmöller Diagram of Analysis Errors (exp.{exp})", fontsize=14)

def validation(exp: str):
    x_t = np.load("nature_run.npz")["truth_trajectory"]
    print(x_t.shape)
    fig, ax = plot_rmse([f"x_a_{exp}_OI.npz", f"x_a_{exp}_CG.npz"], x_t)
    
# ==================================================
from time import perf_counter
if __name__ == '__main__':
    start_time = perf_counter()
    main()
    # validation("I")
    end_time = perf_counter()
    print('\ntime :%.3f ms' %((end_time - start_time)*1000))