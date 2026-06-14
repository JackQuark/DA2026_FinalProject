# Benchmarking 3DVar with Different Optimization Methods
Data Assimilation 2026 - Final Project

## Quick Start
1. Run `main.py` to perform data assimilation with different optimization methods (Adam, CG, SGD). The results will be saved as `.npz` files.
2. Run `plot.py` to visualize the results.

## File Descriptions
### ./
- *.pdf: slides for presentation

### ./src/
1. main.py: main script to run the data assimilation and save the results
2. plot.py: script to visualize the results
3. optimizer.py: implementation of the optimization algorithms (Adam, CG, SGD)
4. DASystem.py: implementation of 3D-Var / OI
5. nmc_method.py: implementation of the NMC method for estimating the background error covariance
6. driver.py: numerical solver for the given model (time advance)
7. lorenz96.py: implementation of the Lorenz-96 model