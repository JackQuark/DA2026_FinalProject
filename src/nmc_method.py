# NMC method for estimating background error covariance (B)
# author: Quark
# ==================================================
import numpy as np
# ==================================================
class NMCEstimator:

    def __init__(self, model_driver, dt1_steps, dt2_steps, alpha=1.0, n_grid=None):
        if dt2_steps <= dt1_steps:
            raise ValueError("dt2_steps must be greater than dt1_steps.")

        self.driver = model_driver
        self.dt1 = dt1_steps
        self.dt2 = dt2_steps
        self.alpha = alpha
        self.n_grid = n_grid

        self._errors = None
        self.B = None

    # --------------------------------------------------
    def collect_errors(self, trajectory, perturbation_std=0.1, seed=None):
        rng = np.random.default_rng(seed)
        T = trajectory.shape[0]
        dt1, dt2 = self.dt1, self.dt2

        valid_times = range(dt2, T)

        errors = []
        for t_valid in valid_times:
            x0_short = trajectory[t_valid - dt1] \
                + rng.normal(0, perturbation_std, trajectory.shape[1])
            x0_long = trajectory[t_valid - dt2] \
                + rng.normal(0, perturbation_std, trajectory.shape[1])

            x_short = self.driver.integrate(x0_short, steps=dt1)[-1]
            x_long = self.driver.integrate(x0_long, steps=dt2)[-1]

            eps = x_short - x_long
            errors.append(eps)

        self._errors = np.array(errors)
        return self._errors

    # --------------------------------------------------
    def estimate_B(self, trajectory=None):
        if trajectory is not None:
            self.collect_errors(trajectory)

        if self._errors is None:
            raise RuntimeError(
                "No errors available. Call collect_errors() first "
                "or pass `trajectory`."
            )

        eps = self._errors
        eps_mean = eps.mean(axis=0)
        eps_centered = eps - eps_mean

        M = eps_centered.shape[0]
        cov = (eps_centered.T @ eps_centered) / (M - 1)

        B = self.alpha * cov

        self.B = B
        return B
