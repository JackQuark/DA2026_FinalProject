# model driver
# author: Quark
# ==================================================
import numpy     as np
# ==================================================
class Driver:
    def __init__(self, model, dt=0.01, **kwargs):
        if callable(model):
            self.model = model
        else:
            raise ValueError("Model must be a callable function.")
        
        self.t = 0.0
        self.dt = dt
        self.kwargs = kwargs

    def _RK4_step(self, x, t):
        k1 = self.model(t, x, **self.kwargs)
        k2 = self.model(t + self.dt / 2, x + self.dt * k1 / 2, **self.kwargs)
        k3 = self.model(t + self.dt / 2, x + self.dt * k2 / 2, **self.kwargs)
        k4 = self.model(t + self.dt, x + self.dt * k3, **self.kwargs)
        return x + (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def step(self, x, t=0.0):
        """Integrates the model over a single step."""
        return self._RK4_step(x, t)

    def integrate(self, x0, steps, t0=0.0):
        """Integrates the model over a specified number of steps."""
        x_traj = [x0]
        x_curr = x0
        t_curr = t0
        for _ in range(steps):
            x_curr = self._RK4_step(x_curr, t_curr)
            t_curr += self.dt
            x_traj.append(x_curr)
        return np.array(x_traj)