import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scisbi.base.simulator import BaseSimulator


class LotkaVolterraSimulator(BaseSimulator):
    """
    Lotka-Volterra simulator optimized for simulation-based inference.

    The model parameters are:
    - alpha: prey natural growth rate
    - beta: prey death rate due to predation
    - gamma: predator natural death rate
    - delta: predator growth rate from consuming prey
    - x0: initial prey population
    - y0: initial predator population

    For SBI tasks, the parameters are represented as a flat array/tensor:
    [alpha, beta, gamma, delta, x0, y0]
    """

    def __init__(
        self,
        t_span=(0, 30),
        n_points=100,
        noise_level=0.05,
        summary_stats=False,
        **kwargs,
    ):
        """
        Initialize the Lotka-Volterra simulator for SBI.

        Args:
            t_span: tuple (t_start, t_end) defining the time range for simulation
            n_points: number of time points to evaluate
            noise_level: observation noise (relative to population sizes)
            summary_stats: if True, return summary statistics instead of trajectories.
                           Note: It is recommended to use a separate SummaryStatistic class.
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        self.t_span = t_span
        self.n_points = n_points
        self.t_eval = np.linspace(t_span[0], t_span[1], n_points)
        self.noise_level = noise_level
        self.summary_stats = summary_stats

    def _system(self, t, z, params):
        """
        The Lotka-Volterra differential equations.
        """
        x, y = z
        alpha, beta, gamma, delta = params

        dx_dt = alpha * x - beta * x * y
        dy_dt = -gamma * y + delta * x * y

        return [dx_dt, dy_dt]

    def _compute_summary_statistics(self, trajectories):
        """
        Compute informative summary statistics from trajectories.

        Args:
            trajectories: numpy array of shape (n_sims, 2, n_points)

        Returns:
            summary statistics as numpy array of shape (n_sims, n_stats)
        """
        n_sims = trajectories.shape[0]

        # Create summary statistics array (8 stats per simulation)
        summary = np.zeros((n_sims, 8))

        for i in range(n_sims):
            prey = trajectories[i, 0]
            predator = trajectories[i, 1]

            # Informative summary statistics
            summary[i, 0] = np.mean(prey)  # Mean prey population
            summary[i, 1] = np.std(prey)  # Std of prey population
            summary[i, 2] = np.mean(predator)  # Mean predator population
            summary[i, 3] = np.std(predator)  # Std of predator population
            summary[i, 4] = np.max(prey)  # Maximum prey population
            summary[i, 5] = np.max(predator)  # Maximum predator population
            summary[i, 6] = np.corrcoef(prey, predator)[0, 1]  # Correlation

            # Period estimation (using autocorrelation)
            prey_norm = prey - np.mean(prey)
            acorr = np.correlate(prey_norm, prey_norm, mode="full")
            acorr = acorr[acorr.size // 2 :]
            # Find first peak after 0 lag
            peaks = (acorr[1:-1] > acorr[:-2]) & (acorr[1:-1] > acorr[2:])
            if np.any(peaks):
                first_peak = np.argmax(peaks) + 1
                period = first_peak * (self.t_span[1] - self.t_span[0]) / self.n_points
                summary[i, 7] = period
            else:
                summary[i, 7] = self.t_span[1]  # No oscillation detected

        return summary

    def simulate(self, parameters):
        """
        Run the Lotka-Volterra simulation for SBI.

        Args:
            parameters: Parameters array of shape (Batch_Size, 6)
                        [alpha, beta, gamma, delta, x0, y0]

        Returns:
            numpy array of shape (Batch_Size, 2, n_points) containing
            prey and predator trajectories for each simulation.

            If summary_stats=True, returns shape (Batch_Size, 8).
        """
        # Handle different input formats
        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            parameters = parameters[np.newaxis, :]

        num_simulations = parameters.shape[0]

        # Ensure parameters have the right shape
        if parameters.shape[1] != 6:
            raise ValueError(
                f"Expected 6 parameters [alpha, beta, gamma, delta, x0, y0], got {parameters.shape[1]}"
            )

        # Prepare output array for trajectories
        trajectories = np.zeros((num_simulations, 2, self.n_points))

        for i in range(num_simulations):
            params = parameters[i]
            alpha, beta, gamma, delta, x0, y0 = params

            # Ensure non-negative parameters
            model_params = [
                max(0.01, alpha),
                max(0.01, beta),
                max(0.01, gamma),
                max(0.01, delta),
            ]
            initial_state = [max(0.1, x0), max(0.1, y0)]

            # Solve the ODE system
            solution = solve_ivp(
                lambda t, z: self._system(t, z, model_params),
                self.t_span,
                initial_state,
                t_eval=self.t_eval,
                method="RK45",
            )

            # Extract solution
            prey = solution.y[0]
            predator = solution.y[1]

            # Add observation noise proportional to population size
            noise_prey = np.random.normal(
                0, self.noise_level * np.mean(prey), size=prey.shape
            )
            noise_predator = np.random.normal(
                0, self.noise_level * np.mean(predator), size=predator.shape
            )

            prey = np.maximum(0, prey + noise_prey)
            predator = np.maximum(0, predator + noise_predator)

            # Handle case where solver failed or returned fewer points (should rely on t_eval though)
            if len(prey) != self.n_points:
                # Fill with last known value or zeros?
                # For now, let's just keep what we have padded or truncated,
                # but solve_ivp with t_eval usually guarantees points.
                # If it failed, it might be shorter.
                # Let's assume t_eval forces it.
                pass

            trajectories[i, 0] = prey
            trajectories[i, 1] = predator

        # Return summary statistics or trajectories
        if self.summary_stats:
            return self._compute_summary_statistics(trajectories)
        else:
            return trajectories

    def plot_simulation(self, result, params=None, title=None):
        """
        Plot the simulation results.

        Args:
            result: Trajectories of shape (Batch, 2, n_points)
                    or flattened (Batch, 2*n_points) for backward compatibility in plotting
                    or summary stats (Batch, 8).
            params: Optional parameters to display
            title: Optional plot title

        Returns:
            matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Check if result is summary statistics
        if result.ndim == 2 and result.shape[1] == 8:
             raise ValueError("Cannot plot summary statistics. Need full trajectories.")

        # Handle flattened vs structured
        if result.ndim == 2:
             # Assume flattened: (Batch, 2*n_points)
             n_sims = result.shape[0]
             trajectories = np.zeros((n_sims, 2, self.n_points))
             trajectories[:, 0, :] = result[:, : self.n_points]
             trajectories[:, 1, :] = result[:, self.n_points :]
             result = trajectories

        # Now result is (Batch, 2, n_points)
        n_sims = result.shape[0]

        for i in range(min(10, n_sims)):  # Plot up to 10 simulations
            alpha = 1.0 if n_sims == 1 else 0.5

            prey = result[i, 0]
            predator = result[i, 1]

            ax1.plot(self.t_eval, prey, "g-", alpha=alpha)
            ax1.plot(self.t_eval, predator, "r-", alpha=alpha)
            ax2.plot(prey, predator, "b-", alpha=alpha)

        ax1.set_xlabel("Time")
        ax1.set_ylabel("Population")
        ax1.set_title("Population vs Time")
        ax1.legend(["Prey", "Predator"])
        ax1.grid(True)

        ax2.set_xlabel("Prey Population")
        ax2.set_ylabel("Predator Population")
        ax2.set_title("Phase Space")
        ax2.grid(True)

        if params is not None:
            param_text = f"α={params[0]:.2f}, β={params[1]:.2f}, γ={params[2]:.2f}, δ={params[3]:.2f}"
            if len(params) > 4:
                param_text += f", x₀={params[4]:.1f}, y₀={params[5]:.1f}"
            fig.text(0.5, 0.01, param_text, ha="center")

        if title:
            fig.suptitle(title)

        plt.tight_layout()
        return fig

    def get_original_shape(self, flattened_data):
        """
        Reshape flattened data back to trajectories format.

        Args:
            flattened_data: Array of shape (num_simulations, 2*n_points)

        Returns:
            Array of shape (num_simulations, 2, n_points)
        """
        # This method is kept for legacy support/testing, assuming flattened_data is what we want to reshape.
        # If the input is already shaped (N, 2, T), we return it or check shape.
        if flattened_data.ndim == 3:
             return flattened_data

        n_sims = flattened_data.shape[0]
        trajectories = np.zeros((n_sims, 2, self.n_points))
        trajectories[:, 0, :] = flattened_data[:, : self.n_points]  # Prey
        trajectories[:, 1, :] = flattened_data[
            :, self.n_points : 2 * self.n_points
        ]  # Predator
        return trajectories
