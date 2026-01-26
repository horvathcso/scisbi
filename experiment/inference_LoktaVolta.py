"""
Combined Lotka-Volterra Experiment

This experiment compares multiple inference methods on the Lotka-Volterra problem.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import warnings
import logging

# Import scisbi modules
from scisbi.simulator.LotkaVolterraSimulator import LotkaVolterraSimulator
from scisbi.inference.ABC import ABCRejectionSampling
from scisbi.inference.ABCSMC import ABCSMC
from scisbi.inference.JANA import JANA
from scisbi.inference.Simformer import Simformer

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
warnings.filterwarnings("ignore")

# Setup directories
results_dir = Path("experiment/results-LotkaVolterra")
if not results_dir.exists():
    results_dir.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(results_dir / "experiment.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ================================
# SETUP CLASSES
# ================================


class UniformPrior:
    """Uniform prior for LV parameters."""

    def __init__(self):
        # [alpha, beta, gamma, delta, x0, y0]
        self.lower = np.array([0.1, 0.01, 0.1, 0.01, 1.0, 1.0])
        self.upper = np.array([2.0, 1.0, 2.0, 1.0, 20.0, 20.0])

    def sample(self, num_samples=None):
        if num_samples is None:
            return np.random.uniform(self.lower, self.upper)
        else:
            return np.random.uniform(self.lower, self.upper, size=(num_samples, 6))

    def log_prob(self, x):
        x = np.atleast_2d(x)
        log_probs = []
        volume = np.prod(self.upper - self.lower)
        log_prob_val = -np.log(volume)

        for params in x:
            if np.all(params >= self.lower) and np.all(params <= self.upper):
                log_probs.append(log_prob_val)
            else:
                log_probs.append(-np.inf)

        return np.array(log_probs) if len(log_probs) > 1 else log_probs[0]


class SummaryStatistic:
    """Summary statistic for Lotka-Volterra trajectories."""

    def __init__(self, simulator):
        self.simulator = simulator

    def compute(self, data):
        # data: (Batch, 2, T) or (Batch, 2*T) if flattened
        # LotkaVolterraSimulator returns (Batch, 2, T) now.
        # But if it's flattened, we might need to handle it.
        # The simulator has get_original_shape but that's an instance method.
        # Let's assume input is (Batch, 2, T)

        if data.ndim == 2:
            # Could be (Batch, 2*T) or (Batch, 2, T) is 3D.
            # If flattened:
            # But the new simulator returns (Batch, 2, T).
            # If batch=1, (1, 2, T).
            pass

        # Ensure 3D
        if data.ndim == 2:  # flattened (Batch, 2*T)
            # Use simulator to reshape?
            # Or reimplement reshape logic
            n_points = data.shape[1] // 2
            trajectories = np.zeros((data.shape[0], 2, n_points))
            trajectories[:, 0, :] = data[:, :n_points]
            trajectories[:, 1, :] = data[:, n_points:]
            data = trajectories

        # Compute summary stats: (Batch, 8)
        # Use the simulator's internal method?
        # simulator._compute_summary_statistics takes (Batch, 2, T)
        return self.simulator._compute_summary_statistics(data)


def distance_function(x, y):
    # x, y: (1, 8) or (8,)
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    # Normalize if needed, but simple euclidean for now
    return np.linalg.norm(x - y)


class PerturbationKernel:
    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, theta):
        return theta + np.random.normal(0, self.std, size=theta.shape)


# ================================
# MODELS
# ================================


class JANANetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, z):
        return self.net(torch.cat([x, z], dim=1))


# ================================
# MAIN
# ================================


def main():
    logger.info("Starting Lotka-Volterra Experiment")

    # Setup
    true_params = np.array([1.0, 0.2, 1.5, 0.1, 10.0, 5.0])
    simulator = LotkaVolterraSimulator(t_span=(0, 30), n_points=50, noise_level=0.05)
    prior = UniformPrior()
    summary_stat = SummaryStatistic(simulator)

    # Generate observed data
    true_params_batch = true_params.reshape(1, -1)
    observed_data = simulator.simulate(true_params_batch)  # (1, 2, 50)

    # Methods
    results = {}

    # 1. ABC Rejection
    logger.info("Running ABC Rejection...")
    try:
        abc = ABCRejectionSampling(
            simulator,
            prior,
            distance_function,
            tolerance=10.0,
            summary_statistic=summary_stat,
        )
        posterior = abc.infer(observed_data, num_simulations=1000)
        results["ABC"] = posterior.get_samples()
        logger.info("ABC success")
    except Exception as e:
        logger.error(f"ABC failed: {e}")
        import traceback

        traceback.print_exc()

    # 2. ABC-SMC
    logger.info("Running ABC-SMC...")
    try:
        smc = ABCSMC(
            simulator,
            prior,
            distance_function,
            tolerance_schedule=[50.0, 20.0, 10.0, 5.0],
            perturbation_kernel=PerturbationKernel(0.1),
            summary_statistic=summary_stat,
            num_particles=500,
        )
        posterior = smc.infer(observed_data)
        results["ABC-SMC"] = posterior.sample(1000)
        logger.info("ABC-SMC success")
    except Exception as e:
        logger.error(f"ABC-SMC failed: {e}")
        import traceback

        traceback.print_exc()

    # 3. JANA
    logger.info("Running JANA...")
    try:
        # summary stat returns 8 values
        model = JANANetwork(input_dim=8, output_dim=6)
        jana = JANA(
            simulator, prior, model, summary_statistic=summary_stat, device="auto"
        )
        posterior = jana.infer(observed_data, num_simulations=5000, num_epochs=50)
        results["JANA"] = posterior.sample(observed_data, num_samples=1000)
        logger.info("JANA success")
    except Exception as e:
        logger.error(f"JANA failed: {e}")
        import traceback

        traceback.print_exc()

    # 4. Simformer
    logger.info("Running Simformer...")
    try:
        # Data dim is 8 (summary stats)
        simformer = Simformer(
            simulator,
            prior,
            data_dim=8,
            param_dim=6,
            summary_statistic=summary_stat,
            device="auto",
        )
        posterior = simformer.infer(
            observed_data, num_simulations=1000, sequence_length=20, num_epochs=50
        )
        results["Simformer"] = posterior.sample(observed_data, num_samples=1000)
        logger.info("Simformer success")
    except Exception as e:
        logger.error(f"Simformer failed: {e}")
        import traceback

        traceback.print_exc()

    # Metrics and Plotting
    # ... (Simplified for this file, keeping mostly to running inference)

    # Save results
    for method, samples in results.items():
        if samples is not None:
            samples = np.array(samples)
            mse = np.mean((np.mean(samples, axis=0) - true_params) ** 2)
            logger.info(f"{method} MSE: {mse}")

    logger.info("Experiment Completed")


if __name__ == "__main__":
    main()
