"""
Combined Two Moon Experiment

This experiment compares multiple inference methods on the Two Moon problem.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import warnings
import logging

# Import scisbi modules
from scisbi.simulator.TwoMoonSimulator import TwoMoonsSimulator
from scisbi.inference.ABC import ABCRejectionSampling
from scisbi.inference.ABCSMC import ABCSMC
from scisbi.inference.JANA import JANA
from scisbi.inference.Simformer import Simformer

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
warnings.filterwarnings("ignore")

# Setup directories
results_dir = Path("experiment/results-2moon")
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
    def __init__(self):
        # [noise, moon_radius, moon_width, moon_distance]
        self.lower = np.array([0.01, 0.5, 0.1, -1.0])
        self.upper = np.array([0.5, 2.0, 2.0, 1.0])

    def sample(self, num_samples=None):
        if num_samples is None:
            return np.random.uniform(self.lower, self.upper)
        else:
            return np.random.uniform(self.lower, self.upper, size=(num_samples, 4))

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
    def compute(self, data):
        # data: (Batch, N, 2)
        if data.ndim == 2:  # (N, 2) -> (1, N, 2)
            data = data[np.newaxis, :, :]

        # Summary: mean, std, min, max per coordinate
        # (Batch, 8)
        mean = np.mean(data, axis=1)  # (Batch, 2)
        std = np.std(data, axis=1)
        min_val = np.min(data, axis=1)
        max_val = np.max(data, axis=1)

        return np.concatenate([mean, std, min_val, max_val], axis=1)


def distance_function(x, y):
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    return np.linalg.norm(x - y)


class PerturbationKernel:
    def __init__(self, std=0.05):
        self.std = std

    def __call__(self, theta):
        return theta + np.random.normal(0, self.std, size=theta.shape)


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
    logger.info("Starting Two Moon Experiment")

    # Setup
    true_params = np.array([0.1, 1.0, 0.8, -0.25])
    # Ensure n_points is set for simulator
    simulator = TwoMoonsSimulator(n_points=200)
    prior = UniformPrior()
    summary_stat = SummaryStatistic()

    # Generate observed data
    true_params_batch = true_params.reshape(1, -1)
    observed_data = simulator.simulate(true_params_batch)  # (1, 200, 2)

    # Methods
    results = {}

    # 1. ABC Rejection
    logger.info("Running ABC Rejection...")
    try:
        abc = ABCRejectionSampling(
            simulator,
            prior,
            distance_function,
            tolerance=0.5,
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
            tolerance_schedule=[2.0, 1.0, 0.5, 0.25],
            perturbation_kernel=PerturbationKernel(0.05),
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
        # summary stat returns 8 values (mean(2), std(2), min(2), max(2))
        model = JANANetwork(input_dim=8, output_dim=4)
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
        simformer = Simformer(
            simulator,
            prior,
            data_dim=8,
            param_dim=4,
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

    # Save results
    for method, samples in results.items():
        if samples is not None:
            samples = np.array(samples)
            mse = np.mean((np.mean(samples, axis=0) - true_params) ** 2)
            logger.info(f"{method} MSE: {mse}")

    logger.info("Experiment Completed")


if __name__ == "__main__":
    main()
