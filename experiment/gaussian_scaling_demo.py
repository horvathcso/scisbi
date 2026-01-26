"""
Gaussian Scaling Demo

This script demonstrates how ABC methods scale with different parameters
(sample size, tolerance) on a simple Gaussian problem.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import logging
from pathlib import Path
import seaborn as sns

# Import scisbi modules
from scisbi.simulator.GaussianSimulator import GaussianSimulator
from scisbi.inference.ABC import ABCRejectionSampling
from scisbi.inference.ABCMCMC import ABCMCMC

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Setup directories
results_dir = Path("experiment/results-scale")
if not results_dir.exists():
    results_dir.mkdir(parents=True, exist_ok=True)

# ================================
# SETUP CLASSES
# ================================

class UniformPrior:
    def __init__(self):
        self.bounds = np.array([[-2.0, 2.0], [0.1, 5.0]])

    def sample(self, num_samples=None):
        if num_samples is None:
            return np.array(
                [
                    np.random.uniform(self.bounds[i, 0], self.bounds[i, 1])
                    for i in range(2)
                ]
            )
        else:
            return np.array(
                [
                    [
                        np.random.uniform(self.bounds[i, 0], self.bounds[i, 1])
                        for i in range(2)
                    ]
                    for _ in range(num_samples)
                ]
            )

    def log_prob(self, x):
        x = np.atleast_2d(x)
        log_probs = []
        for params in x:
            if len(params) != 2:
                log_probs.append(-np.inf)
                continue
            in_bounds = all(
                self.bounds[i, 0] <= params[i] <= self.bounds[i, 1] for i in range(2)
            )
            if in_bounds:
                volume = np.prod(self.bounds[:, 1] - self.bounds[:, 0])
                log_probs.append(-np.log(volume))
            else:
                log_probs.append(-np.inf)
        return np.array(log_probs) if len(log_probs) > 1 else log_probs[0]

class SummaryStatistic:
    def compute(self, data):
        # data: (Batch, N)
        if isinstance(data, list):
            data = np.array(data)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        means = np.mean(data, axis=1, keepdims=True)
        stds = np.std(data, axis=1, ddof=1, keepdims=True)
        return np.concatenate([means, stds], axis=1)

def euclidean_distance(x, y):
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    return np.sqrt(np.sum((x - y) ** 2))

class GaussianProposal:
    def __init__(self, std=0.1):
        self.std = std
    def __call__(self, current_state):
        proposal = current_state + np.random.normal(0, self.std, size=len(current_state))
        if len(proposal) > 1:
            proposal[1] = max(proposal[1], 0.1)
        return proposal

# Wrapper for Simulator to generate N samples
class IIDGaussianSimulator(GaussianSimulator):
    def __init__(self, n_samples=100, **kwargs):
        super().__init__(**kwargs)
        self.n_samples = n_samples

    def simulate(self, parameters):
        batch_size = parameters.shape[0]
        mean = parameters[:, 0:1]
        std = parameters[:, 1:2]
        return self.rng.normal(loc=mean, scale=std, size=(batch_size, self.n_samples))

# ================================
# RUNNERS
# ================================

def run_abc_rejection_demo(
    simulator, prior, summary_stat, observed_data, num_samples, tolerance, max_attempts
):
    try:
        start_time = time.time()

        # observed_data (1, N)
        if observed_data.ndim == 1:
            observed_data = observed_data.reshape(1, -1)

        abc = ABCRejectionSampling(
            simulator=simulator,
            prior=prior,
            distance_function=euclidean_distance,
            tolerance=tolerance,
            max_attempts=max_attempts,
            summary_statistic=summary_stat,
            verbose=False,
        )

        result = abc.infer(observed_data, num_simulations=num_samples)
        samples = result.get_samples()

        runtime = time.time() - start_time

        # Compute MSE
        true_params = np.array([1.5, 2.0])
        samples = np.array(samples)
        if samples.ndim == 1:
            samples = samples.reshape(-1, 2)
        mse = np.mean((np.mean(samples, axis=0) - true_params)**2)

        return {
            "method": "ABC-Rejection",
            "num_samples": num_samples,
            "tolerance": tolerance,
            "runtime": runtime,
            "mse": mse,
            "success": True,
            "error": None
        }
    except Exception as e:
        return {
            "method": "ABC-Rejection",
            "num_samples": num_samples,
            "tolerance": tolerance,
            "runtime": 0,
            "mse": 0,
            "success": False,
            "error": str(e)
        }

def run_abc_mcmc_demo(
    simulator, prior, summary_stat, observed_data, num_samples, tolerance, max_iterations
):
    try:
        start_time = time.time()
        if observed_data.ndim == 1:
            observed_data = observed_data.reshape(1, -1)

        proposal = GaussianProposal(std=0.1)

        abc = ABCMCMC(
            simulator=simulator,
            prior=prior,
            distance_function=euclidean_distance,
            tolerance=tolerance,
            proposal_distribution=proposal,
            summary_statistic=summary_stat,
            verbose=False,
        )

        result = abc.infer(observed_data, num_iterations=max_iterations)
        samples = result.get_samples()

        # Take last num_samples if we have more
        if len(samples) > num_samples:
            samples = samples[-num_samples:]

        runtime = time.time() - start_time

        true_params = np.array([1.5, 2.0])
        samples = np.array(samples)
        if samples.ndim == 1:
            samples = samples.reshape(-1, 2)
        mse = np.mean((np.mean(samples, axis=0) - true_params)**2)

        return {
            "method": "ABC-MCMC",
            "num_samples": num_samples,
            "tolerance": tolerance,
            "runtime": runtime,
            "mse": mse,
            "success": True,
            "error": None
        }
    except Exception as e:
        return {
            "method": "ABC-MCMC",
            "num_samples": num_samples,
            "tolerance": tolerance,
            "runtime": 0,
            "mse": 0,
            "success": False,
            "error": str(e)
        }

def run_scaling_demo():
    logger.info("Starting Scaling Demo...")

    # Setup
    true_params = np.array([1.5, 2.0])
    simulator = IIDGaussianSimulator(dimensions=1, seed=42, n_samples=100)
    prior = UniformPrior()
    summary_stat = SummaryStatistic()

    true_params_batch = true_params.reshape(1, -1)
    observed_data = simulator.simulate(true_params_batch) # (1, 100)

    # Configs
    configs = [
        # (samples, tolerance, max_attempts, max_iterations)
        (10, 5.0, 1000, 200),
        (50, 1.0, 5000, 500),
        # Reduced set for quick demo
        (100, 0.5, 10000, 1000),
    ]

    results = []

    for i, (samples, tolerance, max_attempts, max_iterations) in enumerate(configs):
        logger.info(f"Config {i+1}: samples={samples}, tol={tolerance}")

        res_abc = run_abc_rejection_demo(
            simulator, prior, summary_stat, observed_data, samples, tolerance, max_attempts
        )
        results.append(res_abc)

        res_mcmc = run_abc_mcmc_demo(
            simulator, prior, summary_stat, observed_data, samples, tolerance, max_iterations
        )
        results.append(res_mcmc)

    df = pd.DataFrame(results)
    df.to_csv(results_dir / "scaling_demo_results.csv", index=False)
    logger.info("Demo Completed")
    return df

def create_demo_plots(df):
    if df.empty or not df["success"].any():
        return

    # Plot Runtime
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df[df["success"]], x="num_samples", y="runtime", hue="method", style="tolerance")
    plt.title("Runtime Scaling")
    plt.savefig(results_dir / "runtime_scaling.png")

    # Plot MSE
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df[df["success"]], x="num_samples", y="mse", hue="method", style="tolerance")
    plt.title("MSE Scaling")
    plt.savefig(results_dir / "mse_scaling.png")

def create_demo_report(df):
    pass # Skipped for brevity

def main():
    try:
        df = run_scaling_demo()
        create_demo_plots(df)
        create_demo_report(df)
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
