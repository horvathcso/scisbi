"""
Combined Gaussian Distribution Experiment

This experiment compares 6 inference methods on a Gaussian distribution problem:
1. ABC Rejection Sampling
2. ABC-MCMC
3. ABC-SMC
4. Simple Neural Network
5. JANA
6. Simformer

Target: Gaussian distribution with mean=1.5, std=2.0
Prior: Uniform prior with mean∈[-2,2], std∈[0.1,5]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
import time
import pandas as pd
import warnings
import logging
from datetime import datetime

# Import scisbi modules
from scisbi.simulator.GaussianSimulator import GaussianSimulator
from scisbi.inference.ABC import ABCRejectionSampling
from scisbi.inference.ABCMCMC import ABCMCMC
from scisbi.inference.ABCSMC import ABCSMC
from scisbi.inference.JANA import JANA
from scisbi.inference.Simformer import Simformer

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
warnings.filterwarnings("ignore")

# Setup directories
results_dir = Path("experiment/results-gaussian")

# Setup logging
if not results_dir.exists():
    results_dir.mkdir(parents=True, exist_ok=True)

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
    """Uniform prior for Gaussian parameters (mean, std)."""

    def __init__(self):
        self.bounds = np.array([[-2.0, 2.0], [0.1, 5.0]])

    def sample(self, num_samples=None):
        if num_samples is None:
            # Single sample (P,)
            return np.array(
                [
                    np.random.uniform(self.bounds[i, 0], self.bounds[i, 1])
                    for i in range(2)
                ]
            )
        else:
            # Batch sample (N, P)
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
        """Log probability under uniform prior."""
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
                # Uniform density = 1 / (volume of rectangle)
                volume = np.prod(self.bounds[:, 1] - self.bounds[:, 0])
                log_probs.append(-np.log(volume))
            else:
                log_probs.append(-np.inf)

        return np.array(log_probs) if len(log_probs) > 1 else log_probs[0]


class SummaryStatistic:
    """Summary statistic for Gaussian data."""

    def compute(self, data):
        # data can be (Batch, N) or (N,)
        if isinstance(data, list):
            data = np.array(data)

        # If input is (N,), treat as (1, N) for calculation then squeeze?
        # Standard: (Batch, N). Returns (Batch, S)
        if data.ndim == 1:
            # Assume single batch
            data = data.reshape(1, -1)

        means = np.mean(data, axis=1, keepdims=True)
        stds = np.std(data, axis=1, ddof=1, keepdims=True)
        return np.concatenate([means, stds], axis=1)


class GaussianPerturbationKernel:
    """Gaussian perturbation kernel for ABC-SMC."""

    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, particle):
        perturbed = particle + np.random.normal(0, self.std, size=len(particle))
        # Ensure std parameter (index 1) is positive
        if len(perturbed) > 1:
            perturbed[1] = max(perturbed[1], 0.1)  # Minimum std of 0.1
        return perturbed


class GaussianProposal:
    """Gaussian proposal distribution for ABC-MCMC."""

    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, current_state):
        proposal = current_state + np.random.normal(
            0, self.std, size=len(current_state)
        )
        # Ensure std parameter (index 1) is positive
        if len(proposal) > 1:
            proposal[1] = max(proposal[1], 0.1)  # Minimum std of 0.1
        return proposal


def euclidean_distance(x, y):
    """Euclidean distance between summary statistics."""
    # x, y: (1, S) or (S,)
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    return np.sqrt(np.sum((x - y) ** 2))


# ================================
# NEURAL NETWORK MODELS
# ================================


class SimpleNeuralNetwork(nn.Module):
    """Simple neural network for direct parameter estimation."""

    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Initialize weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        output = self.net(x)

        # Ensure positive std
        if output.shape[1] == 2:
            mean_part = output[:, 0:1]
            std_part = torch.nn.functional.softplus(output[:, 1:2]) + 0.1
            output = torch.cat([mean_part, std_part], dim=1)

        return output


class ImprovedJANANetwork(nn.Module):
    """Improved JANA network with better stability."""

    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()

        # Network with LayerNorm for stability
        self.net = nn.Sequential(
            nn.Linear(input_dim + output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

        # Conservative initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)

    def forward(self, x, z):
        """Forward pass with summary statistics x and parameter samples z."""
        combined = torch.cat([x, z], dim=1)
        output = self.net(combined)

        # Ensure positive std using softplus (non-inplace)
        if output.shape[1] == 2:
            mean_part = output[:, 0:1]
            std_part = torch.nn.functional.softplus(output[:, 1:2]) + 0.1
            output = torch.cat([mean_part, std_part], dim=1)

        return output


class ImprovedSimformerNetwork(nn.Module):
    """Improved Simformer network with better stability."""

    # Note: Simformer already has a network class in its module, but we can reuse this
    # or use the one from the module. The experiment used to import modules but define networks here.
    # The new Simformer implementation uses its own transformer class.
    # We should use the one from the module if we use the Simformer class.
    pass


# ================================
# INFERENCE METHODS
# ================================


def run_abc_rejection(simulator, prior, summary_stat, observed_data, num_samples=2500):
    """Run ABC rejection sampling."""

    logger.info("Starting ABC Rejection Sampling...")
    # observed_data should be (1, N)
    if observed_data.ndim == 1:
        observed_data = observed_data.reshape(1, -1)

    obs_summary = summary_stat.compute(observed_data)

    abc = ABCRejectionSampling(
        simulator=simulator,
        prior=prior,
        distance_function=euclidean_distance,
        tolerance=0.25,
        max_attempts=500000,
        verbose=False,
        summary_statistic=summary_stat
    )

    try:
        # Note: infer usually takes observed_data, not summary if summary_stat is passed to init.
        # But if summary_stat is passed to init, ABC handles it.
        # Here we passed summary_stat to init. So pass observed_data.

        abc_posterior = abc.infer(observed_data, num_simulations=num_samples)
        samples = abc_posterior.get_samples()

        # Convert samples to numpy array if needed
        if isinstance(samples, list):
            samples = np.array(samples)

        # Handle different sample formats
        if hasattr(samples, "shape") and samples.ndim == 1:
            # If samples is 1D (should be 2D N, P), reshape to 2D
            samples = samples.reshape(-1, 2)

        logger.info(f"ABC Rejection: Successfully generated {len(samples)} samples")
        return samples, None
    except Exception as e:
        logger.error(f"ABC Rejection failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def run_abc_mcmc(simulator, prior, summary_stat, observed_data, num_samples=2500):
    """Run ABC-MCMC."""

    logger.info("Starting ABC-MCMC...")
    if observed_data.ndim == 1:
        observed_data = observed_data.reshape(1, -1)

    proposal = GaussianProposal(std=0.1)

    abc_mcmc = ABCMCMC(
        simulator=simulator,
        prior=prior,
        distance_function=euclidean_distance,
        tolerance=0.25,
        proposal_distribution=proposal,
        verbose=False,
        burn_in=100,
        thin=2,
        summary_statistic=summary_stat
    )

    try:
        abc_mcmc_posterior = abc_mcmc.infer(
            observed_data, num_iterations=num_samples + 1000 # ensure enough
        )
        samples = abc_mcmc_posterior.get_samples()

        # Convert samples to numpy array if needed
        if isinstance(samples, list):
            samples = np.array(samples)

        if hasattr(samples, "shape") and samples.ndim == 1:
            samples = samples.reshape(-1, 2)

        logger.info(f"ABC-MCMC: Successfully generated {len(samples)} samples")
        return samples, None
    except Exception as e:
        logger.error(f"ABC-MCMC failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def run_abc_smc(simulator, prior, summary_stat, observed_data, num_samples=2500):
    """Run ABC-SMC."""

    logger.info("Starting ABC-SMC...")
    if observed_data.ndim == 1:
        observed_data = observed_data.reshape(1, -1)

    perturbation_kernel = GaussianPerturbationKernel(std=0.1)

    abc_smc = ABCSMC(
        simulator=simulator,
        prior=prior,
        distance_function=euclidean_distance,
        tolerance_schedule=[0.8, 0.4, 0.2],
        perturbation_kernel=perturbation_kernel,
        verbose=False,
        summary_statistic=summary_stat,
        num_particles=num_samples
    )

    try:
        abc_smc_posterior = abc_smc.infer(
            observed_data
        )

        # Use the sample method to get samples
        samples = abc_smc_posterior.sample(num_samples)

        # Convert to numpy array
        if isinstance(samples, list):
            samples = np.array(samples)

        logger.info(f"ABC-SMC: Successfully generated {len(samples)} samples")
        return samples, None

    except Exception as e:
        logger.error(f"ABC-SMC failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def train_simple_nn(
    simulator, prior, summary_stat, observed_data, num_epochs=250, device="cuda"
):
    """Train simple neural network with device parameter."""

    logger.info("Training Simple Neural Network...")
    if observed_data.ndim == 1:
        observed_data = observed_data.reshape(1, -1)

    obs_summary = summary_stat.compute(observed_data)

    # Generate training data
    num_sims = 5000

    # Use batch simulation
    params = prior.sample(num_sims) # (N, P)
    sim_data = simulator.simulate(params) # (N, D)
    sim_summary = summary_stat.compute(sim_data) # (N, S)

    X_train = sim_summary
    y_train = params

    # Normalize data
    X_mean, X_std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
    X_train_norm = (X_train - X_mean) / (X_std + 1e-8)

    y_mean, y_std = np.mean(y_train, axis=0), np.std(y_train, axis=0)
    y_train_norm = (y_train - y_mean) / (y_std + 1e-8)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Create model
    model = SimpleNeuralNetwork(X_train.shape[1], y_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        X_tensor = torch.FloatTensor(X_train_norm).to(device)
        y_tensor = torch.FloatTensor(y_train_norm).to(device)

        predictions = model(X_tensor)
        loss = nn.MSELoss()(predictions, y_tensor)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if epoch % 20 == 0:
            logger.info(f"Simple NN Epoch {epoch}: Loss = {loss.item():.6f}")

    # Generate samples
    model.eval()
    with torch.no_grad():
        obs_norm = (obs_summary - X_mean) / (X_std + 1e-8)
        obs_tensor = torch.FloatTensor(obs_norm).to(device) # (1, S)

        samples = []

        # Strategy 1: Add meaningful noise to capture uncertainty
        for _ in range(300):
            noise_scale = 0.3  # Scale factor for uncertainty
            noisy_obs = obs_tensor + torch.randn_like(obs_tensor) * noise_scale

            sample_norm = model(noisy_obs)
            sample = sample_norm.cpu().numpy()[0] * (y_std + 1e-8) + y_mean
            samples.append(sample)

        # Strategy 2: Ensemble-like sampling by adding different noise levels
        noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
        for noise_level in noise_levels:
            for _ in range(140):  # 140 * 5 = 700 more samples
                noisy_obs = obs_tensor + torch.randn_like(obs_tensor) * noise_level
                sample_norm = model(noisy_obs)
                sample = sample_norm.cpu().numpy()[0] * (y_std + 1e-8) + y_mean
                samples.append(sample)

        samples = np.array(samples)

        # Filter valid samples
        valid_mask = ~np.isnan(samples).any(axis=1)
        samples = samples[valid_mask]

        # Ensure std parameter is positive
        samples[:, 1] = np.maximum(samples[:, 1], 0.1)

        logger.info(f"Simple NN: Successfully generated {len(samples)} samples")
        return samples, train_losses


def train_jana(
    simulator,
    prior,
    summary_stat,
    observed_data,
    num_epochs=150,
    batch_size=32,
    device="cuda",
):
    """Train JANA."""

    logger.info("Training JANA...")
    if observed_data.ndim == 1:
        observed_data = observed_data.reshape(1, -1)

    # Use JANA class from library
    jana = JANA(
        simulator=simulator,
        prior=prior,
        model=ImprovedJANANetwork(2, 2), # 2 summary stats, 2 params
        summary_statistic=summary_stat,
        device="auto"
    )

    # Infer
    try:
        jana_posterior = jana.infer(
            observed_data=observed_data,
            num_simulations=5000,
            batch_size=batch_size,
            num_epochs=num_epochs,
            verbose=True
        )

        samples = jana_posterior.sample(observed_data, num_samples=1000)

        # Training loss history not directly available from infer return, but we can return None for plots or capture log
        return samples, None

    except Exception as e:
        logger.error(f"JANA failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def train_simformer(
    simulator,
    prior,
    summary_stat,
    observed_data,
    num_epochs=150,
    batch_size=32,
    device="cuda",
):
    """Train Simformer."""

    logger.info("Training Simformer...")
    if observed_data.ndim == 1:
        observed_data = observed_data.reshape(1, -1)

    # Use Simformer class from library
    simformer = Simformer(
        simulator=simulator,
        prior=prior,
        data_dim=1, # Univariate gaussian
        param_dim=2, # Mean, std
        summary_statistic=summary_stat, # Wait, Simformer processes raw sequences usually?
        # The library implementation supports summary_statistic.
        # But if we use summary statistic (which returns 2 values: mean, std), data_dim should be 2.
        # If we pass summary_stat, the simulator output (100 samples) is reduced to (2,).
        # Then sequence is of (2,) vectors? No, Simformer usually takes sequence of simulations.
        # The library implementation:
        # seq_data.append(sim_data) where sim_data is summary.
        # If we use SummaryStatistic class above, it returns (2,) array for (100,) input.
        # So data_dim should be 2 if we use summary stat.
        device="auto"
    )

    # If we use summary statistic, the data dim is 2 (mean, std).
    # If we don't, data dim is 1 (univariate gaussian samples), but then sequence length is 100?
    # Simformer library implementation generates sequences of parameters.
    # For each parameter step, it simulates data.
    # If summary_stat is used, it simulates (100,) -> computes (2,) -> sequence of (2,)s.

    # Let's try WITHOUT summary statistic for Simformer to let it learn from raw data sequences?
    # But Simformer needs a sequence of (param, data) pairs.
    # The library implementation does:
    # for j in sequence_length: param=prior.sample(), data=sim(param).
    # So it learns p(theta | data).

    # If we use summary statistic, data_dim=2.
    # Let's stick to using summary stat to be comparable with others?
    # Or let it handle raw data?
    # GaussianSimulator returns (100,) per simulation.
    # If we use raw data, data_dim=100.

    # Let's use summary statistic for consistency and lower dim.
    simformer.data_dim = 2
    simformer.model = simformer.model = Simformer.model = None # re-init model in init... wait
    # We need to re-initialize the model inside Simformer with correct dimensions if we change them.
    # Better to init correctly.

    simformer = Simformer(
        simulator=simulator,
        prior=prior,
        data_dim=2,
        param_dim=2,
        summary_statistic=summary_stat,
        device="auto"
    )

    try:
        sf_posterior = simformer.infer(
            observed_data=observed_data,
            num_simulations=1000, # Sequences
            sequence_length=20,
            batch_size=batch_size,
            num_epochs=num_epochs,
            verbose=True
        )

        samples = sf_posterior.sample(observed_data, num_samples=1000)
        return samples, None

    except Exception as e:
        logger.error(f"Simformer failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ================================
# EVALUATION METRICS
# ================================


def compute_metrics(samples, true_params):
    """Compute evaluation metrics for samples."""

    if samples is None or len(samples) == 0:
        return {
            "mse": np.inf,
            "coverage": 0.0,
            "interval_width": np.inf,
            "mean_estimate": np.array([np.nan, np.nan]),
            "std_estimate": np.array([np.nan, np.nan]),
        }

    # Mean squared error
    mean_estimate = np.mean(samples, axis=0)
    mse = np.mean((mean_estimate - true_params) ** 2)

    # Standard deviation of estimates
    std_estimate = np.std(samples, axis=0)

    # Coverage and interval width (95% credible interval)
    coverage = 0.0
    interval_width = 0.0

    for i in range(len(true_params)):
        lower = np.percentile(samples[:, i], 2.5)
        upper = np.percentile(samples[:, i], 97.5)

        if lower <= true_params[i] <= upper:
            coverage += 1.0

        interval_width += upper - lower

    coverage /= len(true_params)
    interval_width /= len(true_params)

    return {
        "mse": mse,
        "coverage": coverage,
        "interval_width": interval_width,
        "mean_estimate": mean_estimate,
        "std_estimate": std_estimate,
    }


# ================================
# VISUALIZATION
# ================================


def create_comparison_plots(results, true_params):
    """Create comparison plots for all methods."""

    successful_methods = {k: v for k, v in results.items() if v["samples"] is not None}

    if not successful_methods:
        logger.warning("No successful methods to plot")
        return

    num_methods = len(successful_methods)
    fig, axes = plt.subplots(2, num_methods, figsize=(5 * num_methods, 10))

    # Handle case where there's only one method
    if num_methods == 1:
        axes = axes.reshape(2, 1)

    method_colors = ["blue", "green", "orange", "red", "purple", "brown"]

    for i, (method_name, result) in enumerate(successful_methods.items()):
        samples = result["samples"]
        metrics = result["metrics"]
        color = method_colors[i % len(method_colors)]

        # Convert samples to numpy array if it's a list
        if isinstance(samples, list):
            samples = np.array(samples)

        # Ensure samples is 2D
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)

        # Plot mean parameter
        if num_methods == 1:
            ax1 = axes[0]
            ax2 = axes[1]
        else:
            ax1 = axes[0, i]
            ax2 = axes[1, i]

        ax1.hist(samples[:, 0], bins=50, alpha=0.7, density=True, color=color)
        ax1.axvline(
            true_params[0],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"True = {true_params[0]:.2f}",
        )
        ax1.axvline(
            metrics["mean_estimate"][0],
            color="black",
            linestyle="-",
            linewidth=2,
            label=f"Est = {metrics['mean_estimate'][0]:.2f}",
        )
        ax1.set_xlabel("Mean Parameter")
        ax1.set_ylabel("Density")
        ax1.set_title(f"{method_name} - Mean Posterior\nMSE: {metrics['mse']:.4f}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot std parameter
        ax2.hist(samples[:, 1], bins=50, alpha=0.7, density=True, color=color)
        ax2.axvline(
            true_params[1],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"True = {true_params[1]:.2f}",
        )
        ax2.axvline(
            metrics["mean_estimate"][1],
            color="black",
            linestyle="-",
            linewidth=2,
            label=f"Est = {metrics['mean_estimate'][1]:.2f}",
        )
        ax2.set_xlabel("Std Parameter")
        ax2.set_ylabel("Density")
        ax2.set_title(f"{method_name} - Std Posterior\nTime: {result['time']:.2f}s")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        results_dir / "combined_gaussian_comparison.png", dpi=300, bbox_inches="tight"
    )
    # plt.show()
    logger.info("Comparison plots saved to combined_gaussian_comparison.png")


def create_timing_plot(results):
    """Create timing comparison plot."""

    successful_methods = {k: v for k, v in results.items() if v["samples"] is not None}

    if not successful_methods:
        logger.warning("No successful methods for timing plot")
        return

    methods = list(successful_methods.keys())
    times = [successful_methods[m]["time"] for m in methods]

    plt.figure(figsize=(12, 6))
    colors = ["blue", "green", "orange", "red", "purple", "brown"]
    bars = plt.bar(methods, times, color=colors[: len(methods)])
    plt.ylabel("Time (seconds)")
    plt.title("Timing Comparison - Combined Gaussian Inference Experiment")
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar, t in zip(bars, times):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{t:.2f}s",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(results_dir / "timing_comparison.png", dpi=300, bbox_inches="tight")
    # plt.show()
    logger.info("Timing plot saved to timing_comparison.png")


def plot_training_curves(results):
    """Plot training loss curves for neural network methods."""

    nn_methods = {
        k: v
        for k, v in results.items()
        if v["train_losses"] is not None and v["samples"] is not None
    }

    if not nn_methods:
        logger.warning("No neural network methods with training curves to plot")
        return

    num_methods = len(nn_methods)
    fig, axes = plt.subplots(1, num_methods, figsize=(6 * num_methods, 5))

    if num_methods == 1:
        axes = [axes]

    colors = ["blue", "green", "orange", "red", "purple", "brown"]

    for i, (method_name, result) in enumerate(nn_methods.items()):
        color = colors[i % len(colors)]
        axes[i].plot(result["train_losses"], color=color, linewidth=2)
        axes[i].set_title(f"{method_name} - Training Loss")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Loss")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "training_curves.png", dpi=300, bbox_inches="tight")
    # plt.show()
    logger.info("Training curves saved to training_curves.png")


def create_summary_table(results, true_params):
    """Create summary table of results."""

    summary_data = []

    for method_name, result in results.items():
        if result["samples"] is not None:
            metrics = result["metrics"]

            summary_data.append(
                {
                    "Method": method_name,
                    "Time (s)": f"{result['time']:.2f}",
                    "MSE": f"{metrics['mse']:.4f}",
                    "Coverage": f"{metrics['coverage']:.2f}",
                    "Interval Width": f"{metrics['interval_width']:.2f}",
                    "Mean Est.": f"[{metrics['mean_estimate'][0]:.2f}, {metrics['mean_estimate'][1]:.2f}]",
                    "Std Est.": f"[{metrics['std_estimate'][0]:.2f}, {metrics['std_estimate'][1]:.2f}]",
                    "Status": "SUCCESS",
                }
            )
        else:
            summary_data.append(
                {
                    "Method": method_name,
                    "Time (s)": f"{result['time']:.2f}",
                    "MSE": "FAILED",
                    "Coverage": "FAILED",
                    "Interval Width": "FAILED",
                    "Mean Est.": "FAILED",
                    "Std Est.": "FAILED",
                    "Status": "FAILED",
                }
            )

    df = pd.DataFrame(summary_data)

    logger.info("=" * 70)
    logger.info("SUMMARY TABLE")
    logger.info("=" * 70)
    logger.info(f"\n{df.to_string(index=False)}")

    # Save to CSV
    df.to_csv(results_dir / "combined_gaussian_results.csv", index=False)
    logger.info("Summary table saved to combined_gaussian_results.csv")


def plot_inferred_model_samples(results, true_params, simulator, filename=None):
    """Plot samples drawn from inferred parameter values for all methods."""

    successful_methods = {k: v for k, v in results.items() if v["samples"] is not None}

    if not successful_methods:
        logger.warning("No successful methods for inferred model samples plot")
        return

    num_methods = len(successful_methods)
    fig, axes = plt.subplots(2, num_methods, figsize=(5 * num_methods, 10))

    # Handle case where there's only one method
    if num_methods == 1:
        axes = axes.reshape(2, 1)

    colors = ["blue", "green", "orange", "red", "purple", "brown"]

    # Generate true data for comparison
    # true_params is (P,) -> simulator expects (1, P)
    true_params_batch = true_params.reshape(1, -1)
    true_data = simulator.simulate(true_params_batch).flatten() # (100,)

    for i, (method_name, result) in enumerate(successful_methods.items()):
        samples = result["samples"]
        color = colors[i % len(colors)]

        # Convert samples to numpy array if needed
        if isinstance(samples, list):
            samples = np.array(samples)

        # Get mean estimate for this method
        mean_estimate = np.mean(samples, axis=0)

        # Generate data from mean estimate
        mean_estimate_batch = mean_estimate.reshape(1, -1)
        mean_data = simulator.simulate(mean_estimate_batch).flatten()

        # Plot 1: Histogram comparison
        ax1 = axes[0, i] if num_methods > 1 else axes[0]
        ax1.hist(
            true_data,
            bins=50,
            alpha=0.7,
            density=True,
            color="red",
            label=f"True (μ={true_params[0]:.2f}, σ={true_params[1]:.2f})",
        )
        ax1.hist(
            mean_data,
            bins=50,
            alpha=0.7,
            density=True,
            color=color,
            label=f"Inferred (μ={mean_estimate[0]:.2f}, σ={mean_estimate[1]:.2f})",
        )
        ax1.set_title(f"{method_name} - Data Comparison")
        ax1.set_xlabel("Value")
        ax1.set_ylabel("Density")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Multiple samples from posterior
        ax2 = axes[1, i] if num_methods > 1 else axes[1]

        # Draw samples from posterior for visualization
        n_posterior_samples = min(10, len(samples))
        indices = np.random.choice(len(samples), n_posterior_samples, replace=False)

        for j, idx in enumerate(indices):
            sample_params = samples[idx]
            sample_params_batch = sample_params.reshape(1, -1)
            sample_data = simulator.simulate(sample_params_batch).flatten()

            alpha = 0.3 if j > 0 else 0.7
            label = "Posterior Samples" if j == 0 else None
            ax2.hist(
                sample_data,
                bins=30,
                alpha=alpha,
                density=True,
                color=color,
                label=label,
            )

        # Overlay true data
        ax2.hist(
            true_data,
            bins=30,
            alpha=0.7,
            density=True,
            color="red",
            label="True Data",
            linestyle="--",
            histtype="step",
            linewidth=2,
        )

        ax2.set_title(f"{method_name} - Posterior Uncertainty")
        ax2.set_xlabel("Value")
        ax2.set_ylabel("Density")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if filename:
        plt.savefig(results_dir / filename, dpi=300, bbox_inches="tight")
    # plt.show()
    logger.info(f"Inferred model samples plot saved to {filename}")


# ================================
# MAIN EXPERIMENT
# ================================


def main():
    """Run the main experiment."""

    logger.info("=" * 70)
    logger.info("COMBINED GAUSSIAN INFERENCE EXPERIMENT")
    logger.info("=" * 70)
    logger.info(
        f"Experiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Setup
    true_params = np.array([1.5, 2.0])
    # GaussianSimulator(dimensions=1) returns (Batch, 1) by default or (Batch, N)?
    # GaussianSimulator implementation:
    # return self.rng.normal(loc=mean, scale=std, size=(batch_size, dimensions))
    # This generates 1 sample per simulation.
    # But usually we observe multiple data points (e.g. 100).
    # How to generate 100 samples per parameter set with GaussianSimulator?
    # The default GaussianSimulator outputs (batch_size, dimensions).
    # If we want (100,) for a single parameter set, we need to batch the parameters 100 times?
    # Or modify simulator to return N samples?
    #
    # In the new interface, "num_simulations" is gone.
    # "simulator.simulate(parameters)" returns 1 output per parameter vector.
    # So if we want 100 i.i.d. samples, we need 100 copies of parameter vector in batch.

    # However, SummaryStatistic expects (N,) data?
    # The old experiment generated 100 data points.

    # We should define a wrapper or use broadcasting.
    # If we pass (100, 2) parameters (all same), we get (100, 1) data.
    # Then summary statistic computes mean/std of these 100 points.

    # Let's adjust how we use the simulator in this experiment.

    # Wrapper simulator for this specific experiment that treats "1 simulation" as "100 i.i.d samples"
    class IIDGaussianSimulator(GaussianSimulator):
        def __init__(self, n_samples=100, **kwargs):
            super().__init__(**kwargs)
            self.n_samples = n_samples

        def simulate(self, parameters):
            # parameters: (Batch, 2)
            # We want output (Batch, n_samples)
            # Default simulate returns (Batch, 1) if dimensions=1.
            # We can't easily change default simulate behavior without changing base class or using loop?
            # Or we can exploit numpy broadcasting if simulate used it?
            # Default simulate: samples = rng.normal(loc=mean, scale=std, size=(batch_size, dimensions))
            # If we want size=(batch_size, n_samples), we need to override.

            # Since we are allowed to implement other simulators, let's use a local class or override.
            # But we are importing GaussianSimulator.

            # Let's just implement simulate here correctly.
            batch_size = parameters.shape[0]
            mean = parameters[:, 0:1]
            std = parameters[:, 1:2]

            # (Batch, n_samples)
            return self.rng.normal(loc=mean, scale=std, size=(batch_size, self.n_samples))

    simulator = IIDGaussianSimulator(dimensions=1, seed=42, n_samples=100)
    prior = UniformPrior()
    summary_stat = SummaryStatistic()

    # Generate observed data
    # True params (1, 2)
    true_params_batch = true_params.reshape(1, -1)
    observed_data = simulator.simulate(true_params_batch).flatten() # (100,)

    obs_summary = summary_stat.compute(observed_data)

    logger.info(f"True parameters: {true_params}")
    logger.info(f"Observed summary: {obs_summary}")

    # Define methods
    methods = {
        "ABC": run_abc_rejection,
        "ABC-MCMC": run_abc_mcmc,
        "ABC-SMC": run_abc_smc,
        "Simple-NN": train_simple_nn,
        "JANA": train_jana,
        "Simformer": train_simformer,
    }

    results = {}

    # Run each method
    for method_name, method_func in methods.items():
        logger.info("\n" + "=" * 50)
        logger.info(f"RUNNING {method_name}")
        logger.info("=" * 50)

        try:
            start_time = time.time()

            if method_name in ["Simple-NN", "JANA", "Simformer"]:
                method_result = method_func(
                    simulator, prior, summary_stat, observed_data
                )
                if method_result is not None:
                    samples, train_losses = method_result
                else:
                    samples, train_losses = None, None
            else:
                method_result = method_func(
                    simulator, prior, summary_stat, observed_data, num_samples=1000
                )
                if method_result is not None:
                    samples, train_losses = method_result
                else:
                    samples, train_losses = None, None

            run_time = time.time() - start_time

            if samples is not None and len(samples) > 0:
                metrics = compute_metrics(samples, true_params)

                results[method_name] = {
                    "samples": samples,
                    "time": run_time,
                    "metrics": metrics,
                    "train_losses": train_losses,
                }

                logger.info(f"  Time: {run_time:.2f}s")
                logger.info(f"  Samples: {len(samples)}")
                logger.info(f"  MSE: {metrics['mse']:.4f}")
                logger.info(f"  Coverage: {metrics['coverage']:.2f}")
                logger.info(f"  Mean estimate: {metrics['mean_estimate']}")
                logger.info("  SUCCESS")
            else:
                logger.warning("  FAILED - No valid samples")
                results[method_name] = {
                    "samples": None,
                    "time": run_time,
                    "metrics": None,
                    "train_losses": train_losses,
                }

        except Exception as e:
            logger.error(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[method_name] = {
                "samples": None,
                "time": 0,
                "metrics": None,
                "train_losses": None,
            }

    # Create summary table
    create_summary_table(results, true_params)

    # Create visualizations
    create_comparison_plots(results, true_params)
    create_timing_plot(results)
    plot_training_curves(results)
    plot_inferred_model_samples(
        results, true_params, simulator, filename="inferred_model_samples.png"
    )

    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT COMPLETED")
    logger.info(
        f"Experiment finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    logger.info("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
