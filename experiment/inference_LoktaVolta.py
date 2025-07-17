"""
Simplified Comprehensive Inference Methods Comparison on Two Moon Simulator

This experiment compares available inference methods on the Two Moon Simulator
with extensive plotting and timing analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import time as time_module
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Import scisbi modules

from scisbi.simulator import LotkaVolterraSimulator
from scisbi.inference.ABC import ABCRejectionSampling
from scisbi.inference.ABCMCMC import ABCMCMC
from scisbi.inference.ABCSMC import ABCSMC
from scisbi.inference.Simformer import Simformer
from scisbi.inference.JANA import JANA

import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Create results directory
results_dir = Path("experiment/results-LotkaVolterra")
results_dir.mkdir(parents=True, exist_ok=True)

# ================================
# HELPER CLASSES AND FUNCTIONS
# ================================


class UniformPrior:
    """Uniform prior for Lotka-Volterra parameters."""

    def __init__(self):
        # Prior ranges: [α, β, γ, δ, x0, y0]
        # More reasonable bounds for Lotka-Volterra parameters
        self.bounds = np.array(
            [
                [0.1, 3.0],  # α (prey growth rate)
                [0.01, 1.0],  # β (predation rate)
                [0.1, 3.0],  # γ (predator efficiency)
                [0.01, 1.0],  # δ (predator death rate)
                [1.0, 20.0],  # x0 (initial prey population)
                [1.0, 20.0],  # y0 (initial predator population)
            ]
        )

    def sample(self, num_samples=None):
        if num_samples is None:
            sample = np.array(
                [
                    np.random.uniform(self.bounds[i, 0], self.bounds[i, 1])
                    for i in range(6)
                ]
            )
            # Ensure all values are positive
            sample = np.maximum(sample, 1e-6)
            return sample
        else:
            samples = np.array(
                [
                    [
                        np.random.uniform(self.bounds[i, 0], self.bounds[i, 1])
                        for i in range(6)
                    ]
                    for _ in range(num_samples)
                ]
            )
            # Ensure all values are positive
            samples = np.maximum(samples, 1e-6)
            return samples

    def log_prob(self, x):
        """Log probability under uniform prior."""
        x = np.atleast_2d(x)
        log_probs = []

        for params in x:
            if len(params) != 6:
                log_probs.append(-np.inf)
                continue

            in_bounds = all(
                self.bounds[i, 0] <= params[i] <= self.bounds[i, 1] for i in range(6)
            )

            if in_bounds:
                # Uniform density = 1 / (volume of hypercube)
                volume = np.prod(self.bounds[:, 1] - self.bounds[:, 0])
                log_probs.append(-np.log(volume))
            else:
                log_probs.append(-np.inf)

        return np.array(log_probs) if len(log_probs) > 1 else log_probs[0]


class GaussianPerturbationKernel:
    """Gaussian perturbation kernel for ABC-SMC."""

    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, particle):
        perturbed = particle + np.random.normal(0, self.std, size=len(particle))
        # Ensure all parameters remain positive
        perturbed = np.maximum(perturbed, 1e-6)
        return perturbed


class GaussianProposal:
    """Gaussian proposal distribution for ABC-MCMC."""

    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, current_state):
        proposal = current_state + np.random.normal(
            0, self.std, size=len(current_state)
        )
        # Ensure all parameters remain positive
        proposal = np.maximum(proposal, 1e-6)
        return proposal


def euclidean_distance(x, y):
    """Euclidean distance between flattened arrays or summary statistics."""
    # Convert inputs to numpy arrays
    x_arr = np.asarray(x).flatten()
    y_arr = np.asarray(y).flatten()

    # Ensure same length by truncating to minimum length
    min_len = min(len(x_arr), len(y_arr))
    x_arr = x_arr[:min_len]
    y_arr = y_arr[:min_len]

    return np.sqrt(np.sum((x_arr - y_arr) ** 2))


def calculate_accuracy_metrics(samples, true_params):
    """Calculate accuracy metrics for posterior samples."""
    if isinstance(samples, list) and len(samples) > 0:
        samples = np.array(samples)

    if samples.ndim == 1:
        # Single sample
        mse = mean_squared_error([true_params], [samples])
        return {"MSE": mse, "MAE": np.mean(np.abs(samples - true_params))}

    if samples.ndim == 2:
        # Multiple samples
        mean_estimate = np.mean(samples, axis=0)
        mse = mean_squared_error([true_params], [mean_estimate])
        mae = np.mean(np.abs(mean_estimate - true_params))

        # Coverage (what fraction of true params are within credible intervals)
        coverage = []
        for i in range(len(true_params)):
            param_samples = samples[:, i]
            lower, upper = np.percentile(param_samples, [2.5, 97.5])
            coverage.append(lower <= true_params[i] <= upper)

        return {
            "MSE": mse,
            "MAE": mae,
            "Coverage_95": np.mean(coverage),
            "Std_Error": np.std(np.mean(samples, axis=1)),
        }

    return {"MSE": float("inf"), "MAE": float("inf")}


# ================================
# MAIN EXPERIMENT
# ================================


def main():
    print("=" * 60)
    print("COMPREHENSIVE INFERENCE METHODS COMPARISON")
    print("Two Moon Simulator Experiment")
    print("=" * 60)

    # ================================
    # 1. SETUP AND DATA GENERATION
    # ================================

    print("\n1. Setting up simulator and generating data...")

    true_params = np.array([1.0, 0.2, 1.5, 0.1, 10.0, 5.0])

    # Create a simulator instance with desired settings
    simulator = SafeLotkaVolterraSimulator(
        t_span=(0, 30), n_points=50, noise_level=0.05
    )

    n_observations = 10
    observed_data = simulator.simulate(
        parameters=true_params, num_simulations=n_observations
    )

    print(f"True parameters: {true_params}")
    print(
        f"Generated {observed_data.shape[0]} data points from {n_observations} simulations"
    )

    # Plot observed data
    plot_lotka_volterra_data(
        observed_data, true_params, "Observed Lotka-Volterra Data", "observed_data.png"
    )

    # Plot phase space
    plot_phase_space(
        observed_data,
        true_params,
        "Observed Lotka-Volterra Phase Space",
        "phase_space.png",
    )

    # ================================
    # 2. PRIOR SETUP AND VISUALIZATION
    # ================================

    print("\n2. Setting up prior distribution...")

    prior = UniformPrior()

    # Plot prior samples
    plot_prior_samples(prior, filename="prior_samples.png")

    # ================================
    # 3. INFERENCE METHODS COMPARISON
    # ================================

    print("\n3. Running inference methods...")

    results = {}
    timing_results = {}

    # Prepare summary statistic version of observed data
    # observed_summary = summary_statistic(observed_data)
    # print(f"Observed summary statistics: {observed_summary}")

    # --------------------------------
    # 3.1 ABC Rejection Sampling
    # --------------------------------

    print("\n3.1 Running ABC Rejection Sampling...")

    start_time = time_module.time()

    abc = ABCRejectionSampling(
        simulator=simulator,
        prior=prior,
        distance_function=euclidean_distance,
        tolerance=350.0,  # Very lenient tolerance
        max_attempts=10000,
        verbose=True,
    )

    try:
        abc_posterior = abc.infer(observed_data, num_simulations=150)
        abc_samples = abc_posterior.get_samples()
        abc_duration = time_module.time() - start_time

        results["ABC"] = {
            "samples": abc_samples,
            "posterior": abc_posterior,
            "success": True,
        }
        timing_results["ABC"] = abc_duration

        print(f"ABC completed in {abc_duration:.2f}s with {len(abc_samples)} samples")

    except Exception as e:
        print(f"ABC failed: {e}")
        results["ABC"] = {"success": False, "error": str(e)}
        timing_results["ABC"] = float("inf")

    # --------------------------------
    # 3.2 ABC-MCMC
    # --------------------------------

    print("\n3.2 Running ABC-MCMC...")

    start_time = time_module.time()

    proposal = GaussianProposal(std=0.05)

    abc_mcmc = ABCMCMC(
        simulator=simulator,
        prior=prior,
        distance_function=euclidean_distance,
        tolerance=350,  # Very lenient tolerance
        proposal_distribution=proposal,
        verbose=True,
        burn_in=50,
        thin=2,
        max_attempts_per_step=150,
    )

    try:
        abc_mcmc_posterior = abc_mcmc.infer(observed_data, num_iterations=250)
        abc_mcmc_samples = abc_mcmc_posterior.get_samples()
        abc_mcmc_duration = time_module.time() - start_time

        results["ABC-MCMC"] = {
            "samples": abc_mcmc_samples,
            "posterior": abc_mcmc_posterior,
            "success": True,
        }
        timing_results["ABC-MCMC"] = abc_mcmc_duration

        print(
            f"ABC-MCMC completed in {abc_mcmc_duration:.2f}s with {len(abc_mcmc_samples)} samples"
        )

    except Exception as e:
        print(f"ABC-MCMC failed: {e}")
        results["ABC-MCMC"] = {"success": False, "error": str(e)}
        timing_results["ABC-MCMC"] = float("inf")

    # --------------------------------
    # 3.3 ABC-SMC
    # --------------------------------

    print("\n3.3 Running ABC-SMC...")

    start_time = time_module.time()

    perturbation_kernel = GaussianPerturbationKernel(
        std=0.1
    )  # Reduced std for stability
    tolerance_schedule = [500.0, 400.0, 350.0]  # More gradual schedule

    abc_smc = ABCSMC(
        simulator=simulator,
        prior=prior,
        distance_function=euclidean_distance,
        tolerance_schedule=tolerance_schedule,
        perturbation_kernel=perturbation_kernel,
        num_particles=100,  # Reduced for faster execution
        verbose=True,
    )

    try:
        abc_smc_posterior = abc_smc.infer(observed_data)
        # ABC-SMC returns particles directly, not through get_samples()
        abc_smc_samples = abc_smc_posterior.particles  # Use particles attribute instead
        abc_smc_duration = time_module.time() - start_time

        results["ABC-SMC"] = {
            "samples": abc_smc_samples,
            "posterior": abc_smc_posterior,
            "success": True,
        }
        timing_results["ABC-SMC"] = abc_smc_duration

        print(
            f"ABC-SMC completed in {abc_smc_duration:.2f}s with {len(abc_smc_samples)} samples"
        )

    except Exception as e:
        print(f"ABC-SMC failed: {e}")
        import traceback

        traceback.print_exc()
        results["ABC-SMC"] = {"success": False, "error": str(e)}
        timing_results["ABC-SMC"] = float("inf")

    # --------------------------------
    # 3.4 JANA
    # --------------------------------
    class SimpleJANANet(nn.Module):
        def __init__(self, x_dim, theta_dim, hidden_dim=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(x_dim + theta_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, theta_dim),
            )

        def forward(self, x, z):
            # x: observed data, z: noise from prior
            # Flatten x if it has more than 2 dimensions
            if x.dim() > 2:
                batch_size = x.shape[0]
                x = x.view(batch_size, -1)  # Flatten all dimensions except batch

            # Ensure z is 2D
            if z.dim() == 1:
                z = z.unsqueeze(0)

            combined = torch.cat([x, z], dim=-1)
            return self.net(combined)

    print("\n3.4 Running JANA...")

    start_time = time_module.time()

    # Create JANA network with simpler architecture
    data_dim = 100  # summary statistics dimension
    param_dim = 6  # parameter dimension
    jana_net = SimpleJANANet(data_dim, param_dim, hidden_dim=32)  # Smaller network

    jana = JANA(
        simulator=simulator,
        prior=prior,  # Use the custom prior
        model=jana_net,
        device="cpu",  # Use CPU for compatibility
    )

    try:
        # Make sure observed_data is properly formatted
        if isinstance(observed_data, list):
            observed_data = np.array(observed_data)
        if observed_data.ndim == 1:
            observed_data = observed_data.reshape(1, -1)

        jana_posterior = jana.infer(
            observed_data,  # Pass observed data to JANA
            num_simulations=500,  # Reduced for faster execution
            num_epochs=15,  # Reduced epochs
            batch_size=32,  # Smaller batch size
            learning_rate=1e-3,
            verbose=True,
        )

        # Sample from JANA posterior - use same number of samples as observed data
        jana_samples = jana_posterior.sample(
            observed_data, num_samples=len(observed_data)
        )
        jana_time = time_module.time() - start_time

        results["JANA"] = {
            "samples": jana_samples,
            "posterior": jana_posterior,
            "success": True,
        }
        timing_results["JANA"] = jana_time

        print(f"JANA completed in {jana_time:.2f}s with {len(jana_samples)} samples")

    except Exception as e:
        print(f"JANA failed: {e}")
        import traceback

        traceback.print_exc()
        results["JANA"] = {"success": False, "error": str(e)}
        timing_results["JANA"] = float("inf")

    # --------------------------------
    # 3.5 Simformer
    # --------------------------------

    print("\n3.5 Running Simformer...")

    start_time = time_module.time()

    # Create a summary statistic function for Simformer
    class LotkaVolterraSummaryStatistic:
        def compute(self, data):
            """Compute summary statistics for Lotka-Volterra data."""
            if data.ndim == 1:
                # Single time series
                return np.array(
                    [
                        np.mean(data),
                        np.std(data),
                        np.max(data) - np.min(data),
                        np.mean(np.diff(data)) if len(data) > 1 else 0.0,
                    ]
                )
            elif data.ndim == 2:
                # Multiple time series or prey-predator pairs
                if data.shape[1] == 2:
                    # Prey-predator pairs
                    prey = data[:, 0]
                    predator = data[:, 1]
                    return np.array(
                        [
                            np.mean(prey),
                            np.std(prey),
                            np.mean(predator),
                            np.std(predator),
                            np.max(prey) - np.min(prey),
                            np.max(predator) - np.min(predator),
                            np.mean(np.diff(prey)) if len(prey) > 1 else 0.0,
                            np.mean(np.diff(predator)) if len(predator) > 1 else 0.0,
                        ]
                    )
                else:
                    # Flatten and compute basic statistics
                    flat_data = data.flatten()
                    return np.array(
                        [
                            np.mean(flat_data),
                            np.std(flat_data),
                            np.max(flat_data) - np.min(flat_data),
                            np.mean(np.diff(flat_data)) if len(flat_data) > 1 else 0.0,
                        ]
                    )
            else:
                # Higher dimensional data - flatten
                flat_data = data.flatten()
                return np.array(
                    [
                        np.mean(flat_data),
                        np.std(flat_data),
                        np.max(flat_data) - np.min(flat_data),
                        np.mean(np.diff(flat_data)) if len(flat_data) > 1 else 0.0,
                    ]
                )

    summary_stat = LotkaVolterraSummaryStatistic()

    try:
        # Determine data dimension based on summary statistics
        sample_summary = summary_stat.compute(observed_data)
        data_dim = len(sample_summary)
        param_dim = 6  # Lotka-Volterra parameters

        print(f"Data dimension: {data_dim}, Parameter dimension: {param_dim}")

        # Create Simformer instance
        simformer = Simformer(
            simulator=simulator,
            prior=prior,
            data_dim=data_dim,
            param_dim=param_dim,
            d_model=128,  # Transformer model dimension
            nhead=4,  # Number of attention heads
            num_layers=3,  # Number of transformer layers
            dim_feedforward=256,
            summary_statistic=summary_stat,
            device="cpu",  # Use CPU for compatibility
        )

        print("Starting Simformer inference...")

        # Run inference
        simformer_posterior = simformer.infer(
            observed_data=observed_data,
            num_simulations=200,  # Number of training simulations
            sequence_length=20,  # Length of training sequences
            batch_size=16,  # Batch size for training
            num_epochs=30,  # Number of training epochs
            learning_rate=1e-4,  # Learning rate
            verbose=True,
        )

        # Sample from posterior
        simformer_samples = simformer_posterior.sample(observed_data, num_samples=200)
        simformer_duration = time_module.time() - start_time

        results["Simformer"] = {
            "samples": simformer_samples,
            "posterior": simformer_posterior,
            "success": True,
        }
        timing_results["Simformer"] = simformer_duration

        print(
            f"Simformer completed in {simformer_duration:.2f}s with {len(simformer_samples)} samples"
        )

    except Exception as e:
        print(f"Simformer failed: {e}")
        import traceback

        traceback.print_exc()
        results["Simformer"] = {"success": False, "error": str(e)}
        timing_results["Simformer"] = float("inf")

    # ================================
    # 4. PLOTTING AND ANALYSIS
    # ================================

    print("\n4. Generating plots and analysis...")

    # Plot posterior comparisons
    plot_posterior_comparison(results, true_params, prior, "posterior_comparison.png")

    # Plot timing comparison
    plot_timing_comparison(timing_results, "timing_comparison.png")

    # Plot parameter recovery
    plot_parameter_recovery(results, true_params, "parameter_recovery.png")

    # Save results summary
    summary_df = save_results_summary(
        results, timing_results, true_params, "lotka_volterra_results.csv"
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    # Calculate and display success rates
    successful_methods = [
        method for method, result in results.items() if result.get("success", False)
    ]

    print(f"\nSuccessful methods: {len(successful_methods)}/5")
    for method in successful_methods:
        samples = results[method]["samples"]
        if isinstance(samples, list):
            samples = np.array(samples)
        print(f"  - {method}: {len(samples)} samples in {timing_results[method]:.2f}s")

    # Display accuracy metrics for successful methods
    print("\nAccuracy Metrics:")
    for method in successful_methods:
        samples = results[method]["samples"]
        if isinstance(samples, list):
            samples = np.array(samples)
        metrics = calculate_accuracy_metrics(samples, true_params)
        print(f"  {method}:")
        print(f"    MSE: {metrics.get('MSE', 'N/A'):.4f}")
        print(f"    MAE: {metrics.get('MAE', 'N/A'):.4f}")
        print(f"    95% Coverage: {metrics.get('Coverage_95', 'N/A'):.2f}")

    print(f"\nAll plots and results saved to: {results_dir}")
    print("=" * 60)


# ================================
# PLOTTING FUNCTIONS
# ================================


def plot_lotka_volterra_data(
    data, params, title="Lotka-Volterra Simulation", filename=None
):
    """Plot the Lotka-Volterra simulation data."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Assume data is time series with prey and predator populations
    # Data shape should be (n_observations, n_timepoints, 2) or similar
    if data.ndim == 3:
        # Multiple simulations
        time_points = np.linspace(0, 30, data.shape[1])
        for i in range(min(5, data.shape[0])):  # Plot first 5 simulations
            ax.plot(
                time_points,
                data[i, :, 0],
                "b-",
                alpha=0.7,
                label="Prey" if i == 0 else "",
            )
            ax.plot(
                time_points,
                data[i, :, 1],
                "r-",
                alpha=0.7,
                label="Predator" if i == 0 else "",
            )
    elif data.ndim == 2:
        # Single simulation
        time_points = np.linspace(0, 30, data.shape[0])
        ax.plot(time_points, data[:, 0], "b-", linewidth=2, label="Prey")
        ax.plot(time_points, data[:, 1], "r-", linewidth=2, label="Predator")

    ax.set_xlabel("Time")
    ax.set_ylabel("Population")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if filename:
        plt.savefig(results_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return fig


def plot_phase_space(data, params, title="Lotka-Volterra Phase Space", filename=None):
    """Plot the phase space (prey vs predator) for Lotka-Volterra."""
    fig, ax = plt.subplots(figsize=(8, 8))

    if data.ndim == 3:
        # Multiple simulations
        for i in range(min(5, data.shape[0])):
            ax.plot(data[i, :, 0], data[i, :, 1], "-", alpha=0.7)
            ax.plot(data[i, 0, 0], data[i, 0, 1], "go", markersize=8)  # Start point
            ax.plot(data[i, -1, 0], data[i, -1, 1], "ro", markersize=8)  # End point
    elif data.ndim == 2:
        ax.plot(data[:, 0], data[:, 1], "b-", linewidth=2)
        ax.plot(data[0, 0], data[0, 1], "go", markersize=10, label="Start")
        ax.plot(data[-1, 0], data[-1, 1], "ro", markersize=10, label="End")
        ax.legend()

    ax.set_xlabel("Prey Population")
    ax.set_ylabel("Predator Population")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if filename:
        plt.savefig(results_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return fig


def plot_prior_samples(prior, num_samples=1000, filename=None):
    """Plot samples from the prior distribution."""
    param_names = [
        "α (prey growth)",
        "β (predation rate)",
        "γ (predator efficiency)",
        "δ (predator death)",
        "x0 (initial prey)",
        "y0 (initial predator)",
    ]

    samples = prior.sample(num_samples)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(6):
        axes[i].hist(
            samples[:, i],
            bins=50,
            alpha=0.7,
            density=True,
            color="skyblue",
            edgecolor="black",
        )
        axes[i].set_title(f"{param_names[i]}")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Density")
        axes[i].grid(True, alpha=0.3)

    plt.suptitle("Prior Distribution Samples", fontsize=16)
    plt.tight_layout()

    if filename:
        plt.savefig(results_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return fig


def plot_posterior_comparison(results, true_params, prior, filename=None):
    """Plot posterior samples comparison with prior and true parameters."""
    param_names = [
        "α (prey growth)",
        "β (predation rate)",
        "γ (predator efficiency)",
        "δ (predator death)",
        "x0 (initial prey)",
        "y0 (initial predator)",
    ]

    # Get successful methods
    successful_methods = [
        method for method, result in results.items() if result.get("success", False)
    ]

    if not successful_methods:
        print("No successful inference methods to plot.")
        return None

    n_methods = len(successful_methods)
    fig, axes = plt.subplots(6, n_methods + 1, figsize=(4 * (n_methods + 1), 20))

    if n_methods == 1:
        axes = axes.reshape(6, -1)

    # Plot prior in first column
    prior_samples = prior.sample(1000)
    for i in range(6):
        axes[i, 0].hist(
            prior_samples[:, i],
            bins=50,
            alpha=0.7,
            density=True,
            color="lightgray",
            edgecolor="black",
            label="Prior",
        )
        axes[i, 0].axvline(
            true_params[i], color="red", linestyle="--", linewidth=2, label="True"
        )
        axes[i, 0].set_title(f"Prior\n{param_names[i]}")
        axes[i, 0].set_xlabel("Value")
        axes[i, 0].set_ylabel("Density")
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)

    # Plot posteriors
    for j, method in enumerate(successful_methods):
        samples = results[method]["samples"]
        if isinstance(samples, list):
            samples = np.array(samples)

        if samples.ndim == 1:
            samples = samples.reshape(1, -1)

        for i in range(6):
            if samples.shape[1] > i:
                # Plot prior as background
                axes[i, j + 1].hist(
                    prior_samples[:, i],
                    bins=50,
                    alpha=0.3,
                    density=True,
                    color="lightgray",
                    edgecolor="none",
                    label="Prior",
                )

                # Plot posterior
                axes[i, j + 1].hist(
                    samples[:, i],
                    bins=30,
                    alpha=0.7,
                    density=True,
                    color="skyblue",
                    edgecolor="black",
                    label=f"{method}",
                )
                axes[i, j + 1].axvline(
                    true_params[i],
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label="True",
                )

                # Add mean and credible intervals
                mean_val = np.mean(samples[:, i])
                ci_lower, ci_upper = np.percentile(samples[:, i], [2.5, 97.5])
                axes[i, j + 1].axvline(
                    mean_val, color="blue", linestyle="-", linewidth=2, label="Mean"
                )
                axes[i, j + 1].axvspan(
                    ci_lower, ci_upper, alpha=0.2, color="blue", label="95% CI"
                )

            axes[i, j + 1].set_title(f"{method}\n{param_names[i]}")
            axes[i, j + 1].set_xlabel("Value")
            axes[i, j + 1].set_ylabel("Density")
            axes[i, j + 1].legend()
            axes[i, j + 1].grid(True, alpha=0.3)

    plt.suptitle("Posterior Comparison with Prior and True Parameters", fontsize=16)
    plt.tight_layout()

    if filename:
        plt.savefig(results_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return fig


def plot_timing_comparison(timing_results, filename=None):
    """Plot timing comparison of different inference methods."""
    # Filter out failed methods (infinite time)
    valid_methods = {k: v for k, v in timing_results.items() if v != float("inf")}

    if not valid_methods:
        print("No valid timing results to plot.")
        return None

    methods = list(valid_methods.keys())
    times = list(valid_methods.values())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Bar plot
    bars = ax1.bar(
        methods,
        times,
        color=["skyblue", "lightcoral", "lightgreen", "gold"][: len(methods)],
    )
    ax1.set_ylabel("Time (seconds)")
    ax1.set_title("Inference Method Timing Comparison")
    ax1.set_xlabel("Method")

    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{time:.2f}s",
            ha="center",
            va="bottom",
        )

    ax1.grid(True, alpha=0.3)

    # Log scale plot for better visualization if times vary significantly
    ax2.bar(
        methods,
        times,
        color=["skyblue", "lightcoral", "lightgreen", "gold"][: len(methods)],
    )
    ax2.set_ylabel("Time (seconds, log scale)")
    ax2.set_title("Inference Method Timing (Log Scale)")
    ax2.set_xlabel("Method")
    ax2.set_yscale("log")

    # Add value labels on bars
    for i, (method, time) in enumerate(valid_methods.items()):
        ax2.text(i, time, f"{time:.2f}s", ha="center", va="bottom")

    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if filename:
        plt.savefig(results_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return fig


def plot_parameter_recovery(results, true_params, filename=None):
    """Plot parameter recovery performance for each method."""
    param_names = ["α", "β", "γ", "δ", "x0", "y0"]
    successful_methods = [
        method for method, result in results.items() if result.get("success", False)
    ]

    if not successful_methods:
        print("No successful methods to plot parameter recovery.")
        return None

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, param_name in enumerate(param_names):
        ax = axes[i]

        # Plot true parameter as horizontal line
        ax.axhline(
            y=true_params[i],
            color="red",
            linestyle="--",
            linewidth=2,
            label="True Value",
        )

        method_means = []
        method_errors = []
        method_names = []

        for j, method in enumerate(successful_methods):
            samples = results[method]["samples"]
            if isinstance(samples, list):
                samples = np.array(samples)

            if samples.ndim == 1:
                samples = samples.reshape(1, -1)

            if samples.shape[1] > i:
                param_samples = samples[:, i]
                mean_val = np.mean(param_samples)
                std_val = np.std(param_samples)

                # Plot mean with error bars
                ax.errorbar(
                    j,
                    mean_val,
                    yerr=std_val,
                    fmt="o",
                    capsize=5,
                    markersize=8,
                    label=f"{method}",
                )
                ax.scatter(j, mean_val, s=100, alpha=0.7)  # Larger dot for mean

                method_means.append(mean_val)
                method_errors.append(std_val)
                method_names.append(method)

        ax.set_xlabel("Method")
        ax.set_ylabel(f"{param_name} Value")
        ax.set_title(f"Parameter Recovery: {param_name}")
        ax.set_xticks(range(len(method_names)))
        ax.set_xticklabels(method_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Parameter Recovery Performance", fontsize=16)
    plt.tight_layout()

    if filename:
        plt.savefig(results_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return fig


def save_results_summary(results, timing_results, true_params, filename=None):
    """Save a summary of all results to a CSV file."""
    summary_data = []

    for method, result in results.items():
        if result.get("success", False):
            samples = result["samples"]
            if isinstance(samples, list):
                samples = np.array(samples)

            if samples.ndim == 1:
                samples = samples.reshape(1, -1)

            # Calculate metrics
            metrics = calculate_accuracy_metrics(samples, true_params)

            summary_data.append(
                {
                    "Method": method,
                    "Success": True,
                    "Num_Samples": len(samples),
                    "Time_seconds": timing_results.get(method, float("inf")),
                    "MSE": metrics.get("MSE", float("inf")),
                    "MAE": metrics.get("MAE", float("inf")),
                    "Coverage_95": metrics.get("Coverage_95", 0),
                    "Std_Error": metrics.get("Std_Error", float("inf")),
                }
            )
        else:
            summary_data.append(
                {
                    "Method": method,
                    "Success": False,
                    "Num_Samples": 0,
                    "Time_seconds": timing_results.get(method, float("inf")),
                    "MSE": float("inf"),
                    "MAE": float("inf"),
                    "Coverage_95": 0,
                    "Std_Error": float("inf"),
                }
            )

    df = pd.DataFrame(summary_data)

    if filename:
        df.to_csv(results_dir / filename, index=False)
        print(f"Results summary saved to {results_dir / filename}")

    return df


class SafeLotkaVolterraSimulator:
    """Wrapper around LotkaVolterraSimulator that ensures positive noise levels."""

    def __init__(self, t_span=(0, 30), n_points=50, noise_level=0.05):
        self.t_span = t_span
        self.n_points = n_points
        self.noise_level = max(noise_level, 1e-6)  # Ensure positive noise

    def simulate(self, parameters, num_simulations=1):
        """Simulate with safe noise handling."""
        import numpy as np

        # Create a new simulator instance for each call to avoid state issues
        base_simulator = LotkaVolterraSimulator(
            t_span=self.t_span,
            n_points=self.n_points,
            noise_level=abs(self.noise_level),  # Ensure positive noise
        )

        # Ensure parameters are positive
        if isinstance(parameters, (list, tuple)):
            parameters = np.array(parameters)
        parameters = np.maximum(parameters, 1e-6)  # Ensure all params are positive

        try:
            result = base_simulator.simulate(parameters, num_simulations)
            return result
        except Exception as e:
            # If simulation fails, return a small positive result
            print(f"Simulation failed with parameters {parameters}, error: {e}")
            # Return a minimal valid result
            return np.random.uniform(
                0.1, 1.0, size=(num_simulations, self.n_points * 2)
            ).reshape(num_simulations, -1)


if __name__ == "__main__":
    main()
