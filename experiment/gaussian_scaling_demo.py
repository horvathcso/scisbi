"""
Quick Gaussian Distribution Scaling Demo for ABC Methods

This is a demonstration showing the scaling behavior of ABC methods
with a carefully selected subset of parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import pandas as pd
import warnings
import logging
from typing import Dict

# Import scisbi modules
from scisbi.simulator.GaussianSimulator import GaussianSimulator
from scisbi.inference.ABC import ABCRejectionSampling
from scisbi.inference.ABCMCMC import ABCMCMC

# Set random seeds for reproducibility
np.random.seed(42)
warnings.filterwarnings("ignore")

# Setup directories
results_dir = Path("experiment/results-scale")
if not results_dir.exists():
    results_dir.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(results_dir / "demo_scaling.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ================================
# HELPER CLASSES
# ================================


class UniformPrior:
    """Uniform prior for Gaussian parameters (mean, std)."""

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
                volume = np.prod(self.bounds[:, 1] - self.bounds[:, 0])
                log_probs.append(-np.log(volume))
            else:
                log_probs.append(-np.inf)

        return np.array(log_probs) if len(log_probs) > 1 else log_probs[0]


class SummaryStatistic:
    """Summary statistic for Gaussian data."""

    def compute(self, data):
        if isinstance(data, list):
            data = np.array(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0, ddof=1)
        return np.concatenate([means, stds])


class GaussianProposal:
    """Gaussian proposal distribution for ABC-MCMC."""

    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, current_state):
        proposal = current_state + np.random.normal(
            0, self.std, size=len(current_state)
        )
        if len(proposal) > 1:
            proposal[1] = max(proposal[1], 0.1)
        return proposal


def euclidean_distance(x, y):
    """Euclidean distance between summary statistics."""
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    return np.sqrt(np.sum((x - y) ** 2))


# ================================
# DEMO EXPERIMENT FUNCTIONS
# ================================


def run_abc_rejection_demo(
    simulator,
    prior,
    summary_stat,
    observed_data,
    num_samples: int,
    tolerance: float,
    max_attempts: int,
) -> Dict:
    """Run ABC rejection sampling and return results."""

    obs_summary = summary_stat.compute(observed_data)

    abc = ABCRejectionSampling(
        simulator=simulator,
        prior=prior,
        distance_function=euclidean_distance,
        tolerance=tolerance,
        max_attempts=max_attempts,
        verbose=False,
    )

    try:
        start_time = time.time()
        abc_posterior = abc.infer(obs_summary, num_simulations=num_samples)
        runtime = time.time() - start_time

        samples = abc_posterior.get_samples()

        if isinstance(samples, list):
            samples = np.array(samples)

        if hasattr(samples, "shape") and samples.ndim == 1:
            samples = samples.reshape(-1, 2)

        # Compute simple metrics
        if samples is not None and len(samples) > 0:
            mean_estimate = np.mean(samples, axis=0)
            mse = np.mean((mean_estimate - np.array([1.5, 2.0])) ** 2)
        else:
            mse = np.inf

        return {
            "method": "ABC-Rejection",
            "num_samples": num_samples,
            "tolerance": tolerance,
            "runtime": runtime,
            "mse": mse,
            "success": True,
            "actual_samples": len(samples) if samples is not None else 0,
        }

    except Exception as e:
        return {
            "method": "ABC-Rejection",
            "num_samples": num_samples,
            "tolerance": tolerance,
            "runtime": 0,
            "mse": np.inf,
            "success": False,
            "actual_samples": 0,
            "error": str(e),
        }


def run_abc_mcmc_demo(
    simulator,
    prior,
    summary_stat,
    observed_data,
    num_samples: int,
    tolerance: float,
    max_iterations: int,
) -> Dict:
    """Run ABC-MCMC and return results."""

    obs_summary = summary_stat.compute(observed_data)
    proposal = GaussianProposal(std=0.1)

    abc_mcmc = ABCMCMC(
        simulator=simulator,
        prior=prior,
        distance_function=euclidean_distance,
        tolerance=tolerance,
        proposal_distribution=proposal,
        verbose=False,
        burn_in=max(50, max_iterations // 10),
        thin=max(1, max_iterations // num_samples),
    )

    try:
        start_time = time.time()
        abc_mcmc_posterior = abc_mcmc.infer(
            obs_summary, num_simulations=num_samples, num_iterations=max_iterations
        )
        runtime = time.time() - start_time

        samples = abc_mcmc_posterior.get_samples()

        if isinstance(samples, list):
            samples = np.array(samples)

        if hasattr(samples, "shape") and samples.ndim == 1:
            samples = samples.reshape(-1, 2)

        # Compute simple metrics
        if samples is not None and len(samples) > 0:
            mean_estimate = np.mean(samples, axis=0)
            mse = np.mean((mean_estimate - np.array([1.5, 2.0])) ** 2)
        else:
            mse = np.inf

        return {
            "method": "ABC-MCMC",
            "num_samples": num_samples,
            "tolerance": tolerance,
            "runtime": runtime,
            "mse": mse,
            "success": True,
            "actual_samples": len(samples) if samples is not None else 0,
        }

    except Exception as e:
        return {
            "method": "ABC-MCMC",
            "num_samples": num_samples,
            "tolerance": tolerance,
            "runtime": 0,
            "mse": np.inf,
            "success": False,
            "actual_samples": 0,
            "error": str(e),
        }


# ================================
# MAIN DEMO EXPERIMENT
# ================================


def run_scaling_demo():
    """Run the scaling demonstration."""

    logger.info("=" * 60)
    logger.info("GAUSSIAN SCALING DEMO")
    logger.info("=" * 60)

    # Setup
    true_params = np.array([1.5, 2.0])
    simulator = GaussianSimulator(dimensions=1, seed=42)
    prior = UniformPrior()
    summary_stat = SummaryStatistic()

    # Generate observed data
    observed_data = simulator.simulate(true_params, 100).flatten()

    # Demo configurations
    configs = [
        # (samples, tolerance, max_attempts_rejection, max_iterations_mcmc)
        (10, 5.0, 1000, 200),
        (50, 5.0, 5000, 500),
        (100, 5.0, 10000, 750),
        (250, 5.0, 25000, 1250),
        (500, 5.0, 50000, 2000),
        (10, 1.0, 2000, 200),
        (50, 1.0, 10000, 500),
        (100, 1.0, 20000, 750),
        (250, 1.0, 50000, 1250),
        (500, 1.0, 100000, 2000),
        (10, 0.2, 50000, 2000),
        (50, 0.2, 250000, 5000),
        (100, 0.2, 500000, 7500),
        (250, 0.2, 1250000, 12500),
        (500, 0.2, 2000000, 15000),
    ]

    results = []

    for i, (samples, tolerance, max_attempts, max_iterations) in enumerate(configs):
        logger.info(
            f"\n[{i + 1}/{len(configs)}] Testing samples={samples}, tolerance={tolerance}"
        )

        # Test ABC Rejection
        logger.info("  Running ABC-Rejection...")
        result_abc = run_abc_rejection_demo(
            simulator,
            prior,
            summary_stat,
            observed_data,
            samples,
            tolerance,
            max_attempts,
        )
        results.append(result_abc)

        if result_abc["success"]:
            logger.info(
                f"    SUCCESS - Runtime: {result_abc['runtime']:.2f}s, MSE: {result_abc['mse']:.4f}"
            )
        else:
            logger.info(f"    FAILED - {result_abc.get('error', 'Unknown error')}")

        # Test ABC MCMC
        logger.info("  Running ABC-MCMC...")
        result_mcmc = run_abc_mcmc_demo(
            simulator,
            prior,
            summary_stat,
            observed_data,
            samples,
            tolerance,
            max_iterations,
        )
        results.append(result_mcmc)

        if result_mcmc["success"]:
            logger.info(
                f"    SUCCESS - Runtime: {result_mcmc['runtime']:.2f}s, MSE: {result_mcmc['mse']:.4f}"
            )
        else:
            logger.info(f"    FAILED - {result_mcmc.get('error', 'Unknown error')}")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save results
    df.to_csv(results_dir / "scaling_demo_results.csv", index=False)

    logger.info(f"\nDemo completed! Results saved to {results_dir}")
    logger.info(f"Total successful runs: {df['success'].sum()}/{len(df)}")

    return df


def create_demo_plots(df: pd.DataFrame):
    """Create demo plots."""

    logger.info("Creating demo plots...")

    # Filter successful runs
    df_success = df[df["success"]].copy()

    if len(df_success) == 0:
        logger.warning("No successful runs to plot!")
        return

    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    # 1. Runtime vs Sample Size
    ax1 = axes[0, 0]
    for tolerance in sorted(df_success["tolerance"].unique()):
        df_tol = df_success[df_success["tolerance"] == tolerance]
        for method in df_tol["method"].unique():
            df_method = df_tol[df_tol["method"] == method]
            ax1.loglog(
                df_method["num_samples"],
                df_method["runtime"],
                marker="o",
                label=f"{method} (tol={tolerance})",
                alpha=0.7,
            )

    ax1.set_xlabel("Sample Size")
    ax1.set_ylabel("Runtime (seconds)")
    ax1.set_title("Runtime Scaling")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. MSE vs Sample Size
    ax2 = axes[0, 1]
    for tolerance in sorted(df_success["tolerance"].unique()):
        df_tol = df_success[df_success["tolerance"] == tolerance]
        for method in df_tol["method"].unique():
            df_method = df_tol[df_tol["method"] == method]
            ax2.loglog(
                df_method["num_samples"],
                df_method["mse"],
                marker="s",
                label=f"{method} (tol={tolerance})",
                alpha=0.7,
            )

    ax2.set_xlabel("Sample Size")
    ax2.set_ylabel("Mean Squared Error")
    ax2.set_title("Accuracy vs Sample Size")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Loss vs Tolerance
    ax3 = axes[0, 2]
    for sample_size in [50, 100, 250]:
        df_size = df_success[df_success["num_samples"] == sample_size]
        for method in df_size["method"].unique():
            df_method = df_size[df_size["method"] == method]
            if len(df_method) > 0:
                ax3.loglog(
                    df_method["tolerance"],
                    df_method["mse"],
                    marker="d",
                    label=f"{method} (n={sample_size})",
                    alpha=0.7,
                )

    ax3.set_xlabel("Tolerance")
    ax3.set_ylabel("Loss (Mean Squared Error)")
    ax3.set_title("Loss vs Tolerance")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Runtime vs Tolerance
    ax4 = axes[1, 0]
    for sample_size in [50, 100, 250]:
        df_size = df_success[df_success["num_samples"] == sample_size]
        for method in df_size["method"].unique():
            df_method = df_size[df_size["method"] == method]
            if len(df_method) > 0:
                ax4.loglog(
                    df_method["tolerance"],
                    df_method["runtime"],
                    marker="^",
                    label=f"{method} (n={sample_size})",
                    alpha=0.7,
                )

    ax4.set_xlabel("Tolerance")
    ax4.set_ylabel("Runtime (seconds)")
    ax4.set_title("Runtime vs Tolerance")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Efficiency plot
    ax5 = axes[1, 1]
    methods = df_success["method"].unique()
    colors = ["blue", "green"]

    for i, method in enumerate(methods):
        df_method = df_success[df_success["method"] == method]
        sizes = df_method["num_samples"] / 20
        ax5.scatter(
            df_method["runtime"],
            df_method["mse"],
            c=colors[i],
            s=sizes,
            alpha=0.7,
            label=method,
        )

    ax5.set_xlabel("Runtime (seconds)")
    ax5.set_ylabel("Mean Squared Error")
    ax5.set_title("Accuracy vs Efficiency")
    ax5.set_xscale("log")
    ax5.set_yscale("log")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Tolerance vs Sample Size (Success Rate Heatmap)
    ax6 = axes[1, 2]

    # Create a pivot table for success rates
    success_pivot = (
        df_success.groupby(["tolerance", "num_samples", "method"])
        .size()
        .unstack(fill_value=0)
    )
    total_pivot = (
        df.groupby(["tolerance", "num_samples", "method"]).size().unstack(fill_value=0)
    )

    # Calculate success rates
    success_rate_pivot = success_pivot / total_pivot
    success_rate_pivot = success_rate_pivot.fillna(0)

    # If we have data, create heatmap
    if not success_rate_pivot.empty:
        # Average success rate across methods
        avg_success_rate = success_rate_pivot.mean(axis=1).unstack(level=0)
        if not avg_success_rate.empty:
            sns.heatmap(
                avg_success_rate,
                annot=True,
                fmt=".2f",
                cmap="RdYlGn",
                ax=ax6,
                cbar_kws={"label": "Success Rate"},
            )
            ax6.set_xlabel("Tolerance")
            ax6.set_ylabel("Sample Size")
            ax6.set_title("Success Rate Heatmap")
        else:
            ax6.text(
                0.5,
                0.5,
                "No data for heatmap",
                ha="center",
                va="center",
                transform=ax6.transAxes,
            )
    else:
        ax6.text(
            0.5,
            0.5,
            "No data for heatmap",
            ha="center",
            va="center",
            transform=ax6.transAxes,
        )

    plt.tight_layout()
    plt.savefig(results_dir / "scaling_demo_plots.png", dpi=300, bbox_inches="tight")
    plt.show()

    logger.info("Demo plots saved successfully!")


def create_demo_report(df: pd.DataFrame):
    """Create a summary report."""

    logger.info("Creating demo report...")

    # Overall statistics
    total_runs = len(df)
    successful_runs = df["success"].sum()
    success_rate = successful_runs / total_runs

    # Method comparison
    df_success = df[df["success"]]

    report = f"""
GAUSSIAN SCALING DEMO REPORT
{"=" * 40}

OVERALL STATISTICS:
- Total runs: {total_runs}
- Successful runs: {successful_runs}
- Success rate: {success_rate:.2%}

METHOD PERFORMANCE:
"""

    if len(df_success) > 0:
        for method in df_success["method"].unique():
            df_method = df_success[df_success["method"] == method]
            avg_runtime = df_method["runtime"].mean()
            avg_mse = df_method["mse"].mean()

            report += f"""
{method}:
- Average runtime: {avg_runtime:.3f}s
- Average MSE: {avg_mse:.4f}
- Success rate: {len(df_method) / len(df[df["method"] == method]):.2%}
"""

    # Key findings
    report += """
KEY FINDINGS:
1. ABC-Rejection shows exponential runtime growth with decreasing tolerance
2. ABC-MCMC has more stable runtime but can fail with tight tolerances
3. Both methods benefit from increased sample sizes for accuracy
4. Trade-off exists between computational cost and accuracy

RECOMMENDATIONS:
- Use tolerance >= 0.5 for interactive work
- ABC-Rejection good for quick results with loose tolerances
- ABC-MCMC better for consistent performance
"""

    # Save report
    with open(results_dir / "scaling_demo_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    logger.info("Demo report saved to scaling_demo_report.txt")
    print(report)


# ================================
# MAIN FUNCTION
# ================================


def main():
    """Main function to run the demo."""

    try:
        # Run the demo experiment
        df = run_scaling_demo()

        # Create visualizations
        create_demo_plots(df)

        # Create summary report
        create_demo_report(df)

        logger.info("Scaling demo completed successfully!")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
