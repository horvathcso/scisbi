import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sbibm
import time
from sbibm.metrics import c2st, mmd
from tqdm import tqdm
import logging
from typing import List, Dict, Any, Optional, Callable

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from scisbi.base.simulator import BaseSimulator
from scisbi.inference import (
    ABCRejectionSampling,
    ABCMCMC,
    ABCSMC,
    JANA,
    Simformer
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Wrappers for sbibm to scisbi interfaces ---

class SBIBMSimulator(BaseSimulator):
    """Wrapper for sbibm simulators."""
    def __init__(self, task):
        super().__init__()
        self.task = task
        self.simulator = task.get_simulator()

    def simulate(self, parameters: np.ndarray) -> np.ndarray:
        """
        Simulate data given parameters.
        Args:
            parameters: (Batch, Param_Dim)
        Returns:
            data: (Batch, Data_Dim)
        """
        # Convert to torch tensor
        # parameters might be 1D (P,) or 2D (B, P)
        params_tensor = torch.from_numpy(np.atleast_2d(parameters)).float()

        # sbibm simulator expects (B, P)
        # It runs on CPU usually
        with torch.no_grad():
            data_tensor = self.simulator(params_tensor)

        return data_tensor.numpy()

class SBIBMPrior:
    """Wrapper for sbibm priors."""
    def __init__(self, task):
        self.task = task
        self.prior = task.get_prior()
        self.prior_dist = self.task.get_prior_dist() # Sometimes useful if available directly

    def sample(self, num_samples: int = 1) -> np.ndarray:
        """Sample from prior."""
        with torch.no_grad():
            samples = self.prior(num_samples)

        # If num_samples is 1 and caller expects 1D array?
        # But BasePrior usually returns (N, P) or (P,) if N=1?
        # Let's keep it (N, P) always, but check if ABC expects (P,)
        return samples.numpy()

    def log_prob(self, parameters: np.ndarray) -> np.ndarray:
        """Compute log probability."""
        params_tensor = torch.from_numpy(np.atleast_2d(parameters)).float()
        with torch.no_grad():
            # Use prior_dist for log_prob if available, otherwise assume prior is distribution
            if hasattr(self, 'prior_dist') and self.prior_dist is not None:
                log_probs = self.prior_dist.log_prob(params_tensor)
            else:
                log_probs = self.prior.log_prob(params_tensor)
        return log_probs.numpy()

# --- Helper Functions ---

class JANAModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, z):
        # x: (B, Dx), z: (B, Dz)
        # Concatenate
        inp = torch.cat([x, z], dim=1)
        return self.net(inp)

class GaussianKernel:
    """Simple isotropic Gaussian kernel/proposal."""
    def __init__(self, scale=0.1):
        self.scale = scale

    def __call__(self, current_theta: np.ndarray) -> np.ndarray:
        """Call method to act as a callable proposal function."""
        return self.sample(current_theta)

    def sample(self, current_theta: np.ndarray) -> np.ndarray:
        # current_theta: (P,) or (1, P)
        return current_theta + np.random.normal(0, self.scale, size=current_theta.shape)

    def log_prob(self, current_theta: np.ndarray, proposed_theta: np.ndarray) -> float:
        # Log prob of proposing proposed_theta given current_theta
        # proportional to -||diff||^2 / (2*scale^2)
        diff = proposed_theta - current_theta
        return -0.5 * np.sum(diff**2) / (self.scale**2)

def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Simple Euclidean distance between two data points (or batches)."""
    # x, y shape (1, D) or (D,)
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    # If comparing single observation y to batch x
    # x: (B, D), y: (1, D)
    return np.linalg.norm(x - y, axis=1) # Returns (B,) distances

def plot_posterior(
    samples: np.ndarray,
    ref_samples: Optional[np.ndarray],
    task_name: str,
    method_name: str,
    save_path: str
):
    """Plot posterior pairs plot."""
    # Limit samples for plotting speed
    n_plot = min(1000, len(samples))
    indices = np.random.choice(len(samples), n_plot, replace=False)
    samples_plot = samples[indices]

    df_data = pd.DataFrame(samples_plot)
    df_data['Type'] = 'Inference'

    if ref_samples is not None:
        n_ref = min(1000, len(ref_samples))
        ref_indices = np.random.choice(len(ref_samples), n_ref, replace=False)
        ref_plot = ref_samples[ref_indices]
        df_ref = pd.DataFrame(ref_plot)
        df_ref['Type'] = 'Reference'
        df_combined = pd.concat([df_data, df_ref])
    else:
        df_combined = df_data

    # Plot
    g = sns.pairplot(df_combined, hue='Type', diag_kind='kde', plot_kws={'alpha': 0.5, 's': 10})
    g.fig.suptitle(f"{task_name} - {method_name}", y=1.02)
    plt.savefig(save_path)
    plt.close()

# --- Main Benchmark Runner ---

def run_benchmark(
    tasks: List[str],
    num_observation: int = 1,
    results_dir: str = "experiment/sbibm/results",
    dry_run: bool = False
):
    """
    Run the benchmark suite.
    """
    os.makedirs(results_dir, exist_ok=True)
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Store metrics
    all_results = []

    for task_name in tasks:
        logger.info(f"Starting task: {task_name}")

        try:
            task = sbibm.get_task(task_name)

            # Setup Scisbi components
            simulator = SBIBMSimulator(task)
            prior = SBIBMPrior(task)

            # Get observation and reference posterior
            observation = task.get_observation(num_observation=num_observation) # Tensor (1, D)
            observation_np = observation.numpy()

            ref_samples = task.get_reference_posterior_samples(num_observation=num_observation) # Tensor (N, P)
            ref_samples_np = ref_samples.numpy()

            # Define methods configuration
            # Budgets are reduced for dry_run
            budget_factor = 0.01 if dry_run else 1.0

            # Standard configs
            abc_sims = int(10000 * budget_factor) if dry_run else 1000 # Samples to accept? No, total simulations usually.
            # ABCRejection expects num_simulations as desired samples.
            # If acceptance rate is low, this might run forever.
            # I'll use max_attempts.

            methods = {
                "ABC": {
                    "class": ABCRejectionSampling,
                    "kwargs": {
                        "distance_function": euclidean_distance,
                        "tolerance": 2.0, # Very high tolerance for dry-run to ensure acceptance
                        "max_attempts": 1000 if dry_run else 100000
                    },
                    "infer_kwargs": {
                        "num_simulations": 10 if dry_run else 1000
                    }
                },
                "ABCMCMC": {
                    "class": ABCMCMC,
                    "kwargs": {
                         "distance_function": euclidean_distance,
                         "tolerance": 2.0,
                         "proposal_distribution": GaussianKernel(scale=0.5), # Larger steps
                         "max_attempts_per_step": 100 if dry_run else 100
                    },
                    "infer_kwargs": {
                        "num_iterations": 20 if dry_run else 5000
                    }
                },
                "ABCSMC": {
                    "class": ABCSMC,
                    "kwargs": {
                        "distance_function": euclidean_distance,
                        "population_size": 20 if dry_run else 1000, # Minimum 20 for stability
                        "tolerance_schedule": [5.0, 2.0] if dry_run else [2.0, 1.0, 0.5, 0.2, 0.1, 0.05],
                        "perturbation_kernel": GaussianKernel(scale=0.1)
                    },
                    "infer_kwargs": {
                        "num_iterations": 2 if dry_run else 6
                    }
                },
                "JANA": {
                    "class": JANA,
                    "kwargs": {
                        "model": None
                    },
                    "infer_kwargs": {
                        "num_simulations": 100 if dry_run else 10000,
                        "num_epochs": 1 if dry_run else 50,
                        "batch_size": 16
                    }
                },
                "Simformer": {
                    "class": Simformer,
                    "kwargs": {
                        "data_dim": observation_np.shape[1],
                        "param_dim": ref_samples_np.shape[1]
                    },
                    "infer_kwargs": {
                        "num_simulations": 10 if dry_run else 2000,
                        "sequence_length": 5 if dry_run else 50,
                        "num_epochs": 1 if dry_run else 50
                    }
                }
            }

            for method_name, config in methods.items():
                logger.info(f"  Running {method_name}...")
                start_time = time.time()

                try:
                    # Initialize algorithm
                    kwargs = config["kwargs"].copy()

                    # Special initialization for JANA model (needs dimensions)
                    if method_name == "JANA":
                        input_dim = observation_np.shape[1] # x size
                        z_dim = ref_samples_np.shape[1] # z size (using prior dimension)
                        output_dim = ref_samples_np.shape[1] # theta

                        model = JANAModel(input_dim + z_dim, 64, output_dim)
                        kwargs["model"] = model

                    algo = config["class"](
                        simulator=simulator,
                        prior=prior,
                        **kwargs
                    )

                    # Run inference
                    infer_kwargs = config["infer_kwargs"].copy()
                    posterior = algo.infer(
                        observed_data=observation_np,
                        **infer_kwargs
                    )

                    # Sample posterior
                    num_samples = 1000
                    if method_name in ["JANA", "Simformer"]:
                        # For neural methods, sample method takes observed_data
                        samples = posterior.sample(observation_np, num_samples=num_samples)
                    else:
                        # For ABC, result object already has samples or provides resampling
                        if hasattr(posterior, "sample"):
                            # Check if sample requires arguments?
                            # ABCPosterior.sample(n)
                            # SimformerPosterior.sample(obs, n)
                            # Uniformity issue here in BaseInference?
                            # BaseInferenceAlgorithm return type isn't strictly defined as a common Posterior class with common sample method
                            # ABCPosterior.sample(n)
                            # ABCMCMCPosterior.sample(n)
                            # ABCSMCPosterior.sample(n)
                            # JANAPosterior.sample(obs, n)

                            # Inspect signature or try/except
                            try:
                                samples = posterior.sample(num_samples)
                            except TypeError:
                                samples = posterior.sample(observation_np, num_samples=num_samples)
                        elif hasattr(posterior, "samples"):
                            # ABCPosterior has .samples attribute (list of arrays)
                            # But we want exactly num_samples
                            # It has .sample(n) method too.
                            samples = posterior.sample(num_samples)
                        else:
                            raise RuntimeError(f"Unknown posterior type for {method_name}")

                    # Ensure samples are numpy array (N, P)
                    if isinstance(samples, list):
                        samples = np.array(samples)
                    if samples.ndim == 1:
                         samples = samples.reshape(-1, 1)
                    elif samples.ndim == 3 and samples.shape[1] == 1:
                         # Squeeze middle dimension (N, 1, P) -> (N, P)
                         samples = samples.squeeze(1)

                    # Calculate Metrics
                    runtime = time.time() - start_time

                    # C2ST
                    samples_tensor = torch.from_numpy(samples).float()
                    # Resample reference if sizes mismatch too much for C2ST stability?
                    # sbibm c2st expects (N, P) tensors

                    # Ensure we have enough samples
                    if len(samples) < 100:
                        logger.warning(f"Not enough samples for C2ST ({len(samples)}). Skipping.")
                        c2st_score = np.nan
                        mmd_score = np.nan
                    else:
                        # Crop to min length to match
                        n_eval = min(len(samples), len(ref_samples))
                        n_eval = min(n_eval, 10000) # Cap at 10k

                        s_eval = samples_tensor[:n_eval]
                        r_eval = ref_samples[:n_eval]

                        c2st_score = c2st(r_eval, s_eval).item()
                        mmd_score = mmd(r_eval, s_eval).item()

                    logger.info(f"    {method_name} Results: C2ST={c2st_score:.4f}, MMD={mmd_score:.4f}, Time={runtime:.2f}s")

                    all_results.append({
                        "task": task_name,
                        "method": method_name,
                        "c2st": c2st_score,
                        "mmd": mmd_score,
                        "runtime": runtime,
                        "num_simulations": config["infer_kwargs"].get("num_simulations", np.nan)
                    })

                    # Plot
                    plot_path = os.path.join(figures_dir, f"{task_name}_{method_name}.png")
                    plot_posterior(samples, ref_samples_np, task_name, method_name, plot_path)

                except Exception as e:
                    logger.error(f"    Failed {method_name}: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Failed task {task_name}: {e}", exc_info=True)

    # Save summary
    df_results = pd.DataFrame(all_results)
    csv_path = os.path.join(results_dir, "benchmark_results.csv")
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Benchmark completed. Results saved to {csv_path}")

if __name__ == "__main__":
    # Select tasks
    # Available: 'lotka_volterra', 'gaussian_mixture', 'sir', 'two_moons', 'bernoulli_glm',
    # 'gaussian_linear_uniform', 'slcp', 'gaussian_linear', 'slcp_distractors', 'bernoulli_glm_raw'

    # Run a subset for testing by default, or all if specified
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        tasks_to_run = sbibm.get_available_tasks()
    else:
        # Default test run
        tasks_to_run = ["two_moons", "gaussian_linear"]

    logger.info(f"Running benchmark on tasks: {tasks_to_run}")

    # Check if dry run for quick testing
    is_dry_run = "--dry-run" in sys.argv

    run_benchmark(tasks_to_run, dry_run=is_dry_run)
