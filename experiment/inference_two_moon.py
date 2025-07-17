"""
Combined Two Moon Distribution Experiment

This experiment compares 6 different inference methods on a Two Moon distribution problem:
1. ABC Rejection Sampling
2. ABC-MCMC
3. ABC-SMC
4. Simple Neural Network
5. JANA
6. Simformer

Target: Two Moon distribution with specific parameters
Prior: Uniform priors for all parameters
Features: Extensive plotting and comprehensive analysis
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
from scisbi.simulator.TwoMoonSimulator import TwoMoonsSimulator
from scisbi.inference.ABC import ABCRejectionSampling
from scisbi.inference.ABCMCMC import ABCMCMC
from scisbi.inference.ABCSMC import ABCSMC

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
warnings.filterwarnings("ignore")

# Setup directories
results_dir = Path("experiment/results-2moon")

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
    """Uniform prior for Two Moon Simulator parameters."""

    def __init__(self):
        # Prior ranges: [noise, moon_radius, moon_width, moon_distance]
        self.bounds = np.array(
            [
                [0.01, 0.3],  # noise: 0.01 to 0.3
                [0.5, 2.0],  # moon_radius: 0.5 to 2.0
                [0.3, 1.5],  # moon_width: 0.3 to 1.5
                [-0.8, 0.2],  # moon_distance: -0.8 to 0.2
            ]
        )

    def sample(self, num_samples=None):
        if num_samples is None:
            return np.array(
                [
                    np.random.uniform(self.bounds[i, 0], self.bounds[i, 1])
                    for i in range(4)
                ]
            )
        else:
            return np.array(
                [
                    [
                        np.random.uniform(self.bounds[i, 0], self.bounds[i, 1])
                        for i in range(4)
                    ]
                    for _ in range(num_samples)
                ]
            )

    def log_prob(self, x):
        """Log probability under uniform prior."""
        x = np.atleast_2d(x)
        log_probs = []

        for params in x:
            if len(params) != 4:
                log_probs.append(-np.inf)
                continue

            in_bounds = all(
                self.bounds[i, 0] <= params[i] <= self.bounds[i, 1] for i in range(4)
            )

            if in_bounds:
                # Uniform density = 1 / (volume of hypercube)
                volume = np.prod(self.bounds[:, 1] - self.bounds[:, 0])
                log_probs.append(-np.log(volume))
            else:
                log_probs.append(-np.inf)

        return np.array(log_probs) if len(log_probs) > 1 else log_probs[0]


class SummaryStatistic:
    """Summary statistic for Two Moon data."""

    def compute(self, data):
        """Compute summary statistics: mean x, mean y, std x, std y."""
        if isinstance(data, list):
            data = np.array(data)

        if data.ndim == 1:
            # If 1D, assume it's already a summary statistic
            return (
                data[:4]
                if len(data) >= 4
                else np.pad(data, (0, 4 - len(data)), "constant")
            )

        # Ensure data is 2D with shape (n_samples, 2)
        if data.ndim == 2 and data.shape[1] == 2:
            return np.array(
                [
                    np.mean(data[:, 0]),  # mean x
                    np.mean(data[:, 1]),  # mean y
                    np.std(data[:, 0]),  # std x
                    np.std(data[:, 1]),  # std y
                ]
            )
        else:
            # Fallback: return first 4 elements or pad with zeros
            flat = data.flatten()
            if len(flat) >= 4:
                return flat[:4]
            else:
                return np.pad(flat, (0, 4 - len(flat)), "constant")


class GaussianPerturbationKernel:
    """Gaussian perturbation kernel for ABC-SMC."""

    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, particle):
        perturbed = particle + np.random.normal(0, self.std, size=len(particle))
        # Ensure noise parameter is positive
        perturbed[0] = max(perturbed[0], 0.01)
        return perturbed


class GaussianProposal:
    """Gaussian proposal distribution for ABC-MCMC."""

    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, current_state):
        proposal = current_state + np.random.normal(
            0, self.std, size=len(current_state)
        )
        # Ensure noise parameter is positive
        proposal[0] = max(proposal[0], 0.01)
        return proposal


def euclidean_distance(x, y):
    """Euclidean distance between summary statistics."""
    x = np.array(x).flatten()
    y = np.array(y).flatten()

    # Ensure same length
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]

    return np.sqrt(np.sum((x - y) ** 2))


# ================================
# NEURAL NETWORK MODELS
# ================================


class SimpleNeuralNetwork(nn.Module):
    """Simple neural network for direct parameter estimation."""

    def __init__(self, input_dim, output_dim, hidden_dim=128):
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

        # Ensure positive noise parameter
        if output.shape[1] == 4:
            noise_part = torch.nn.functional.softplus(output[:, 0:1]) + 0.01
            other_parts = output[:, 1:]
            output = torch.cat([noise_part, other_parts], dim=1)

        return output


class ImprovedJANANetwork(nn.Module):
    """Improved JANA network with better stability."""

    def __init__(self, input_dim, output_dim, hidden_dim=128):
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

        # Ensure positive noise parameter
        if output.shape[1] == 4:
            noise_part = torch.nn.functional.softplus(output[:, 0:1]) + 0.01
            other_parts = output[:, 1:]
            output = torch.cat([noise_part, other_parts], dim=1)

        return output


class ImprovedSimformerNetwork(nn.Module):
    """Improved Simformer network with better stability."""

    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Transformer encoder layer
        self.transformer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)

        # Initialize weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # x shape: (batch, input_dim)
        x = self.input_projection(x)  # (batch, hidden_dim)
        x = x.unsqueeze(1)  # (batch, 1, hidden_dim)
        x = x + self.positional_encoding

        # Apply transformer
        x = self.transformer(x)  # (batch, 1, hidden_dim)
        x = x.squeeze(1)  # (batch, hidden_dim)

        output = self.output_projection(x)  # (batch, output_dim)

        # Ensure positive noise parameter
        if output.shape[1] == 4:
            noise_part = torch.nn.functional.softplus(output[:, 0:1]) + 0.01
            other_parts = output[:, 1:]
            output = torch.cat([noise_part, other_parts], dim=1)

        return output


# ================================
# INFERENCE METHODS
# ================================


def run_abc_rejection(simulator, prior, summary_stat, observed_data, num_samples=5000):
    """Run ABC rejection sampling."""

    logger.info("Starting ABC Rejection Sampling...")
    obs_summary = summary_stat.compute(observed_data)

    abc = ABCRejectionSampling(
        simulator=simulator,
        prior=prior,
        distance_function=euclidean_distance,
        tolerance=0.3,  # Adjusted for Two Moon problem
        max_attempts=1000000,
        verbose=False,
    )

    try:
        abc_posterior = abc.infer(obs_summary, num_simulations=num_samples)
        samples = abc_posterior.get_samples()

        # Convert samples to numpy array if needed
        if isinstance(samples, list):
            samples = np.array(samples)

        # Handle different sample formats
        if hasattr(samples, "shape") and samples.ndim == 1:
            samples = samples.reshape(-1, 4)

        logger.info(f"ABC Rejection: Successfully generated {len(samples)} samples")
        return samples, None
    except Exception as e:
        logger.error(f"ABC Rejection failed: {e}")
        return None, None


def run_abc_mcmc(simulator, prior, summary_stat, observed_data, num_samples=5000):
    """Run ABC-MCMC."""

    logger.info("Starting ABC-MCMC...")
    obs_summary = summary_stat.compute(observed_data)
    proposal = GaussianProposal(std=0.1)

    abc_mcmc = ABCMCMC(
        simulator=simulator,
        prior=prior,
        distance_function=euclidean_distance,
        tolerance=0.3,  # Adjusted for Two Moon problem
        proposal_distribution=proposal,
        verbose=False,
        burn_in=200,
        thin=2,
        max_attempts_per_step=1000,
    )

    try:
        abc_mcmc_posterior = abc_mcmc.infer(
            obs_summary, num_simulations=num_samples, num_iterations=2000
        )
        samples = abc_mcmc_posterior.get_samples()

        # Convert samples to numpy array if needed
        if isinstance(samples, list):
            samples = np.array(samples)

        # Handle different sample formats
        if hasattr(samples, "shape") and samples.ndim == 1:
            samples = samples.reshape(-1, 4)

        logger.info(f"ABC-MCMC: Successfully generated {len(samples)} samples")
        return samples, None
    except Exception as e:
        logger.error(f"ABC-MCMC failed: {e}")
        return None, None


def run_abc_smc(simulator, prior, summary_stat, observed_data, num_samples=5000):
    """Run ABC-SMC."""

    logger.info("Starting ABC-SMC...")
    obs_summary = summary_stat.compute(observed_data)
    perturbation_kernel = GaussianPerturbationKernel(std=0.1)

    abc_smc = ABCSMC(
        simulator=simulator,
        prior=prior,
        distance_function=euclidean_distance,
        tolerance_schedule=[1.0, 0.5, 0.3],
        perturbation_kernel=perturbation_kernel,
        verbose=False,
    )

    try:
        abc_smc_posterior = abc_smc.infer(
            obs_summary, num_simulations=num_samples, num_populations=3
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
        return None, None


def train_simple_nn(simulator, prior, summary_stat, observed_data, num_epochs=200):
    """Train simple neural network."""

    logger.info("Training Simple Neural Network...")
    obs_summary = summary_stat.compute(observed_data)

    # Generate training data
    num_sims = 10000
    X_train = []
    y_train = []

    for i in range(num_sims):
        params = prior.sample()
        sim_data = simulator.simulate(params, 100)
        sim_summary = summary_stat.compute(sim_data)

        X_train.append(sim_summary)
        y_train.append(params)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Normalize data
    X_mean, X_std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
    X_train_norm = (X_train - X_mean) / (X_std + 1e-8)

    y_mean, y_std = np.mean(y_train, axis=0), np.std(y_train, axis=0)
    y_train_norm = (y_train - y_mean) / (y_std + 1e-8)

    # Create model
    model = SimpleNeuralNetwork(X_train.shape[1], y_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        X_tensor = torch.FloatTensor(X_train_norm)
        y_tensor = torch.FloatTensor(y_train_norm)

        predictions = model(X_tensor)
        loss = nn.MSELoss()(predictions, y_tensor)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if epoch % 20 == 0:
            logger.info(f"Simple NN Epoch {epoch}: Loss = {loss.item():.6f}")

    # Generate samples with improved uncertainty estimation
    model.eval()
    with torch.no_grad():
        obs_norm = (obs_summary - X_mean) / (X_std + 1e-8)
        obs_tensor = torch.FloatTensor(obs_norm).unsqueeze(0)

        samples = []

        # Strategy 1: Add meaningful noise to capture uncertainty
        for _ in range(400):
            noise_scale = 0.2  # Adjusted for Two Moon problem
            noisy_obs = obs_tensor + torch.randn_like(obs_tensor) * noise_scale

            sample_norm = model(noisy_obs)
            sample = sample_norm.numpy()[0] * (y_std + 1e-8) + y_mean
            samples.append(sample)

        # Strategy 2: Ensemble-like sampling
        noise_levels = [0.1, 0.15, 0.2, 0.25, 0.3]
        for noise_level in noise_levels:
            for _ in range(120):  # 120 * 5 = 600 more samples
                noisy_obs = obs_tensor + torch.randn_like(obs_tensor) * noise_level
                sample_norm = model(noisy_obs)
                sample = sample_norm.numpy()[0] * (y_std + 1e-8) + y_mean
                samples.append(sample)

        samples = np.array(samples)

        # Filter valid samples and ensure constraints
        valid_mask = ~np.isnan(samples).any(axis=1)
        samples = samples[valid_mask]

        # Ensure noise parameter is positive
        samples[:, 0] = np.maximum(samples[:, 0], 0.01)

        logger.info(f"Simple NN: Successfully generated {len(samples)} samples")
        return samples, train_losses


def train_jana(
    simulator, prior, summary_stat, observed_data, num_epochs=40, batch_size=32
):
    """Train JANA with improved stability."""

    logger.info("Training JANA...")
    obs_summary = summary_stat.compute(observed_data)

    # Generate training data
    num_sims = 8000
    X_train = []
    y_train = []

    for i in range(num_sims):
        params = prior.sample()
        sim_data = simulator.simulate(params, 100)
        sim_summary = summary_stat.compute(sim_data)

        X_train.append(sim_summary)
        y_train.append(params)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Normalize data
    X_mean, X_std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
    X_train_norm = (X_train - X_mean) / (X_std + 1e-8)

    y_mean, y_std = np.mean(y_train, axis=0), np.std(y_train, axis=0)
    y_train_norm = (y_train - y_mean) / (y_std + 1e-8)

    # Create model
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    model = ImprovedJANANetwork(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.8)

    # Training loop with early stopping
    best_loss = float("inf")
    patience_counter = 0
    train_losses = []

    for epoch in range(num_epochs):
        model.train()

        # Create mini-batches
        indices = np.random.permutation(len(X_train))
        epoch_loss = 0
        num_batches = 0

        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i : i + batch_size]

            X_batch = torch.FloatTensor(X_train_norm[batch_indices])
            y_batch = torch.FloatTensor(y_train_norm[batch_indices])

            optimizer.zero_grad()

            # Forward pass: use actual parameters as z input
            predictions = model(X_batch, y_batch)

            # JANA loss: network should predict same parameters
            loss = nn.MSELoss()(predictions, y_batch)

            # Check for NaN
            if torch.isnan(loss):
                logger.error(f"NaN loss detected at epoch {epoch}, batch {i}")
                return None, train_losses

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 10:
            logger.info(f"JANA: Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0:
            logger.info(f"JANA Epoch {epoch}: Loss = {avg_loss:.6f}")

    # Sample from posterior with consistency checking
    model.eval()
    with torch.no_grad():
        obs_norm = (obs_summary - X_mean) / (X_std + 1e-8)
        obs_tensor = torch.FloatTensor(obs_norm).unsqueeze(0)

        samples = []

        # JANA sampling with consistency checking
        num_prior_samples = 10000
        prior_samples = prior.sample(num_prior_samples)

        for i in range(num_prior_samples):
            z_sample = prior_samples[i]
            z_norm = (z_sample - y_mean) / (y_std + 1e-8)
            z_tensor = torch.FloatTensor(z_norm).unsqueeze(0)

            sample_norm = model(obs_tensor, z_tensor)
            sample = sample_norm.numpy()[0] * (y_std + 1e-8) + y_mean

            # Accept sample based on consistency
            consistency_threshold = 0.5
            consistency_error = np.linalg.norm(sample - z_sample)

            if consistency_error < consistency_threshold:
                samples.append(sample)

        # If not enough samples, use lenient approach
        if len(samples) < 100:
            logger.warning(
                f"JANA: Only {len(samples)} consistent samples found, using all prior samples"
            )
            samples = []
            for i in range(min(1000, num_prior_samples)):
                z_sample = prior_samples[i]
                z_norm = (z_sample - y_mean) / (y_std + 1e-8)
                z_tensor = torch.FloatTensor(z_norm).unsqueeze(0)

                sample_norm = model(obs_tensor, z_tensor)
                sample = sample_norm.numpy()[0] * (y_std + 1e-8) + y_mean
                samples.append(sample)

        samples = np.array(samples)

        # Filter out invalid samples and ensure constraints
        valid_mask = ~np.isnan(samples).any(axis=1)
        valid_samples = samples[valid_mask]

        # Ensure noise parameter is positive
        valid_samples[:, 0] = np.maximum(valid_samples[:, 0], 0.01)

        logger.info(f"JANA: Successfully generated {len(valid_samples)} valid samples")

        if len(valid_samples) > 0:
            return valid_samples, train_losses
        else:
            logger.warning("JANA: No valid samples generated")
            return None, train_losses


def train_simformer(
    simulator, prior, summary_stat, observed_data, num_epochs=40, batch_size=32
):
    """Train Simformer with improved stability."""

    logger.info("Training Simformer...")
    obs_summary = summary_stat.compute(observed_data)

    # Generate training data
    num_sims = 8000
    X_train = []
    y_train = []

    for i in range(num_sims):
        params = prior.sample()
        sim_data = simulator.simulate(params, 100)
        sim_summary = summary_stat.compute(sim_data)

        X_train.append(sim_summary)
        y_train.append(params)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Normalize data
    X_mean, X_std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
    X_train_norm = (X_train - X_mean) / (X_std + 1e-8)

    y_mean, y_std = np.mean(y_train, axis=0), np.std(y_train, axis=0)
    y_train_norm = (y_train - y_mean) / (y_std + 1e-8)

    # Create model
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    model = ImprovedSimformerNetwork(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.8)

    # Training loop with early stopping
    best_loss = float("inf")
    patience_counter = 0
    train_losses = []

    for epoch in range(num_epochs):
        model.train()

        # Create mini-batches
        indices = np.random.permutation(len(X_train))
        epoch_loss = 0
        num_batches = 0

        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i : i + batch_size]

            X_batch = torch.FloatTensor(X_train_norm[batch_indices])
            y_batch = torch.FloatTensor(y_train_norm[batch_indices])

            optimizer.zero_grad()

            # Forward pass
            predictions = model(X_batch)

            # Simple MSE loss
            loss = nn.MSELoss()(predictions, y_batch)

            # Check for NaN
            if torch.isnan(loss):
                logger.error(f"NaN loss detected at epoch {epoch}, batch {i}")
                return None, train_losses

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 10:
            logger.info(f"Simformer: Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0:
            logger.info(f"Simformer Epoch {epoch}: Loss = {avg_loss:.6f}")

    # Sample from posterior with improved uncertainty estimation
    model.eval()
    with torch.no_grad():
        obs_norm = (obs_summary - X_mean) / (X_std + 1e-8)
        obs_tensor = torch.FloatTensor(obs_norm).unsqueeze(0)

        samples = []

        # Strategy 1: Add meaningful noise to capture uncertainty
        for _ in range(400):
            noise_scale = 0.2  # Adjusted for Two Moon problem
            noisy_obs = obs_tensor + torch.randn_like(obs_tensor) * noise_scale

            sample_norm = model(noisy_obs)
            sample = sample_norm.numpy()[0] * (y_std + 1e-8) + y_mean
            samples.append(sample)

        # Strategy 2: Ensemble-like sampling
        noise_levels = [0.1, 0.15, 0.2, 0.25, 0.3]
        for noise_level in noise_levels:
            for _ in range(120):  # 120 * 5 = 600 more samples
                noisy_obs = obs_tensor + torch.randn_like(obs_tensor) * noise_level
                sample_norm = model(noisy_obs)
                sample = sample_norm.numpy()[0] * (y_std + 1e-8) + y_mean
                samples.append(sample)

        samples = np.array(samples)

        # Filter out invalid samples and ensure constraints
        valid_mask = ~np.isnan(samples).any(axis=1)
        valid_samples = samples[valid_mask]

        # Ensure noise parameter is positive
        valid_samples[:, 0] = np.maximum(valid_samples[:, 0], 0.01)

        logger.info(
            f"Simformer: Successfully generated {len(valid_samples)} valid samples"
        )

        if len(valid_samples) > 0:
            return valid_samples, train_losses
        else:
            logger.warning("Simformer: No valid samples generated")
            return None, train_losses


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
            "mean_estimate": np.array([np.nan, np.nan, np.nan, np.nan]),
            "std_estimate": np.array([np.nan, np.nan, np.nan, np.nan]),
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
# VISUALIZATION FUNCTIONS
# ================================


def plot_two_moons_data(data, title="Two Moons Data", filename=None):
    """Plot two moons data."""
    plt.figure(figsize=(10, 8))

    if isinstance(data, list):
        data = np.array(data)

    plt.scatter(
        data[:, 0],
        data[:, 1],
        alpha=0.6,
        s=20,
        color="blue",
        edgecolors="black",
        linewidth=0.5,
    )
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")

    if filename:
        plt.savefig(results_dir / filename, dpi=300, bbox_inches="tight")
    plt.show()
    logger.info(f"Two moons data plot saved to {filename}")


def plot_prior_samples(prior, n_samples=1000, filename=None):
    """Plot samples from the prior distribution."""
    samples = prior.sample(n_samples)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    param_names = ["Noise", "Moon Radius", "Moon Width", "Moon Distance"]
    colors = ["skyblue", "lightgreen", "lightcoral", "lightyellow"]

    for i, (ax, name, color) in enumerate(zip(axes.flat, param_names, colors)):
        ax.hist(
            samples[:, i],
            bins=50,
            alpha=0.7,
            density=True,
            color=color,
            edgecolor="black",
        )
        ax.set_title(f"Prior Distribution: {name}", fontsize=12, fontweight="bold")
        ax.set_xlabel(name, fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if filename:
        plt.savefig(results_dir / filename, dpi=300, bbox_inches="tight")
    plt.show()
    logger.info(f"Prior samples plot saved to {filename}")


def plot_parameter_variations(simulator, prior, filename=None):
    """Plot data examples from different parameter configurations."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Define parameter configurations
    param_configs = [
        [0.05, 0.8, 0.6, -0.3],  # Small noise, small radius
        [0.15, 1.2, 1.0, -0.1],  # Medium noise, medium radius
        [0.25, 1.5, 1.2, 0.1],  # Large noise, large radius
        [0.1, 1.0, 0.5, -0.5],  # Narrow width
        [0.1, 1.0, 1.3, -0.2],  # Wide width
        [0.1, 1.0, 0.8, -0.25],  # Default parameters
    ]

    config_names = [
        "Small Noise, Small Radius",
        "Medium Noise, Medium Radius",
        "Large Noise, Large Radius",
        "Narrow Width",
        "Wide Width",
        "Default Parameters",
    ]

    colors = ["red", "blue", "green", "orange", "purple", "brown"]

    for i, (params, name, color) in enumerate(zip(param_configs, config_names, colors)):
        # Create simulator with specific parameters
        sim = TwoMoonsSimulator(
            noise=params[0],
            moon_radius=params[1],
            moon_width=params[2],
            moon_distance=params[3],
        )
        data = sim.simulate(params, 200)

        axes[i].scatter(
            data[:, 0],
            data[:, 1],
            alpha=0.6,
            s=15,
            color=color,
            edgecolors="black",
            linewidth=0.3,
        )
        axes[i].set_title(f"{name}\n{params}", fontsize=11, fontweight="bold")
        axes[i].set_xlabel("X", fontsize=10)
        axes[i].set_ylabel("Y", fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].axis("equal")

    plt.tight_layout()
    if filename:
        plt.savefig(results_dir / filename, dpi=300, bbox_inches="tight")
    plt.show()
    logger.info(f"Parameter variations plot saved to {filename}")


def create_posterior_comparison_plots(results, true_params, prior, filename=None):
    """Create comprehensive posterior comparison plots."""

    successful_methods = {k: v for k, v in results.items() if v["samples"] is not None}

    if not successful_methods:
        logger.warning("No successful methods to plot")
        return

    num_methods = len(successful_methods)
    num_params = 4
    param_names = ["Noise", "Moon Radius", "Moon Width", "Moon Distance"]

    fig, axes = plt.subplots(
        num_params, num_methods, figsize=(5 * num_methods, 4 * num_params)
    )

    # Handle case with only one method
    if num_methods == 1:
        axes = axes.reshape(num_params, 1)

    # Generate prior samples for plotting
    prior_samples = prior.sample(5000)

    colors = ["blue", "green", "orange", "red", "purple", "brown"]

    for j, (method_name, result) in enumerate(successful_methods.items()):
        samples = result["samples"]
        metrics = result["metrics"]
        color = colors[j % len(colors)]

        # Convert samples to numpy array if it's a list
        if isinstance(samples, list):
            samples = np.array(samples)

        # Ensure samples is 2D
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)

        for i, param_name in enumerate(param_names):
            ax = axes[i, j] if num_methods > 1 else axes[i]

            # Plot prior distribution
            ax.hist(
                prior_samples[:, i],
                bins=50,
                alpha=0.3,
                density=True,
                color="gray",
                label="Prior",
                edgecolor="none",
            )

            # Plot posterior distribution
            ax.hist(
                samples[:, i],
                bins=30,
                alpha=0.7,
                density=True,
                color=color,
                edgecolor="black",
                label="Posterior",
            )

            # Plot mean estimate
            ax.axvline(
                metrics["mean_estimate"][i],
                color="blue",
                linewidth=2,
                label=f"Mean: {metrics['mean_estimate'][i]:.3f}",
            )

            # Plot true parameter value
            ax.axvline(
                true_params[i],
                color="red",
                linewidth=2,
                linestyle="--",
                label=f"True: {true_params[i]:.3f}",
            )

            ax.set_title(
                f"{method_name}\n{param_name}\nMSE: {metrics['mse']:.4f}",
                fontsize=10,
                fontweight="bold",
            )
            ax.set_xlabel(param_name, fontsize=9)
            ax.set_ylabel("Density", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if filename:
        plt.savefig(results_dir / filename, dpi=300, bbox_inches="tight")
    plt.show()
    logger.info(f"Posterior comparison plots saved to {filename}")


def create_timing_plot(results, filename=None):
    """Create timing comparison plot."""

    successful_methods = {k: v for k, v in results.items() if v["samples"] is not None}

    if not successful_methods:
        logger.warning("No successful methods for timing plot")
        return

    methods = list(successful_methods.keys())
    times = [successful_methods[m]["time"] for m in methods]

    plt.figure(figsize=(12, 8))
    colors = ["blue", "green", "orange", "red", "purple", "brown"]
    bars = plt.bar(
        methods, times, color=colors[: len(methods)], alpha=0.7, edgecolor="black"
    )

    plt.ylabel("Time (seconds)", fontsize=12)
    plt.title(
        "Timing Comparison - Two Moon Inference Experiment",
        fontsize=14,
        fontweight="bold",
    )
    plt.xticks(rotation=45, fontsize=11)
    plt.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, t in zip(bars, times):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01 * max(times),
            f"{t:.2f}s",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    if filename:
        plt.savefig(results_dir / filename, dpi=300, bbox_inches="tight")
    plt.show()
    logger.info(f"Timing plot saved to {filename}")


def plot_training_curves(results, filename=None):
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
        axes[i].set_title(
            f"{method_name} - Training Loss", fontsize=12, fontweight="bold"
        )
        axes[i].set_xlabel("Epoch", fontsize=11)
        axes[i].set_ylabel("Loss", fontsize=11)
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    if filename:
        plt.savefig(results_dir / filename, dpi=300, bbox_inches="tight")
    plt.show()
    logger.info(f"Training curves saved to {filename}")


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
                    "Mean Est.": f"[{metrics['mean_estimate'][0]:.3f}, {metrics['mean_estimate'][1]:.3f}, {metrics['mean_estimate'][2]:.3f}, {metrics['mean_estimate'][3]:.3f}]",
                    "Std Est.": f"[{metrics['std_estimate'][0]:.3f}, {metrics['std_estimate'][1]:.3f}, {metrics['std_estimate'][2]:.3f}, {metrics['std_estimate'][3]:.3f}]",
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

    logger.info("=" * 100)
    logger.info("SUMMARY TABLE")
    logger.info("=" * 100)
    logger.info(f"\n{df.to_string(index=False)}")

    # Save to CSV
    df.to_csv(results_dir / "two_moon_results.csv", index=False)
    logger.info("Summary table saved to two_moon_results.csv")


def plot_inferred_model_samples(results, true_params, simulator, filename=None):
    """Plot samples drawn from inferred parameter values for all methods."""

    successful_methods = {k: v for k, v in results.items() if v["samples"] is not None}

    if not successful_methods:
        logger.warning("No successful methods for inferred model samples plot")
        return

    num_methods = len(successful_methods)
    fig, axes = plt.subplots(2, num_methods, figsize=(6 * num_methods, 12))

    # Handle case where there's only one method
    if num_methods == 1:
        axes = axes.reshape(2, 1)

    colors = ["blue", "green", "orange", "red", "purple", "brown"]

    # Generate true data for comparison
    true_data = simulator.simulate(true_params, 1000)

    for i, (method_name, result) in enumerate(successful_methods.items()):
        samples = result["samples"]
        color = colors[i % len(colors)]

        # Convert samples to numpy array if needed
        if isinstance(samples, list):
            samples = np.array(samples)

        # Get mean estimate for this method
        mean_estimate = np.mean(samples, axis=0)

        # Generate data from mean estimate
        mean_data = simulator.simulate(mean_estimate, 1000)

        # Plot 1: Scatter plot comparison (2D data)
        ax1 = axes[0, i] if num_methods > 1 else axes[0]
        ax1.scatter(
            true_data[:, 0],
            true_data[:, 1],
            alpha=0.5,
            color="red",
            s=20,
            label=f"True Data (n={len(true_data)})",
        )
        ax1.scatter(
            mean_data[:, 0],
            mean_data[:, 1],
            alpha=0.5,
            color=color,
            s=20,
            label=f"Mean Estimate (n={len(mean_data)})",
        )
        ax1.set_title(f"{method_name} - Data Comparison")
        ax1.set_xlabel("X1")
        ax1.set_ylabel("X2")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect("equal")

        # Plot 2: Multiple samples from posterior
        ax2 = axes[1, i] if num_methods > 1 else axes[1]

        # Plot true data as background
        ax2.scatter(
            true_data[:, 0],
            true_data[:, 1],
            alpha=0.3,
            color="red",
            s=10,
            label="True Data",
        )

        # Draw samples from posterior for visualization
        n_posterior_samples = min(5, len(samples))
        indices = np.random.choice(len(samples), n_posterior_samples, replace=False)

        for j, idx in enumerate(indices):
            sample_params = samples[idx]
            sample_data = simulator.simulate(sample_params, 200)

            alpha = 0.4 if j > 0 else 0.7
            size = 15 if j == 0 else 10
            label = "Posterior Samples" if j == 0 else None
            ax2.scatter(
                sample_data[:, 0],
                sample_data[:, 1],
                alpha=alpha,
                color=color,
                s=size,
                label=label,
            )

        ax2.set_title(f"{method_name} - Posterior Uncertainty")
        ax2.set_xlabel("X1")
        ax2.set_ylabel("X2")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect("equal")

    plt.tight_layout()
    if filename:
        plt.savefig(results_dir / filename, dpi=300, bbox_inches="tight")
    plt.show()
    logger.info(f"Inferred model samples plot saved to {filename}")


# ================================
# MAIN EXPERIMENT
# ================================


def main():
    """Run the main experiment."""

    logger.info("=" * 70)
    logger.info("COMBINED TWO MOON INFERENCE EXPERIMENT")
    logger.info("=" * 70)
    logger.info(
        f"Experiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Setup
    true_params = np.array([0.1, 1.0, 0.8, -0.25])  # [noise, radius, width, distance]
    simulator = TwoMoonsSimulator(
        noise=true_params[0],
        moon_radius=true_params[1],
        moon_width=true_params[2],
        moon_distance=true_params[3],
    )
    prior = UniformPrior()
    summary_stat = SummaryStatistic()

    # Generate observed data
    observed_data = simulator.simulate(true_params, 200)
    obs_summary = summary_stat.compute(observed_data)

    logger.info(f"True parameters: {true_params}")
    logger.info(f"Observed summary: {obs_summary}")

    # Create initial visualizations
    plot_two_moons_data(observed_data, "Observed Two Moons Data", "observed_data.png")
    plot_prior_samples(prior, filename="prior_samples.png")
    plot_parameter_variations(simulator, prior, filename="parameter_variations.png")

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
                    simulator, prior, summary_stat, observed_data, num_samples=2000
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
            results[method_name] = {
                "samples": None,
                "time": 0,
                "metrics": None,
                "train_losses": None,
            }

    # Create summary table
    create_summary_table(results, true_params)

    # Create comprehensive visualizations
    create_posterior_comparison_plots(
        results, true_params, prior, "posterior_comparison.png"
    )
    create_timing_plot(results, "timing_comparison.png")
    plot_training_curves(results, "training_curves.png")
    plot_inferred_model_samples(
        results, true_params, simulator, "inferred_model_samples.png"
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
