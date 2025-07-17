import pytest
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import io
import sys
from unittest.mock import patch

from scisbi.inference.JANA import JANA, JANAPosterior
from scisbi.base.simulator import BaseSimulator
from scisbi.base.summary_statistic import BaseSummaryStatistic


# Mock classes for basic testing
class MockSimulator(BaseSimulator):
    def __init__(self, return_value=None):
        self.return_value = (
            return_value if return_value is not None else np.array([1.0, 2.0])
        )

    def simulate(self, parameters, **kwargs):
        return self.return_value


class MockPrior:
    def __init__(self, sample_dim=2, sample_value=None):
        self.sample_dim = sample_dim
        self.sample_value = (
            sample_value if sample_value is not None else np.ones(sample_dim)
        )

    def sample(self, n=1):
        if n == 1:
            return self.sample_value.reshape(1, -1)
        return np.tile(self.sample_value, (n, 1))

    def log_prob(self, parameters):
        """Dummy implementation for testing purposes."""
        return np.zeros(parameters.shape[0] if parameters.ndim > 1 else 1)


class MockSummaryStatistic(BaseSummaryStatistic):
    def compute(self, data):
        return np.mean(data, axis=-1, keepdims=True)

    def _normalize(self, stats):
        """Dummy implementation for testing purposes."""
        return stats

    def visualize(self, stats):
        """Dummy implementation for testing purposes."""
        pass


class SimpleJanaNet(nn.Module):
    def __init__(self, x_dim=2, theta_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + theta_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, theta_dim),
        )

    def forward(self, x, z):
        xz = torch.cat((x, z), dim=1)
        return self.net(xz)


# Test JANA initialization
def test_jana_initialization_valid():
    simulator = MockSimulator()
    prior = MockPrior()
    model = SimpleJanaNet()

    jana = JANA(simulator, prior, model)

    assert jana.simulator == simulator
    assert jana.prior == prior
    assert jana.model == model
    assert jana.device in ["cpu", "cuda"]


def test_jana_initialization_with_summary_statistic():
    simulator = MockSimulator()
    prior = MockPrior()
    model = SimpleJanaNet()
    summary_stat = MockSummaryStatistic()

    jana = JANA(simulator, prior, model, summary_statistic=summary_stat, device="cpu")

    assert jana.summary_statistic == summary_stat
    assert jana.device == "cpu"


# Test JANAPosterior
def test_jana_posterior_initialization():
    model = SimpleJanaNet()
    prior = MockPrior()
    posterior = JANAPosterior(model, prior)
    assert posterior.model == model
    assert posterior.prior == prior


@patch("torch.from_numpy")
@patch.object(SimpleJanaNet, "forward")
def test_jana_posterior_sample(mock_forward, mock_from_numpy):
    model = SimpleJanaNet(x_dim=1, theta_dim=1)
    prior = MockPrior(sample_dim=1)
    summary_stat = MockSummaryStatistic()
    posterior = JANAPosterior(model, prior, summary_statistic=summary_stat)

    observed_data = np.random.randn(10)
    num_samples = 1

    # Mock returns
    mock_forward.return_value = torch.randn(num_samples, 1)
    # Make from_numpy pass through tensors
    mock_from_numpy.side_effect = lambda x: torch.tensor(x).float()

    samples = posterior.sample(observed_data, num_samples=num_samples)

    assert samples.shape == (num_samples, 1)
    # Check that summary statistic was called implicitly
    assert mock_from_numpy.call_args_list[0][0][0].shape == (1, 1)  # for observed data
    assert mock_from_numpy.call_args_list[1][0][0].shape == (
        num_samples,
        1,
    )  # for prior samples (z)


def test_jana_posterior_log_prob_raises_error():
    posterior = JANAPosterior(SimpleJanaNet(), MockPrior())
    with pytest.raises(NotImplementedError):
        posterior.log_prob(None, None)


# Integration tests with Gaussian distributions
class GaussianSimulator(BaseSimulator):
    def simulate(self, parameters, num_simulations=200):
        if isinstance(parameters, dict):
            mean = parameters["mean"]
            std = parameters["std"]
        else:
            try:
                mean, std = parameters
            except Exception:
                raise ValueError(
                    "Parameters must be a dict with keys 'mean' and 'std' or an iterable with two elements."
                )
        return np.random.normal(mean, np.abs(std) + 1e-6, num_simulations)


class GaussianPrior:
    def __init__(self, dim=2, prior_mean=None, prior_std=None):
        self.dim = dim
        self.prior_mean = prior_mean if prior_mean is not None else np.zeros(dim)
        self.prior_std = prior_std if prior_std is not None else np.ones(dim)

    def sample(self, n=1):
        samples = np.random.normal(
            loc=self.prior_mean, scale=self.prior_std, size=(n, self.dim)
        )
        return samples if n > 1 else samples.squeeze(axis=0)

    def log_prob(self, parameters):
        """Dummy implementation for testing purposes."""
        return np.zeros(parameters.shape[0] if parameters.ndim > 1 else 1)


class SummaryStats(BaseSummaryStatistic):
    def compute(self, data):
        mean = np.mean(data, axis=-1)
        std = np.std(data, axis=-1)
        return np.stack([mean, std], axis=-1)

    def _normalize(self, stats):
        """Dummy implementation for testing purposes."""
        return stats

    def visualize(self, stats):
        """Dummy implementation for testing purposes."""
        pass


def test_jana_inference_2d_gaussian():
    """Test JANA inference on 2D Gaussian (mean, std) parameters."""
    np.random.seed(42)
    torch.manual_seed(42)

    true_params = np.array([2.0, 1.5])  # true mean=2.0, true std=1.5
    simulator = GaussianSimulator()
    prior = GaussianPrior(
        dim=2, prior_mean=np.array([0.0, 2.0]), prior_std=np.array([3.0, 2.0])
    )
    summary_stat = SummaryStats()

    # Generate observed data
    observed_data = simulator.simulate(true_params)

    model = SimpleJanaNet(x_dim=2, theta_dim=2)

    jana = JANA(
        simulator=simulator,
        prior=prior,
        model=model,
        summary_statistic=summary_stat,
        device="cpu",
    )

    posterior = jana.infer(
        num_simulations=500, num_epochs=5, batch_size=32, verbose=False
    )

    samples = posterior.sample(observed_data, num_samples=1)

    assert samples.shape == (1, 2)

    # Check that posterior samples are reasonable (means should be close to true_params)
    posterior_mean = np.mean(samples, axis=0)
    # This is a weak test as training is short, but it checks the pipeline
    assert np.all(np.abs(posterior_mean - true_params) < 2.0)


def test_jana_inference_verbose_mode():
    """Test JANA inference with verbose output."""
    np.random.seed(42)
    torch.manual_seed(42)

    simulator = MockSimulator(return_value=np.random.randn(10))
    prior = MockPrior(sample_dim=1, sample_value=np.array([0.5]))
    model = SimpleJanaNet(x_dim=10, theta_dim=1)

    jana = JANA(simulator, prior, model, device="cpu")

    captured_output = io.StringIO()
    sys.stdout = captured_output

    try:
        jana.infer(num_simulations=20, num_epochs=1, batch_size=5, verbose=True)
    finally:
        sys.stdout = sys.__stdout__

    output = captured_output.getvalue()
    assert "Using device: cpu" in output
    assert "Simulating data..." in output
    assert "Training the neural network..." in output
    assert "Epoch 1/1" in output
    assert "Training finished." in output


# Main execution with plotting
def main():
    """
    Main function to demonstrate JANA functionality with plots.
    This runs a complete example and generates visualizations.
    """
    print("=" * 60)
    print("JANA Demonstration and Testing")
    print("=" * 60)

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Define true parameters for a 2D Gaussian
    true_params = np.array([2.0, 1.5])  # True mean and std
    print(f"True parameters: mean={true_params[0]}, std={true_params[1]}")

    # Generate observed data
    simulator = GaussianSimulator()
    observed_data = simulator.simulate(true_params, num_simulations=1000)

    summary_stat = SummaryStats()
    observed_summary = summary_stat.compute(observed_data)
    print(
        f"Observed data summary: mean={observed_summary[0]:.3f}, std={observed_summary[1]:.3f}"
    )

    # Set up JANA components
    prior = GaussianPrior(
        dim=2, prior_mean=np.array([0.0, 2.0]), prior_std=np.array([3.0, 2.0])
    )
    model = SimpleJanaNet(x_dim=2, theta_dim=2)

    jana = JANA(
        simulator=simulator,
        prior=prior,
        model=model,
        summary_statistic=summary_stat,
        device="auto",
    )

    print("\nRunning JANA inference...")
    # Run inference
    posterior = jana.infer(
        num_simulations=10000,
        num_epochs=100,
        learning_rate=1e-3,
        batch_size=128,
        stop_after_epochs=10,
        verbose=True,
    )

    # Generate posterior samples
    print("\nGenerating posterior samples...")
    num_samples = 5000
    samples = posterior.sample(observed_data, num_samples=num_samples)

    # Print results
    print("\nJANA Posterior Summary:")
    posterior_mean = np.mean(samples, axis=0)
    posterior_std = np.std(samples, axis=0)
    print(f"Posterior mean: {posterior_mean[0]:.3f}, {posterior_mean[1]:.3f}")
    print(f"Posterior std:  {posterior_std[0]:.3f}, {posterior_std[1]:.3f}")

    # Plotting
    print("\nPlotting results...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot for parameter 1 (mean)
    axes[0].hist(
        samples[:, 0], bins=50, density=True, alpha=0.7, label="JANA Posterior"
    )
    axes[0].axvline(true_params[0], color="r", linestyle="--", label="True Mean")
    axes[0].set_title("Posterior of Mean")
    axes[0].set_xlabel("Mean value")
    axes[0].legend()

    # Plot for parameter 2 (std)
    axes[1].hist(
        samples[:, 1], bins=50, density=True, alpha=0.7, label="JANA Posterior"
    )
    axes[1].axvline(true_params[1], color="r", linestyle="--", label="True Std")
    axes[1].set_title("Posterior of Std")
    axes[1].set_xlabel("Std value")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    # 2D scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.1)
    plt.axvline(true_params[0], color="r", linestyle="--")
    plt.axhline(true_params[1], color="r", linestyle="--")
    plt.xlabel("Mean")
    plt.ylabel("Std")
    plt.title("Posterior Samples")
    plt.show()

    print("\n" + "=" * 60)
    print("JANA demonstration completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
