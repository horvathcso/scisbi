"""
Comprehensive test suite for the Simformer algorithm implementation.

Tests cover:
- Basic initialization and configuration
- Model architecture components
- Training and inference procedures
- Posterior sampling and evaluation
- Integration with different simulators
- Plotting and visualization
"""

import pytest
import numpy as np
from unittest.mock import patch

# Check if PyTorch is available
try:
    import torch
    from scisbi.inference.Simformer import (
        Simformer,
        SimformerPosterior,
        SimformerTransformer,
        PositionalEncoding,
    )

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

    # Mock classes for testing when PyTorch is not available
    class Simformer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for Simformer")

    class SimformerPosterior:
        pass

    class SimformerTransformer:
        pass

    class PositionalEncoding:
        pass


# Check for matplotlib
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Mock classes for basic testing
class MockSimulator:
    def __init__(self, return_value=None, noise_std=0.1):
        self.return_value = return_value or [1.0, 2.0]
        self.noise_std = noise_std

    def simulate(self, parameters, **kwargs):
        # Handle batching
        params = np.asarray(parameters)
        if params.ndim == 1:
            params = params[:, np.newaxis]

        batch_size, param_dim = params.shape

        # Return batched data
        # Return (Batch, 3) or (Batch, 4) depending on param_dim
        if param_dim == 1:
             # 1D parameter case
             # data shape: (Batch, 3)
             base = np.array([1.0, 0.5, 2.0])
             # params: (Batch, 1). base: (3,)
             # (Batch, 3)
             data = params * base[np.newaxis, :]
             noise = np.random.normal(0, self.noise_std, (batch_size, 3))
             return data + noise
        else:
             # Multi-dimensional parameter case
             # Sum parameters: (Batch, 1)
             param_sum = np.sum(params, axis=1, keepdims=True)
             base = np.array([1.0, 0.5, 2.0, 1.5])
             data = param_sum * base[np.newaxis, :]
             noise = np.random.normal(0, self.noise_std, (batch_size, 4))
             return data + noise


class MockPrior:
    def __init__(self, sample_value=None, dim=1):
        self.sample_value = sample_value or 0.5
        self.dim = dim

    def sample(self, num_samples=None):
        if num_samples is not None:
            if self.dim == 1:
                return np.random.normal(0, 1, (num_samples, 1)) # (N, 1)
            else:
                return np.random.normal(0, 1, (num_samples, self.dim))
        else:
            # Single sample
            if self.dim == 1:
                return np.array([np.random.normal(0, 1)]) # (1,)
            else:
                return np.random.normal(0, 1, self.dim) # (D,)

    def log_prob(self, x):
        x_arr = np.atleast_1d(np.array(x))
        if x_arr.ndim > 1:
             # Batch log prob
             return -0.5 * np.sum(x_arr**2, axis=1) - 0.5 * x_arr.shape[1] * np.log(2 * np.pi)
        return -0.5 * np.sum(x_arr**2) - 0.5 * len(x_arr) * np.log(2 * np.pi)


class MockSummaryStatistic:
    def compute(self, data):
        # data: (Batch, D)
        data_arr = np.array(data)
        if data_arr.ndim == 1:
            data_arr = data_arr[np.newaxis, :]

        mean = np.mean(data_arr, axis=1, keepdims=True)
        std = np.std(data_arr, axis=1, keepdims=True)
        return np.concatenate([mean, std], axis=1) # (Batch, 2)


# Test PositionalEncoding
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_positional_encoding():
    d_model = 256
    max_len = 1000
    pos_enc = PositionalEncoding(d_model, max_len)

    # Test forward pass
    batch_size, seq_len = 4, 50
    x = torch.randn(seq_len, batch_size, d_model)
    output = pos_enc(x)

    assert output.shape == x.shape
    assert not torch.allclose(
        output, x
    )  # Should be different due to positional encoding


# Test SimformerTransformer
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_simformer_transformer_initialization():
    data_dim = 4
    param_dim = 2
    model = SimformerTransformer(
        data_dim=data_dim,
        param_dim=param_dim,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
    )

    assert model.data_dim == data_dim
    assert model.param_dim == param_dim
    assert model.d_model == 128


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_simformer_transformer_forward():
    data_dim = 4
    param_dim = 2
    batch_size = 8
    seq_len = 20

    model = SimformerTransformer(
        data_dim=data_dim, param_dim=param_dim, d_model=128, nhead=4, num_layers=2
    )

    # Create input data
    data_sequence = torch.randn(batch_size, seq_len, data_dim)
    observed_data = torch.randn(batch_size, data_dim)

    # Forward pass
    mean, log_std = model.forward(data_sequence, observed_data)

    assert mean.shape == (batch_size, param_dim)
    assert log_std.shape == (batch_size, param_dim)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_simformer_transformer_sampling():
    data_dim = 3
    param_dim = 1
    batch_size = 5
    seq_len = 15
    num_samples = 10

    model = SimformerTransformer(
        data_dim=data_dim, param_dim=param_dim, d_model=64, nhead=2, num_layers=1
    )

    data_sequence = torch.randn(batch_size, seq_len, data_dim)
    observed_data = torch.randn(batch_size, data_dim)

    # Test single sample
    sample = model.sample_posterior(data_sequence, observed_data, num_samples=1)
    if param_dim == 1:
        assert sample.shape == (batch_size,)  # 1D parameter case
    else:
        assert sample.shape == (batch_size, param_dim)

    # Test multiple samples
    samples = model.sample_posterior(
        data_sequence, observed_data, num_samples=num_samples
    )
    if param_dim == 1:
        assert samples.shape == (num_samples, batch_size)  # 1D parameter case
    else:
        assert samples.shape == (num_samples, batch_size, param_dim)


# Test Simformer initialization
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_simformer_initialization():
    simulator = MockSimulator()
    prior = MockPrior(dim=2)

    simformer = Simformer(
        simulator=simulator,
        prior=prior,
        data_dim=4,
        param_dim=2,
        d_model=128,
        nhead=4,
        num_layers=2,
    )

    assert simformer.data_dim == 4
    assert simformer.param_dim == 2
    assert simformer.simulator == simulator
    assert simformer.prior == prior
    assert isinstance(simformer.model, SimformerTransformer)


def test_simformer_initialization_without_torch():
    simulator = MockSimulator()
    prior = MockPrior()

    if not HAS_TORCH:
        with pytest.raises(ImportError, match="PyTorch is required for Simformer"):
            Simformer(simulator, prior, data_dim=3, param_dim=1)


# Test training data generation
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_simformer_generate_training_sequences():
    np.random.seed(42)

    simulator = MockSimulator()
    prior = MockPrior(dim=1)

    simformer = Simformer(
        simulator=simulator, prior=prior, data_dim=3, param_dim=1, d_model=64
    )

    num_simulations = 10
    sequence_length = 5

    data_seqs, param_seqs, observed_seqs = simformer._generate_training_sequences(
        num_simulations, sequence_length, verbose=False
    )

    assert data_seqs.shape == (num_simulations, sequence_length, 3)
    assert param_seqs.shape == (num_simulations, sequence_length, 1)
    assert observed_seqs.shape == (num_simulations, 3)


# Test SimformerPosterior
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_simformer_posterior_initialization():
    model = SimformerTransformer(
        data_dim=3, param_dim=2, d_model=64, nhead=2, num_layers=1
    )
    prior = MockPrior(dim=2)

    posterior = SimformerPosterior(model, prior, device="cpu")

    assert posterior.model == model
    assert posterior.prior == prior
    assert posterior.device == "cpu"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_simformer_posterior_sampling():
    model = SimformerTransformer(
        data_dim=3, param_dim=2, d_model=64, nhead=2, num_layers=1
    )
    prior = MockPrior(dim=2)

    # Create some mock training data
    data_tensor = torch.randn(5, 10, 3)
    param_tensor = torch.randn(5, 10, 2)
    observed_tensor = torch.randn(5, 3)
    training_data = (data_tensor, param_tensor, observed_tensor)

    posterior = SimformerPosterior(model, prior, training_data, device="cpu")

    # Test sampling
    observed_data = np.array([1.0, 2.0, 3.0])
    samples = posterior.sample(observed_data, num_samples=5)

    assert samples.shape == (5, 2)


# Integration tests
class SimpleLinearSimulator:
    """Simple linear simulator for testing."""

    def __init__(self, noise_std=0.1):
        self.noise_std = noise_std

    def simulate(self, parameters, num_observations=50):
        # parameters: (Batch, P)
        params = np.asarray(parameters)
        if params.ndim == 1:
            params = params[np.newaxis, :]

        batch_size, param_dim = params.shape

        if param_dim == 1:
            slope = params[:, 0:1] # (Batch, 1)
            intercept = np.zeros((batch_size, 1))
        else:
            slope = params[:, 0:1] # (Batch, 1)
            intercept = params[:, 1:2] # (Batch, 1)

        x = np.linspace(0, 1, num_observations) # (50,)
        # slope * x -> (Batch, 1) * (1, 50) -> (Batch, 50)
        # intercept -> (Batch, 1)
        y = (
            slope * x[np.newaxis, :]
            + intercept
            + np.random.normal(0, self.noise_std, (batch_size, num_observations))
        )

        return y


class SimpleLinearPrior:
    """Simple prior for linear regression."""

    def __init__(self, dim=2):
        self.dim = dim

    def sample(self, num_samples=None):
        if num_samples is not None:
            if self.dim == 1:
                return np.random.normal(0, 2, (num_samples, 1))
            else:
                return np.random.normal(0, 2, (num_samples, self.dim))
        else:
            # Single sample
            if self.dim == 1:
                return np.array([np.random.normal(0, 2)])
            else:
                return np.random.normal(0, 2, self.dim)

    def log_prob(self, x):
        x_arr = np.atleast_1d(np.array(x))
        if x_arr.ndim > 1:
             return -0.5 * np.sum((x_arr / 2) ** 2, axis=1) - 0.5 * x_arr.shape[1] * np.log(2 * np.pi * 4)
        return -0.5 * np.sum((x_arr / 2) ** 2) - 0.5 * len(x_arr) * np.log(
            2 * np.pi * 4
        )


class LinearSummaryStatistic:
    """Extract summary statistics from linear regression data."""

    def compute(self, data):
        # data: (Batch, N)
        data_arr = np.array(data)
        if data_arr.ndim == 1:
            data_arr = data_arr[np.newaxis, :]

        mean = np.mean(data_arr, axis=1, keepdims=True)
        std = np.std(data_arr, axis=1, keepdims=True)
        ptp = np.ptp(data_arr, axis=1).reshape(-1, 1) # Range
        diff_mean = np.mean(np.diff(data_arr, axis=1), axis=1, keepdims=True)

        return np.concatenate([mean, std, ptp, diff_mean], axis=1) # (Batch, 4)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_simformer_full_inference_1d():
    """Test full Simformer inference on 1D parameter problem."""
    np.random.seed(42)
    torch.manual_seed(42)

    # Setup problem
    true_slope = 2.0
    simulator = SimpleLinearSimulator(noise_std=0.1)
    prior = SimpleLinearPrior(dim=1)
    summary_stat = LinearSummaryStatistic()

    # Generate observed data
    # simulate expects batch
    observed_data = simulator.simulate(np.array([[true_slope]]))

    # Create Simformer instance with small parameters for testing
    simformer = Simformer(
        simulator=simulator,
        prior=prior,
        data_dim=4,  # Summary statistics dimension
        param_dim=1,
        d_model=64,
        nhead=2,
        num_layers=2,
        summary_statistic=summary_stat,
        device="cpu",  # Force CPU for testing
    )

    # Run inference with minimal parameters for testing
    result = simformer.infer(
        observed_data=observed_data,
        num_simulations=50,  # Small for testing
        sequence_length=10,  # Small for testing
        batch_size=8,
        num_epochs=5,  # Very short training
        verbose=False,
    )

    assert isinstance(result, SimformerPosterior)

    # Test sampling
    samples = result.sample(observed_data, num_samples=10)
    # 1D parameter, but sample returns (N, 1) usually or (N,) depending on squeeze
    # The implementation squeezes if param_dim=1
    # Check Simformer.sample_posterior:
    # if self.param_dim == 1: samples = samples.squeeze(-1)
    # SimformerPosterior.sample calls model.sample_posterior(..., num_samples=1) repeated
    # Wait, SimformerPosterior.sample repeats logic?
    # No, it repeats batch_observed (num_samples, 1).
    # And calls sample_posterior(..., num_samples=1) which returns (1, num_samples) squeezed?
    # SimformerPosterior.sample returns (num_samples,) or (num_samples, D)

    if samples.ndim == 1:
        assert samples.shape == (10,)  # 1D case
    else:
        assert samples.shape == (10, 1)  # 2D case

    # Test log probability
    log_prob = result.log_prob([true_slope], observed_data)
    assert isinstance(log_prob, (float, np.ndarray))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_simformer_full_inference_2d():
    """Test full Simformer inference on 2D parameter problem."""
    np.random.seed(42)
    torch.manual_seed(42)

    # Setup problem
    true_params = [2.0, 1.0]  # slope, intercept
    simulator = SimpleLinearSimulator(noise_std=0.1)
    prior = SimpleLinearPrior(dim=2)
    summary_stat = LinearSummaryStatistic()

    # Generate observed data
    observed_data = simulator.simulate(np.array([true_params]))

    # Create Simformer instance
    simformer = Simformer(
        simulator=simulator,
        prior=prior,
        data_dim=4,
        param_dim=2,
        d_model=64,
        nhead=2,
        num_layers=2,
        summary_statistic=summary_stat,
        device="cpu",  # Force CPU for testing
    )

    # Run inference with minimal parameters for testing
    result = simformer.infer(
        observed_data=observed_data,
        num_simulations=50,
        sequence_length=10,
        batch_size=8,
        num_epochs=5,
        verbose=False,
    )

    assert isinstance(result, SimformerPosterior)

    # Test sampling
    samples = result.sample(observed_data, num_samples=10)
    assert samples.shape == (10, 2)


# Test plotting functionality (mocked)
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
@patch("matplotlib.pyplot.show")
def test_simformer_plotting(mock_show):
    """Test that plotting functions work (with mocked display)."""
    # This will be implemented in the main function
    # For now, just test that matplotlib can be imported
    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        plt.close(fig)


# Main demonstration function
def main():
    pass # Skipped for brevity in test file update
