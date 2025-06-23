import pytest
import numpy as np
from unittest.mock import Mock, patch
from scisbi.inference.ABC import ABCRejectionSampling, ABCPosterior
import io
import sys

# Mock classes for basic testing
class MockSimulator:
    def __init__(self, return_value=None):
        self.return_value = return_value or [1.0, 2.0]
    
    def simulate(self, parameters, **kwargs):
        return self.return_value


class MockPrior:
    def __init__(self, sample_value=None):
        self.sample_value = sample_value or 0.5
        
    def sample(self):
        return self.sample_value
        
    def log_prob(self, x):
        return -0.5 * x**2


class MockSummaryStatistic:
    def compute(self, data):
        return np.mean(data)


class MockDistanceFunction:
    def __init__(self, return_value=0.1):
        self.return_value = return_value
        
    def __call__(self, x, y):
        return self.return_value


# Test ABCRejectionSampling initialization
def test_abc_initialization_valid():
    simulator = MockSimulator()
    prior = MockPrior()
    distance_fn = MockDistanceFunction()
    tolerance = 0.1
    
    abc = ABCRejectionSampling(simulator, prior, distance_fn, tolerance)
    
    assert abc.simulator == simulator
    assert abc.prior == prior
    assert abc.distance_function == distance_fn
    assert abc.tolerance == tolerance
    assert abc.max_attempts == 1000000
    assert abc.verbose == False


def test_abc_initialization_with_summary_statistic():
    simulator = MockSimulator()
    prior = MockPrior()
    distance_fn = MockDistanceFunction()
    tolerance = 0.1
    summary_stat = MockSummaryStatistic()
    
    abc = ABCRejectionSampling(
        simulator, prior, distance_fn, tolerance, 
        summary_statistic=summary_stat, max_attempts=5000, verbose=True
    )
    
    assert abc.summary_statistic == summary_stat
    assert abc.max_attempts == 5000
    assert abc.verbose == True


def test_abc_initialization_invalid_distance_function():
    simulator = MockSimulator()
    prior = MockPrior()
    tolerance = 0.1
    
    with pytest.raises(TypeError, match="distance_function must be callable"):
        ABCRejectionSampling(simulator, prior, "not_callable", tolerance)


def test_abc_initialization_negative_tolerance():
    simulator = MockSimulator()
    prior = MockPrior()
    distance_fn = MockDistanceFunction()
    
    with pytest.raises(ValueError, match="tolerance must be non-negative"):
        ABCRejectionSampling(simulator, prior, distance_fn, -0.1)


# Test ABCRejectionSampling.infer method
def test_abc_infer_basic():
    simulator = MockSimulator([1.0, 2.0])
    prior = MockPrior(0.5)
    distance_fn = MockDistanceFunction(0.05)  # Always accept
    tolerance = 0.1
    
    abc = ABCRejectionSampling(simulator, prior, distance_fn, tolerance)
    observed_data = [1.1, 1.9]
    
    result = abc.infer(observed_data, num_simulations=3)
    
    assert isinstance(result, ABCPosterior)
    assert len(result.samples) == 3
    assert len(result.distances) == 3
    assert result.tolerance == tolerance
    assert all(d <= tolerance for d in result.distances)


def test_abc_infer_with_summary_statistic():
    simulator = MockSimulator([1.0, 2.0])
    prior = MockPrior(0.5)
    distance_fn = MockDistanceFunction(0.05)
    tolerance = 0.1
    summary_stat = None
    
    abc = ABCRejectionSampling(simulator, prior, distance_fn, tolerance, summary_stat)
    observed_data = [1.1, 1.9]
    
    result = abc.infer(observed_data, num_simulations=2)
    
    assert len(result.samples) == 2


def test_abc_infer_invalid_num_simulations():
    simulator = MockSimulator()
    prior = MockPrior()
    distance_fn = MockDistanceFunction()
    tolerance = 0.1
    
    abc = ABCRejectionSampling(simulator, prior, distance_fn, tolerance)
    
    with pytest.raises(ValueError, match="num_simulations must be positive"):
        abc.infer([1, 2], num_simulations=0)


def test_abc_infer_max_attempts_exceeded():
    simulator = MockSimulator()
    prior = MockPrior()
    distance_fn = MockDistanceFunction(1.0)  # Always reject
    tolerance = 0.1
    
    abc = ABCRejectionSampling(simulator, prior, distance_fn, tolerance, max_attempts=10)
    
    with pytest.raises(RuntimeError, match="Maximum attempts.*exceeded"):
        abc.infer([1, 2], num_simulations=5)


def test_abc_infer_with_kwargs_override():
    simulator = MockSimulator()
    prior = MockPrior()
    distance_fn = MockDistanceFunction(0.05)
    tolerance = 0.1
    
    abc = ABCRejectionSampling(simulator, prior, distance_fn, tolerance, verbose=False)
    
    # Test with verbose override
    result = abc.infer([1, 2], num_simulations=1, verbose=True, tolerance=0.2)
    
    assert len(result.samples) == 1
    assert result.tolerance == 0.2


# Test ABCPosterior class
def test_abc_posterior_initialization():
    samples = [0.1, 0.2, 0.3]
    distances = [0.05, 0.08, 0.02]
    tolerance = 0.1
    num_attempts = 100
    
    posterior = ABCPosterior(samples, distances, tolerance, num_attempts)
    
    assert posterior.samples == samples
    assert posterior.distances == distances
    assert posterior.tolerance == tolerance
    assert posterior.num_attempts == num_attempts
    assert posterior.acceptance_rate == 0.03


def test_abc_posterior_sample_single():
    samples = [0.1, 0.2, 0.3]
    distances = [0.05, 0.08, 0.02]
    posterior = ABCPosterior(samples, distances, 0.1, 100)
    
    np.random.seed(42)
    sample = posterior.sample(1)
    assert sample in samples


def test_abc_posterior_sample_multiple():
    samples = [0.1, 0.2, 0.3]
    distances = [0.05, 0.08, 0.02]
    posterior = ABCPosterior(samples, distances, 0.1, 100)
    
    np.random.seed(42)
    sample_list = posterior.sample(5)
    assert len(sample_list) == 5
    assert all(s in samples for s in sample_list)


def test_abc_posterior_get_methods():
    samples = [0.1, 0.2, 0.3]
    distances = [0.05, 0.08, 0.02]
    posterior = ABCPosterior(samples, distances, 0.1, 100)
    
    assert posterior.get_samples() == samples
    assert posterior.get_distances() == distances
    # Ensure they return copies
    assert posterior.get_samples() is not posterior.samples


def test_abc_posterior_summary_statistics():
    samples = [0.1, 0.2, 0.3]
    distances = [0.05, 0.08, 0.02]
    posterior = ABCPosterior(samples, distances, 0.1, 100)
    
    stats = posterior.summary_statistics()
    
    assert stats['num_samples'] == 3
    assert stats['acceptance_rate'] == 0.03
    assert stats['tolerance'] == 0.1
    assert stats['num_attempts'] == 100
    assert stats['mean_distance'] == np.mean(distances)
    assert stats['max_distance'] == max(distances)
    assert stats['min_distance'] == min(distances)


def test_abc_posterior_len():
    samples = [0.1, 0.2, 0.3]
    distances = [0.05, 0.08, 0.02]
    posterior = ABCPosterior(samples, distances, 0.1, 100)
    
    assert len(posterior) == 3


def test_abc_posterior_repr():
    samples = [0.1, 0.2, 0.3]
    distances = [0.05, 0.08, 0.02]
    posterior = ABCPosterior(samples, distances, 0.1, 100)
    
    repr_str = repr(posterior)
    assert "ABCPosterior" in repr_str
    assert "num_samples=3" in repr_str
    assert "acceptance_rate=0.0300" in repr_str
    assert "tolerance=0.1" in repr_str


@patch('matplotlib.pyplot.show')
def test_abc_posterior_plot_1d(mock_show):
    samples = [0.1, 0.2, 0.3, 0.4, 0.5]
    distances = [0.05, 0.08, 0.02, 0.07, 0.03]
    posterior = ABCPosterior(samples, distances, 0.1, 100)
    
    posterior.plot()
    mock_show.assert_called_once()


@patch('matplotlib.pyplot.show')
def test_abc_posterior_plot_multidimensional(mock_show):
    samples = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    distances = [0.05, 0.08, 0.02]
    posterior = ABCPosterior(samples, distances, 0.1, 100)
    
    posterior.plot(parameter_index=0)
    mock_show.assert_called_once()


def test_abc_posterior_plot_no_samples():
    posterior = ABCPosterior([], [], 0.1, 100)
    
    with pytest.raises(ValueError, match="No samples available to plot"):
        posterior.plot()


def test_abc_posterior_plot_multidimensional_no_index():
    samples = [[0.1, 0.2], [0.3, 0.4]]
    distances = [0.05, 0.08]
    posterior = ABCPosterior(samples, distances, 0.1, 100)
    
    with pytest.raises(ValueError, match="Samples are multi-dimensional. Specify parameter_index"):
        posterior.plot()


# Integration tests with Gaussian distributions
class GaussianSimulator:
    def __init__(self, true_params):
        self.true_params = true_params
        
    def simulate(self, parameters, num_simulations=100):
        if len(parameters) == 2:  # mean, std
            mean, std = parameters
            return np.random.normal(mean, std, num_simulations)
        elif len(parameters) == 3:  # mean1, mean2, std
            mean1, mean2, std = parameters
            return np.column_stack([
                np.random.normal(mean1, std, num_simulations),
                np.random.normal(mean2, std, num_simulations)
            ])
        else:
            raise ValueError("Unsupported parameter dimension")


class GaussianPrior:
    def __init__(self, dim):
        self.dim = dim
        
    def sample(self):
        if self.dim == 2:
            return [np.random.uniform(-5, 5), np.random.uniform(0.1, 3)]
        elif self.dim == 3:
            return [np.random.uniform(-5, 5), np.random.uniform(-5, 5), np.random.uniform(0.1, 3)]
            
    def log_prob(self, x):
        return 0.0  # Uniform prior


def euclidean_distance(x, y):
    """Euclidean distance between sample means and standard deviations."""
    x_arr = np.array(x)
    y_arr = np.array(y)
    
    if x_arr.ndim == 1 and y_arr.ndim == 1:
        # 1D case - compare mean and std
        x_stats = [np.mean(x_arr), np.std(x_arr)]
        y_stats = [np.mean(y_arr), np.std(y_arr)]
    else:
        # 2D case - compare means and std
        x_stats = [np.mean(x_arr[:, 0]), np.mean(x_arr[:, 1]), np.std(x_arr)]
        y_stats = [np.mean(y_arr[:, 0]), np.mean(y_arr[:, 1]), np.std(y_arr)]
    
    return np.linalg.norm(np.array(x_stats) - np.array(y_stats))


def test_abc_inference_2d_gaussian():
    """Test ABC inference on 2D Gaussian (mean, std) parameters."""
    np.random.seed(42)
    
    true_params = [2.0, 1.5]  # true mean=2.0, true std=1.5
    simulator = GaussianSimulator(true_params)
    prior = GaussianPrior(dim=2)
    
    # Generate observed data
    observed_data = np.random.normal(true_params[0], true_params[1], 100)
    
    abc = ABCRejectionSampling(
        simulator=simulator,
        prior=prior,
        distance_function=euclidean_distance,
        tolerance=1.0,
        max_attempts=1000
    )
    
    result = abc.infer(observed_data, num_simulations=20)
    
    assert len(result.samples) == 20
    assert all(len(sample) == 2 for sample in result.samples)
    assert result.acceptance_rate > 0
    
    # Check that posterior samples are reasonable
    mean_estimates = [sample[0] for sample in result.samples]
    std_estimates = [sample[1] for sample in result.samples]
    
    assert np.mean(mean_estimates) == pytest.approx(true_params[0], abs=1.0)
    assert np.mean(std_estimates) == pytest.approx(true_params[1], abs=1.0)


def test_abc_inference_3d_gaussian():
    """Test ABC inference on 3D Gaussian (mean1, mean2, std) parameters."""
    np.random.seed(123)
    
    true_params = [1.0, -1.0, 2.0]  # true mean1=1.0, true mean2=-1.0, true std=2.0
    simulator = GaussianSimulator(true_params)
    prior = GaussianPrior(dim=3)
    
    # Generate observed data
    observed_data = np.column_stack([
        np.random.normal(true_params[0], true_params[2], 100),
        np.random.normal(true_params[1], true_params[2], 100)
    ])
    
    abc = ABCRejectionSampling(
        simulator=simulator,
        prior=prior,
        distance_function=euclidean_distance,
        tolerance=1.5,
        max_attempts=1000
    )
    
    result = abc.infer(observed_data, num_simulations=15)
    
    assert len(result.samples) == 15
    assert all(len(sample) == 3 for sample in result.samples)
    assert result.acceptance_rate > 0
    
    # Check that posterior samples are reasonable
    mean1_estimates = [sample[0] for sample in result.samples]
    mean2_estimates = [sample[1] for sample in result.samples]
    std_estimates = [sample[2] for sample in result.samples]
    
    assert np.mean(mean1_estimates) == pytest.approx(true_params[0], abs=1.5)
    assert np.mean(mean2_estimates) == pytest.approx(true_params[1], abs=1.5)
    assert np.mean(std_estimates) == pytest.approx(true_params[2], abs=1.5)


def test_abc_inference_with_summary_statistic_integration():
    """Test ABC inference using summary statistics with Gaussian example."""
    np.random.seed(42)
    
    class MeanStdSummary:
        def compute(self, data):
            return [np.mean(data), np.std(data)]
    
    true_params = [0.0, 1.0]
    simulator = GaussianSimulator(true_params)
    prior = GaussianPrior(dim=2)
    summary_stat = MeanStdSummary()
    
    observed_data = np.random.normal(true_params[0], true_params[1], 100)
    
    def summary_distance(x, y):
        return np.linalg.norm(np.array(x) - np.array(y))
    
    abc = ABCRejectionSampling(
        simulator=simulator,
        prior=prior,
        distance_function=summary_distance,
        tolerance=0.5,
        summary_statistic=summary_stat,
        max_attempts=500
    )
    
    result = abc.infer(observed_data, num_simulations=10)
    
    assert len(result.samples) == 10
    assert result.acceptance_rate > 0


def test_abc_inference_verbose_mode():
    """Test ABC inference with verbose output."""
    np.random.seed(42)
    
    simulator = MockSimulator([1.0, 2.0])
    prior = MockPrior(0.5)
    distance_fn = MockDistanceFunction(0.05)
    tolerance = 0.1
    
    abc = ABCRejectionSampling(simulator, prior, distance_fn, tolerance, verbose=True)
    
    # Capture stdout to verify verbose output
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        result = abc.infer([1, 2], num_simulations=5)
        output = captured_output.getvalue()
        
        assert "Starting ABC rejection sampling" in output
        assert "ABC rejection sampling completed" in output
        assert len(result.samples) == 5
    finally:
        sys.stdout = sys.__stdout__