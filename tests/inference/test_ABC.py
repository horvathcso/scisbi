import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch
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
    assert not abc.verbose


def test_abc_initialization_with_summary_statistic():
    simulator = MockSimulator()
    prior = MockPrior()
    distance_fn = MockDistanceFunction()
    tolerance = 0.1
    summary_stat = MockSummaryStatistic()

    abc = ABCRejectionSampling(
        simulator,
        prior,
        distance_fn,
        tolerance,
        summary_statistic=summary_stat,
        max_attempts=5000,
        verbose=True,
    )

    assert abc.summary_statistic == summary_stat
    assert abc.max_attempts == 5000
    assert abc.verbose


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

    abc = ABCRejectionSampling(
        simulator, prior, distance_fn, tolerance, max_attempts=10
    )

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

    assert stats["num_samples"] == 3
    assert stats["acceptance_rate"] == 0.03
    assert stats["tolerance"] == 0.1
    assert stats["num_attempts"] == 100
    assert stats["mean_distance"] == np.mean(distances)
    assert stats["max_distance"] == max(distances)
    assert stats["min_distance"] == min(distances)


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


@patch("matplotlib.pyplot.show")
def test_abc_posterior_plot_1d(mock_show):
    samples = [0.1, 0.2, 0.3, 0.4, 0.5]
    distances = [0.05, 0.08, 0.02, 0.07, 0.03]
    posterior = ABCPosterior(samples, distances, 0.1, 100)

    posterior.plot()
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
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

    with pytest.raises(
        ValueError, match="Samples are multi-dimensional. Specify parameter_index"
    ):
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
            return np.column_stack(
                [
                    np.random.normal(mean1, std, num_simulations),
                    np.random.normal(mean2, std, num_simulations),
                ]
            )
        else:
            raise ValueError("Unsupported parameter dimension")


class GaussianPrior:
    def __init__(self, dim):
        self.dim = dim

    def sample(self):
        if self.dim == 2:
            return [np.random.uniform(-5, 5), np.random.uniform(0.1, 3)]
        elif self.dim == 3:
            return [
                np.random.uniform(-5, 5),
                np.random.uniform(-5, 5),
                np.random.uniform(0.1, 3),
            ]

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
        max_attempts=1000,
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
    observed_data = np.column_stack(
        [
            np.random.normal(true_params[0], true_params[2], 100),
            np.random.normal(true_params[1], true_params[2], 100),
        ]
    )

    abc = ABCRejectionSampling(
        simulator=simulator,
        prior=prior,
        distance_function=euclidean_distance,
        tolerance=1.5,
        max_attempts=1000,
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
        max_attempts=500,
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


# Main execution with plotting
def main():
    """
    Main function to demonstrate ABC rejection sampling functionality with plots.
    This runs multiple examples and generates comprehensive visualizations.
    """
    print("=" * 60)
    print("ABC Rejection Sampling Demonstration and Analysis")
    print("=" * 60)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Example 1: Simple 1D Gaussian parameter inference
    print("\n1. SINGLE PARAMETER GAUSSIAN INFERENCE")
    print("-" * 40)

    # True parameter: mean of a Gaussian with known std=1
    true_mean = 2.5
    true_std = 1.0
    print(f"True parameters: mean={true_mean}, std={true_std} (std fixed)")

    # Generate observed data
    n_obs = 500
    observed_data_1d = np.random.normal(true_mean, true_std, n_obs)
    print(f"Generated {n_obs} observed data points")
    print(
        f"Observed data statistics: mean={np.mean(observed_data_1d):.3f}, std={np.std(observed_data_1d):.3f}"
    )

    # Simple 1D simulator and prior for mean estimation
    class Simple1DSimulator:
        def simulate(self, parameters, num_simulations=100):
            mean = parameters[0] if isinstance(parameters, list) else parameters
            return np.random.normal(mean, 1.0, num_simulations)  # Fixed std=1

    class Simple1DPrior:
        def sample(self):
            return np.random.uniform(-2, 6)  # Uniform prior on mean

        def log_prob(self, x):
            return 0.0 if -2 <= x <= 6 else -np.inf

    def simple_distance(x, y):
        return abs(np.mean(x) - np.mean(y))

    # Run ABC for different tolerance values
    tolerances = [0.1, 0.3, 0.5, 1.0]
    results_1d = {}

    for tol in tolerances:
        print(f"\nRunning ABC with tolerance = {tol}")
        abc_1d = ABCRejectionSampling(
            simulator=Simple1DSimulator(),
            prior=Simple1DPrior(),
            distance_function=simple_distance,
            tolerance=tol,
            verbose=True,
            max_attempts=10000,
        )

        try:
            result = abc_1d.infer(observed_data_1d, num_simulations=200)
            results_1d[tol] = result
            print(
                f"Success: {len(result)} samples, acceptance rate: {result.acceptance_rate:.4f}"
            )
        except RuntimeError as e:
            print(f"Failed: {e}")
            results_1d[tol] = None

    # Plot tolerance comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, tol in enumerate(tolerances):
        if results_1d[tol] is not None:
            result = results_1d[tol]
            axes[i].hist(
                result.samples,
                bins=25,
                density=True,
                alpha=0.7,
                color="skyblue",
                edgecolor="black",
            )
            axes[i].axvline(
                true_mean,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"True mean: {true_mean}",
            )
            axes[i].axvline(
                np.mean(result.samples),
                color="blue",
                linestyle="-",
                linewidth=2,
                label=f"ABC mean: {np.mean(result.samples):.3f}",
            )
            axes[i].set_title(
                f"Tolerance = {tol}\nAcceptance rate: {result.acceptance_rate:.4f}"
            )
            axes[i].set_xlabel("Mean Parameter")
            axes[i].set_ylabel("Density")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        else:
            axes[i].text(
                0.5,
                0.5,
                f"Failed\nTolerance = {tol}",
                transform=axes[i].transAxes,
                ha="center",
                va="center",
                fontsize=14,
            )
            axes[i].set_title(f"Tolerance = {tol}")

    plt.suptitle(
        "ABC Rejection Sampling: Effect of Tolerance on Posterior", fontsize=16
    )
    plt.tight_layout()
    plt.show()

    # Example 2: 2D Gaussian parameter inference with detailed analysis
    print("\n2. TWO-PARAMETER GAUSSIAN INFERENCE")
    print("-" * 40)

    true_params_2d = [1.5, 2.0]  # true mean=1.5, true std=2.0
    print(f"True parameters: mean={true_params_2d[0]}, std={true_params_2d[1]}")

    # Generate observed data
    observed_data_2d = np.random.normal(true_params_2d[0], true_params_2d[1], 500)
    print(
        f"Observed data statistics: mean={np.mean(observed_data_2d):.3f}, std={np.std(observed_data_2d):.3f}"
    )

    # Set up 2D ABC
    simulator_2d = GaussianSimulator(true_params_2d)
    prior_2d = GaussianPrior(dim=2)

    abc_2d = ABCRejectionSampling(
        simulator=simulator_2d,
        prior=prior_2d,
        distance_function=euclidean_distance,
        tolerance=0.5,
        verbose=True,
        max_attempts=100000,
    )

    print("\nRunning 2D ABC inference...")
    result_2d = abc_2d.infer(observed_data_2d, num_simulations=300)

    # Extract parameter estimates
    mean_estimates = [sample[0] for sample in result_2d.samples]
    std_estimates = [sample[1] for sample in result_2d.samples]

    print("\n2D ABC Results:")
    print(f"Number of samples: {len(result_2d)}")
    print(f"Acceptance rate: {result_2d.acceptance_rate:.4f}")
    print(
        f"Mean estimate: {np.mean(mean_estimates):.3f} ± {np.std(mean_estimates):.3f}"
    )
    print(f"Std estimate: {np.mean(std_estimates):.3f} ± {np.std(std_estimates):.3f}")

    # Comprehensive 2D plotting
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Marginal distributions
    axes[0, 0].hist(
        mean_estimates,
        bins=30,
        density=True,
        alpha=0.7,
        color="green",
        edgecolor="black",
    )
    axes[0, 0].axvline(
        true_params_2d[0],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"True: {true_params_2d[0]}",
    )
    axes[0, 0].axvline(
        np.mean(mean_estimates),
        color="blue",
        linestyle="-",
        linewidth=2,
        label=f"ABC: {np.mean(mean_estimates):.3f}",
    )
    axes[0, 0].set_title("Posterior: Mean Parameter")
    axes[0, 0].set_xlabel("Mean")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(
        std_estimates,
        bins=30,
        density=True,
        alpha=0.7,
        color="orange",
        edgecolor="black",
    )
    axes[0, 1].axvline(
        true_params_2d[1],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"True: {true_params_2d[1]}",
    )
    axes[0, 1].axvline(
        np.mean(std_estimates),
        color="blue",
        linestyle="-",
        linewidth=2,
        label=f"ABC: {np.mean(std_estimates):.3f}",
    )
    axes[0, 1].set_title("Posterior: Std Parameter")
    axes[0, 1].set_xlabel("Standard Deviation")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Joint distribution
    axes[0, 2].scatter(mean_estimates, std_estimates, alpha=0.6, s=15, color="purple")
    axes[0, 2].scatter(
        true_params_2d[0],
        true_params_2d[1],
        color="red",
        s=100,
        marker="x",
        linewidth=3,
        label="True parameters",
    )
    axes[0, 2].scatter(
        np.mean(mean_estimates),
        np.mean(std_estimates),
        color="blue",
        s=100,
        marker="o",
        label="ABC estimates",
    )
    axes[0, 2].set_title("Joint Posterior Distribution")
    axes[0, 2].set_xlabel("Mean Parameter")
    axes[0, 2].set_ylabel("Std Parameter")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Distance analysis
    axes[1, 0].hist(
        result_2d.distances,
        bins=25,
        density=True,
        alpha=0.7,
        color="red",
        edgecolor="black",
    )
    axes[1, 0].axvline(
        result_2d.tolerance,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Tolerance: {result_2d.tolerance}",
    )
    axes[1, 0].set_title("Distribution of Accepted Distances")
    axes[1, 0].set_xlabel("Distance")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Prior vs Posterior comparison
    prior_samples_mean = []
    prior_samples_std = []
    for _ in range(1000):
        sample = prior_2d.sample()
        prior_samples_mean.append(sample[0])
        prior_samples_std.append(sample[1])

    axes[1, 1].scatter(
        prior_samples_mean,
        prior_samples_std,
        alpha=0.3,
        s=5,
        color="gray",
        label="Prior samples",
    )
    axes[1, 1].scatter(
        mean_estimates,
        std_estimates,
        alpha=0.6,
        s=15,
        color="purple",
        label="Posterior samples",
    )
    axes[1, 1].scatter(
        true_params_2d[0],
        true_params_2d[1],
        color="red",
        s=100,
        marker="x",
        linewidth=3,
        label="True parameters",
    )
    axes[1, 1].set_title("Prior vs Posterior")
    axes[1, 1].set_xlabel("Mean Parameter")
    axes[1, 1].set_ylabel("Std Parameter")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Data comparison
    axes[1, 2].hist(
        observed_data_2d,
        bins=30,
        density=True,
        alpha=0.7,
        color="lightblue",
        edgecolor="black",
        label="Observed data",
    )

    # Sample some posterior predictive datasets
    for i in range(5):
        if i < len(result_2d.samples):
            sample_params = result_2d.samples[i]
            pred_data = np.random.normal(sample_params[0], sample_params[1], 200)
            axes[1, 2].hist(
                pred_data,
                bins=30,
                density=True,
                alpha=0.2,
                color="red",
                histtype="step",
            )

    axes[1, 2].set_title("Data: Observed vs Posterior Predictive")
    axes[1, 2].set_xlabel("Value")
    axes[1, 2].set_ylabel("Density")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle("ABC Rejection Sampling: Complete 2D Analysis", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Example 3: Summary statistics demonstration
    print("\n3. SUMMARY STATISTICS DEMONSTRATION")
    print("-" * 40)

    class ComprehensiveSummary:
        def compute(self, data):
            return [
                np.mean(data),
                np.std(data),
                np.percentile(data, 25),
                np.percentile(data, 75),
            ]

    def comprehensive_distance(x, y):
        return np.linalg.norm(np.array(x) - np.array(y))

    summary_stat = ComprehensiveSummary()

    print("Running ABC with comprehensive summary statistics...")
    abc_summary = ABCRejectionSampling(
        simulator=simulator_2d,
        prior=prior_2d,
        distance_function=comprehensive_distance,
        tolerance=0.8,
        summary_statistic=summary_stat,
        verbose=True,
        max_attempts=100000,
    )

    result_summary = abc_summary.infer(observed_data_2d, num_simulations=200)

    # Compare with and without summary statistics
    mean_estimates_summ = [sample[0] for sample in result_summary.samples]
    std_estimates_summ = [sample[1] for sample in result_summary.samples]

    print("\nSummary Statistics Results:")
    print(f"Number of samples: {len(result_summary)}")
    print(f"Acceptance rate: {result_summary.acceptance_rate:.4f}")
    print(
        f"Mean estimate: {np.mean(mean_estimates_summ):.3f} ± {np.std(mean_estimates_summ):.3f}"
    )
    print(
        f"Std estimate: {np.mean(std_estimates_summ):.3f} ± {np.std(std_estimates_summ):.3f}"
    )

    # Comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].hist(
        mean_estimates,
        bins=25,
        density=True,
        alpha=0.7,
        color="blue",
        edgecolor="black",
        label="Full data",
    )
    axes[0].hist(
        mean_estimates_summ,
        bins=25,
        density=True,
        alpha=0.7,
        color="red",
        edgecolor="black",
        label="Summary statistics",
    )
    axes[0].axvline(
        true_params_2d[0],
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"True: {true_params_2d[0]}",
    )
    axes[0].set_title("Mean Parameter: Full Data vs Summary Statistics")
    axes[0].set_xlabel("Mean")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(
        std_estimates,
        bins=25,
        density=True,
        alpha=0.7,
        color="blue",
        edgecolor="black",
        label="Full data",
    )
    axes[1].hist(
        std_estimates_summ,
        bins=25,
        density=True,
        alpha=0.7,
        color="red",
        edgecolor="black",
        label="Summary statistics",
    )
    axes[1].axvline(
        true_params_2d[1],
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"True: {true_params_2d[1]}",
    )
    axes[1].set_title("Std Parameter: Full Data vs Summary Statistics")
    axes[1].set_xlabel("Standard Deviation")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("ABC: Impact of Summary Statistics", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Summary statistics table
    print("\nSummary Statistics Comparison:")
    print(
        f"{'Method':<20} {'Acceptance Rate':<15} {'Mean Error':<12} {'Std Error':<12}"
    )
    print("-" * 65)

    mean_error_full = abs(np.mean(mean_estimates) - true_params_2d[0])
    std_error_full = abs(np.mean(std_estimates) - true_params_2d[1])
    mean_error_summ = abs(np.mean(mean_estimates_summ) - true_params_2d[0])
    std_error_summ = abs(np.mean(std_estimates_summ) - true_params_2d[1])

    print(
        f"{'Full Data':<20} {result_2d.acceptance_rate:<15.4f} {mean_error_full:<12.4f} {std_error_full:<12.4f}"
    )
    print(
        f"{'Summary Stats':<20} {result_summary.acceptance_rate:<15.4f} {mean_error_summ:<12.4f} {std_error_summ:<12.4f}"
    )

    print("\n" + "=" * 60)
    print("ABC Rejection Sampling demonstration completed!")
    print("Key insights:")
    print("1. Lower tolerance → More accurate but lower acceptance rate")
    print("2. ABC can recover true parameters from simulated data")
    print("3. Summary statistics can improve efficiency")
    print("4. Joint posterior shows parameter correlations")
    print("=" * 60)


if __name__ == "__main__":
    main()
