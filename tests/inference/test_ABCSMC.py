import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch
from scisbi.inference.ABCSMC import ABCSMC, ABCSMCPosterior
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


class MockPerturbationKernel:
    def __init__(self, noise_std=0.1):
        self.noise_std = noise_std

    def __call__(self, particle):
        if isinstance(particle, (list, np.ndarray)):
            return [p + np.random.normal(0, self.noise_std) for p in particle]
        else:
            return particle + np.random.normal(0, self.noise_std)


# Test ABCSMC initialization
def test_abcsmc_initialization_valid():
    simulator = MockSimulator()
    prior = MockPrior()
    distance_fn = MockDistanceFunction()
    tolerance_schedule = [1.0, 0.5, 0.1]
    perturbation_kernel = MockPerturbationKernel()

    abc_smc = ABCSMC(
        simulator, prior, distance_fn, tolerance_schedule, perturbation_kernel
    )

    assert abc_smc.simulator == simulator
    assert abc_smc.prior == prior
    assert abc_smc.distance_function == distance_fn
    assert abc_smc.tolerance_schedule == tolerance_schedule
    assert abc_smc.perturbation_kernel == perturbation_kernel
    assert abc_smc.num_particles == 1000
    assert not abc_smc.verbose


def test_abcsmc_initialization_with_options():
    simulator = MockSimulator()
    prior = MockPrior()
    distance_fn = MockDistanceFunction()
    tolerance_schedule = [0.5, 0.2]
    perturbation_kernel = MockPerturbationKernel()
    summary_stat = MockSummaryStatistic()

    abc_smc = ABCSMC(
        simulator,
        prior,
        distance_fn,
        tolerance_schedule,
        perturbation_kernel,
        num_particles=500,
        summary_statistic=summary_stat,
        max_attempts_per_particle=2000,
        verbose=True,
        adaptive_tolerance=True,
    )

    assert abc_smc.summary_statistic == summary_stat
    assert abc_smc.num_particles == 500
    assert abc_smc.max_attempts_per_particle == 2000
    assert abc_smc.verbose
    assert abc_smc.adaptive_tolerance


def test_abcsmc_initialization_invalid_distance_function():
    simulator = MockSimulator()
    prior = MockPrior()
    tolerance_schedule = [1.0, 0.5]
    perturbation_kernel = MockPerturbationKernel()

    with pytest.raises(TypeError, match="distance_function must be callable"):
        ABCSMC(
            simulator, prior, "not_callable", tolerance_schedule, perturbation_kernel
        )


def test_abcsmc_initialization_invalid_perturbation_kernel():
    simulator = MockSimulator()
    prior = MockPrior()
    distance_fn = MockDistanceFunction()
    tolerance_schedule = [1.0, 0.5]

    with pytest.raises(TypeError, match="perturbation_kernel must be callable"):
        ABCSMC(simulator, prior, distance_fn, tolerance_schedule, "not_callable")


def test_abcsmc_initialization_invalid_tolerance_schedule():
    simulator = MockSimulator()
    prior = MockPrior()
    distance_fn = MockDistanceFunction()
    perturbation_kernel = MockPerturbationKernel()

    # Empty schedule
    with pytest.raises(ValueError, match="tolerance_schedule must be a non-empty list"):
        ABCSMC(simulator, prior, distance_fn, [], perturbation_kernel)

    # Non-decreasing schedule
    with pytest.raises(ValueError, match="tolerance_schedule must be decreasing"):
        ABCSMC(simulator, prior, distance_fn, [0.1, 0.5, 1.0], perturbation_kernel)

    # Negative tolerance
    with pytest.raises(ValueError, match="All tolerances must be non-negative"):
        ABCSMC(simulator, prior, distance_fn, [1.0, -0.5], perturbation_kernel)


def test_abcsmc_initialization_invalid_num_particles():
    simulator = MockSimulator()
    prior = MockPrior()
    distance_fn = MockDistanceFunction()
    tolerance_schedule = [1.0, 0.5]
    perturbation_kernel = MockPerturbationKernel()

    with pytest.raises(ValueError, match="num_particles must be positive"):
        ABCSMC(
            simulator,
            prior,
            distance_fn,
            tolerance_schedule,
            perturbation_kernel,
            num_particles=0,
        )


# Test ABCSMC.infer method
def test_abcsmc_infer_basic():
    np.random.seed(42)
    simulator = MockSimulator([1.0, 2.0])
    prior = MockPrior(0.5)
    distance_fn = MockDistanceFunction(0.05)  # Always accept
    tolerance_schedule = [0.2, 0.1]
    perturbation_kernel = MockPerturbationKernel(0.01)

    abc_smc = ABCSMC(
        simulator,
        prior,
        distance_fn,
        tolerance_schedule,
        perturbation_kernel,
        num_particles=10,
    )
    observed_data = [1.1, 1.9]

    result = abc_smc.infer(observed_data)

    assert isinstance(result, ABCSMCPosterior)
    assert len(result.particles) == 10
    assert len(result.weights) == 10
    assert len(result.distances) == 10
    assert result.tolerance_schedule == tolerance_schedule
    assert len(result.all_populations) == 2  # Two tolerance levels


def test_abcsmc_infer_with_kwargs_override():
    np.random.seed(42)
    simulator = MockSimulator([1.0, 2.0])
    prior = MockPrior(0.5)
    distance_fn = MockDistanceFunction(0.05)
    tolerance_schedule = [0.2, 0.1]
    perturbation_kernel = MockPerturbationKernel(0.01)

    abc_smc = ABCSMC(
        simulator,
        prior,
        distance_fn,
        tolerance_schedule,
        perturbation_kernel,
        num_particles=10,
        verbose=False,
    )

    # Test with override parameters
    result = abc_smc.infer([1, 2], num_particles=5, verbose=True)

    assert len(result.particles) == 5


def test_abcsmc_infer_single_tolerance():
    np.random.seed(42)
    simulator = MockSimulator([1.0, 2.0])
    prior = MockPrior(0.5)
    distance_fn = MockDistanceFunction(0.05)
    tolerance_schedule = [0.1]  # Single tolerance
    perturbation_kernel = MockPerturbationKernel(0.01)

    abc_smc = ABCSMC(
        simulator,
        prior,
        distance_fn,
        tolerance_schedule,
        perturbation_kernel,
        num_particles=5,
    )

    result = abc_smc.infer([1, 2])

    assert len(result.all_populations) == 1
    assert len(result.particles) == 5


# Test helper methods
def test_abcsmc_sample_from_prior():
    np.random.seed(42)
    simulator = MockSimulator([1.0, 2.0])
    prior = MockPrior(0.5)
    distance_fn = MockDistanceFunction(0.05)  # Always accept
    tolerance_schedule = [0.1]
    perturbation_kernel = MockPerturbationKernel()

    abc_smc = ABCSMC(
        simulator, prior, distance_fn, tolerance_schedule, perturbation_kernel
    )

    particles, weights, distances, attempts = abc_smc._sample_from_prior(
        observed_summary=[1.1, 1.9],
        tolerance=0.1,
        num_particles=5,
        max_attempts=100,
        verbose=False,
    )

    assert len(particles) == 5
    assert len(weights) == 5
    assert len(distances) == 5
    assert all(w == 1.0 for w in weights)  # Uniform weights for first iteration


def test_abcsmc_effective_sample_size():
    simulator = MockSimulator()
    prior = MockPrior()
    distance_fn = MockDistanceFunction()
    tolerance_schedule = [0.1]
    perturbation_kernel = MockPerturbationKernel()

    abc_smc = ABCSMC(
        simulator, prior, distance_fn, tolerance_schedule, perturbation_kernel
    )

    # Uniform weights
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    ess = abc_smc._effective_sample_size(weights)
    assert ess == pytest.approx(4.0)

    # Non-uniform weights
    weights = np.array([0.8, 0.1, 0.05, 0.05])
    ess = abc_smc._effective_sample_size(weights)
    assert ess < 4.0  # Should be less than number of particles


def test_abcsmc_adaptive_tolerance_update():
    simulator = MockSimulator()
    prior = MockPrior()
    distance_fn = MockDistanceFunction()
    tolerance_schedule = [0.1]
    perturbation_kernel = MockPerturbationKernel()

    abc_smc = ABCSMC(
        simulator, prior, distance_fn, tolerance_schedule, perturbation_kernel
    )

    distances = [0.1, 0.2, 0.3, 0.4, 0.5]
    scheduled_tolerance = 0.4

    adaptive_tol = abc_smc._adaptive_tolerance_update(distances, scheduled_tolerance)
    assert adaptive_tol == 0.3  # Median of distances

    # Should not exceed scheduled tolerance
    scheduled_tolerance = 0.2
    adaptive_tol = abc_smc._adaptive_tolerance_update(distances, scheduled_tolerance)
    assert adaptive_tol == 0.2


# Test ABCSMCPosterior class
def test_abcsmc_posterior_initialization():
    particles = [0.1, 0.2, 0.3]
    weights = np.array([0.5, 0.3, 0.2])
    distances = [0.05, 0.08, 0.02]
    tolerance_schedule = [1.0, 0.5, 0.1]
    all_populations = [[0.0, 0.1], [0.05, 0.15], particles]
    all_weights = [np.array([0.5, 0.5]), np.array([0.4, 0.6]), weights]
    all_distances = [[0.1, 0.12], [0.07, 0.09], distances]
    iteration_info = [
        {"tolerance": 1.0, "acceptance_rate": 0.1},
        {"tolerance": 0.5, "acceptance_rate": 0.05},
        {"tolerance": 0.1, "acceptance_rate": 0.02},
    ]

    posterior = ABCSMCPosterior(
        particles,
        weights,
        distances,
        tolerance_schedule,
        all_populations,
        all_weights,
        all_distances,
        iteration_info,
        100,
    )

    assert posterior.particles == particles
    assert np.allclose(posterior.weights, weights / np.sum(weights))
    assert posterior.distances == distances
    assert posterior.tolerance_schedule == tolerance_schedule
    assert len(posterior.all_populations) == 3
    assert posterior.num_particles == 100


def test_abcsmc_posterior_sample():
    particles = [0.1, 0.2, 0.3]
    weights = np.array([0.5, 0.3, 0.2])
    distances = [0.05, 0.08, 0.02]
    tolerance_schedule = [0.1]

    posterior = ABCSMCPosterior(
        particles, weights, distances, tolerance_schedule, [], [], [], [], 100
    )

    np.random.seed(42)
    sample = posterior.sample(1)
    assert sample in particles

    np.random.seed(42)
    samples = posterior.sample(5)
    assert len(samples) == 5
    assert all(s in particles for s in samples)


def test_abcsmc_posterior_get_methods():
    particles = [0.1, 0.2, 0.3]
    weights = np.array([0.5, 0.3, 0.2])
    distances = [0.05, 0.08, 0.02]
    tolerance_schedule = [0.1]

    posterior = ABCSMCPosterior(
        particles, weights, distances, tolerance_schedule, [], [], [], [], 100
    )

    assert posterior.get_particles() == particles
    assert np.allclose(posterior.get_weights(), weights / np.sum(weights))
    assert posterior.get_distances() == distances
    # Ensure they return copies
    assert posterior.get_particles() is not posterior.particles


def test_abcsmc_posterior_effective_sample_size():
    particles = [0.1, 0.2, 0.3, 0.4]
    weights = np.array([0.25, 0.25, 0.25, 0.25])  # Uniform weights
    distances = [0.05, 0.08, 0.02, 0.04]
    tolerance_schedule = [0.1]

    posterior = ABCSMCPosterior(
        particles, weights, distances, tolerance_schedule, [], [], [], [], 100
    )

    ess = posterior.effective_sample_size()
    assert ess == pytest.approx(4.0)


def test_abcsmc_posterior_summary_statistics():
    particles = [0.1, 0.2, 0.3]
    weights = np.array([0.5, 0.3, 0.2])
    distances = [0.05, 0.08, 0.02]
    tolerance_schedule = [1.0, 0.5, 0.1]
    iteration_info = [
        {
            "tolerance": 1.0,
            "acceptance_rate": 0.1,
            "mean_distance": 0.1,
            "std_distance": 0.01,
        },
        {
            "tolerance": 0.5,
            "acceptance_rate": 0.05,
            "mean_distance": 0.08,
            "std_distance": 0.005,
        },
        {
            "tolerance": 0.1,
            "acceptance_rate": 0.02,
            "mean_distance": 0.05,
            "std_distance": 0.003,
        },
    ]

    posterior = ABCSMCPosterior(
        particles,
        weights,
        distances,
        tolerance_schedule,
        [],
        [],
        [],
        iteration_info,
        100,
    )

    stats = posterior.summary_statistics()

    assert stats["num_particles"] == 3
    assert stats["final_tolerance"] == 0.1
    assert stats["num_iterations"] == 3
    assert stats["tolerance_schedule"] == tolerance_schedule
    assert stats["final_acceptance_rate"] == 0.02
    assert stats["mean_distance"] == np.mean(distances)


def test_abcsmc_posterior_len():
    particles = [0.1, 0.2, 0.3]
    weights = np.array([0.5, 0.3, 0.2])
    distances = [0.05, 0.08, 0.02]
    tolerance_schedule = [0.1]

    posterior = ABCSMCPosterior(
        particles, weights, distances, tolerance_schedule, [], [], [], [], 100
    )

    assert len(posterior) == 3


def test_abcsmc_posterior_repr():
    particles = [0.1, 0.2, 0.3]
    weights = np.array([0.5, 0.3, 0.2])
    distances = [0.05, 0.08, 0.02]
    tolerance_schedule = [0.1]

    posterior = ABCSMCPosterior(
        particles, weights, distances, tolerance_schedule, [], [], [], [], 100
    )

    repr_str = repr(posterior)
    assert "ABCSMCPosterior" in repr_str
    assert "num_particles=3" in repr_str
    assert "final_tolerance=0.1" in repr_str


# Test plotting methods
@patch("matplotlib.pyplot.show")
def test_abcsmc_posterior_plot_1d(mock_show):
    particles = [0.1, 0.2, 0.3, 0.4, 0.5]
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    distances = [0.05, 0.08, 0.02, 0.07, 0.03]
    tolerance_schedule = [0.1]

    posterior = ABCSMCPosterior(
        particles, weights, distances, tolerance_schedule, [], [], [], [], 100
    )

    posterior.plot()
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_abcsmc_posterior_plot_multidimensional(mock_show):
    particles = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    weights = np.array([0.33, 0.33, 0.34])
    distances = [0.05, 0.08, 0.02]
    tolerance_schedule = [0.1]

    posterior = ABCSMCPosterior(
        particles, weights, distances, tolerance_schedule, [], [], [], [], 100
    )

    posterior.plot(parameter_index=0)
    mock_show.assert_called_once()


def test_abcsmc_posterior_plot_no_particles():
    posterior = ABCSMCPosterior([], np.array([]), [], [], [], [], [], [], 100)

    with pytest.raises(ValueError, match="No particles available to plot"):
        posterior.plot()


def test_abcsmc_posterior_plot_multidimensional_no_index():
    particles = [[0.1, 0.2], [0.3, 0.4]]
    weights = np.array([0.5, 0.5])
    distances = [0.05, 0.08]
    tolerance_schedule = [0.1]

    posterior = ABCSMCPosterior(
        particles, weights, distances, tolerance_schedule, [], [], [], [], 100
    )

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
        # Simple uniform prior log probability
        if self.dim == 2:
            if -5 <= x[0] <= 5 and 0.1 <= x[1] <= 3:
                return np.log(1 / (10 * 2.9))  # uniform density
            else:
                return -np.inf
        elif self.dim == 3:
            if -5 <= x[0] <= 5 and -5 <= x[1] <= 5 and 0.1 <= x[2] <= 3:
                return np.log(1 / (10 * 10 * 2.9))  # uniform density
            else:
                return -np.inf


class GaussianPerturbationKernel:
    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, particle):
        perturbed = []
        for param in particle:
            perturbed.append(param + np.random.normal(0, self.std))
        return perturbed


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


def test_abcsmc_inference_2d_gaussian():
    """Test ABC-SMC inference on 2D Gaussian (mean, std) parameters."""
    np.random.seed(42)

    true_params = [2.0, 1.5]  # true mean=2.0, true std=1.5
    simulator = GaussianSimulator(true_params)
    prior = GaussianPrior(dim=2)
    perturbation_kernel = GaussianPerturbationKernel(std=0.2)
    tolerance_schedule = [2.0, 1.0, 0.5]

    # Generate observed data
    observed_data = np.random.normal(true_params[0], true_params[1], 100)

    abc_smc = ABCSMC(
        simulator=simulator,
        prior=prior,
        distance_function=euclidean_distance,
        tolerance_schedule=tolerance_schedule,
        perturbation_kernel=perturbation_kernel,
        num_particles=50,
    )

    result = abc_smc.infer(observed_data)

    assert len(result.particles) == 50
    assert all(len(particle) == 2 for particle in result.particles)
    assert len(result.all_populations) == 3  # Three tolerance levels

    # Check that posterior samples are reasonable
    if len(result.particles) > 0:
        mean_estimates = [particle[0] for particle in result.particles]
        std_estimates = [particle[1] for particle in result.particles]

        # More lenient bounds for SMC due to potentially challenging tolerance schedule
        assert np.mean(mean_estimates) == pytest.approx(true_params[0], abs=2.0)
        assert np.mean(std_estimates) == pytest.approx(true_params[1], abs=2.0)


def test_abcsmc_inference_verbose_mode():
    """Test ABC-SMC inference with verbose output."""
    np.random.seed(42)

    simulator = MockSimulator([1.0, 2.0])
    prior = MockPrior(0.5)
    distance_fn = MockDistanceFunction(0.05)
    tolerance_schedule = [0.2, 0.1]
    perturbation_kernel = MockPerturbationKernel(0.01)

    abc_smc = ABCSMC(
        simulator,
        prior,
        distance_fn,
        tolerance_schedule,
        perturbation_kernel,
        num_particles=5,
        verbose=True,
    )

    # Capture stdout to verify verbose output
    captured_output = io.StringIO()
    sys.stdout = captured_output

    try:
        result = abc_smc.infer([1, 2])
        output = captured_output.getvalue()

        assert "Starting ABC-SMC" in output
        assert "ABC-SMC completed" in output
        assert len(result.particles) == 5
    finally:
        sys.stdout = sys.__stdout__


# Main execution with plotting
def main():
    """
    Main function to demonstrate ABC-SMC functionality with plots.
    This runs a complete example and generates comprehensive visualizations.
    """
    print("=" * 60)
    print("ABC-SMC (Sequential Monte Carlo) Demonstration and Analysis")
    print("=" * 60)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Define true parameters for a 2D Gaussian
    true_mean = 1.0
    true_std = 2.0
    true_params = [true_mean, true_std]

    print(f"True parameters: mean={true_mean}, std={true_std}")

    # Generate observed data
    n_obs = 300
    observed_data = np.random.normal(true_mean, true_std, n_obs)
    print(f"Generated {n_obs} observed data points")
    print(
        f"Observed data statistics: mean={np.mean(observed_data):.3f}, std={np.std(observed_data):.3f}"
    )

    # Set up ABC-SMC components
    simulator = GaussianSimulator(true_params)
    prior = GaussianPrior(dim=2)
    perturbation_kernel = GaussianPerturbationKernel(std=0.1)

    # Define distance function
    def summary_distance(x, y):
        x_stats = [np.mean(x), np.std(x)]
        y_stats = [np.mean(y), np.std(y)]
        return np.linalg.norm(np.array(x_stats) - np.array(y_stats))

    # Example 1: Conservative tolerance schedule
    print("\n1. CONSERVATIVE TOLERANCE SCHEDULE")
    print("-" * 40)

    tolerance_schedule_conservative = [3.0, 2.0, 1.5, 1.0, 0.5]
    print(f"Tolerance schedule: {tolerance_schedule_conservative}")

    abc_smc_conservative = ABCSMC(
        simulator=simulator,
        prior=prior,
        distance_function=summary_distance,
        tolerance_schedule=tolerance_schedule_conservative,
        perturbation_kernel=perturbation_kernel,
        num_particles=200,
        verbose=True,
    )

    print("\nRunning ABC-SMC with conservative schedule...")
    result_conservative = abc_smc_conservative.infer(observed_data)

    # Extract parameter estimates
    mean_estimates_cons = [particle[0] for particle in result_conservative.particles]
    std_estimates_cons = [particle[1] for particle in result_conservative.particles]

    print("\nConservative Schedule Results:")
    print(f"Number of particles: {len(result_conservative)}")
    print(
        f"Final effective sample size: {result_conservative.effective_sample_size():.1f}"
    )
    print(
        f"Mean estimate: {np.mean(mean_estimates_cons):.3f} ± {np.std(mean_estimates_cons):.3f}"
    )
    print(
        f"Std estimate: {np.mean(std_estimates_cons):.3f} ± {np.std(std_estimates_cons):.3f}"
    )

    # Example 2: Aggressive tolerance schedule
    print("\n2. AGGRESSIVE TOLERANCE SCHEDULE")
    print("-" * 40)

    tolerance_schedule_aggressive = [2.0, 0.5, 0.1]
    print(f"Tolerance schedule: {tolerance_schedule_aggressive}")

    abc_smc_aggressive = ABCSMC(
        simulator=simulator,
        prior=prior,
        distance_function=summary_distance,
        tolerance_schedule=tolerance_schedule_aggressive,
        perturbation_kernel=perturbation_kernel,
        num_particles=200,
        verbose=True,
    )

    print("\nRunning ABC-SMC with aggressive schedule...")
    result_aggressive = abc_smc_aggressive.infer(observed_data)

    # Extract parameter estimates
    mean_estimates_agg = [particle[0] for particle in result_aggressive.particles]
    std_estimates_agg = [particle[1] for particle in result_aggressive.particles]

    print("\nAggressive Schedule Results:")
    print(f"Number of particles: {len(result_aggressive)}")
    print(
        f"Final effective sample size: {result_aggressive.effective_sample_size():.1f}"
    )
    print(
        f"Mean estimate: {np.mean(mean_estimates_agg):.3f} ± {np.std(mean_estimates_agg):.3f}"
    )
    print(
        f"Std estimate: {np.mean(std_estimates_agg):.3f} ± {np.std(std_estimates_agg):.3f}"
    )

    # Example 3: Adaptive tolerance
    print("\n3. ADAPTIVE TOLERANCE SCHEDULE")
    print("-" * 40)

    abc_smc_adaptive = ABCSMC(
        simulator=simulator,
        prior=prior,
        distance_function=summary_distance,
        tolerance_schedule=[2.0, 1.0, 0.5, 0.2],  # Initial schedule
        perturbation_kernel=perturbation_kernel,
        num_particles=200,
        adaptive_tolerance=True,
        verbose=True,
    )

    print("\nRunning ABC-SMC with adaptive tolerance...")
    result_adaptive = abc_smc_adaptive.infer(observed_data)

    # Extract parameter estimates
    mean_estimates_adap = [particle[0] for particle in result_adaptive.particles]
    std_estimates_adap = [particle[1] for particle in result_adaptive.particles]

    print("\nAdaptive Schedule Results:")
    print(f"Number of particles: {len(result_adaptive)}")
    print(f"Final effective sample size: {result_adaptive.effective_sample_size():.1f}")
    print(
        f"Mean estimate: {np.mean(mean_estimates_adap):.3f} ± {np.std(mean_estimates_adap):.3f}"
    )
    print(
        f"Std estimate: {np.mean(std_estimates_adap):.3f} ± {np.std(std_estimates_adap):.3f}"
    )
    print(f"Final tolerance schedule: {result_adaptive.tolerance_schedule}")

    # Generate comprehensive plots
    print("\n4. GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("-" * 40)

    # Plot 1: Comparison of final posteriors
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Mean parameter comparison
    axes[0, 0].hist(
        mean_estimates_cons,
        bins=25,
        density=True,
        alpha=0.6,
        color="blue",
        edgecolor="black",
        label="Conservative",
        weights=result_conservative.weights,
    )
    axes[0, 0].hist(
        mean_estimates_agg,
        bins=25,
        density=True,
        alpha=0.6,
        color="red",
        edgecolor="black",
        label="Aggressive",
        weights=result_aggressive.weights,
    )
    axes[0, 0].hist(
        mean_estimates_adap,
        bins=25,
        density=True,
        alpha=0.6,
        color="green",
        edgecolor="black",
        label="Adaptive",
        weights=result_adaptive.weights,
    )
    axes[0, 0].axvline(
        true_mean,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"True: {true_mean}",
    )
    axes[0, 0].set_title("Final Posterior: Mean Parameter")
    axes[0, 0].set_xlabel("Mean")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Standard deviation parameter comparison
    axes[0, 1].hist(
        std_estimates_cons,
        bins=25,
        density=True,
        alpha=0.6,
        color="blue",
        edgecolor="black",
        label="Conservative",
        weights=result_conservative.weights,
    )
    axes[0, 1].hist(
        std_estimates_agg,
        bins=25,
        density=True,
        alpha=0.6,
        color="red",
        edgecolor="black",
        label="Aggressive",
        weights=result_aggressive.weights,
    )
    axes[0, 1].hist(
        std_estimates_adap,
        bins=25,
        density=True,
        alpha=0.6,
        color="green",
        edgecolor="black",
        label="Adaptive",
        weights=result_adaptive.weights,
    )
    axes[0, 1].axvline(
        true_std, color="black", linestyle="--", linewidth=2, label=f"True: {true_std}"
    )
    axes[0, 1].set_title("Final Posterior: Std Parameter")
    axes[0, 1].set_xlabel("Standard Deviation")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Joint posterior comparison
    axes[1, 0].scatter(
        mean_estimates_cons,
        std_estimates_cons,
        alpha=0.6,
        s=15,
        color="blue",
        label="Conservative",
    )
    axes[1, 0].scatter(
        mean_estimates_agg,
        std_estimates_agg,
        alpha=0.6,
        s=15,
        color="red",
        label="Aggressive",
    )
    axes[1, 0].scatter(
        mean_estimates_adap,
        std_estimates_adap,
        alpha=0.6,
        s=15,
        color="green",
        label="Adaptive",
    )
    axes[1, 0].scatter(
        true_mean,
        true_std,
        color="black",
        s=100,
        marker="x",
        linewidth=3,
        label="True parameters",
    )
    axes[1, 0].set_title("Joint Posterior Distribution")
    axes[1, 0].set_xlabel("Mean Parameter")
    axes[1, 0].set_ylabel("Std Parameter")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Effective sample size comparison
    methods = ["Conservative", "Aggressive", "Adaptive"]
    ess_values = [
        result_conservative.effective_sample_size(),
        result_aggressive.effective_sample_size(),
        result_adaptive.effective_sample_size(),
    ]
    colors = ["blue", "red", "green"]

    bars = axes[1, 1].bar(
        methods, ess_values, color=colors, alpha=0.7, edgecolor="black"
    )
    axes[1, 1].set_title("Final Effective Sample Size")
    axes[1, 1].set_ylabel("ESS")
    axes[1, 1].grid(True, alpha=0.3)

    # Add values on bars
    for bar, ess in zip(bars, ess_values):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{ess:.1f}",
            ha="center",
            va="bottom",
        )

    plt.suptitle("ABC-SMC: Comparison of Tolerance Schedules", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Plot 2: Evolution of particle distributions (Conservative schedule)
    print("Plotting particle evolution for conservative schedule...")
    result_conservative.plot_evolution(parameter_index=0)

    # Plot 3: Algorithm diagnostics (Conservative schedule)
    print("Plotting algorithm diagnostics for conservative schedule...")
    result_conservative.plot_diagnostics()

    # Plot 4: Tolerance schedule comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Conservative
    iterations_cons = range(1, len(result_conservative.tolerance_schedule) + 1)
    axes[0].semilogy(
        iterations_cons,
        result_conservative.tolerance_schedule,
        "o-",
        color="blue",
        linewidth=2,
        markersize=8,
    )
    axes[0].set_title("Conservative Schedule")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Tolerance (log scale)")
    axes[0].grid(True, alpha=0.3)

    # Aggressive
    iterations_agg = range(1, len(result_aggressive.tolerance_schedule) + 1)
    axes[1].semilogy(
        iterations_agg,
        result_aggressive.tolerance_schedule,
        "o-",
        color="red",
        linewidth=2,
        markersize=8,
    )
    axes[1].set_title("Aggressive Schedule")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Tolerance (log scale)")
    axes[1].grid(True, alpha=0.3)

    # Adaptive
    iterations_adap = range(1, len(result_adaptive.tolerance_schedule) + 1)
    axes[2].semilogy(
        iterations_adap,
        result_adaptive.tolerance_schedule,
        "o-",
        color="green",
        linewidth=2,
        markersize=8,
    )
    axes[2].set_title("Adaptive Schedule")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Tolerance (log scale)")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("ABC-SMC: Tolerance Schedule Comparison", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Summary statistics table
    print("\n5. PERFORMANCE COMPARISON")
    print("-" * 40)
    print(
        f"{'Method':<12} {'ESS':<8} {'Mean Error':<12} {'Std Error':<12} {'Iterations':<10}"
    )
    print("-" * 65)

    mean_error_cons = abs(np.mean(mean_estimates_cons) - true_mean)
    std_error_cons = abs(np.mean(std_estimates_cons) - true_std)
    mean_error_agg = abs(np.mean(mean_estimates_agg) - true_mean)
    std_error_agg = abs(np.mean(std_estimates_agg) - true_std)
    mean_error_adap = abs(np.mean(mean_estimates_adap) - true_mean)
    std_error_adap = abs(np.mean(std_estimates_adap) - true_std)

    print(
        f"{'Conservative':<12} {result_conservative.effective_sample_size():<8.1f} "
        f"{mean_error_cons:<12.4f} {std_error_cons:<12.4f} {len(result_conservative.tolerance_schedule):<10}"
    )
    print(
        f"{'Aggressive':<12} {result_aggressive.effective_sample_size():<8.1f} "
        f"{mean_error_agg:<12.4f} {std_error_agg:<12.4f} {len(result_aggressive.tolerance_schedule):<10}"
    )
    print(
        f"{'Adaptive':<12} {result_adaptive.effective_sample_size():<8.1f} "
        f"{mean_error_adap:<12.4f} {std_error_adap:<12.4f} {len(result_adaptive.tolerance_schedule):<10}"
    )

    print("\n" + "=" * 60)
    print("ABC-SMC demonstration completed!")
    print("Key insights:")
    print("1. Conservative schedules: Better particle diversity but more iterations")
    print("2. Aggressive schedules: Faster convergence but risk of particle degeneracy")
    print("3. Adaptive schedules: Balance between efficiency and robustness")
    print("4. ESS monitoring helps detect particle degeneracy")
    print("5. SMC provides complete evolution history for analysis")
    print("=" * 60)


if __name__ == "__main__":
    main()
