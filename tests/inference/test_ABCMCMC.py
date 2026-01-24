import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch
from scisbi.inference.ABCMCMC import ABCMCMC, ABCMCMCPosterior
import io
import sys


# Mock classes for basic testing
class MockSimulator:
    def __init__(self, return_value=None):
        if return_value is not None:
            self.return_value = return_value
        else:
            self.return_value = np.array([[1.0, 2.0]])

    def simulate(self, parameters, **kwargs):
        # Handle batching if necessary, but return_value is fixed
        # Ensure return shape matches batch size
        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            parameters = parameters[np.newaxis, :]
        batch_size = parameters.shape[0]

        if self.return_value.shape[0] == 1 and batch_size > 1:
             return np.tile(self.return_value, (batch_size, 1))
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
        # data: (Batch, D)
        return np.mean(data, axis=1, keepdims=True)


class MockDistanceFunction:
    def __init__(self, return_value=0.1):
        self.return_value = return_value

    def __call__(self, x, y):
        return self.return_value


class MockProposalDistribution:
    def __init__(self, offset=0.1):
        self.offset = offset

    def __call__(self, current_state):
        if isinstance(current_state, (list, np.ndarray)):
            return [x + np.random.normal(0, self.offset) for x in current_state]
        else:
            return current_state + np.random.normal(0, self.offset)


# Test ABCMCMC initialization
def test_abcmcmc_initialization_valid():
    simulator = MockSimulator()
    prior = MockPrior()
    distance_fn = MockDistanceFunction()
    tolerance = 0.1
    proposal = MockProposalDistribution()

    abc_mcmc = ABCMCMC(simulator, prior, distance_fn, tolerance, proposal)

    assert abc_mcmc.simulator == simulator
    assert abc_mcmc.prior == prior
    assert abc_mcmc.distance_function == distance_fn
    assert abc_mcmc.tolerance == tolerance
    assert abc_mcmc.proposal_distribution == proposal
    assert abc_mcmc.max_attempts_per_step == 100
    assert not abc_mcmc.verbose


def test_abcmcmc_initialization_with_summary_statistic():
    simulator = MockSimulator()
    prior = MockPrior()
    distance_fn = MockDistanceFunction()
    tolerance = 0.1
    proposal = MockProposalDistribution()
    summary_stat = MockSummaryStatistic()

    abc_mcmc = ABCMCMC(
        simulator,
        prior,
        distance_fn,
        tolerance,
        proposal,
        summary_statistic=summary_stat,
        max_attempts_per_step=50,
        verbose=True,
        thin=2,
        burn_in=100,
    )

    assert abc_mcmc.summary_statistic == summary_stat
    assert abc_mcmc.max_attempts_per_step == 50
    assert abc_mcmc.verbose
    assert abc_mcmc.thin == 2
    assert abc_mcmc.burn_in == 100


def test_abcmcmc_initialization_invalid_distance_function():
    simulator = MockSimulator()
    prior = MockPrior()
    tolerance = 0.1
    proposal = MockProposalDistribution()

    with pytest.raises(TypeError, match="distance_function must be callable"):
        ABCMCMC(simulator, prior, "not_callable", tolerance, proposal)


def test_abcmcmc_initialization_invalid_proposal():
    simulator = MockSimulator()
    prior = MockPrior()
    distance_fn = MockDistanceFunction()
    tolerance = 0.1

    with pytest.raises(TypeError, match="proposal_distribution must be callable"):
        ABCMCMC(simulator, prior, distance_fn, tolerance, "not_callable")


def test_abcmcmc_initialization_negative_tolerance():
    simulator = MockSimulator()
    prior = MockPrior()
    distance_fn = MockDistanceFunction()
    proposal = MockProposalDistribution()

    with pytest.raises(ValueError, match="tolerance must be non-negative"):
        ABCMCMC(simulator, prior, distance_fn, -0.1, proposal)


# Test ABCMCMC.infer method
def test_abcmcmc_infer_basic():
    np.random.seed(42)
    simulator = MockSimulator(np.array([[1.0, 2.0]]))
    prior = MockPrior(0.5)
    distance_fn = MockDistanceFunction(0.05)  # Always accept
    tolerance = 0.1
    proposal = MockProposalDistribution(0.1)

    abc_mcmc = ABCMCMC(simulator, prior, distance_fn, tolerance, proposal)
    observed_data = np.array([1.1, 1.9])

    result = abc_mcmc.infer(observed_data, num_iterations=10)

    assert isinstance(result, ABCMCMCPosterior)
    assert len(result.samples) == 10  # No burn-in, thin=1
    assert len(result.distances) == 10
    assert result.tolerance == tolerance
    assert result.num_iterations == 10


def test_abcmcmc_infer_with_burn_in_and_thinning():
    np.random.seed(42)
    simulator = MockSimulator(np.array([[1.0, 2.0]]))
    prior = MockPrior(0.5)
    distance_fn = MockDistanceFunction(0.05)
    tolerance = 0.1
    proposal = MockProposalDistribution(0.1)

    abc_mcmc = ABCMCMC(
        simulator, prior, distance_fn, tolerance, proposal, burn_in=5, thin=2
    )
    observed_data = np.array([1.1, 1.9])

    result = abc_mcmc.infer(observed_data, num_iterations=20)

    # With burn_in=5, thin=2, iterations=20: (20-5)//2 = 7 samples expected
    expected_samples = (20 - 5) // 2
    assert len(result.samples) == expected_samples
    assert result.burn_in == 5
    assert result.thin == 2


def test_abcmcmc_infer_with_initial_state():
    np.random.seed(42)
    simulator = MockSimulator(np.array([[1.0, 2.0]]))
    prior = MockPrior(0.5)
    distance_fn = MockDistanceFunction(0.05)
    tolerance = 0.1
    proposal = MockProposalDistribution(0.1)

    abc_mcmc = ABCMCMC(simulator, prior, distance_fn, tolerance, proposal)
    observed_data = np.array([1.1, 1.9])
    initial_state = 1.0

    result = abc_mcmc.infer(
        observed_data, num_iterations=5, initial_state=initial_state
    )

    assert len(result.samples) == 5


def test_abcmcmc_infer_invalid_num_iterations():
    simulator = MockSimulator()
    prior = MockPrior()
    distance_fn = MockDistanceFunction()
    tolerance = 0.1
    proposal = MockProposalDistribution()

    abc_mcmc = ABCMCMC(simulator, prior, distance_fn, tolerance, proposal)

    with pytest.raises(ValueError, match="num_iterations must be positive"):
        abc_mcmc.infer(np.array([1, 2]), num_iterations=0)


def test_abcmcmc_infer_with_kwargs_override():
    np.random.seed(42)
    simulator = MockSimulator()
    prior = MockPrior()
    distance_fn = MockDistanceFunction(0.05)
    tolerance = 0.1
    proposal = MockProposalDistribution()

    abc_mcmc = ABCMCMC(
        simulator, prior, distance_fn, tolerance, proposal, verbose=False
    )

    # Test with override parameters
    result = abc_mcmc.infer(
        np.array([1, 2]), num_iterations=5, verbose=True, tolerance=0.2, burn_in=2, thin=1
    )

    assert len(result.samples) == 3  # (5-2)//1 = 3
    assert result.tolerance == 0.2
    assert result.burn_in == 2


# Test helper methods
def test_abcmcmc_find_valid_initial_state():
    np.random.seed(42)
    simulator = MockSimulator(np.array([[1.0, 2.0]]))
    prior = MockPrior(0.5)
    distance_fn = MockDistanceFunction(0.05)  # Always valid
    tolerance = 0.1
    proposal = MockProposalDistribution()

    abc_mcmc = ABCMCMC(simulator, prior, distance_fn, tolerance, proposal)
    observed_summary = np.array([[1.5]]) # MockSummary returns (1, 1)

    initial_state = abc_mcmc._find_valid_initial_state(observed_summary, tolerance, 10)
    assert initial_state == 0.5  # MockPrior returns 0.5


def test_abcmcmc_find_valid_initial_state_failure():
    simulator = MockSimulator(np.array([[1.0, 2.0]]))
    prior = MockPrior(0.5)
    distance_fn = MockDistanceFunction(1.0)  # Always invalid
    tolerance = 0.1
    proposal = MockProposalDistribution()

    abc_mcmc = ABCMCMC(simulator, prior, distance_fn, tolerance, proposal)
    observed_summary = np.array([[1.5]])

    with pytest.raises(RuntimeError, match="Could not find valid initial state"):
        abc_mcmc._find_valid_initial_state(observed_summary, tolerance, 5)


def test_abcmcmc_is_valid_state():
    simulator = MockSimulator(np.array([[1.0, 2.0]]))
    prior = MockPrior()
    distance_fn = MockDistanceFunction(0.05)
    tolerance = 0.1
    proposal = MockProposalDistribution()

    abc_mcmc = ABCMCMC(simulator, prior, distance_fn, tolerance, proposal)
    observed_summary = np.array([[1.5]])

    assert abc_mcmc._is_valid_state(0.5, observed_summary, tolerance)


def test_abcmcmc_is_valid_state_invalid():
    simulator = MockSimulator(np.array([[1.0, 2.0]]))
    prior = MockPrior()
    distance_fn = MockDistanceFunction(1.0)  # Distance > tolerance
    tolerance = 0.1
    proposal = MockProposalDistribution()

    abc_mcmc = ABCMCMC(simulator, prior, distance_fn, tolerance, proposal)
    observed_summary = np.array([[1.5]])

    assert not abc_mcmc._is_valid_state(0.5, observed_summary, tolerance)


# Test ABCMCMCPosterior class (same as before)
def test_abcmcmc_posterior_initialization():
    samples = [0.1, 0.2, 0.3]
    log_priors = [-1.0, -0.5, -0.8]
    distances = [0.05, 0.08, 0.02]
    acceptance_indicators = [True, False, True]
    tolerance = 0.1
    num_iterations = 100
    burn_in = 10
    thin = 2

    posterior = ABCMCMCPosterior(
        samples,
        log_priors,
        distances,
        acceptance_indicators,
        tolerance,
        num_iterations,
        burn_in,
        thin,
    )

    assert posterior.samples == samples
    assert posterior.log_priors == log_priors
    assert posterior.distances == distances
    assert posterior.acceptance_indicators == acceptance_indicators
    assert posterior.tolerance == tolerance
    assert posterior.num_iterations == num_iterations
    assert posterior.burn_in == burn_in
    assert posterior.thin == thin
    assert posterior.acceptance_rate == 2 / 3


def test_abcmcmc_posterior_sample_single():
    samples = [0.1, 0.2, 0.3]
    log_priors = [-1.0, -0.5, -0.8]
    distances = [0.05, 0.08, 0.02]
    acceptance_indicators = [True, False, True]

    posterior = ABCMCMCPosterior(
        samples, log_priors, distances, acceptance_indicators, 0.1, 100, 10, 2
    )

    np.random.seed(42)
    sample = posterior.sample(1)
    assert sample in samples


def test_abcmcmc_posterior_sample_multiple():
    samples = [0.1, 0.2, 0.3]
    log_priors = [-1.0, -0.5, -0.8]
    distances = [0.05, 0.08, 0.02]
    acceptance_indicators = [True, False, True]

    posterior = ABCMCMCPosterior(
        samples, log_priors, distances, acceptance_indicators, 0.1, 100, 10, 2
    )

    np.random.seed(42)
    sample_list = posterior.sample(5)
    assert len(sample_list) == 5
    assert all(s in samples for s in sample_list)


def test_abcmcmc_posterior_get_methods():
    samples = [0.1, 0.2, 0.3]
    log_priors = [-1.0, -0.5, -0.8]
    distances = [0.05, 0.08, 0.02]
    acceptance_indicators = [True, False, True]

    posterior = ABCMCMCPosterior(
        samples, log_priors, distances, acceptance_indicators, 0.1, 100, 10, 2
    )

    assert posterior.get_samples() == samples
    assert posterior.get_log_priors() == log_priors
    assert posterior.get_distances() == distances
    assert posterior.get_acceptance_indicators() == acceptance_indicators
    # Ensure they return copies
    assert posterior.get_samples() is not posterior.samples


def test_abcmcmc_posterior_summary_statistics():
    samples = [0.1, 0.2, 0.3]
    log_priors = [-1.0, -0.5, -0.8]
    distances = [0.05, 0.08, 0.02]
    acceptance_indicators = [True, False, True]

    posterior = ABCMCMCPosterior(
        samples, log_priors, distances, acceptance_indicators, 0.1, 100, 10, 2
    )

    stats = posterior.summary_statistics()

    assert stats["num_samples"] == 3
    assert stats["acceptance_rate"] == 2 / 3
    assert stats["tolerance"] == 0.1
    assert stats["num_iterations"] == 100
    assert stats["burn_in"] == 10
    assert stats["thin"] == 2
    assert stats["effective_sample_size"] == 3  # Simplified implementation
    assert stats["mean_distance"] == np.mean(distances)
    assert stats["max_distance"] == max(distances)
    assert stats["min_distance"] == min(distances)


def test_abcmcmc_posterior_effective_sample_size():
    samples = [0.1, 0.2, 0.3]
    log_priors = [-1.0, -0.5, -0.8]
    distances = [0.05, 0.08, 0.02]
    acceptance_indicators = [True, False, True]

    posterior = ABCMCMCPosterior(
        samples, log_priors, distances, acceptance_indicators, 0.1, 100, 10, 2
    )

    # Simplified implementation returns len(samples)
    assert posterior.effective_sample_size() == 3


def test_abcmcmc_posterior_len():
    samples = [0.1, 0.2, 0.3]
    log_priors = [-1.0, -0.5, -0.8]
    distances = [0.05, 0.08, 0.02]
    acceptance_indicators = [True, False, True]

    posterior = ABCMCMCPosterior(
        samples, log_priors, distances, acceptance_indicators, 0.1, 100, 10, 2
    )

    assert len(posterior) == 3


def test_abcmcmc_posterior_repr():
    samples = [0.1, 0.2, 0.3]
    log_priors = [-1.0, -0.5, -0.8]
    distances = [0.05, 0.08, 0.02]
    acceptance_indicators = [True, False, True]

    posterior = ABCMCMCPosterior(
        samples, log_priors, distances, acceptance_indicators, 0.1, 100, 10, 2
    )

    repr_str = repr(posterior)
    assert "ABCMCMCPosterior" in repr_str
    assert "num_samples=3" in repr_str
    assert "acceptance_rate=0.6667" in repr_str
    assert "tolerance=0.1" in repr_str


# Test plotting methods
@patch("matplotlib.pyplot.show")
def test_abcmcmc_posterior_plot_1d(mock_show):
    samples = [0.1, 0.2, 0.3, 0.4, 0.5]
    log_priors = [-1.0, -0.5, -0.8, -0.3, -0.9]
    distances = [0.05, 0.08, 0.02, 0.07, 0.03]
    acceptance_indicators = [True, False, True, True, False]

    posterior = ABCMCMCPosterior(
        samples, log_priors, distances, acceptance_indicators, 0.1, 100, 10, 2
    )

    posterior.plot()
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_abcmcmc_posterior_plot_multidimensional(mock_show):
    samples = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    log_priors = [-1.0, -0.5, -0.8]
    distances = [0.05, 0.08, 0.02]
    acceptance_indicators = [True, False, True]

    posterior = ABCMCMCPosterior(
        samples, log_priors, distances, acceptance_indicators, 0.1, 100, 10, 2
    )

    posterior.plot(parameter_index=0)
    mock_show.assert_called_once()


def test_abcmcmc_posterior_plot_no_samples():
    posterior = ABCMCMCPosterior([], [], [], [], 0.1, 100, 10, 2)

    with pytest.raises(ValueError, match="No samples available to plot"):
        posterior.plot()


def test_abcmcmc_posterior_plot_multidimensional_no_index():
    samples = [[0.1, 0.2], [0.3, 0.4]]
    log_priors = [-1.0, -0.5]
    distances = [0.05, 0.08]
    acceptance_indicators = [True, False]

    posterior = ABCMCMCPosterior(
        samples, log_priors, distances, acceptance_indicators, 0.1, 100, 10, 2
    )

    with pytest.raises(
        ValueError, match="Samples are multi-dimensional. Specify parameter_index"
    ):
        posterior.plot()


# Integration tests with Gaussian distributions
class GaussianSimulator:
    def __init__(self, true_params):
        self.true_params = true_params

    def simulate(self, parameters):
        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            parameters = parameters[np.newaxis, :]
        batch_size = parameters.shape[0]
        param_dim = parameters.shape[1]

        if param_dim == 2:  # mean, std
            mean = parameters[:, 0:1]
            std = parameters[:, 1:2]
            return np.random.normal(mean, std, (batch_size, 100)) # (Batch, 100)
        elif param_dim == 3:  # mean1, mean2, std
            mean1 = parameters[:, 0:1]
            mean2 = parameters[:, 1:2]
            std = parameters[:, 2:3]
            return np.concatenate(
                [
                    np.random.normal(mean1, std, (batch_size, 100)),
                    np.random.normal(mean2, std, (batch_size, 100)),
                ], axis=1
            )
        else:
            raise ValueError("Unsupported parameter dimension")


class GaussianPrior:
    def __init__(self, dim):
        self.dim = dim

    def sample(self):
        if self.dim == 2:
            return np.array([np.random.uniform(-5, 5), np.random.uniform(0.1, 3)])
        elif self.dim == 3:
            return np.array([
                np.random.uniform(-5, 5),
                np.random.uniform(-5, 5),
                np.random.uniform(0.1, 3),
            ])

    def log_prob(self, x):
        if self.dim == 2:
            if -5 <= x[0] <= 5 and 0.1 <= x[1] <= 3:
                return np.log(1 / (10 * 2.9))
            else:
                return -np.inf
        elif self.dim == 3:
            if -5 <= x[0] <= 5 and -5 <= x[1] <= 5 and 0.1 <= x[2] <= 3:
                return np.log(1 / (10 * 10 * 2.9))
            else:
                return -np.inf


class GaussianProposal:
    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, current_state):
        return current_state + np.random.normal(0, self.std, size=len(current_state))


def euclidean_distance(x, y):
    """Euclidean distance between sample means and standard deviations."""
    x_arr = np.array(x).flatten()
    y_arr = np.array(y).flatten()
    # If using summary stats already in tests, this works.
    return np.linalg.norm(x_arr - y_arr)


def test_abcmcmc_inference_2d_gaussian():
    """Test ABC-MCMC inference on 2D Gaussian (mean, std) parameters."""
    np.random.seed(42)

    true_params = [2.0, 1.5]
    simulator = GaussianSimulator(true_params)
    prior = GaussianPrior(dim=2)
    proposal = GaussianProposal(std=0.2)

    # Generate observed data
    # (100,)
    observed_data = np.random.normal(true_params[0], true_params[1], 100)

    # Use summary statistic
    class Stats:
        def compute(self, data):
            data = np.array(data)
            if data.ndim == 1:
                data = data[np.newaxis, :]
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True)
            return np.concatenate([mean, std], axis=1)

    summary_stat = Stats()

    abc_mcmc = ABCMCMC(
        simulator=simulator,
        prior=prior,
        distance_function=euclidean_distance,
        tolerance=1.0,
        proposal_distribution=proposal,
        max_attempts_per_step=50,
        summary_statistic=summary_stat
    )

    result = abc_mcmc.infer(observed_data, num_iterations=50, burn_in=10)

    assert len(result.samples) == 40
    assert all(len(sample) == 2 for sample in result.samples)
    assert result.acceptance_rate >= 0

    if len(result.samples) > 0:
        mean_estimates = [sample[0] for sample in result.samples]
        std_estimates = [sample[1] for sample in result.samples]
        assert np.mean(mean_estimates) == pytest.approx(true_params[0], abs=2.0)
        assert np.mean(std_estimates) == pytest.approx(true_params[1], abs=2.0)


def test_abcmcmc_inference_verbose_mode():
    """Test ABC-MCMC inference with verbose output."""
    np.random.seed(42)

    simulator = MockSimulator(np.array([[1.0, 2.0]]))
    prior = MockPrior(0.5)
    distance_fn = MockDistanceFunction(0.05)
    tolerance = 0.1
    proposal = MockProposalDistribution(0.1)

    abc_mcmc = ABCMCMC(simulator, prior, distance_fn, tolerance, proposal, verbose=True)

    captured_output = io.StringIO()
    sys.stdout = captured_output

    try:
        result = abc_mcmc.infer(np.array([1, 2]), num_iterations=20, burn_in=5)
        output = captured_output.getvalue()

        assert "Starting ABC-MCMC sampling" in output
        assert "ABC-MCMC completed" in output
        assert len(result.samples) == 15  # 20 - 5 burn-in
    finally:
        sys.stdout = sys.__stdout__


if __name__ == "__main__":
    pass
