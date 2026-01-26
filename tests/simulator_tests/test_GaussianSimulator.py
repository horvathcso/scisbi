import pytest
import numpy as np
from scipy import stats
from scisbi.simulator import GaussianSimulator


class TestGaussianSimulator:
    def test_init_default(self):
        """Test initialization with default parameters."""
        sim = GaussianSimulator()
        assert sim.dimensions == 1
        assert not sim.use_covariance

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        sim = GaussianSimulator(
            dimensions=3, use_covariance=True, seed=42, custom_param="value"
        )
        assert sim.dimensions == 3
        assert sim.use_covariance
        assert sim.config["custom_param"] == "value"

    def test_1d_array_params(self):
        """Test 1D Gaussian with array parameters."""
        seed = 12345
        sim = GaussianSimulator(dimensions=1, seed=seed)

        # Parameters: [mean, std]
        params_single = np.array([5.0, 2.0])
        num_sims = 10000

        # Batch parameters
        params = np.tile(params_single, (num_sims, 1))

        samples = sim.simulate(params)

        # Check shape
        assert samples.shape == (num_sims, 1)

        # Check statistical properties (with tolerance for randomness)
        assert 4.8 < np.mean(samples) < 5.2
        assert 1.9 < np.std(samples) < 2.1

        # Test normality using Kolmogorov-Smirnov test
        _, p_value = stats.kstest(
            samples.flatten(), "norm", args=(params_single[0], params_single[1])
        )
        assert p_value > 0.01  # Should not reject null hypothesis that data is normal

    def test_1d_dict_params(self):
        """Test 1D Gaussian with dictionary parameters."""
        sim = GaussianSimulator(dimensions=1, seed=42)

        num_sims = 5000
        params = {
            "mean": np.full((num_sims, 1), 3.0),
            "std": np.full((num_sims, 1), 1.5),
        }

        samples = sim.simulate(params)

        # Check shape and statistical properties
        assert samples.shape == (num_sims, 1)
        assert 2.85 < np.mean(samples) < 3.15
        assert 1.4 < np.std(samples) < 1.6

    def test_multivariate_independent(self):
        """Test multivariate Gaussian with diagonal covariance."""
        sim = GaussianSimulator(dimensions=3, use_covariance=False, seed=42)

        mean = np.array([1.0, 2.0, 3.0])
        std = np.array([0.5, 1.0, 1.5])
        params_single = np.concatenate([mean, std])

        num_sims = 5000
        params = np.tile(params_single, (num_sims, 1))

        samples = sim.simulate(params)

        # Check shape
        assert samples.shape == (num_sims, 3)

        # Check means and standard deviations for each dimension
        sample_means = np.mean(samples, axis=0)
        sample_stds = np.std(samples, axis=0)

        assert np.allclose(sample_means, mean, rtol=0.1)
        assert np.allclose(sample_stds, std, rtol=0.1)

        # Check independence between dimensions
        corr_matrix = np.corrcoef(samples.T)
        off_diagonals = corr_matrix[~np.eye(3, dtype=bool)]
        assert np.all(np.abs(off_diagonals) < 0.1)  # Low correlation expected

    def test_multivariate_covariance_dict(self):
        """Test multivariate Gaussian with full covariance matrix using dict parameters."""
        sim = GaussianSimulator(dimensions=2, use_covariance=True, seed=42)

        mean = np.array([5.0, 10.0])
        # Non-diagonal covariance matrix
        cov = np.array([[4.0, 3.0], [3.0, 9.0]])

        num_sims = 10000
        params = {
            "mean": np.tile(mean, (num_sims, 1)),
            "cov": np.tile(cov, (num_sims, 1, 1)),
        }

        samples = sim.simulate(params)

        # Check shape
        assert samples.shape == (num_sims, 2)

        # Check means
        sample_means = np.mean(samples, axis=0)
        assert np.allclose(sample_means, mean, rtol=0.1)

        # Check covariance matrix
        sample_cov = np.cov(samples.T)
        assert np.allclose(sample_cov, cov, rtol=0.2)

        # Check correlation
        expected_corr = 3.0 / (2.0 * 3.0)  # correlation = cov(X,Y) / (std(X) * std(Y))
        sample_corr = np.corrcoef(samples.T)[0, 1]
        assert np.isclose(sample_corr, expected_corr, rtol=0.1)

    def test_multivariate_covariance_array(self):
        """Test multivariate Gaussian with full covariance matrix using array parameters."""
        sim = GaussianSimulator(dimensions=3, use_covariance=True, seed=123)

        mean = np.array([1.0, 2.0, 3.0])
        # Create a valid covariance matrix (symmetric, positive semi-definite)
        A = np.random.rand(3, 3)
        cov = np.dot(A, A.T)  # Ensures positive semi-definiteness

        # Flatten cov matrix for parameter input
        params_single = np.concatenate([mean, cov.flatten()])

        num_sims = 8000
        params = np.tile(params_single, (num_sims, 1))

        samples = sim.simulate(params)

        # Check shape
        assert samples.shape == (num_sims, 3)

        # Check statistical properties
        sample_means = np.mean(samples, axis=0)
        sample_cov = np.cov(samples.T)

        assert np.allclose(sample_means, mean, rtol=0.1)
        assert np.allclose(sample_cov, cov, rtol=0.2)

    def test_reproducibility(self):
        """Test that setting the same seed gives reproducible results."""
        sim1 = GaussianSimulator(dimensions=2, seed=999)
        sim2 = GaussianSimulator(dimensions=2, seed=999)

        params = {"mean": np.zeros((100, 2)), "std": np.ones((100, 2))}

        samples1 = sim1.simulate(params)
        samples2 = sim2.simulate(params)

        assert np.allclose(samples1, samples2)

    def test_parameter_errors(self):
        """Test that invalid parameters raise appropriate errors."""
        sim = GaussianSimulator(dimensions=2)

        # Invalid parameter length (single item batch)
        # Expected dim: 2*2=4.
        with pytest.raises(ValueError):
            sim.simulate(np.array([[1.0, 2.0, 3.0]]))  # Missing std values

        # Invalid parameter type
        with pytest.raises(Exception):
            sim.simulate("not_a_valid_parameter")
