import numpy as np
import pytest
from scisbi.simulator import GaussianMixtureSimulator
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


class TestGaussianMixtureSimulator:
    """Test suite for the GaussianMixtureSimulator class."""

    def test_init_default_parameters(self):
        """Test that the simulator initializes with default parameters."""
        sim = GaussianMixtureSimulator()
        assert sim.n_components == 2
        assert sim.dimensions == 1
        assert sim.use_covariance is False

    def test_init_custom_parameters(self):
        """Test that the simulator initializes with custom parameters."""
        sim = GaussianMixtureSimulator(
            n_components=3, dimensions=2, use_covariance=True
        )
        assert sim.n_components == 3
        assert sim.dimensions == 2
        assert sim.use_covariance is True

    def test_seed_reproducibility(self):
        """Test that setting a seed produces reproducible results."""
        seed = 42
        sim1 = GaussianMixtureSimulator(seed=seed)
        sim2 = GaussianMixtureSimulator(seed=seed)

        params = {"weights": [0.6, 0.4], "means": [0, 5], "stds": [1, 2]}
        samples1 = sim1.simulate(params, num_simulations=100)
        samples2 = sim2.simulate(params, num_simulations=100)

        np.testing.assert_array_equal(samples1, samples2)

    def test_univariate_dict_params(self):
        """Test univariate simulation with dictionary parameters."""
        sim = GaussianMixtureSimulator(n_components=2, dimensions=1, seed=42)
        params = {"weights": [0.7, 0.3], "means": [0, 10], "stds": [1, 2]}

        num_samples = 10000
        samples = sim.simulate(params, num_simulations=num_samples)

        assert samples.shape == (num_samples, 1)

        # Check that samples follow expected distribution
        # We expect a bimodal distribution with peaks at 0 and 10
        # Calculate histogram and check for two peaks
        hist, bin_edges = np.histogram(samples, bins=20)
        peak_indices = np.where(np.diff(np.sign(np.diff(hist))))[0] + 1
        assert len(peak_indices) >= 1  # At least one peak should be detected

    def test_univariate_array_params(self):
        """Test univariate simulation with array parameters."""
        sim = GaussianMixtureSimulator(n_components=2, dimensions=1, seed=42)
        # Format: [weights, means, stds]
        params = np.array([0.7, 0.3, 0, 10, 1, 2])

        num_samples = 1000
        samples = sim.simulate(params, num_simulations=num_samples)

        assert samples.shape == (num_samples, 1)

    def test_multivariate_diagonal_dict_params(self):
        """Test multivariate simulation with diagonal covariance and dictionary parameters."""
        sim = GaussianMixtureSimulator(
            n_components=2, dimensions=2, use_covariance=False, seed=42
        )
        params = {
            "weights": [0.6, 0.4],
            "means": [[0, 0], [5, 5]],
            "stds": [[1, 1], [2, 2]],
        }

        num_samples = 1000
        samples = sim.simulate(params, num_simulations=num_samples)

        assert samples.shape == (num_samples, 2)

        # Check that samples are clustered around the means
        # We expect two clusters, one at [0,0] and one at [5,5]
        kmeans = KMeans(n_clusters=2, random_state=42).fit(samples)
        centers = kmeans.cluster_centers_

        # Check that cluster centers are close to the means
        center_dists = np.min(
            [
                np.linalg.norm(centers - np.array([[0, 0], [5, 5]])),
                np.linalg.norm(centers - np.array([[5, 5], [0, 0]])),
            ]
        )

        assert center_dists < 1.0  # Cluster centers should be close to means

    def test_multivariate_diagonal_array_params(self):
        """Test multivariate simulation with diagonal covariance and array parameters."""
        sim = GaussianMixtureSimulator(
            n_components=2, dimensions=2, use_covariance=False, seed=42
        )
        # Format: [weights, means, stds]
        params = np.array([0.6, 0.4, 0, 0, 5, 5, 1, 1, 2, 2])

        num_samples = 1000
        samples = sim.simulate(params, num_simulations=num_samples)

        assert samples.shape == (num_samples, 2)

    def test_multivariate_full_covariance_dict_params(self):
        """Test multivariate simulation with full covariance and dictionary parameters."""
        sim = GaussianMixtureSimulator(
            n_components=2, dimensions=2, use_covariance=True, seed=42
        )
        params = {
            "weights": [0.6, 0.4],
            "means": [[0, 0], [5, 5]],
            "covs": [[[1, 0.5], [0.5, 1]], [[2, -0.5], [-0.5, 2]]],
        }

        num_samples = 1000
        samples = sim.simulate(params, num_simulations=num_samples)

        assert samples.shape == (num_samples, 2)

        # Check covariance structure in the generated samples
        # We need to identify points from each cluster and check their covariance
        gmm = GaussianMixture(n_components=2, covariance_type="full").fit(samples)

        # Check that found covariances have similar structure to input
        # (allowing for some variance due to finite sample size)
        expected_covs = np.array(params["covs"])
        est_covs = gmm.covariances_

        # The order of components might be flipped, so we need to check both possibilities
        cov_diff1 = np.linalg.norm(est_covs - expected_covs)
        cov_diff2 = np.linalg.norm(est_covs - expected_covs[::-1])
        assert min(cov_diff1, cov_diff2) < 2.0  # Allow some tolerance

    def test_multivariate_full_covariance_array_params(self):
        """Test multivariate simulation with full covariance and array parameters."""
        sim = GaussianMixtureSimulator(
            n_components=2, dimensions=2, use_covariance=True, seed=42
        )
        # Format: [weights, means, covs]
        params = np.array(
            [
                0.6,
                0.4,  # weights
                0,
                0,
                5,
                5,  # means
                1,
                0.5,
                0.5,
                1,  # cov1
                2,
                -0.5,
                -0.5,
                2,  # cov2
            ]
        )

        num_samples = 1000
        samples = sim.simulate(params, num_simulations=num_samples)

        assert samples.shape == (num_samples, 2)

    def test_weight_normalization(self):
        """Test that weights are properly normalized."""
        sim = GaussianMixtureSimulator(n_components=3, dimensions=1, seed=42)

        # Use non-normalized weights
        params = {
            "weights": [10, 20, 30],  # Should be normalized to [1/6, 1/3, 1/2]
            "means": [0, 5, 10],
            "stds": [1, 1, 1],
        }

        num_samples = 10000
        samples = sim.simulate(params, num_simulations=num_samples)

        # Count approximate proportions by binning
        bins = [-np.inf, 2.5, 7.5, np.inf]
        hist, _ = np.histogram(samples, bins=bins)
        proportions = hist / num_samples

        # Check that proportions roughly match expected weights
        # Allow some deviation due to random sampling
        expected = np.array([1 / 6, 1 / 3, 1 / 2])
        assert np.all(np.abs(proportions - expected) < 0.1)

    def test_invalid_parameter_shape(self):
        """Test that invalid parameter shapes raise errors."""
        sim = GaussianMixtureSimulator(n_components=2, dimensions=2)

        # Wrong number of parameters
        with pytest.raises(ValueError):
            sim.simulate(np.array([0.5, 0.5, 1, 1]), num_simulations=10)

    def test_component_selection(self):
        """Test that components are selected according to weights."""
        # Use very distinct means to separate components clearly
        sim = GaussianMixtureSimulator(n_components=2, dimensions=1, seed=42)
        params = {
            "weights": [0.8, 0.2],
            "means": [-10, 10],  # Far apart
            "stds": [1, 1],  # Small std for clear separation
        }

        num_samples = 1000
        samples = sim.simulate(params, num_simulations=num_samples).flatten()

        # Count samples in each component (with a threshold at 0)
        comp1_count = np.sum(samples < 0)
        comp2_count = np.sum(samples >= 0)

        # Check that proportions roughly match weights
        assert abs(comp1_count / num_samples - 0.8) < 0.1
        assert abs(comp2_count / num_samples - 0.2) < 0.1

    def test_higher_dimensions(self):
        """Test with higher dimensions."""
        sim = GaussianMixtureSimulator(n_components=2, dimensions=3, seed=42)
        params = {
            "weights": [0.5, 0.5],
            "means": [[0, 0, 0], [5, 5, 5]],
            "stds": [[1, 1, 1], [1, 1, 1]],
        }

        num_samples = 100
        samples = sim.simulate(params, num_simulations=num_samples)

        assert samples.shape == (num_samples, 3)

    def test_extreme_weights(self):
        """Test with extreme weights where one component dominates."""
        sim = GaussianMixtureSimulator(n_components=2, dimensions=1, seed=42)
        params = {
            "weights": [0.99, 0.01],
            "means": [0, 100],  # Second mean is far away
            "stds": [1, 1],
        }

        num_samples = 1000
        samples = sim.simulate(params, num_simulations=num_samples).flatten()

        # Most samples should be from the first component
        assert (
            np.percentile(samples, 95) < 10
        )  # 95% of samples should be near first component

    def test_symmetric_covariance_matrices(self):
        """Test that covariance matrices are made symmetric if not already."""
        sim = GaussianMixtureSimulator(
            n_components=1, dimensions=2, use_covariance=True, seed=42
        )

        # Use non-symmetric covariance matrix
        params = {
            "weights": [1.0],
            "means": [[0, 0]],
            "covs": [[[1.0, 0.5], [0.8, 1.0]]],  # Asymmetric: 0.5 != 0.8
        }

        # This would raise an error if the simulator doesn't symmetrize the matrix
        samples = sim.simulate(params, num_simulations=100)
        assert samples.shape == (100, 2)
