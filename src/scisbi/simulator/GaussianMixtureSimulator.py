import numpy as np
from scisbi.base.simulator import BaseSimulator


class GaussianMixtureSimulator(BaseSimulator):
    """
    Simulator for data from a Gaussian Mixture distribution.

    This simulator generates observations from a mixture of Gaussian (normal) distributions
    with configurable means, standard deviations/covariances, and mixture weights.
    """

    def __init__(
        self, n_components=2, dimensions=1, use_covariance=False, seed=None, **kwargs
    ):
        """
        Initialize the Gaussian Mixture simulator.

        Args:
            n_components (int): Number of mixture components. Default is 2.
            dimensions (int): Number of dimensions for each Gaussian. Default is 1 (univariate).
            use_covariance (bool): If True, use full covariance matrices for multivariate
                                  Gaussians. Only applies when dimensions > 1. Default is False.
            seed (int, optional): Random seed for reproducibility.
            **kwargs: Additional configuration parameters passed to BaseSimulator.
        """
        super().__init__(**kwargs)
        self.n_components = n_components
        self.dimensions = dimensions
        self.use_covariance = use_covariance and dimensions > 1
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

    def simulate(self, parameters):
        """
        Generate samples from a Gaussian Mixture distribution with given parameters.

        Args:
            parameters (dict or array-like): Parameters for the Gaussian Mixture distribution.
                Expected shape: (Batch_Size, Num_Params)
                If dict: Should contain 'weights', 'means', and 'stds' or 'covs' keys.
                    Values should be batched.
                If array-like:
                    - With use_covariance=False: [weights, means, stds] flattened per batch item
                    - With use_covariance=True: [weights, means, covs] flattened per batch item

        Returns:
            np.ndarray: Array of shape (Batch_Size, dimensions) containing
                        the simulated observations.
        """
        # Process parameters based on their type
        if isinstance(parameters, dict):
            weights = np.asarray(parameters["weights"])
            means = np.asarray(parameters["means"])
            if self.use_covariance:
                covs = np.asarray(parameters["covs"])
            else:
                stds = np.asarray(parameters["stds"])

            # Ensure batch dimension and correct shapes
            if weights.ndim == 1:
                weights = weights[np.newaxis, :]

            # Means should be (Batch, K, D)
            if means.ndim == 2:
                # Ambiguous: could be (Batch, K) if D=1, or (K, D) if Batch=1?
                # We assume if it came from parameters dict, it's (Batch, ...).
                # So if D=1, input might be (Batch, K).
                # Or if K=1, input might be (Batch, D).
                # But K and D are fixed in __init__.
                if self.dimensions == 1:
                    means = means[:, :, np.newaxis]
                elif self.n_components == 1:
                    # Rare case, but means could be (Batch, D) -> (Batch, 1, D)
                    means = means[:, np.newaxis, :]
                else:
                    # If Batch=1, input might be (K, D).
                    # We need to distinguish (Batch, K*D) flat? No, dict values should be structured.
                    # Let's assume (Batch, K) for D=1 is common.
                    pass

            # Stds should be (Batch, K, D)
            if not self.use_covariance:
                if stds.ndim == 2 and self.dimensions == 1:
                    stds = stds[:, :, np.newaxis]

            # Covs should be (Batch, K, D, D)
            if self.use_covariance:
                if covs.ndim == 3:
                    # (Batch, K, D*D) or (Batch, K, D, D) ?
                    # If dimensions=2, D*D=4.
                    # If user passed (Batch, K, D, D) it is 4D.
                    pass

            batch_size = weights.shape[0]

            # Reshape to ensure correctness
            means = means.reshape(batch_size, self.n_components, self.dimensions)
            if self.use_covariance:
                covs = covs.reshape(
                    batch_size, self.n_components, self.dimensions, self.dimensions
                )
            else:
                stds = stds.reshape(batch_size, self.n_components, self.dimensions)

        else:
            # Parse parameters from flat array
            parameters = np.asarray(parameters)
            if parameters.ndim == 1:
                parameters = parameters[np.newaxis, :]

            batch_size = parameters.shape[0]
            idx = 0

            # Extract weights (N, n_components)
            weights = parameters[:, idx : idx + self.n_components]
            idx += self.n_components

            # Extract means (N, n_components * dimensions)
            means_flat = parameters[:, idx : idx + self.n_components * self.dimensions]
            means = means_flat.reshape(batch_size, self.n_components, self.dimensions)
            idx += self.n_components * self.dimensions

            # Extract standard deviations or covariance matrices
            if self.use_covariance:
                # For full covariance matrices: n_components * dimensions^2 elements
                covs_flat = parameters[
                    :, idx : idx + self.n_components * self.dimensions**2
                ]
                covs = covs_flat.reshape(
                    batch_size, self.n_components, self.dimensions, self.dimensions
                )
            else:
                # For diagonal covariance: n_components * dimensions elements
                stds_flat = parameters[
                    :, idx : idx + self.n_components * self.dimensions
                ]
                stds = stds_flat.reshape(batch_size, self.n_components, self.dimensions)

        # Normalize weights
        # weights: (N, K)
        weights_sum = np.sum(weights, axis=1, keepdims=True)
        # Avoid division by zero
        weights = weights / (weights_sum + 1e-10)

        samples = np.zeros((batch_size, self.dimensions))

        for i in range(batch_size):
            # Choose component k
            k = self.rng.choice(self.n_components, p=weights[i])

            if self.dimensions == 1 or not self.use_covariance:
                # means[i, k] is (D,), stds[i, k] is (D,)
                # self.rng.normal returns array if loc/scale are arrays
                samples[i] = self.rng.normal(loc=means[i, k], scale=stds[i, k])
            else:
                # Multivariate with full covariance
                cov_k = covs[i, k]
                cov_k = (cov_k + cov_k.T) / 2
                samples[i] = self.rng.multivariate_normal(mean=means[i, k], cov=cov_k)

        return samples
