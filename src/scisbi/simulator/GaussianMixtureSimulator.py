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

    def simulate(self, parameters, num_simulations=1):
        """
        Generate samples from a Gaussian Mixture distribution with given parameters.

        Args:
            parameters (dict or array-like): Parameters for the Gaussian Mixture distribution.
                If dict: Should contain 'weights', 'means', and 'stds' or 'covs' keys.
                    - 'weights': Array of shape (n_components,) containing mixture weights.
                    - 'means': Array of shape (n_components, dimensions) containing component means.
                    - 'stds' or 'covs': Component standard deviations or covariance matrices.
                If array-like: Format depends on configuration:
                    - With use_covariance=False: [weights, means, stds] flattened
                    - With use_covariance=True: [weights, means, covs] flattened
            num_simulations (int): Number of samples to generate.

        Returns:
            np.ndarray: Array of shape (num_simulations, dimensions) containing
                        the simulated observations.
        """
        super().simulate(parameters, num_simulations)

        # Process parameters based on their type
        if isinstance(parameters, dict):
            weights = np.asarray(parameters["weights"])
            means = np.asarray(parameters["means"])
            if self.use_covariance:
                covs = np.asarray(parameters["covs"])
            else:
                stds = np.asarray(parameters["stds"])
        else:
            # Parse parameters from flat array
            parameters = np.asarray(parameters)
            idx = 0

            # Extract weights (n_components elements)
            weights = parameters[idx : idx + self.n_components]
            idx += self.n_components

            # Extract means (n_components * dimensions elements)
            means = parameters[idx : idx + self.n_components * self.dimensions]
            means = means.reshape(self.n_components, self.dimensions)
            idx += self.n_components * self.dimensions

            # Extract standard deviations or covariance matrices
            if self.use_covariance:
                # For full covariance matrices: n_components * dimensions^2 elements
                covs = parameters[idx : idx + self.n_components * self.dimensions**2]
                covs = covs.reshape(self.n_components, self.dimensions, self.dimensions)
            else:
                # For diagonal covariance: n_components * dimensions elements
                stds = parameters[idx : idx + self.n_components * self.dimensions]
                stds = stds.reshape(self.n_components, self.dimensions)

        # Normalize weights
        weights = weights / np.sum(weights)

        # Sample component indices based on weights
        component_indices = self.rng.choice(
            self.n_components, size=num_simulations, p=weights
        )

        # Initialize output array
        samples = np.zeros((num_simulations, self.dimensions))

        # Generate samples for each component
        for k in range(self.n_components):
            # Find indices where this component was selected
            mask = component_indices == k
            count = np.sum(mask)

            if count > 0:
                if self.dimensions == 1 or not self.use_covariance:
                    # Univariate case or multivariate with diagonal covariance
                    component_samples = self.rng.normal(
                        loc=means[k], scale=stds[k], size=(count, self.dimensions)
                    )
                else:
                    # Multivariate with full covariance
                    # Ensure covariance matrix is symmetric and positive semi-definite
                    cov_k = covs[k]
                    cov_k = (cov_k + cov_k.T) / 2
                    component_samples = self.rng.multivariate_normal(
                        mean=means[k], cov=cov_k, size=count
                    )

                samples[mask] = component_samples

        return samples
