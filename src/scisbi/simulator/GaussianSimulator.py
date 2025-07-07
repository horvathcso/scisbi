import numpy as np
from scisbi.base.simulator import BaseSimulator


class GaussianSimulator(BaseSimulator):
    """
    Simulator for data from a Gaussian distribution.

    This simulator generates observations from a Gaussian (normal) distribution
    with configurable mean and standard deviation parameters.
    """

    def __init__(self, dimensions=1, use_covariance=False, seed=None, **kwargs):
        """
        Initialize the Gaussian simulator.

        Args:
            dimensions (int): Number of dimensions for the Gaussian. Default is 1 (univariate).
            use_covariance (bool): If True, use full covariance matrix for multivariate
                                   Gaussian. Only applies when dimensions > 1. Default is False.
            seed (int, optional): Random seed for reproducibility.
            **kwargs: Additional configuration parameters passed to BaseSimulator.
        """
        super().__init__(**kwargs)
        self.dimensions = dimensions
        self.use_covariance = use_covariance and dimensions > 1
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

    def simulate(self, parameters, num_simulations=1):
        """
        Generate samples from a Gaussian distribution with given parameters.

        Args:
            parameters (dict or array-like): Parameters for the Gaussian distribution.
                If dict: Should contain 'mean' and 'std' or 'cov' keys.
                If array-like: First half represents mean, second half represents std or
                flattened covariance (if use_covariance=True).
            num_simulations (int): Number of samples to generate.

        Returns:
            np.ndarray: Array of shape (num_simulations, dimensions) containing
                        the simulated observations.
        """
        super().simulate(parameters, num_simulations)

        # Process parameters based on their type
        if isinstance(parameters, dict):
            mean = np.asarray(parameters["mean"])
            if self.use_covariance:
                cov = np.asarray(parameters["cov"])
            else:
                std = np.asarray(parameters["std"])
        else:
            parameters = np.asarray(parameters)
            if len(parameters) == 2 * self.dimensions:
                mean = parameters[: self.dimensions]
                std = parameters[self.dimensions :]
            elif (
                self.use_covariance
                and len(parameters) == self.dimensions + self.dimensions**2
            ):
                mean = parameters[: self.dimensions]
                cov = parameters[self.dimensions :].reshape(
                    self.dimensions, self.dimensions
                )
            else:
                raise ValueError(
                    f"Parameters shape {parameters.shape} does not match expected format"
                )

        # Generate samples
        if self.dimensions == 1:
            # Univariate case
            samples = self.rng.normal(
                loc=mean, scale=std, size=(num_simulations, self.dimensions)
            )
        else:
            # Multivariate case
            if self.use_covariance:
                # Ensure covariance matrix is symmetric and positive semi-definite
                cov = (cov + cov.T) / 2
                samples = self.rng.multivariate_normal(
                    mean=mean, cov=cov, size=num_simulations
                )
            else:
                # Independent multivariate sampling using diagonal covariance
                samples = self.rng.normal(
                    loc=mean, scale=std, size=(num_simulations, self.dimensions)
                )

        return samples
