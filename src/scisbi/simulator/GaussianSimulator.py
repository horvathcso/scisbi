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

    def simulate(self, parameters):
        """
        Generate samples from a Gaussian distribution with given parameters.

        Args:
            parameters (dict or array-like): Parameters for the Gaussian distribution.
                If dict: Should contain 'mean' and 'std' or 'cov' keys.
                         Values should be arrays of shape (Batch_Size, Dimensions)
                         or (Batch_Size, Dimensions, Dimensions) for cov.
                If array-like: Should be shape (Batch_Size, Num_Params).
                               If use_covariance=False, Num_Params = 2 * Dimensions.
                               First half is mean, second half is std.
                               If use_covariance=True, Num_Params = Dimensions + Dimensions^2.

        Returns:
            np.ndarray: Array of shape (Batch_Size, Dimensions) containing
                        the simulated observations.
        """
        # Process parameters based on their type
        if isinstance(parameters, dict):
            mean = np.asarray(parameters["mean"])
            if self.use_covariance:
                cov = np.asarray(parameters["cov"])
            else:
                std = np.asarray(parameters["std"])

            # Ensure batch dimension
            if mean.ndim == 1:
                mean = mean[np.newaxis, :]
            if not self.use_covariance and std.ndim == 1:
                std = std[np.newaxis, :]
            if self.use_covariance and cov.ndim == 2:
                cov = cov[np.newaxis, :, :]

            batch_size = mean.shape[0]

        else:
            parameters = np.asarray(parameters)
            # Ensure batch dimension
            if parameters.ndim == 1:
                parameters = parameters[np.newaxis, :]

            batch_size = parameters.shape[0]

            if not self.use_covariance:
                if parameters.shape[1] == 2 * self.dimensions:
                    mean = parameters[:, : self.dimensions]
                    std = parameters[:, self.dimensions :]
                else:
                    raise ValueError(
                        f"Parameters shape {parameters.shape} does not match expected format (Batch, {2 * self.dimensions})"
                    )
            else:
                expected_dim = self.dimensions + self.dimensions**2
                if parameters.shape[1] == expected_dim:
                    mean = parameters[:, : self.dimensions]
                    cov_flat = parameters[:, self.dimensions :]
                    cov = cov_flat.reshape(batch_size, self.dimensions, self.dimensions)
                else:
                    raise ValueError(
                        f"Parameters shape {parameters.shape} does not match expected format (Batch, {expected_dim})"
                    )

        # Generate samples
        if self.dimensions == 1:
            # Univariate case
            # mean shape: (Batch, 1), std shape: (Batch, 1)
            # Output: (Batch, 1)
            samples = self.rng.normal(
                loc=mean, scale=std, size=(batch_size, self.dimensions)
            )
        else:
            # Multivariate case
            if self.use_covariance:
                # Ensure covariance matrix is symmetric and positive semi-definite
                # cov shape: (Batch, D, D)
                cov = (cov + np.transpose(cov, (0, 2, 1))) / 2

                # numpy multivariate_normal doesn't support broadcasting mean/cov for batch
                # efficiently in old versions, but let's check.
                # It does NOT support batching mean/cov naturally to produce (Batch, D) directly
                # if size is not provided, it draws 1 sample.
                # If we want 1 sample per mean/cov pair, we iterate.

                samples = np.empty((batch_size, self.dimensions))
                for i in range(batch_size):
                    samples[i] = self.rng.multivariate_normal(mean=mean[i], cov=cov[i])
            else:
                # Independent multivariate sampling using diagonal covariance
                # mean shape: (Batch, D), std shape: (Batch, D)
                # Output: (Batch, D)
                # numpy normal DOES support broadcasting.
                samples = self.rng.normal(
                    loc=mean, scale=std, size=(batch_size, self.dimensions)
                )

        return samples
