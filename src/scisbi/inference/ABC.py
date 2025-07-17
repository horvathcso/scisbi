import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Callable, List, Optional, Union
from ..base.inference import BaseInferenceAlgorithm
from ..base.simulator import BaseSimulator
from ..base.summary_statistic import BaseSummaryStatistic


class ABCRejectionSampling(BaseInferenceAlgorithm):
    """
    Approximate Bayesian Computation (ABC) using Rejection Sampling.

    ABC rejection sampling is a likelihood-free inference method that approximates
    the posterior distribution π(θ|x_obs) by accepting parameter samples θ* from
    the prior π(θ) only when the simulated data x* = f(x|θ*) is sufficiently
    close to the observed data x_obs according to a distance metric ρ(·,·) and
    tolerance threshold ε.

    The algorithm implements the following procedure:

    1. Sample candidate parameters θ* from the prior distribution π(θ).
    2. Generate synthetic data x* by running the simulator f(x|θ*) with θ*.
    3. Compute the distance ρ(x*, x_obs) between simulated and observed data.
    4. Accept θ* if ρ(x*, x_obs) ≤ ε, otherwise reject.
    5. Repeat until N accepted samples are collected.

    The accepted samples approximate draws from the ABC posterior:

    π_ABC(θ|x_obs) ∝ π(θ) · I[ρ(f(x|θ), x_obs) ≤ ε]

    where I[·] is the indicator function. As ε → 0, the ABC posterior approaches
    the true posterior π(θ|x_obs), but smaller ε values require more computational
    effort due to lower acceptance rates.

    The choice of distance function ρ and tolerance ε are critical:

    - ρ should capture relevant differences between datasets.
    - ε controls the approximation quality vs. computational cost trade-off.
    - Summary statistics can be used to reduce dimensionality and focus on
      informative features of the data.

    References:

    - Beaumont, M. A., Zhang, W., & Balding, D. J. (2002). Approximate Bayesian
      computation in population genetics. Genetics, 162(2), 2025-2035.
    - Scott A. Sisson, Yanan Fan, Mark Beaumont (2018). Handbook of
      approximate Bayesian computation. CRC press.
    - Pritchard, J. K., Seielstad, M. T., Perez-Lezaun, A., & Feldman, M. W. (1999).
      Population growth of human Y chromosomes: a study of Y chromosome microsatellites.

    """

    def __init__(
        self,
        simulator: BaseSimulator,
        prior: Any,
        distance_function: Callable[[Any, Any], float],
        tolerance: float,
        summary_statistic: Optional[BaseSummaryStatistic] = None,
        **kwargs: Any,
    ):
        """
        Initialize the ABC rejection sampling algorithm.

        Args:
            simulator (BaseSimulator): Simulator object with a 'simulate' method
                that generates synthetic data given parameters.

            prior (Any): Prior distribution object with 'sample' and 'log_prob' methods.
                Should support sampling parameter vectors θ ~ π(θ).

            distance_function (Callable[[Any, Any], float]): Function ρ(x_sim, x_obs)
                that computes the distance between simulated and observed
                data. Should return a non-negative float value.

            tolerance (float): Acceptance threshold ε ≥ 0. Smaller values yield more
                accurate approximations but lower acceptance rates.

            summary_statistic (Optional[BaseSummaryStatistic]): Optional summary statistic
                function to reduce data dimensionality before distance computation.

            **kwargs (Optional): Additional configuration parameters such as:

                - max_attempts (int): Maximum number of simulation attempts
                  before raising an error (default: 1000000).
                - verbose (bool): Whether to print progress information.

        """
        super().__init__(simulator, prior, summary_statistic, **kwargs)

        if not callable(distance_function):
            raise TypeError("distance_function must be callable")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")

        self.distance_function = distance_function
        self.tolerance = tolerance
        self.max_attempts = kwargs.get("max_attempts", 1000000)
        self.verbose = kwargs.get("verbose", False)

    def infer(
        self,
        observed_data: Any,
        num_simulations: int,
        **kwargs: Any,
    ) -> "ABCPosterior":
        """
        Perform ABC rejection sampling to approximate the posterior distribution.

        This method implements the core ABC rejection sampling algorithm as described
        in the class docstring. It repeatedly samples from the prior, simulates data,
        and accepts parameters when the distance to observed data is below tolerance.

        Args:
            observed_data (Any): The observed dataset x_obs to perform inference on.
                               If summary_statistic was provided during initialization,
                               this will be processed through the summary statistic.
            num_simulations (int): Target number N of accepted posterior samples.
                                 Must be positive.
            **kwargs: Runtime parameters that override initialization settings:
                     - tolerance (float): Override the tolerance threshold for this run
                     - max_attempts (int): Override maximum simulation attempts
                     - verbose (bool): Override verbosity setting

        Returns:
            ABCPosterior: Object containing the accepted parameter samples and
                         providing methods to query the approximate posterior.

        Raises:
            ValueError: If num_simulations <= 0
            RuntimeError: If maximum attempts exceeded before collecting enough samples

        Note:
            The acceptance rate depends heavily on the tolerance ε and the mismatch
            between prior and posterior. For complex problems, consider using
            adaptive tolerance selection or sequential ABC methods.
        """
        if num_simulations <= 0:
            raise ValueError("num_simulations must be positive")

        # Override settings with runtime parameters
        tolerance = kwargs.get("tolerance", self.tolerance)
        max_attempts = kwargs.get("max_attempts", self.max_attempts)
        verbose = kwargs.get("verbose", self.verbose)

        # Process observed data through summary statistic if provided
        if self.summary_statistic is not None:
            observed_summary = self.summary_statistic.compute(observed_data)
        else:
            observed_summary = observed_data

        # Initialize storage for accepted samples
        accepted_samples = []
        accepted_distances = []
        attempt_count = 0

        if verbose:
            print("Starting ABC rejection sampling...")
            print(f"Target samples: {num_simulations}, Tolerance: {tolerance}")

        # Main ABC rejection sampling loop
        while len(accepted_samples) < num_simulations:
            if attempt_count >= max_attempts:
                raise RuntimeError(
                    f"Maximum attempts ({max_attempts}) exceeded. "
                    f"Only {len(accepted_samples)}/{num_simulations} samples accepted. "
                    f"Consider increasing tolerance or max_attempts."
                )

            # Step 1: Sample parameters from prior
            theta_candidate = self.prior.sample()
            # Step 2: Simulate data with candidate parameters
            simulated_data = self.simulator.simulate(
                theta_candidate, num_simulations=len(observed_summary)
            )

            # Step 3: Apply summary statistic if provided
            if self.summary_statistic is not None:
                simulated_summary = self.summary_statistic.compute(simulated_data)
            else:
                simulated_summary = simulated_data

            # Step 4: Compute distance between simulated and observed data
            distance = self.distance_function(simulated_summary, observed_summary)

            # Step 5: Accept or reject based on tolerance threshold
            if distance <= tolerance:
                accepted_samples.append(theta_candidate)
                accepted_distances.append(distance)

                if (
                    verbose
                    and len(accepted_samples) % max(1, num_simulations // 10) == 0
                ):
                    acceptance_rate = (
                        len(accepted_samples) / attempt_count
                        if attempt_count > 0
                        else 0
                    )
                    print(
                        f"Progress: {len(accepted_samples)}/{num_simulations} "
                        f"(acceptance rate: {acceptance_rate:.3f})"
                    )

            attempt_count += 1

        if verbose:
            final_acceptance_rate = len(accepted_samples) / attempt_count
            print("ABC rejection sampling completed!")
            print(f"Final acceptance rate: {final_acceptance_rate:.4f}")
            print(f"Total attempts: {attempt_count}")

        # Return ABC posterior object
        return ABCPosterior(
            samples=accepted_samples,
            distances=accepted_distances,
            tolerance=tolerance,
            num_attempts=attempt_count,
        )


class ABCPosterior:
    """
    Container for ABC rejection sampling results representing the approximate posterior.

    This class stores the accepted parameter samples from ABC rejection sampling
    and provides methods to query and analyze the approximate posterior distribution.
    The samples can be used for posterior inference, uncertainty quantification,
    and statistical analysis.
    """

    def __init__(
        self,
        samples: List[Any],
        distances: List[float],
        tolerance: float,
        num_attempts: int,
    ):
        """
        Initialize the ABC posterior object.

        Args:
            samples (List[Any]): List of accepted parameter samples θ*
            distances (List[float]): Corresponding distance values ρ(x*, x_obs)
            tolerance (float): Tolerance threshold ε used for acceptance
            num_attempts (int): Total number of simulation attempts
        """
        self.samples = samples
        self.distances = distances
        self.tolerance = tolerance
        self.num_attempts = num_attempts
        self.acceptance_rate = len(samples) / num_attempts if num_attempts > 0 else 0

    def sample(self, num_samples: int = 1) -> Union[Any, List[Any]]:
        """
        Draw samples from the ABC posterior.

        Args:
            num_samples (int): Number of samples to draw

        Returns:
            Single sample if num_samples=1, otherwise list of samples
        """
        if num_samples == 1:
            return np.random.choice(self.samples)
        else:
            return np.random.choice(
                self.samples, size=num_samples, replace=True
            ).tolist()

    def get_samples(self) -> List[Any]:
        """Return all accepted samples."""
        return self.samples.copy()

    def get_distances(self) -> List[float]:
        """Return all accepted distances."""
        return self.distances.copy()

    def summary_statistics(self) -> dict:
        """
        Compute summary statistics of the ABC posterior.

        Returns:
            Dictionary with posterior summary information
        """
        return {
            "num_samples": len(self.samples),
            "acceptance_rate": self.acceptance_rate,
            "tolerance": self.tolerance,
            "num_attempts": self.num_attempts,
            "mean_distance": np.mean(self.distances),
            "max_distance": np.max(self.distances),
            "min_distance": np.min(self.distances),
        }

    def plot(self, parameter_index: Optional[int] = None, bins: int = 30) -> None:
        """
        Plot a histogram of the posterior samples for a specific parameter or 1D samples.

        Args:
            parameter_index (Optional[int]): Index of the parameter to plot. If None,
                                             assumes samples are 1D (default: None).
            bins (int): Number of bins for the histogram (default: 30).
        """
        if not self.samples:
            raise ValueError("No samples available to plot.")

        # Handle 1D samples or extract the specified parameter
        if parameter_index is None:
            if isinstance(self.samples[0], (list, tuple)):
                raise ValueError(
                    "Samples are multi-dimensional. Specify parameter_index."
                )
            parameter_values = self.samples
        else:
            parameter_values = [sample[parameter_index] for sample in self.samples]

        plt.hist(parameter_values, bins=bins, density=True, alpha=0.7, color="blue")
        title = "Posterior Distribution" + (
            f" (Parameter {parameter_index})" if parameter_index is not None else ""
        )
        plt.title(title)
        plt.xlabel(
            f"Parameter {parameter_index}" if parameter_index is not None else "Samples"
        )
        plt.ylabel("Density")
        plt.grid(True)
        plt.show()

    def __len__(self) -> int:
        """Return number of accepted samples."""
        return len(self.samples)

    def __repr__(self) -> str:
        return (
            f"ABCPosterior(num_samples={len(self.samples)}, "
            f"acceptance_rate={self.acceptance_rate:.4f}, "
            f"tolerance={self.tolerance})"
        )
