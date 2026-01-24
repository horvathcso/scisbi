import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Callable, List, Optional, Union
from ..base.inference import BaseInferenceAlgorithm
from ..base.simulator import BaseSimulator
from ..base.summary_statistic import BaseSummaryStatistic


class ABCRejectionSampling(BaseInferenceAlgorithm):
    """
    Approximate Bayesian Computation (ABC) using Rejection Sampling.
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
        Perform ABC rejection sampling.
        """
        if num_simulations <= 0:
            raise ValueError("num_simulations must be positive")

        # Override settings with runtime parameters
        tolerance = kwargs.get("tolerance", self.tolerance)
        max_attempts = kwargs.get("max_attempts", self.max_attempts)
        verbose = kwargs.get("verbose", self.verbose)

        # Handle observed_data shape
        observed_data = np.asarray(observed_data)
        if observed_data.ndim == 1:
             observed_data = observed_data[np.newaxis, :]

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
            theta_candidate = np.asarray(theta_candidate)
            # Ensure batch dim (1, P)
            if theta_candidate.ndim == 0:
                theta_candidate = theta_candidate.reshape(1, 1)
            elif theta_candidate.ndim == 1:
                theta_candidate = theta_candidate[np.newaxis, :]

            # Step 2: Simulate data with candidate parameters
            simulated_data = self.simulator.simulate(theta_candidate)

            # Step 3: Apply summary statistic if provided
            if self.summary_statistic is not None:
                simulated_summary = self.summary_statistic.compute(simulated_data)
            else:
                simulated_summary = simulated_data

            # Step 4: Compute distance
            distance = self.distance_function(simulated_summary, observed_summary)
            if np.ndim(distance) > 0:
                distance = distance.item()

            # Step 5: Accept or reject based on tolerance threshold
            if distance <= tolerance:
                # Store sample
                accepted_samples.append(theta_candidate.flatten())
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
    """

    def __init__(
        self,
        samples: List[Any],
        distances: List[float],
        tolerance: float,
        num_attempts: int,
    ):
        self.samples = samples
        self.distances = distances
        self.tolerance = tolerance
        self.num_attempts = num_attempts
        self.acceptance_rate = len(samples) / num_attempts if num_attempts > 0 else 0

    def sample(self, num_samples: int = 1) -> Union[Any, List[Any]]:
        if num_samples == 1:
            idx = np.random.choice(len(self.samples))
            return self.samples[idx]
        else:
            indices = np.random.choice(len(self.samples), size=num_samples, replace=True)
            return [self.samples[i] for i in indices]

    def get_samples(self) -> List[Any]:
        return self.samples.copy()

    def get_distances(self) -> List[float]:
        return self.distances.copy()

    def summary_statistics(self) -> dict:
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
        if not self.samples:
            raise ValueError("No samples available to plot.")

        # Samples are list of arrays
        samples_arr = np.array(self.samples)

        # Handle 1D samples or extract the specified parameter
        if parameter_index is None:
            if samples_arr.ndim > 1 and samples_arr.shape[1] > 1:
                 raise ValueError(
                    "Samples are multi-dimensional. Specify parameter_index."
                )
            parameter_values = samples_arr.flatten()
        else:
            parameter_values = samples_arr[:, parameter_index]

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
        return len(self.samples)

    def __repr__(self) -> str:
        return (
            f"ABCPosterior(num_samples={len(self.samples)}, "
            f"acceptance_rate={self.acceptance_rate:.4f}, "
            f"tolerance={self.tolerance})"
        )
