import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Callable, List, Optional, Union, Tuple
from ..base.inference import BaseInferenceAlgorithm
from ..base.simulator import BaseSimulator
from ..base.summary_statistic import BaseSummaryStatistic


class ABCSMC(BaseInferenceAlgorithm):
    """
    Approximate Bayesian Computation using Sequential Monte Carlo (ABC-SMC).
    """

    def __init__(
        self,
        simulator: BaseSimulator,
        prior: Any,
        distance_function: Callable[[Any, Any], float],
        tolerance_schedule: List[float],
        perturbation_kernel: Callable[[Any], Any],
        num_particles: int = 1000,
        summary_statistic: Optional[BaseSummaryStatistic] = None,
        **kwargs: Any,
    ):
        """
        Initialize the ABC-SMC algorithm.
        """
        super().__init__(simulator, prior, summary_statistic, **kwargs)

        if not callable(distance_function):
            raise TypeError("distance_function must be callable")
        if not callable(perturbation_kernel):
            raise TypeError("perturbation_kernel must be callable")
        if not isinstance(tolerance_schedule, list) or len(tolerance_schedule) == 0:
            raise ValueError("tolerance_schedule must be a non-empty list")
        if not all(tol >= 0 for tol in tolerance_schedule):
            raise ValueError("All tolerances must be non-negative")
        if tolerance_schedule != sorted(tolerance_schedule, reverse=True):
            raise ValueError("tolerance_schedule must be decreasing")
        if num_particles <= 0:
            raise ValueError("num_particles must be positive")

        self.distance_function = distance_function
        self.tolerance_schedule = tolerance_schedule
        self.perturbation_kernel = perturbation_kernel
        self.num_particles = num_particles
        self.max_attempts_per_particle = kwargs.get("max_attempts_per_particle", 1000)
        self.verbose = kwargs.get("verbose", False)
        self.adaptive_tolerance = kwargs.get("adaptive_tolerance", False)
        self.ess_threshold = kwargs.get("effective_sample_size_threshold", 0.5)

    def infer(
        self,
        observed_data: Any,
        **kwargs: Any,
    ) -> "ABCSMCPosterior":
        """
        Perform ABC-SMC sampling to approximate the posterior distribution.
        """
        # Override settings with runtime parameters
        tolerance_schedule = kwargs.get("tolerance_schedule", self.tolerance_schedule)
        num_particles = kwargs.get("num_particles", self.num_particles)
        max_attempts = kwargs.get(
            "max_attempts_per_particle", self.max_attempts_per_particle
        )
        verbose = kwargs.get("verbose", self.verbose)
        adaptive_tolerance = kwargs.get("adaptive_tolerance", self.adaptive_tolerance)

        # Process observed data through summary statistic if provided
        observed_data = np.asarray(observed_data)
        if observed_data.ndim == 1:
             observed_data = observed_data[np.newaxis, :]

        if self.summary_statistic is not None:
            observed_summary = self.summary_statistic.compute(observed_data)
        else:
            observed_summary = observed_data

        # Initialize storage for population evolution
        all_populations = []
        all_weights = []
        all_distances = []
        iteration_info = []

        if verbose:
            print(f"Starting ABC-SMC with {num_particles} particles...")
            print(f"Tolerance schedule: {tolerance_schedule}")

        # Main ABC-SMC loop over tolerance levels
        current_particles = None
        current_weights = None

        for t, tolerance in enumerate(tolerance_schedule):
            if verbose:
                print(
                    f"\nIteration {t + 1}/{len(tolerance_schedule)}, tolerance = {tolerance:.6f}"
                )

            # S2: Generate particles for current tolerance level
            if t == 0:
                # S2.1: First iteration - sample from prior
                particles, weights, distances, attempts = self._sample_from_prior(
                    observed_summary, tolerance, num_particles, max_attempts, verbose
                )
            else:
                # S2.1: Subsequent iterations - resample and perturb
                particles, weights, distances, attempts = self._resample_and_perturb(
                    current_particles,
                    current_weights,
                    observed_summary,
                    tolerance,
                    num_particles,
                    max_attempts,
                    verbose,
                )

            # S3: Normalize weights
            if len(weights) > 0:
                weights = np.array(weights)
                weights = weights / np.sum(weights)
            else:
                raise RuntimeError(f"No particles accepted at tolerance {tolerance}")

            # Store population
            all_populations.append(particles.copy())
            all_weights.append(weights.copy())
            all_distances.append(distances.copy())

            # Compute diagnostics
            ess = self._effective_sample_size(weights)
            acceptance_rate = len(particles) / attempts if attempts > 0 else 0

            iteration_info.append(
                {
                    "tolerance": tolerance,
                    "num_particles": len(particles),
                    "effective_sample_size": ess,
                    "acceptance_rate": acceptance_rate,
                    "total_attempts": attempts,
                    "mean_distance": np.mean(distances),
                    "std_distance": np.std(distances),
                }
            )

            if verbose:
                print(f"Accepted {len(particles)} particles (attempts: {attempts})")
                print(f"Acceptance rate: {acceptance_rate:.4f}")
                print(f"Effective sample size: {ess:.1f} ({ess / len(particles):.3f})")
                print(f"Distance: {np.mean(distances):.4f} ± {np.std(distances):.4f}")

            # Update for next iteration
            current_particles = particles
            current_weights = weights

            # Adaptive tolerance adjustment
            if adaptive_tolerance and t < len(tolerance_schedule) - 1:
                next_tolerance = self._adaptive_tolerance_update(
                    distances, tolerance_schedule[t + 1]
                )
                if verbose and next_tolerance != tolerance_schedule[t + 1]:
                    print(
                        f"Adaptive tolerance: {tolerance_schedule[t + 1]:.6f} → {next_tolerance:.6f}"
                    )
                tolerance_schedule[t + 1] = next_tolerance

        if verbose:
            print("\nABC-SMC completed!")
            print(f"Final population: {len(current_particles)} particles")
            final_ess = self._effective_sample_size(current_weights)
            print(f"Final effective sample size: {final_ess:.1f}")

        # Return ABC-SMC posterior object
        return ABCSMCPosterior(
            particles=current_particles,
            weights=current_weights,
            distances=distances,
            tolerance_schedule=tolerance_schedule,
            all_populations=all_populations,
            all_weights=all_weights,
            all_distances=all_distances,
            iteration_info=iteration_info,
            num_particles=num_particles,
        )

    def _sample_from_prior(
        self,
        observed_summary: Any,
        tolerance: float,
        num_particles: int,
        max_attempts: int,
        verbose: bool,
    ) -> Tuple[List[Any], List[float], List[float], int]:
        """Sample initial population from prior distribution."""
        particles = []
        distances = []
        attempts = 0

        while (
            len(particles) < num_particles and attempts < max_attempts * num_particles
        ):
            # Sample from prior
            theta = self.prior.sample()
            attempts += 1

            try:
                # Simulate data
                theta_batch = np.asarray(theta)
                if theta_batch.ndim == 0:
                    theta_batch = theta_batch.reshape(1, 1)
                elif theta_batch.ndim == 1:
                    theta_batch = theta_batch[np.newaxis, :]

                simulated_data = self.simulator.simulate(theta_batch)

                # Apply summary statistic if provided
                if self.summary_statistic is not None:
                    simulated_summary = self.summary_statistic.compute(simulated_data)
                else:
                    simulated_summary = simulated_data

                # Compute distance
                distance = self.distance_function(simulated_summary, observed_summary)
                if np.ndim(distance) > 0:
                    distance = distance.item()

                # Accept if within tolerance
                if distance <= tolerance:
                    particles.append(theta)
                    distances.append(distance)

                    if verbose and len(particles) % max(1, num_particles // 10) == 0:
                        acceptance_rate = len(particles) / attempts
                        print(
                            f"  Progress: {len(particles)}/{num_particles} "
                            f"(acceptance rate: {acceptance_rate:.4f})"
                        )

            except Exception as e:
                # print(f"DEBUG ABCSMC: {e}") # Uncomment for debugging
                continue  # Skip failed simulations

        # Initialize uniform weights for first iteration
        weights = [1.0] * len(particles)

        return particles, weights, distances, attempts

    def _resample_and_perturb(
        self,
        previous_particles: List[Any],
        previous_weights: np.ndarray,
        observed_summary: Any,
        tolerance: float,
        num_particles: int,
        max_attempts: int,
        verbose: bool,
    ) -> Tuple[List[Any], List[float], List[float], int]:
        """Resample from previous population and perturb particles."""
        particles = []
        weights = []
        distances = []
        attempts = 0

        # Normalize previous weights for resampling
        if np.sum(previous_weights) > 0:
            resample_weights = previous_weights / np.sum(previous_weights)
        else:
            resample_weights = np.ones(len(previous_particles)) / len(
                previous_particles
            )

        while (
            len(particles) < num_particles and attempts < max_attempts * num_particles
        ):
            attempts += 1

            # Sample particle from previous population with probability proportional to weights
            sampled_idx = np.random.choice(len(previous_particles), p=resample_weights)
            theta_star = previous_particles[sampled_idx]

            # Perturb particle using perturbation kernel
            theta_candidate = self.perturbation_kernel(theta_star)

            # Check prior probability
            if self.prior.log_prob(theta_candidate) == -np.inf:
                continue  # Reject if outside prior support

            try:
                # Simulate data
                theta_batch = np.asarray(theta_candidate)
                if theta_batch.ndim == 0:
                    theta_batch = theta_batch.reshape(1, 1)
                elif theta_batch.ndim == 1:
                    theta_batch = theta_batch[np.newaxis, :]

                simulated_data = self.simulator.simulate(theta_batch)

                # Apply summary statistic if provided
                if self.summary_statistic is not None:
                    simulated_summary = self.summary_statistic.compute(simulated_data)
                else:
                    simulated_summary = simulated_data

                # Compute distance
                distance = self.distance_function(simulated_summary, observed_summary)
                if np.ndim(distance) > 0:
                    distance = distance.item()

                # Accept if within tolerance
                if distance <= tolerance:
                    particles.append(theta_candidate)
                    distances.append(distance)

                    # Calculate importance weight
                    # w_i = π(θ_i) / Σ_j w_{t-1}^j K_t(θ_{t-1}^j, θ_i)
                    numerator = np.exp(self.prior.log_prob(theta_candidate))

                    # Compute denominator: sum over all previous particles
                    denominator = 0.0
                    for j, prev_theta in enumerate(previous_particles):
                        kernel_prob = self._kernel_probability(
                            prev_theta, theta_candidate
                        )
                        denominator += previous_weights[j] * kernel_prob

                    if denominator > 0:
                        weight = numerator / denominator
                    else:
                        weight = 1.0  # Fallback for numerical issues

                    weights.append(weight)

                    if verbose and len(particles) % max(1, num_particles // 10) == 0:
                        acceptance_rate = len(particles) / attempts
                        print(
                            f"  Progress: {len(particles)}/{num_particles} "
                            f"(acceptance rate: {acceptance_rate:.4f})"
                        )

            except Exception:
                continue  # Skip failed simulations

        return particles, weights, distances, attempts

    def _kernel_probability(self, theta_from: Any, theta_to: Any) -> float:
        """
        Compute probability density of perturbation kernel K(θ_to | θ_from).
        """
        # For symmetric kernels, return 1.0
        # In practice, this should compute the actual kernel probability
        return 1.0

    def _effective_sample_size(self, weights: np.ndarray) -> float:
        """Compute effective sample size of weighted particles."""
        if len(weights) == 0:
            return 0.0
        normalized_weights = weights / np.sum(weights)
        return 1.0 / np.sum(normalized_weights**2)

    def _adaptive_tolerance_update(
        self, distances: List[float], scheduled_tolerance: float
    ) -> float:
        """
        Adaptively update tolerance based on current distance distribution.
        """
        if len(distances) == 0:
            return scheduled_tolerance

        # Use median of current distances, but don't exceed scheduled tolerance
        adaptive_tol = np.median(distances)
        return min(adaptive_tol, scheduled_tolerance)


class ABCSMCPosterior:
    """
    Container for ABC-SMC sampling results representing the approximate posterior.
    """

    def __init__(
        self,
        particles: List[Any],
        weights: np.ndarray,
        distances: List[float],
        tolerance_schedule: List[float],
        all_populations: List[List[Any]],
        all_weights: List[np.ndarray],
        all_distances: List[List[float]],
        iteration_info: List[dict],
        num_particles: int,
    ):
        self.particles = particles
        self.weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
        self.distances = distances
        self.tolerance_schedule = tolerance_schedule
        self.all_populations = all_populations
        self.all_weights = all_weights
        self.all_distances = all_distances
        self.iteration_info = iteration_info
        self.num_particles = num_particles
        self.final_tolerance = tolerance_schedule[-1] if tolerance_schedule else 0.0

    def sample(self, num_samples: int = 1) -> Union[Any, List[Any]]:
        if len(self.particles) == 0:
            raise ValueError("No particles available for sampling")

        # Sample with probability proportional to weights
        # Ensure weights are 1D for choice
        weights_1d = np.asarray(self.weights).flatten()
        # Re-normalize if needed (should be normalized in constructor)
        if np.abs(np.sum(weights_1d) - 1.0) > 1e-5:
             weights_1d = weights_1d / np.sum(weights_1d)

        indices = np.random.choice(
            len(self.particles), size=num_samples, replace=True, p=weights_1d
        )

        if num_samples == 1:
            return self.particles[indices[0]]
        else:
            return [self.particles[i] for i in indices]

    def get_particles(self) -> List[Any]:
        return self.particles.copy()

    def get_weights(self) -> np.ndarray:
        return self.weights.copy()

    def get_distances(self) -> List[float]:
        return self.distances.copy()

    def get_iteration_info(self) -> List[dict]:
        return self.iteration_info.copy()

    def effective_sample_size(self) -> float:
        if len(self.weights) == 0:
            return 0.0
        return 1.0 / np.sum(self.weights**2)

    def summary_statistics(self) -> dict:
        final_info = self.iteration_info[-1] if self.iteration_info else {}

        return {
            "num_particles": len(self.particles),
            "effective_sample_size": self.effective_sample_size(),
            "final_tolerance": self.final_tolerance,
            "num_iterations": len(self.tolerance_schedule),
            "tolerance_schedule": self.tolerance_schedule,
            "final_acceptance_rate": final_info.get("acceptance_rate", 0.0),
            "final_mean_distance": final_info.get("mean_distance", np.nan),
            "final_std_distance": final_info.get("std_distance", np.nan),
            "mean_distance": np.mean(self.distances) if self.distances else np.nan,
            "max_distance": np.max(self.distances) if self.distances else np.nan,
            "min_distance": np.min(self.distances) if self.distances else np.nan,
        }

    def plot(self, parameter_index: Optional[int] = None, bins: int = 30) -> None:
        if not self.particles:
            raise ValueError("No particles available to plot.")

        # Handle 1D samples or extract the specified parameter
        particles_arr = np.array(self.particles)

        if parameter_index is None:
            if particles_arr.ndim > 1 and particles_arr.shape[1] > 1:
                raise ValueError(
                    "Samples are multi-dimensional. Specify parameter_index."
                )
            parameter_values = particles_arr.flatten()
        else:
            parameter_values = particles_arr[:, parameter_index]

        plt.hist(
            parameter_values,
            bins=bins,
            density=True,
            alpha=0.7,
            color="purple",
            edgecolor="black",
            weights=self.weights,
        )
        title = "SMC Posterior Distribution" + (
            f" (Parameter {parameter_index})" if parameter_index is not None else ""
        )
        plt.title(title)
        plt.xlabel(
            f"Parameter {parameter_index}" if parameter_index is not None else "Samples"
        )
        plt.ylabel("Density")
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_evolution(self, parameter_index: Optional[int] = None) -> None:
        if not self.all_populations:
            raise ValueError("No population history available to plot.")

        num_iterations = len(self.all_populations)
        fig, axes = plt.subplots(1, min(num_iterations, 4), figsize=(15, 4))

        if num_iterations == 1:
            axes = [axes]
        elif num_iterations > 4:
            # Show first, some middle, and last iterations
            indices = [
                0,
                num_iterations // 3,
                2 * num_iterations // 3,
                num_iterations - 1,
            ]
            indices = list(set(indices))[:4]  # Remove duplicates and limit to 4
        else:
            indices = list(range(num_iterations))

        for i, iter_idx in enumerate(indices):
            particles = self.all_populations[iter_idx]
            weights = self.all_weights[iter_idx]
            tolerance = self.tolerance_schedule[iter_idx]

            # Extract parameter values
            particles_arr = np.array(particles)
            if parameter_index is None:
                if particles_arr.ndim > 1 and particles_arr.shape[1] > 1:
                    raise ValueError(
                        "Samples are multi-dimensional. Specify parameter_index."
                    )
                parameter_values = particles_arr.flatten()
            else:
                parameter_values = particles_arr[:, parameter_index]

            if i < len(axes):
                axes[i].hist(
                    parameter_values,
                    bins=25,
                    density=True,
                    alpha=0.7,
                    color="purple",
                    edgecolor="black",
                    weights=weights,
                )
                axes[i].set_title(f"Iteration {iter_idx + 1}\nε = {tolerance:.4f}")
                axes[i].set_xlabel(
                    f"Parameter {parameter_index}"
                    if parameter_index is not None
                    else "Value"
                )
                if i == 0:
                    axes[i].set_ylabel("Density")
                axes[i].grid(True, alpha=0.3)

        plt.suptitle("ABC-SMC: Evolution of Particle Distribution", fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_diagnostics(self) -> None:
        if not self.iteration_info:
            raise ValueError("No iteration information available.")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        iterations = range(1, len(self.iteration_info) + 1)

        # Tolerance schedule
        tolerances = [info["tolerance"] for info in self.iteration_info]
        axes[0, 0].semilogy(
            iterations, tolerances, "o-", color="blue", linewidth=2, markersize=6
        )
        axes[0, 0].set_title("Tolerance Schedule")
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel("Tolerance (log scale)")
        axes[0, 0].grid(True, alpha=0.3)

        # Effective sample size
        ess_values = [info["effective_sample_size"] for info in self.iteration_info]
        axes[0, 1].plot(
            iterations, ess_values, "o-", color="green", linewidth=2, markersize=6
        )
        axes[0, 1].axhline(
            y=self.num_particles / 2, color="red", linestyle="--", label="50% threshold"
        )
        axes[0, 1].set_title("Effective Sample Size")
        axes[0, 1].set_xlabel("Iteration")
        axes[0, 1].set_ylabel("ESS")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Acceptance rate
        acceptance_rates = [info["acceptance_rate"] for info in self.iteration_info]
        axes[1, 0].plot(
            iterations,
            acceptance_rates,
            "o-",
            color="orange",
            linewidth=2,
            markersize=6,
        )
        axes[1, 0].set_title("Acceptance Rate")
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].set_ylabel("Acceptance Rate")
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)

        # Mean distance
        mean_distances = [info["mean_distance"] for info in self.iteration_info]
        axes[1, 1].semilogy(
            iterations, mean_distances, "o-", color="red", linewidth=2, markersize=6
        )
        axes[1, 1].set_title("Mean Distance")
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("Mean Distance (log scale)")
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle("ABC-SMC: Algorithm Diagnostics", fontsize=16)
        plt.tight_layout()
        plt.show()

    def __len__(self) -> int:
        return len(self.particles)

    def __repr__(self) -> str:
        ess = self.effective_sample_size()
        return (
            f"ABCSMCPosterior(num_particles={len(self.particles)}, "
            f"effective_sample_size={ess:.1f}, "
            f"final_tolerance={self.final_tolerance})"
        )
