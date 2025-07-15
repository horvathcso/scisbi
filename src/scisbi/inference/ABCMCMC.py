import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Callable, List, Optional, Union
from ..base.inference import BaseInferenceAlgorithm
from ..base.simulator import BaseSimulator
from ..base.summary_statistic import BaseSummaryStatistic


class ABCMCMC(BaseInferenceAlgorithm):
    """
    Approximate Bayesian Computation using Markov Chain Monte Carlo (ABC-MCMC).

    ABC-MCMC is a likelihood-free inference method that combines the approximate
    nature of ABC with the efficient exploration of MCMC sampling. Instead of
    independently sampling from the prior as in ABC rejection sampling, ABC-MCMC
    constructs a Markov chain that explores the parameter space more efficiently
    by proposing new states based on the current state.

    The algorithm implements the following procedure:

    1. Initialize the chain at θ₀.
    2. At iteration i, propose θ* from proposal distribution q(θ|θᵢ).
    3. Simulate data x* using θ* with the forward model f(x|θ*).
    4. If ρ(x*, x_obs) ≤ ε, proceed to acceptance step, otherwise reject.
    5. Accept θ* with probability α = min(1, [π(θ*)q(θᵢ|θ*)] / [π(θᵢ)q(θ*|θᵢ)]).
    6. Set θᵢ₊₁ = θ* if accepted, otherwise θᵢ₊₁ = θᵢ.
    7. Increment i and repeat from step 2.

    The key advantage over ABC rejection sampling is that ABC-MCMC can achieve
    better mixing and exploration of the posterior, especially in high-dimensional
    parameter spaces or when the prior-posterior mismatch is large. However, it
    requires careful tuning of the proposal distribution and may suffer from
    correlation between successive samples.

    The choice of proposal distribution q(θ|θᵢ) is crucial:

    - Common choices include multivariate normal: q(θ|θᵢ) = N(θᵢ, Σ).
    - Proposal covariance Σ should be tuned to achieve reasonable acceptance rates.
    - Adaptive proposals can be used to improve efficiency during sampling.

    References:

    - Marjoram, P., Molitor, J., Plagnol, V., & Tavaré, S. (2003). Markov chain
      Monte Carlo without likelihoods. Proceedings of the National Academy of
      Sciences, 100(26), 15324-15328.
    - Scott A. Sisson, Yanan Fan, Mark Beaumont (2018). Handbook of
      approximate Bayesian computation. CRC press.
    - Wegmann, D., Leuenberger, C., & Excoffier, L. (2009). Efficient approximate
      Bayesian computation coupled with Markov chain Monte Carlo without
      likelihood. Genetics, 182(4), 1207-1218.
    """

    def __init__(
        self,
        simulator: BaseSimulator,
        prior: Any,
        distance_function: Callable[[Any, Any], float],
        tolerance: float,
        proposal_distribution: Callable[[Any], Any],
        summary_statistic: Optional[BaseSummaryStatistic] = None,
        **kwargs: Any,
    ):
        """
        Initialize the ABC-MCMC algorithm.

        Args:
            simulator (BaseSimulator): Simulator object with a 'simulate' method
                that generates synthetic data given parameters.

            prior (Any): Prior distribution object with 'sample' and 'log_prob' methods.
                Should support sampling parameter vectors θ ~ π(θ) and
                computing log-probabilities log π(θ).

            distance_function (Callable[[Any, Any], float]): Function ρ(x_sim, x_obs)
                that computes the distance between simulated and observed
                data. Should return a non-negative float value.

            tolerance (float): Acceptance threshold ε ≥ 0. Smaller values yield more
                accurate approximations but lower acceptance rates.

            proposal_distribution (Callable[[Any], Any]): Proposal distribution q(θ|θᵢ)
                that takes current state θᵢ and returns a proposed state θ*.
                Should be symmetric for simplicity.

            summary_statistic (Optional[BaseSummaryStatistic]): Optional summary statistic
                function to reduce data dimensionality before distance computation.

            **kwargs (Optional):
                Additional configuration parameters such as:
                    - max_attempts_per_step (int): Maximum simulation attempts per
                    MCMC step before moving to next iteration (default: 100).
                    - verbose (bool): Whether to print progress information.
                    - thin (int): Thinning interval for storing samples (default: 1).
                    - burn_in (int): Number of burn-in iterations to discard (default: 0).


        """
        super().__init__(simulator, prior, summary_statistic, **kwargs)

        if not callable(distance_function):
            raise TypeError("distance_function must be callable")
        if not callable(proposal_distribution):
            raise TypeError("proposal_distribution must be callable")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")

        self.distance_function = distance_function
        self.tolerance = tolerance
        self.proposal_distribution = proposal_distribution
        self.max_attempts_per_step = kwargs.get("max_attempts_per_step", 100)
        self.verbose = kwargs.get("verbose", False)
        self.thin = kwargs.get("thin", 1)
        self.burn_in = kwargs.get("burn_in", 0)

    def infer(
        self,
        observed_data: Any,
        num_iterations: int,
        initial_state: Optional[Any] = None,
        **kwargs: Any,
    ) -> "ABCMCMCPosterior":
        """
        Perform ABC-MCMC sampling to approximate the posterior distribution.

        This method implements the core ABC-MCMC algorithm as described in the
        class docstring. It constructs a Markov chain that explores the parameter
        space while respecting the ABC tolerance constraint.

        Args:
            observed_data (Any): The observed dataset x_obs to perform inference on.
                               If summary_statistic was provided during initialization,
                               this will be processed through the summary statistic.
            num_iterations (int): Number of MCMC iterations to run. The effective
                                number of samples will be smaller due to thinning
                                and burn-in.
            initial_state (Optional[Any]): Starting value θ₀ for the chain. If None,
                                          samples from the prior distribution.
            **kwargs: Runtime parameters that override initialization settings:
                     - tolerance (float): Override the tolerance threshold
                     - max_attempts_per_step (int): Override maximum attempts per step
                     - verbose (bool): Override verbosity setting
                     - thin (int): Override thinning interval
                     - burn_in (int): Override burn-in period

        Returns:
            ABCMCMCPosterior: Object containing the MCMC samples and chain diagnostics.

        Raises:
            ValueError: If num_iterations <= 0
            RuntimeError: If unable to find valid initial state or if chain gets stuck

        Note:
            The effective sample size will be approximately
            (num_iterations - burn_in) // thin. Consider autocorrelation when
            choosing thinning intervals for nearly independent samples.
        """
        if num_iterations <= 0:
            raise ValueError("num_iterations must be positive")

        # Override settings with runtime parameters
        tolerance = kwargs.get("tolerance", self.tolerance)
        max_attempts_per_step = kwargs.get(
            "max_attempts_per_step", self.max_attempts_per_step
        )
        verbose = kwargs.get("verbose", self.verbose)
        thin = kwargs.get("thin", self.thin)
        burn_in = kwargs.get("burn_in", self.burn_in)

        # Process observed data through summary statistic if provided
        if self.summary_statistic is not None:
            observed_summary = self.summary_statistic.compute(observed_data)
        else:
            observed_summary = observed_data

        # Initialize chain state
        if initial_state is None:
            current_state = self._find_valid_initial_state(
                observed_summary, tolerance, max_attempts_per_step
            )
        else:
            current_state = initial_state
            # Verify initial state is valid
            if not self._is_valid_state(current_state, observed_summary, tolerance):
                if verbose:
                    print(
                        "Warning: Initial state does not satisfy tolerance. Searching for valid state..."
                    )
                current_state = self._find_valid_initial_state(
                    observed_summary, tolerance, max_attempts_per_step
                )

        current_log_prior = self.prior.log_prob(current_state)

        # Initialize storage
        samples = []
        log_priors = []
        distances = []
        acceptance_indicators = []

        if verbose:
            print("Starting ABC-MCMC sampling...")
            print(f"Iterations: {num_iterations}, Tolerance: {tolerance}")
            print(f"Burn-in: {burn_in}, Thinning: {thin}")

        # Main ABC-MCMC loop
        for iteration in range(num_iterations):
            # M2: Propose new state
            proposed_state = self.proposal_distribution(current_state)
            proposed_log_prior = self.prior.log_prob(proposed_state)

            # M3 & M4: Simulate and check distance constraint
            accepted = False
            attempts = 0

            while attempts < max_attempts_per_step:
                try:
                    # Simulate data with proposed parameters
                    simulated_data = self.simulator.simulate(
                        proposed_state, num_simulations=len(observed_summary)
                    )

                    # Apply summary statistic if provided
                    if self.summary_statistic is not None:
                        simulated_summary = self.summary_statistic.compute(
                            simulated_data
                        )
                    else:
                        simulated_summary = simulated_data

                    # Compute distance
                    distance = self.distance_function(
                        simulated_summary, observed_summary
                    )

                    # Check tolerance constraint
                    if distance <= tolerance:
                        # M5: Compute acceptance probability
                        # For symmetric proposals: α = min(1, π(θ*)/π(θᵢ))
                        log_acceptance_ratio = proposed_log_prior - current_log_prior
                        acceptance_prob = min(1.0, np.exp(log_acceptance_ratio))

                        # Accept with probability α
                        if np.random.rand() < acceptance_prob:
                            current_state = proposed_state
                            current_log_prior = proposed_log_prior
                            accepted = True

                        break  # Exit attempt loop regardless of acceptance

                    attempts += 1

                except Exception as e:
                    if verbose:
                        print(f"Simulation failed at iteration {iteration}: {e}")
                    attempts += 1

            # If we couldn't find a valid proposal within tolerance, reject
            if attempts >= max_attempts_per_step:
                distance = float("inf")  # Mark as invalid
                accepted = False

            # Store samples after burn-in and according to thinning
            if iteration >= burn_in and iteration % thin == 0:
                samples.append(current_state)
                log_priors.append(current_log_prior)
                distances.append(distance if distance != float("inf") else np.nan)
                acceptance_indicators.append(accepted)

            # Progress reporting
            if verbose and (iteration + 1) % max(1, num_iterations // 10) == 0:
                recent_acceptance = (
                    np.mean(acceptance_indicators[-100:])
                    if acceptance_indicators
                    else 0
                )
                print(
                    f"Iteration {iteration + 1}/{num_iterations}, "
                    f"Recent acceptance rate: {recent_acceptance:.3f}"
                )

        if verbose:
            overall_acceptance_rate = (
                np.mean(acceptance_indicators) if acceptance_indicators else 0
            )
            print("ABC-MCMC completed!")
            print(f"Overall acceptance rate: {overall_acceptance_rate:.4f}")
            print(f"Effective samples: {len(samples)}")

        # Return ABC-MCMC posterior object
        return ABCMCMCPosterior(
            samples=samples,
            log_priors=log_priors,
            distances=distances,
            acceptance_indicators=acceptance_indicators,
            tolerance=tolerance,
            num_iterations=num_iterations,
            burn_in=burn_in,
            thin=thin,
        )

    def _find_valid_initial_state(
        self, observed_summary: Any, tolerance: float, max_attempts: int
    ) -> Any:
        """
        Find a valid initial state that satisfies the distance constraint.

        Args:
            observed_summary: Processed observed data
            tolerance: Distance tolerance threshold
            max_attempts: Maximum attempts to find valid state

        Returns:
            Valid initial state

        Raises:
            RuntimeError: If no valid state found within max_attempts
        """
        for _ in range(max_attempts):
            candidate = self.prior.sample()
            if self._is_valid_state(candidate, observed_summary, tolerance):
                return candidate

        raise RuntimeError(
            f"Could not find valid initial state within {max_attempts} attempts. "
            f"Consider increasing tolerance or max_attempts_per_step."
        )

    def _is_valid_state(
        self, state: Any, observed_summary: Any, tolerance: float
    ) -> bool:
        """
        Check if a state satisfies the distance constraint.

        Args:
            state: Parameter state to check
            observed_summary: Processed observed data
            tolerance: Distance tolerance threshold

        Returns:
            True if state is valid, False otherwise
        """
        try:
            simulated_data = self.simulator.simulate(
                state, num_simulations=len(observed_summary)
            )

            if self.summary_statistic is not None:
                simulated_summary = self.summary_statistic.compute(simulated_data)
            else:
                simulated_summary = simulated_data

            distance = self.distance_function(simulated_summary, observed_summary)
            return distance <= tolerance

        except Exception:
            return False


class ABCMCMCPosterior:
    """
    Container for ABC-MCMC sampling results representing the approximate posterior.

    This class stores the MCMC samples from ABC-MCMC and provides methods to query
    and analyze the approximate posterior distribution, including chain diagnostics
    and convergence assessment tools.
    """

    def __init__(
        self,
        samples: List[Any],
        log_priors: List[float],
        distances: List[float],
        acceptance_indicators: List[bool],
        tolerance: float,
        num_iterations: int,
        burn_in: int,
        thin: int,
    ):
        """
        Initialize the ABC-MCMC posterior object.

        Args:
            samples (List[Any]): MCMC samples after burn-in and thinning
            log_priors (List[float]): Log-prior probabilities for each sample
            distances (List[float]): Distance values for each sample
            acceptance_indicators (List[bool]): Acceptance status for each sample
            tolerance (float): Tolerance threshold used
            num_iterations (int): Total MCMC iterations run
            burn_in (int): Number of burn-in iterations discarded
            thin (int): Thinning interval used
        """
        self.samples = samples
        self.log_priors = log_priors
        self.distances = distances
        self.acceptance_indicators = acceptance_indicators
        self.tolerance = tolerance
        self.num_iterations = num_iterations
        self.burn_in = burn_in
        self.thin = thin
        self.acceptance_rate = (
            np.mean(acceptance_indicators) if acceptance_indicators else 0
        )

    def sample(self, num_samples: int = 1) -> Union[Any, List[Any]]:
        """
        Draw samples from the ABC-MCMC posterior.

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
        """Return all MCMC samples."""
        return self.samples.copy()

    def get_distances(self) -> List[float]:
        """Return all distance values."""
        return self.distances.copy()

    def get_log_priors(self) -> List[float]:
        """Return all log-prior values."""
        return self.log_priors.copy()

    def get_acceptance_indicators(self) -> List[bool]:
        """Return acceptance indicators for chain diagnostics."""
        return self.acceptance_indicators.copy()

    def effective_sample_size(self) -> int:
        """
        Estimate the effective sample size accounting for autocorrelation.

        Returns:
            Estimated effective sample size
        """
        # Simple estimate: total samples / (1 + 2 * sum of autocorrelations)
        # For more sophisticated estimates, consider using external packages
        return len(self.samples)  # Simplified for now

    def summary_statistics(self) -> dict:
        """
        Compute summary statistics of the ABC-MCMC posterior.

        Returns:
            Dictionary with posterior and chain summary information
        """
        valid_distances = [d for d in self.distances if not np.isnan(d)]

        return {
            "num_samples": len(self.samples),
            "acceptance_rate": self.acceptance_rate,
            "tolerance": self.tolerance,
            "num_iterations": self.num_iterations,
            "burn_in": self.burn_in,
            "thin": self.thin,
            "effective_sample_size": self.effective_sample_size(),
            "mean_distance": np.mean(valid_distances) if valid_distances else np.nan,
            "max_distance": np.max(valid_distances) if valid_distances else np.nan,
            "min_distance": np.min(valid_distances) if valid_distances else np.nan,
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

        plt.hist(parameter_values, bins=bins, density=True, alpha=0.7, color="green")
        title = "MCMC Posterior Distribution" + (
            f" (Parameter {parameter_index})" if parameter_index is not None else ""
        )
        plt.title(title)
        plt.xlabel(
            f"Parameter {parameter_index}" if parameter_index is not None else "Samples"
        )
        plt.ylabel("Density")
        plt.grid(True)
        plt.show()

    def plot_trace(self, parameter_index: Optional[int] = None) -> None:
        """
        Plot trace plot for chain diagnostics.

        Args:
            parameter_index (Optional[int]): Index of the parameter to plot. If None,
                                             assumes samples are 1D.
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

        plt.figure(figsize=(10, 4))
        plt.plot(parameter_values, alpha=0.8, color="green")
        title = "MCMC Trace Plot" + (
            f" (Parameter {parameter_index})" if parameter_index is not None else ""
        )
        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel(
            f"Parameter {parameter_index}" if parameter_index is not None else "Value"
        )
        plt.grid(True)
        plt.show()

    def plot_acceptance_rate(self, window_size: int = 100) -> None:
        """
        Plot running acceptance rate for chain diagnostics.

        Args:
            window_size (int): Window size for computing running average
        """
        if len(self.acceptance_indicators) < window_size:
            window_size = len(self.acceptance_indicators)

        running_acceptance = []
        for i in range(len(self.acceptance_indicators)):
            start_idx = max(0, i - window_size + 1)
            window_acceptance = np.mean(self.acceptance_indicators[start_idx : i + 1])
            running_acceptance.append(window_acceptance)

        plt.figure(figsize=(10, 4))
        plt.plot(running_acceptance, color="blue")
        plt.title(f"Running Acceptance Rate (Window: {window_size})")
        plt.xlabel("Iteration")
        plt.ylabel("Acceptance Rate")
        plt.grid(True)
        plt.axhline(
            y=self.acceptance_rate,
            color="red",
            linestyle="--",
            label=f"Overall: {self.acceptance_rate:.3f}",
        )
        plt.legend()
        plt.show()

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __repr__(self) -> str:
        return (
            f"ABCMCMCPosterior(num_samples={len(self.samples)}, "
            f"acceptance_rate={self.acceptance_rate:.4f}, "
            f"tolerance={self.tolerance})"
        )
