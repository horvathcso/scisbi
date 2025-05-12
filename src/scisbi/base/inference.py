# scisbi/base/inference_algorithm.py

import abc
from typing import Any, Dict, Optional

# Assume these base classes are defined elsewhere in your library
from .simulator import BaseSimulator
from .summary_statistic import BaseSummaryStatistic  # New optional dependency


class BaseInferenceAlgorithm(abc.ABC):
    """
    Abstract base class for all Simulation-Based Inference algorithms.

    Concrete implementations should inherit from this class and implement
    the abstract methods. This base class handles the storage of fundamental
    components like the simulator, prior, and optional summary statistic.
    """

    def __init__(
        self,
        simulator: BaseSimulator,
        prior: Any,
        summary_statistic: Optional[BaseSummaryStatistic] = None,
        **kwargs: Any,
    ):
        """
        Initializes the inference algorithm.

        Args:
            simulator (BaseSimulator): An instance of a simulator object.
                                       Must have a 'simulate' method.
            prior (Any): An instance of a prior distribution object.
                               Must have 'log_prob' and 'sample' methods.
            summary_statistic (Optional[BaseSummaryStatistic]): An optional
                                                               summary statistic object.
                                                               If provided, it should
                                                               have a 'compute' method
                                                               to reduce data dimensionality.
                                                               Defaults to None.
            **kwargs: Additional algorithm-specific configuration parameters.
                      These settings often control aspects like neural network
                      architecture hyperparameters, training parameters (epochs,
                      batch size), number of rounds (for sequential methods),
                      acceptance thresholds (for ABC), etc.
        """
        # Basic checks for required components
        if not hasattr(simulator, "simulate") or not callable(simulator.simulate):
            raise TypeError(
                "simulator must be an instance of a class with a 'simulate' method"
            )
        if not hasattr(prior, "log_prob") or not callable(prior.log_prob):
            raise TypeError(
                "prior must be an instance of a class with a 'log_prob' method"
            )
        if not hasattr(prior, "sample") or not callable(prior.sample):
            raise TypeError(
                "prior must be an instance of a class with a 'sample' method"
            )
        if summary_statistic is not None and (
            not hasattr(summary_statistic, "compute")
            or not callable(summary_statistic.compute)
        ):
            raise TypeError(
                "summary_statistic, if provided, must be an instance of a class with a 'compute' method"
            )

        self.simulator: BaseSimulator = simulator
        self.prior: Any = prior
        self.summary_statistic: Optional[BaseSummaryStatistic] = summary_statistic
        self.settings: Dict[str, Any] = kwargs  # Store algorithm-specific config

    @abc.abstractmethod
    def infer(
        self,
        observed_data: Any,  # Type depends on simulator/summary_statistic output
        num_simulations: int,
        **kwargs: Any,
    ) -> Any:
        """
        Runs the simulation-based inference process.

        This method orchestrates the simulation, training (if applicable),
        and posterior estimation steps. The implementation will vary
        significantly between different SBI algorithms (e.g., ABC, SNPE, SNL, SRE).

        Args:
            observed_data (Any): The observed data point(s) to perform inference on.
                                 If a summary statistic was provided during
                                 initialization, this data might be the raw
                                 observed data which will be transformed
                                 internally using the summary statistic.
                                 Otherwise, its type/format depends on the
                                 simulator's output structure.
            num_simulations (int): The total number of simulations the algorithm
                                   is allowed to run across potentially multiple rounds.
            **kwargs: Algorithm-specific parameters for this specific inference run.
                      Examples might include: number of training epochs for a
                      single round, specific random seeds, control over output
                      verbosity for this run. These override settings from __init__.

        Returns:
            Any: An object representing the estimated posterior distribution.
                           This object should provide methods to query the posterior,
                           e.g., `sample`, `log_prob` (if density is available).

        Raises:
            NotImplementedError: If the method is not implemented by a concrete class.
            Any other exception relevant to the specific algorithm's execution.
        """

        # Concrete implementations will typically perform steps like:
        # 1. Get observed data (and apply summary statistic if available).
        # 2. Sample parameters from the prior (or an updated proposal for sequential methods).
        # 3. Run the simulator with these parameters to get simulated data.
        # 4. (If summary statistic is used) Compute summary statistics of simulated data.
        # 5. Use the simulated (parameters, data/summaries) pairs to train a model
        #    (e.g., density estimator, classifier) which is method-specific.
        # 6. Use the trained model and the observed data (or its summaries) to
        #    construct or sample from the estimated posterior distribution.
        # 7. Return an object representing this posterior.
        raise NotImplementedError(
            "Not implemented inference. Call originates from abstract base method"
        )
