# scisbi/base/inference.py

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
                                       Must have a 'simulate' method accepting (Batch, P) parameters.
            prior (Any): An instance of a prior distribution object.
                               Must have 'log_prob' and 'sample' methods.
            summary_statistic (Optional[BaseSummaryStatistic]): An optional
                                                               summary statistic object.
                                                               If provided, it should
                                                               have a 'compute' method
                                                               to reduce data dimensionality.
                                                               Defaults to None.
            **kwargs: Additional algorithm-specific configuration parameters.
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
        observed_data: Any,
        num_simulations: int,
        **kwargs: Any,
    ) -> Any:
        """
        Runs the simulation-based inference process.

        Args:
            observed_data (Any): The observed data point(s) to perform inference on.
                                 Expected shape: (N_obs, *Data_Dims).
                                 Usually N_obs is 1.
                                 If a summary statistic was provided, this data might be transformed
                                 internally using the summary statistic.
            num_simulations (int): The total number of simulations the algorithm
                                   is allowed to run (or number of samples to generate).
            **kwargs: Algorithm-specific parameters for this specific inference run.

        Returns:
            Any: An object representing the estimated posterior distribution.
        """
        raise NotImplementedError(
            "Not implemented inference. Call originates from abstract base method"
        )
