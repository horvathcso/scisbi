# scisbi/base/workflow.py

import abc
from typing import Any, Dict, Optional  # Import necessary types

# Import base classes for type hinting
from .inference import BaseInferenceAlgorithm
from .simulator import BaseSimulator
from .summary_statistic import BaseSummaryStatistic


class BaseWorkflow(abc.ABC):
    """
    Abstract base class for defining and executing a complete SBI workflow.

    A workflow combines a simulator, prior, inference algorithm, summary statistic, etc.
    """

    def __init__(
        self,
        simulator: BaseSimulator,
        prior: Any,
        inference_algorithm: BaseInferenceAlgorithm,
        summary_statistic: Optional[BaseSummaryStatistic] = None,
        **kwargs: Any,
    ):
        """
        Intializes the workflow.

        Args:
            simulator (BaseSimulator): The simulator instance.
            prior (Any): The prior instance.
            inference_algorithm (BaseInferenceAlgorithm): The inference algorithm instance.
            summary_statistic (Optional[BaseSummaryStatistic]): The summary statistic instance.
            **kwargs: Workflow-specific configuration.
        """
        # Type checks (can be kept for runtime validation if desired)
        if not isinstance(simulator, BaseSimulator):
            raise TypeError("simulator must be an instance of BaseSimulator")
        if not isinstance(prior, Any):
            raise TypeError("prior must be an instance of Any")
        if not isinstance(inference_algorithm, BaseInferenceAlgorithm):
            raise TypeError(
                "inference_algorithm must be an instance of BaseInferenceAlgorithm"
            )
        if summary_statistic is not None and not isinstance(
            summary_statistic, BaseSummaryStatistic
        ):
            raise TypeError("summary_statistic must be a BaseSummaryStatistic or None")

        self.simulator: BaseSimulator = simulator
        self.prior: Any = prior
        self.inference_algorithm: BaseInferenceAlgorithm = inference_algorithm
        self.summary_statistic: Optional[BaseSummaryStatistic] = summary_statistic
        self.config: Dict[str, Any] = kwargs

    @abc.abstractmethod
    def run(self, observed_data: Any, num_simulations: int, **kwargs: Any) -> Any:
        """
        Executes the complete SBI workflow.

        Args:
            observed_data (Any): The observed data.
                                 Format depends on the simulator output. Use Any.
            num_simulations (int): The number of simulations to run for inference.
            **kwargs: Additional parameters for running the workflow.

        Returns:
            Any: The estimated posterior distribution.
        """
        pass
