# scisbi/base/summary_statistic.py

import abc
from typing import Any, Dict, Optional, TypeVar

# Define a TypeVar for tensor-like objects, to be used in concrete implementations
TensorLike = TypeVar("TensorLike")


class BaseSummaryStatistic(abc.ABC):
    """
    Abstract base class for functions that compute summary statistics from data.
    """

    def __init__(self, **kwargs: Any):
        """
        In the base class, kwargs can be of any type for flexibility.
        Initializes the summary statistic function.

        Args:
            **kwargs: Configuration parameters for the summary statistic computation.
        """
        self.config: Dict[str, Any] = kwargs

    @abc.abstractmethod
    def compute(self, data: TensorLike) -> TensorLike:
        """
        Computes summary statistics for the given data.

        Args:
            data: The input data.
                  Expected shape: (Batch_Size, *Data_Dims).

        Returns:
            The computed summary statistics.
            Expected shape: (Batch_Size, *Stats_Dims).
        """
        pass

    @abc.abstractmethod
    def _normalize(self, stats: Any) -> Any:
        """
        Normalizes or scales the computed summary statistics.
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration parameters used to initialize the summary statistic.
        """
        return self.config

    @abc.abstractmethod
    def visualize(self, stats: Any):
        """
        Generates a visualization of the summary statistics.
        """
        raise NotImplementedError("Subclass must implement visualization")


class BaseLearnedSummaryStatistic(BaseSummaryStatistic):
    """
    Abstract base class for learned functions that compute summary statistics from data
    using models like Neural Networks or Variational Autoencoders.
    """

    def __init__(
        self, model_architecture: Any, filepath: Optional[str] = None, **kwargs: Any
    ):
        """
        Initializes the learned summary statistic function.

        Args:
            model_architecture: Definition or instance of the learning model architecture.
            filepath: Optional path to a pre-trained model file.
            **kwargs: Configuration parameters.
        """
        self.config: Dict[str, Any] = kwargs
        self.model: Any = self._build_model(model_architecture)
        self._is_trained: bool = False

        if filepath is not None:
            self.load(filepath)
            self._is_trained = True

    @abc.abstractmethod
    def _build_model(self, model_architecture: Any) -> Any:
        """
        Builds and compiles the learning model based on the provided architecture.
        """
        pass

    @abc.abstractmethod
    def train(
        self, train_data: Any, validation_data: Optional[Any] = None, **kwargs: Any
    ) -> None:
        """
        Trains the learning model to compute the summary statistics.

        Args:
            train_data: The data to use for training the model. (Batch, ...)
            validation_data: Optional data to use for validation. (Batch, ...)
            **kwargs: Parameters for the training process.
        """
        pass

    def compute(self, data: Any) -> Any:
        """
        Computes summary statistics for the given data using the trained model.

        Args:
            data: The input data. (Batch, ...)

        Returns:
            The computed summary statistics. (Batch, ...)
        """
        if not self._is_trained:
            raise RuntimeError(
                "The summary statistic model has not been trained yet. Call .train() or provide a filepath to a trained model on initialization."
            )
        raise NotImplementedError("Subclass must implement abstract method 'compute'")

    @abc.abstractmethod
    def save(self, filepath: str) -> None:
        """
        Saves the trained learning model to a file.
        """
        pass

    @abc.abstractmethod
    def load(self, filepath: str) -> None:
        """
        Loads a trained learning model from a file.
        """
        pass

    def is_trained(self) -> bool:
        """
        Checks if the underlying learning model has been trained.
        """
        return self._is_trained
