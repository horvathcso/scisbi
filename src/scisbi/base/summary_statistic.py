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
            data: The input data. Format depends on the simulator output. Can be varied, hence Any.

        Returns:
            The computed summary statistics. Format can vary (e.g., NumPy array, dictionary), hence Any.
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
            model_architecture: Definition or instance of the learning model architecture
                                (e.g., a PyTorch Module or TensorFlow Model). Can be varied, hence Any.
            filepath: Optional path to a pre-trained model file. If provided, the model
                      will be loaded automatically upon initialization.
            **kwargs: Configuration parameters for the summary statistic computation and learning process.
                      Can be varied, hence Any.
        """
        self.config: Dict[str, Any] = kwargs
        # The model attribute will hold the instance of the built learning model.
        # Its specific type depends on the framework used in subclasses, hence Any here.
        self.model: Any = self._build_model(model_architecture)
        self._is_trained: bool = False

        if filepath is not None:
            self.load(filepath)
            self._is_trained = True  # Ensure trained flag is set after loading

    @abc.abstractmethod
    def _build_model(self, model_architecture: Any) -> Any:
        """
        Builds and compiles the learning model based on the provided architecture.
        This method should be implemented by subclasses to define how the model
        is instantiated and prepared for training (e.g., defining optimizer, loss function).

        Args:
            model_architecture: Definition or instance of the learning model architecture.
                                Can be varied, hence Any.

        Returns:
            An instance of the built learning model. Can be varied, hence Any.
        """
        pass

    @abc.abstractmethod
    def train(
        self, train_data: Any, validation_data: Optional[Any] = None, **kwargs: Any
    ) -> None:
        """
        Trains the learning model to compute the summary statistics.

        Args:
            train_data: The data to use for training the model. Format depends on the model, hence Any.
            validation_data: Optional data to use for validation during training. Format depends on the model, hence Optional[Any].
            **kwargs: Parameters for the training process (e.g., epochs, batch size, learning rate). Can be varied, hence Any.
        """
        # The actual setting of _is_trained should happen in the concrete implementation
        # after successful training.
        # self._is_trained = True
        pass

    def compute(self, data: Any) -> Any:
        """
        Computes summary statistics for the given data using the trained model.

        Args:
            data: The input data. Format depends on the expected input of the learned model, hence Any.

        Returns:
            The computed summary statistics (the output of the learned model).
            The format depends on the model's output, hence Any.
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

        Args:
            filepath: The path to save the model.
        """
        pass

    @abc.abstractmethod
    def load(self, filepath: str) -> None:
        """
        Loads a trained learning model from a file.

        Args:
            filepath: The path to the saved model.
        """
        # The actual setting of _is_trained should happen in the concrete implementation
        # after successful loading.
        # self._is_trained = True
        pass

    def is_trained(self) -> bool:
        """
        Checks if the underlying learning model has been trained.

        Returns:
            True if the model is trained, False otherwise.
        """
        return self._is_trained
