# scisbi/base/simulator.py

import abc


class BaseSimulator(abc.ABC):
    """
    Abstract base class for simulators.

    A simulator takes parameters and generates data/observations.
    """

    def __init__(self, **kwargs):
        """
        Initializes the simulator.

        Args:
            **kwargs: Simulator-specific configuration parameters.
        """
        self.config = kwargs

    @abc.abstractmethod
    def simulate(self, parameters):
        """
        Runs the simulator for a given set of parameters.

        Args:
            parameters: The parameters for the simulation(s).
                        Expected shape: (Batch_Size, Parameter_Dim) or (Batch_Size,).
                        The simulator should iterate over the batch or use vectorized operations
                        to generate one observation per parameter set.

        Returns:
            The simulated data/observations.
            Expected shape: (Batch_Size, *Data_Dims).
        """
        pass
