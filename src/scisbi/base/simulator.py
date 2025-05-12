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
    def simulate(self, parameters, num_simulations=1):
        """
        Runs the simulator for a given set of parameters multiple times.

        Args:
            parameters: The parameters for the simulation(s).
                        Format depends on the simulator's input requirements
                        (e.g., a NumPy array, a dictionary, a backend tensor).
                        Assumes these parameters are applied to each simulation run
                        or define the distribution from which parameters for each run are drawn.
            num_simulations: The number of simulations to run for the given parameters. Defaults to 1.

        Returns:
            The simulated data/observations. The output should be structured to
            accommodate multiple simulation results (e.g., a list, a batch tensor
            with an added dimension for the simulation index).
            Format depends on the simulator output.
        """
        self.parameter = parameters
        self.num_simulations = num_simulations
        pass
