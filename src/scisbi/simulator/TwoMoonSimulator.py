import numpy as np
from scisbi.base.simulator import BaseSimulator


class TwoMoonsSimulator(BaseSimulator):
    """
    Simulator for generating two moons dataset.

    Parameters:
        noise: Standard deviation of Gaussian noise to add to the data (default: 0.1)
        moon_radius: Radius of the moons (default: 1.0)
        moon_width: Width of the moons (default: 0.8)
        moon_distance: Vertical distance between moons (default: 0.5)
    """

    def __init__(
        self, noise=0.1, moon_radius=1.0, moon_width=0.8, moon_distance=-0.25, **kwargs
    ):
        super().__init__(**kwargs)
        self.noise = noise
        self.moon_radius = moon_radius
        self.moon_width = moon_width
        self.moon_distance = moon_distance

    def simulate(self, parameters, num_simulations=1):
        """
        Generate samples from the two moons distribution.

        Args:
            parameters: Numpy array of shape [n_params] containing:
                        [noise, moon_radius, moon_width, moon_distance]
                        If None, uses the default values from initialization.
            num_simulations: Number of samples to generate per moon.
                             Total samples will be 2*num_simulations.

        Returns:
            Numpy array of shape [2*num_simulations, 2] containing the flattened data points.
        """
        if parameters is not None and len(parameters) >= 4:
            noise = abs(parameters[0])  # Ensure noise is always positive
            moon_radius = parameters[1]
            moon_width = parameters[2]
            moon_distance = parameters[3]
        else:
            noise = self.noise
            moon_radius = self.moon_radius
            moon_width = self.moon_width
            moon_distance = self.moon_distance
            parameters = np.zeros(4)
            parameters[0] = noise
            parameters[1] = moon_radius
            parameters[2] = moon_width
            parameters[3] = moon_distance

        super().simulate(parameters, num_simulations)

        # Generate first moon
        t = np.linspace(0, np.pi, num_simulations)
        x1 = moon_radius * np.cos(t)
        y1 = moon_radius * np.sin(t)

        # Generate second moon
        x2 = moon_radius * np.cos(t + np.pi) + moon_width
        y2 = moon_radius * np.sin(t + np.pi) - moon_distance

        # Combine moons
        x = np.concatenate([x1, x2])
        y = np.concatenate([y1, y2])

        # Add noise
        x += np.random.normal(0, noise, size=len(x))
        y += np.random.normal(0, noise, size=len(y))

        # Combine into one array and flatten
        points = np.column_stack((x, y))

        return points
