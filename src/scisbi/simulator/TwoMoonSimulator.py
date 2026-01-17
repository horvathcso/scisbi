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
        n_points: Number of points in the generated dataset (default: 100)
    """

    def __init__(
        self,
        noise=0.1,
        moon_radius=1.0,
        moon_width=0.8,
        moon_distance=-0.25,
        n_points=100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.noise = noise
        self.moon_radius = moon_radius
        self.moon_width = moon_width
        self.moon_distance = moon_distance
        self.n_points = n_points

    def simulate(self, parameters):
        """
        Generate samples from the two moons distribution.

        Args:
            parameters: Numpy array of shape (Batch_Size, 4) containing:
                        [noise, moon_radius, moon_width, moon_distance]
                        If parameters is None or empty, uses defaults for 1 simulation.

        Returns:
            Numpy array of shape (Batch_Size, n_points, 2) containing the dataset.
        """
        if parameters is None:
             # Default single simulation
             parameters = np.array([[self.noise, self.moon_radius, self.moon_width, self.moon_distance]])
        else:
             parameters = np.asarray(parameters)
             if parameters.ndim == 1:
                 parameters = parameters[np.newaxis, :]

        batch_size = parameters.shape[0]
        output = np.zeros((batch_size, self.n_points, 2))

        # Loop over batch (could be vectorized but logic varies per param)
        for i in range(batch_size):
            p = parameters[i]
            if len(p) >= 4:
                noise = abs(p[0])
                moon_radius = p[1]
                moon_width = p[2]
                moon_distance = p[3]
            else:
                # Fallback if params provided are incomplete (should not happen if shape enforced)
                noise = self.noise
                moon_radius = self.moon_radius
                moon_width = self.moon_width
                moon_distance = self.moon_distance

            # Generate first moon (half points)
            n_moon1 = self.n_points // 2
            n_moon2 = self.n_points - n_moon1

            t1 = np.linspace(0, np.pi, n_moon1)
            x1 = moon_radius * np.cos(t1)
            y1 = moon_radius * np.sin(t1)

            # Generate second moon (half points)
            t2 = np.linspace(0, np.pi, n_moon2)
            x2 = moon_radius * np.cos(t2 + np.pi) + moon_width
            y2 = moon_radius * np.sin(t2 + np.pi) - moon_distance

            # Combine moons
            x = np.concatenate([x1, x2])
            y = np.concatenate([y1, y2])

            # Add noise
            x += np.random.normal(0, noise, size=len(x))
            y += np.random.normal(0, noise, size=len(y))

            # Combine into one array
            output[i] = np.column_stack((x, y))

        return output
