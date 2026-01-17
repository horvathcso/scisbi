import unittest
import numpy as np
import matplotlib.pyplot as plt

from scisbi.simulator import TwoMoonsSimulator


class TestTwoMoonsSimulator(unittest.TestCase):
    """Test cases for the TwoMoonsSimulator class."""

    def test_initialization(self):
        """Test if the simulator initializes properly with default and custom values."""
        # Test default initialization
        sim_default = TwoMoonsSimulator()
        self.assertEqual(sim_default.noise, 0.1)
        self.assertEqual(sim_default.moon_radius, 1.0)
        self.assertEqual(sim_default.moon_width, 0.8)
        self.assertEqual(sim_default.moon_distance, -0.25)

        # Test custom initialization
        sim_custom = TwoMoonsSimulator(
            noise=0.2, moon_radius=2.0, moon_width=1.0, moon_distance=1.0, n_points=200
        )
        self.assertEqual(sim_custom.noise, 0.2)
        self.assertEqual(sim_custom.moon_radius, 2.0)
        self.assertEqual(sim_custom.moon_width, 1.0)
        self.assertEqual(sim_custom.moon_distance, 1.0)
        self.assertEqual(sim_custom.n_points, 200)

    def test_simulate_output_shape(self):
        """Test if the simulator outputs the correct shape."""
        n_points = 200
        sim = TwoMoonsSimulator(n_points=n_points)

        # Test with default parameters (single batch)
        data = sim.simulate(None)
        # Expected shape: (1, n_points, 2)
        self.assertEqual(data.shape, (1, n_points, 2))

        # Test with custom parameters (batch of 1)
        params = np.array([[0.2, 2.0, 1.0, 1.0]])
        data = sim.simulate(params)
        self.assertEqual(data.shape, (1, n_points, 2))

        # Test with batch of 3
        params_batch = np.array([
            [0.2, 2.0, 1.0, 1.0],
            [0.1, 1.0, 0.8, -0.25],
            [0.3, 1.5, 0.9, 0.0]
        ])
        data = sim.simulate(params_batch)
        self.assertEqual(data.shape, (3, n_points, 2))

    def test_noise_impact(self):
        """Test if noise parameter impacts the data variance."""
        n_points = 1000
        # Generate data with different noise levels
        sim_low_noise = TwoMoonsSimulator(noise=0.01, n_points=n_points)
        sim_high_noise = TwoMoonsSimulator(noise=0.5, n_points=n_points)

        data_low_noise = sim_low_noise.simulate(None) # (1, 1000, 2)
        data_high_noise = sim_high_noise.simulate(None) # (1, 1000, 2)

        # Calculate variance of each dataset (flattened or just per coord)
        var_low = np.var(data_low_noise)
        var_high = np.var(data_high_noise)

        # Higher noise should result in higher variance
        self.assertLess(var_low, var_high)

    def test_moon_separation(self):
        """Test if the moons are properly separated based on the moon_distance parameter."""
        n_points = 1000
        # Compare two simulations with different moon_distance values
        sim_close = TwoMoonsSimulator(
            moon_distance=0.1, noise=0.01, n_points=n_points
        )  # Very small noise to focus on separation
        sim_far = TwoMoonsSimulator(moon_distance=1.0, noise=0.01, n_points=n_points)

        # Generate data
        data_close = sim_close.simulate(None)[0] # Get first batch item: (1000, 2)
        data_far = sim_far.simulate(None)[0]

        # Split data into two moons
        split_idx = n_points // 2
        close_moon1 = data_close[:split_idx]
        close_moon2 = data_close[split_idx:]
        far_moon1 = data_far[:split_idx]
        far_moon2 = data_far[split_idx:]

        # Calculate vertical separation (y-axis) between moons by finding the mean y-coordinate of each moon
        close_separation = np.abs(
            np.mean(close_moon1[:, 1]) - np.mean(close_moon2[:, 1])
        )
        far_separation = np.abs(np.mean(far_moon1[:, 1]) - np.mean(far_moon2[:, 1]))

        # The separation should be greater for the "far" configuration
        self.assertGreater(far_separation, close_separation)

        # Check that the difference in separations is approximately equal to the
        # difference in the moon_distance parameters
        separation_diff = far_separation - close_separation
        parameter_diff = 1.0 - 0.1  # Difference in moon_distance parameters

        self.assertAlmostEqual(separation_diff, parameter_diff, delta=0.15)

    def test_moon_width(self):
        """Test if the moon_width parameter correctly affects the horizontal offset between moons."""
        n_points = 1000
        # Compare two simulations with different moon_width values
        sim_narrow = TwoMoonsSimulator(moon_width=0.5, noise=0.01, n_points=n_points)
        sim_wide = TwoMoonsSimulator(moon_width=1.5, noise=0.01, n_points=n_points)

        # Generate data
        data_narrow = sim_narrow.simulate(None)[0]
        data_wide = sim_wide.simulate(None)[0]

        # Split data into two moons
        split_idx = n_points // 2
        narrow_moon1 = data_narrow[:split_idx]
        narrow_moon2 = data_narrow[split_idx:]
        wide_moon1 = data_wide[:split_idx]
        wide_moon2 = data_wide[split_idx:]

        # Calculate horizontal center difference between moons
        narrow_x_diff = np.mean(narrow_moon2[:, 0]) - np.mean(narrow_moon1[:, 0])
        wide_x_diff = np.mean(wide_moon2[:, 0]) - np.mean(wide_moon1[:, 0])

        # Test that the wider setting produces a larger horizontal offset
        self.assertGreater(wide_x_diff, narrow_x_diff)

        # The width difference should be reflected in the horizontal positioning
        # The difference in x_diff should be approximately equal to the difference in width parameters
        width_diff = 1.5 - 0.5  # Difference in moon_width parameters = 1.0
        x_diff_delta = wide_x_diff - narrow_x_diff

        # Check that the change in horizontal positioning matches the change in the width parameter
        self.assertAlmostEqual(x_diff_delta, width_diff, delta=0.1)

        # Check that each width setting produces the expected offset
        self.assertAlmostEqual(narrow_x_diff, 0.5, delta=0.15)
        self.assertAlmostEqual(wide_x_diff, 1.5, delta=0.15)


def visualize_two_moons(
    simulator, params=None, num_samples=500, title="Two Moons Dataset"
):
    """Helper function to visualize the two moons dataset."""
    # Temporarily set n_points or create new simulator?
    # Better to assume simulator is configured correctly or create new one.
    # Since we can't change n_points easily without accessing internal, let's create a new one.
    # Or just use the simulator's n_points.

    # If params is provided, use it.
    if params is not None and params.ndim == 1:
        params = params[np.newaxis, :]

    data = simulator.simulate(params)[0] # Get first batch

    plt.figure(figsize=(10, 6))
    plt.scatter(data[:, 0], data[:, 1], s=10, alpha=0.8)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    return data


if __name__ == "__main__":
    # Run the tests
    unittest.main(exit=False)
