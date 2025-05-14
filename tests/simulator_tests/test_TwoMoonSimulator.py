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
            noise=0.2, moon_radius=2.0, moon_width=1.0, moon_distance=1.0
        )
        self.assertEqual(sim_custom.noise, 0.2)
        self.assertEqual(sim_custom.moon_radius, 2.0)
        self.assertEqual(sim_custom.moon_width, 1.0)
        self.assertEqual(sim_custom.moon_distance, 1.0)

    def test_simulate_output_shape(self):
        """Test if the simulator outputs the correct shape."""
        sim = TwoMoonsSimulator()

        # Test with default parameters
        num_samples = 100
        data = sim.simulate(None, num_samples)
        self.assertEqual(data.shape, (2 * num_samples, 2))

        # Test with custom parameters
        params = np.array([0.2, 2.0, 1.0, 1.0])
        data = sim.simulate(params, num_samples)
        self.assertEqual(data.shape, (2 * num_samples, 2))

    def test_parameter_assignment(self):
        """Test if the parameters are correctly assigned during simulation."""
        sim = TwoMoonsSimulator()
        params = np.array([0.2, 2.0, 1.0, 1.0])
        sim.simulate(params, 100)

        # Check if parameters are saved correctly
        np.testing.assert_array_equal(sim.parameter, params)
        self.assertEqual(sim.num_simulations, 100)

    def test_noise_impact(self):
        """Test if noise parameter impacts the data variance."""
        # Generate data with different noise levels
        sim_low_noise = TwoMoonsSimulator(noise=0.01)
        sim_high_noise = TwoMoonsSimulator(noise=0.5)

        data_low_noise = sim_low_noise.simulate(None, 500)
        data_high_noise = sim_high_noise.simulate(None, 500)

        # Calculate variance of each dataset
        var_low = np.var(data_low_noise)
        var_high = np.var(data_high_noise)

        # Higher noise should result in higher variance
        self.assertLess(var_low, var_high)

    def test_moon_separation(self):
        """Test if the moons are properly separated based on the moon_distance parameter."""
        # Compare two simulations with different moon_distance values
        sim_close = TwoMoonsSimulator(
            moon_distance=0.1, noise=0.01
        )  # Very small noise to focus on separation
        sim_far = TwoMoonsSimulator(moon_distance=1.0, noise=0.01)

        # Generate data
        data_close = sim_close.simulate(None, 500)
        data_far = sim_far.simulate(None, 500)

        # Split data into two moons
        close_moon1 = data_close[:500]
        close_moon2 = data_close[500:]
        far_moon1 = data_far[:500]
        far_moon2 = data_far[500:]

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
        # Compare two simulations with different moon_width values
        sim_narrow = TwoMoonsSimulator(moon_width=0.5, noise=0.01)
        sim_wide = TwoMoonsSimulator(moon_width=1.5, noise=0.01)

        # Generate data
        data_narrow = sim_narrow.simulate(None, 500)
        data_wide = sim_wide.simulate(None, 500)

        # Split data into two moons
        narrow_moon1 = data_narrow[:500]
        narrow_moon2 = data_narrow[500:]
        wide_moon1 = data_wide[:500]
        wide_moon2 = data_wide[500:]

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
    data = simulator.simulate(params, num_samples)

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

    # Create and visualize different two moons configurations
    print("\nVisualizing Two Moons datasets...")

    # Default parameters
    default_sim = TwoMoonsSimulator()
    visualize_two_moons(default_sim, title="Two Moons (Default Parameters)")

    # Different noise levels
    low_noise_sim = TwoMoonsSimulator(noise=0.05)
    high_noise_sim = TwoMoonsSimulator(noise=0.3)
    visualize_two_moons(low_noise_sim, title="Two Moons (Low Noise: 0.05)")
    visualize_two_moons(high_noise_sim, title="Two Moons (High Noise: 0.3)")

    # Different shapes
    wide_moons_sim = TwoMoonsSimulator(moon_width=1.5, moon_distance=-0.25)
    small_moons_sim = TwoMoonsSimulator(
        moon_radius=0.5, moon_width=0.8, moon_distance=-0.1
    )

    visualize_two_moons(wide_moons_sim, title="Wide Moons")
    visualize_two_moons(small_moons_sim, title="Small Moons")

    # Using parameter array
    custom_params = np.array([0.1, 1.5, 1.2, 0.8])  # [noise, radius, width, distance]
    custom_sim = TwoMoonsSimulator()
    visualize_two_moons(
        custom_sim, params=custom_params, title="Custom Parameters via Array"
    )

    plt.show()
