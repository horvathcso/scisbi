import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import os

from scisbi.simulator import LotkaVolterraSimulator


class TestLotkaVolterraSimulator:
    """Test suite for the Lotka-Volterra simulator."""

    @pytest.fixture
    def simulator(self):
        """Create a default simulator instance."""
        return LotkaVolterraSimulator(t_span=(0, 30), n_points=50, noise_level=0.05)

    @pytest.fixture
    def test_parameters(self):
        """Define test parameters for simulations."""
        return np.array(
            [1.0, 0.2, 1.5, 0.1, 10.0, 5.0]
        )  # [alpha, beta, gamma, delta, x0, y0]

    @pytest.fixture
    def batch_parameters(self):
        """Define a batch of test parameters."""
        return np.array(
            [
                [1.0, 0.2, 1.5, 0.1, 10.0, 5.0],
                [0.8, 0.3, 1.2, 0.2, 15.0, 3.0],
                [1.2, 0.15, 1.8, 0.08, 8.0, 7.0],
            ]
        )

    def test_initialization(self, simulator):
        """Test that the simulator initializes correctly."""
        assert simulator.t_span == (0, 30)
        assert simulator.n_points == 50
        assert simulator.noise_level == 0.05
        assert simulator.summary_stats is False
        assert simulator.t_eval.shape == (50,)

    def test_single_simulation_shape(self, simulator, test_parameters):
        """Test that a single simulation returns the correct shape."""
        result = simulator.simulate(test_parameters)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 2 * simulator.n_points)

    def test_batch_simulation_shape(self, simulator, batch_parameters):
        """Test that batch simulation returns the correct shape."""
        result = simulator.simulate(batch_parameters)
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(batch_parameters), 2 * simulator.n_points)

    def test_summary_statistics(self, test_parameters):
        """Test summary statistics mode."""
        sim_summary = LotkaVolterraSimulator(
            t_span=(0, 30), n_points=50, summary_stats=True
        )
        result = sim_summary.simulate(test_parameters)
        assert isinstance(result, np.ndarray)
        # Should have 8 summary statistics
        assert result.shape == (1, 8)

    def test_batch_summary_statistics(self, batch_parameters):
        """Test batch summary statistics."""
        sim_summary = LotkaVolterraSimulator(
            t_span=(0, 30), n_points=50, summary_stats=True
        )
        result = sim_summary.simulate(batch_parameters)
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(batch_parameters), 8)

    def test_different_n_points(self, test_parameters):
        """Test simulation with different numbers of time points."""
        n_points_values = [10, 50, 100]
        for n_points in n_points_values:
            sim = LotkaVolterraSimulator(n_points=n_points)
            result = sim.simulate(test_parameters)
            assert result.shape == (1, 2 * n_points)

    def test_num_simulations_parameter(self, simulator, test_parameters):
        """Test that num_simulations parameter works correctly."""
        num_sims = 5
        result = simulator.simulate(test_parameters, num_simulations=num_sims)
        assert result.shape == (num_sims, 2 * simulator.n_points)

    def test_get_original_shape(self, simulator, test_parameters):
        """Test reshaping flattened data back to original format."""
        flattened = simulator.simulate(test_parameters)
        reshaped = simulator.get_original_shape(flattened)

        assert reshaped.shape == (1, 2, simulator.n_points)
        # Check that prey and predator are in the right places
        assert np.array_equal(flattened[0, : simulator.n_points], reshaped[0, 0])
        assert np.array_equal(flattened[0, simulator.n_points :], reshaped[0, 1])

    def test_invalid_parameter_shape(self, simulator):
        """Test that an error is raised for invalid parameter shapes."""
        invalid_params = np.array([1.0, 0.2, 1.5])  # Too few parameters
        with pytest.raises(ValueError):
            simulator.simulate(invalid_params)

    def test_negative_parameter_handling(self, simulator):
        """Test that negative parameters are handled properly."""
        # Try with negative values that should be corrected
        params = np.array([-0.5, -0.1, -1.0, -0.05, -5.0, -2.0])
        result = simulator.simulate(params)
        # Should not crash and return expected shape
        assert result.shape == (1, 2 * simulator.n_points)

    def test_zero_noise(self, test_parameters):
        """Test simulation with zero noise."""
        sim_no_noise = LotkaVolterraSimulator(noise_level=0.0)

        # Run two identical simulations - they should be identical with no noise
        result1 = sim_no_noise.simulate(test_parameters, num_simulations=1)
        result2 = sim_no_noise.simulate(test_parameters, num_simulations=1)

        assert np.allclose(result1, result2)

    def test_with_noise(self, test_parameters):
        """Test simulation with noise."""
        # Set a high noise level to ensure difference
        sim_with_noise = LotkaVolterraSimulator(noise_level=0.2)

        # Run two identical simulations - they should differ due to noise
        result1 = sim_with_noise.simulate(test_parameters)
        result2 = sim_with_noise.simulate(test_parameters)

        # At least some values should differ
        assert not np.allclose(result1, result2)

    def test_plot_simulation_single(self, simulator, test_parameters):
        """Test plotting of a single simulation."""
        result = simulator.simulate(test_parameters)
        fig = simulator.plot_simulation(
            result, params=test_parameters, title="Test Plot"
        )

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2

        # Test that figure contains the expected title
        assert fig._suptitle.get_text() == "Test Plot"

        # Close the figure to avoid memory issues
        plt.close(fig)

    def test_plot_simulation_batch(self, simulator, batch_parameters):
        """Test plotting of batch simulations."""
        result = simulator.simulate(batch_parameters)
        fig = simulator.plot_simulation(result, params=batch_parameters[0])

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2

        # Close the figure to avoid memory issues
        plt.close(fig)

    def test_plot_saving(self, simulator, test_parameters, tmp_path):
        """Test saving a plot to file."""
        result = simulator.simulate(test_parameters)
        fig = simulator.plot_simulation(result)

        # Save figure to a temporary file
        test_file = tmp_path / "test_plot.png"
        fig.savefig(test_file)

        # Verify the file exists and has content
        assert os.path.exists(test_file)
        assert os.path.getsize(test_file) > 0

        plt.close(fig)

    def test_simulation_reproduces_expected_dynamics(self, test_parameters):
        """Test that simulations reproduce expected predator-prey dynamics."""
        sim = LotkaVolterraSimulator(t_span=(0, 100), n_points=1000, noise_level=0.0)
        result = sim.simulate(test_parameters)

        # Reshape to get access to prey and predator populations
        trajectories = sim.get_original_shape(result)
        prey = trajectories[0, 0]
        predator = trajectories[0, 1]

        # Test for oscillatory behavior (characteristic of Lotka-Volterra)
        # 1. Find peaks in prey population
        prey_mean = np.mean(prey)
        prey_above_mean = prey > prey_mean

        # Count transitions from below mean to above mean (crude way to count oscillations)
        transitions = np.sum(np.diff(prey_above_mean.astype(int)) > 0)

        # There should be multiple oscillations in a long enough simulation
        assert transitions >= 2, (
            "Expected oscillatory behavior in predator-prey dynamics"
        )

        # 2. Check phase relationship: predator peaks should lag prey peaks
        # Find a prey peak
        for i in range(1, len(prey) - 1):
            if prey[i] > prey[i - 1] and prey[i] > prey[i + 1] and i < len(prey) - 100:
                # Found a prey peak, now find the next predator peak
                predator_window = predator[i : i + 100]
                predator_peak = np.argmax(predator_window) + i

                # Predator peak should come after prey peak
                assert predator_peak > i, "Expected predator peak to lag prey peak"
                break

    def test_edge_case_high_values(self):
        """Test with extremely high parameter values."""
        # Very high growth and interaction rates
        extreme_params = np.array([10.0, 5.0, 8.0, 3.0, 100.0, 50.0])
        sim = LotkaVolterraSimulator()

        # Should run without errors
        result = sim.simulate(extreme_params)
        assert result.shape == (1, 2 * sim.n_points)

    def test_multiple_simulations_statistically_different(self, test_parameters):
        """Test that multiple simulations with noise produce statistically different results."""
        sim = LotkaVolterraSimulator(noise_level=0.1)
        result = sim.simulate(test_parameters, num_simulations=10)

        # Calculate variance across simulations
        variance = np.var(result, axis=0)

        # Variance should not be zero (simulations should differ)
        assert np.mean(variance) > 0, "Expected different outcomes with noise"


if __name__ == "__main__":
    pytest.main(["-v", __file__])

    """
    Uppon running flie, the pytest will be executed and a test plot will be generated and displayed
    """
    # Definition of the Lotka-Volterra model example simulation
    import matplotlib.pyplot as plt

    # Define default parameters for the simulation:
    # [alpha, beta, gamma, delta, x0, y0]
    params = np.array([1.0, 0.2, 1.5, 0.1, 10.0, 5.0])

    # Create a simulator instance with desired settings
    sim = LotkaVolterraSimulator(t_span=(0, 30), n_points=50, noise_level=0.05)

    # Run a single simulation using the defined parameters
    result = sim.simulate(params)

    # Plot the simulation using the inbuilt plot function
    fig = sim.plot_simulation(result, params=params, title="Lotka-Volterra Simulation")

    # Display the plot
    plt.show()
