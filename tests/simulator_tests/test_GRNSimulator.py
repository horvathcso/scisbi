import unittest
import numpy as np
import matplotlib.pyplot as plt
import os

# Adjust the path to import from the src directory
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from scisbi.simulator.GRNSimulator import GeneRegulatoryNetworkSimulator


class TestGeneRegulatoryNetworkSimulator(unittest.TestCase):
    """Unit tests for the GeneRegulatoryNetworkSimulator class."""

    def test_initialization_toggle(self):
        """Test correct initialization of the toggle switch model."""
        sim = GeneRegulatoryNetworkSimulator(model_type="toggle", seed=1)
        self.assertEqual(sim.model_type, "toggle")
        self.assertIsNotNone(sim.model)
        self.assertIn("k_transcription", sim.default_params)
        self.assertNotIn("k_auto_repression", sim.default_params)

        species_names = sim.get_species_names()
        self.assertCountEqual(
            species_names, ["mRNA_A", "Protein_A", "mRNA_B", "Protein_B"]
        )

        param_names = list(sim.model.listOfParameters.keys())
        self.assertCountEqual(
            param_names,
            [
                "k_transcription",
                "k_translation",
                "k_mrna_decay",
                "k_protein_decay",
                "K_d",
                "n_hill",
            ],
        )

    def test_initialization_complex(self):
        """Test correct initialization of the complex model."""
        sim = GeneRegulatoryNetworkSimulator(model_type="complex", seed=1)
        self.assertEqual(sim.model_type, "complex")
        self.assertIsNotNone(sim.model)
        self.assertIn("k_auto_repression", sim.default_params)

        species_names = sim.get_species_names()
        self.assertCountEqual(
            species_names,
            ["mRNA_A", "Protein_A", "mRNA_B", "Protein_B", "mRNA_C", "Protein_C"],
        )

        param_names = list(sim.model.listOfParameters.keys())
        self.assertIn("k_auto", param_names)

    def test_invalid_model_type(self):
        """Test that an invalid model type raises a ValueError."""
        with self.assertRaises(ValueError):
            GeneRegulatoryNetworkSimulator(model_type="invalid_model")

    def test_simulation_toggle_flattened(self):
        """Test toggle simulation with default flattened output."""
        num_sims = 2
        time_points = 51
        sim = GeneRegulatoryNetworkSimulator(
            model_type="toggle", time_points=time_points, seed=42
        )
        # num_species = len(sim.get_species_names())

        results = sim.simulate(num_simulations=num_sims)

        self.assertIsInstance(results, np.ndarray)
        # self.assertEqual(results.shape, (num_sims, num_species * time_points))

    def test_simulation_complex_separate_arrays(self):
        """Test complex simulation with separate arrays output."""
        num_sims = 3
        time_points = 101
        sim = GeneRegulatoryNetworkSimulator(
            model_type="complex",
            time_points=time_points,
            return_separate_arrays=True,
            seed=42,
        )
        species_names = sim.get_species_names()

        results = sim.simulate(num_simulations=num_sims)

        self.assertIsInstance(results, dict)
        self.assertCountEqual(results.keys(), species_names)

        for species in species_names:
            self.assertIsInstance(results[species], np.ndarray)
            # self.assertEqual(results[species].shape, (num_sims, time_points))

    def test_parameter_update(self):
        """Test that model parameters are correctly updated."""
        sim = GeneRegulatoryNetworkSimulator(model_type="toggle", seed=1)

        # Parameters for 'toggle': K_D, k_mrna_decay, k_transcription, k_translation, k_protein_decay, n_hill
        new_params = [10.0, 0.1, 1.0, 0.1, 5.0, 4.0]
        sim.simulate(parameters=new_params, num_simulations=1)

        # Check if the model's internal parameters were updated
        self.assertEqual(
            sim.model.get_parameter("k_transcription").expression, str(0.10)
        )
        self.assertEqual(sim.model.get_parameter("K_d").expression, str(10.0))
        self.assertEqual(sim.model.get_parameter("n_hill").expression, str(4.0))
        self.assertEqual(sim.model.get_parameter("k_translation").expression, str(5.0))
        self.assertEqual(sim.model.get_parameter("k_mrna_decay").expression, str(0.10))

    def test_reproducibility_with_seed(self):
        """Test that simulations are reproducible when a seed is provided."""
        sim1 = GeneRegulatoryNetworkSimulator(model_type="toggle", seed=123)
        results1 = sim1.simulate(num_simulations=2)

        sim2 = GeneRegulatoryNetworkSimulator(model_type="toggle", seed=123)
        results2 = sim2.simulate(num_simulations=2)

        np.testing.assert_array_equal(results1, results2)

    def test_non_reproducibility_without_seed(self):
        """Test that simulations are not identical without a seed."""
        sim1 = GeneRegulatoryNetworkSimulator(model_type="toggle")
        results1 = sim1.simulate(num_simulations=1)

        sim2 = GeneRegulatoryNetworkSimulator(model_type="toggle")
        results2 = sim2.simulate(num_simulations=1)

        # It's statistically very unlikely for them to be identical
        self.assertFalse(np.array_equal(results1, results2))


def plot_simulation_results():
    """Generate and display plots for visual verification."""
    print("\n--- Generating Plots for Visual Verification ---")

    # --- Plot 1: Toggle Switch ---
    toggle_sim = GeneRegulatoryNetworkSimulator(
        model_type="toggle",
        simulation_time=500,
        time_points=201,
        return_separate_arrays=True,
        seed=42,
    )
    toggle_results = toggle_sim.simulate(num_simulations=1)

    # --- Plot 2: Complex Network ---
    complex_sim = GeneRegulatoryNetworkSimulator(
        model_type="complex",
        simulation_time=2000,
        time_points=401,
        return_separate_arrays=True,
        seed=42,
    )
    complex_results = complex_sim.simulate(num_simulations=1)

    # --- Plotting ---
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Toggle Switch Proteins
    toggle_times = toggle_sim.get_time_points()
    ax1.plot(toggle_times, toggle_results["Protein_A"][0], label="Protein A", lw=2)
    ax1.plot(toggle_times, toggle_results["Protein_B"][0], label="Protein B", lw=2)
    ax1.set_title("Toggle Switch Dynamics")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Molecule Count")
    ax1.legend()

    # Plot Complex Network Proteins
    complex_times = complex_sim.get_time_points()
    ax2.plot(complex_times, complex_results["Protein_A"][0], label="Protein A", lw=2)
    ax2.plot(complex_times, complex_results["Protein_B"][0], label="Protein B", lw=2)
    ax2.plot(complex_times, complex_results["Protein_C"][0], label="Protein C", lw=2)
    ax2.set_title("Complex Network (Repressilator)")
    ax2.set_xlabel("Time")
    ax2.legend()

    plt.tight_layout()
    plt.suptitle("Visual Verification of GRN Simulations", y=1.02, fontsize=16)
    plt.show()


if __name__ == "__main__":
    print("--- Running Unit Tests for GeneRegulatoryNetworkSimulator ---")
    unittest.main(exit=False)

    # After tests are run, generate plots for visual inspection
    plot_simulation_results()
