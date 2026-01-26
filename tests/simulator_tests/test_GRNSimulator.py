import unittest
import numpy as np
import os
import sys

# Adjust the path to import from the src directory
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from scisbi.simulator.GRNSimulator import GeneRegulatoryNetworkSimulator


class TestGeneRegulatoryNetworkSimulator(unittest.TestCase):
    """Unit tests for the GeneRegulatoryNetworkSimulator class."""

    def _get_default_param_batch(self, sim, batch_size):
        """Helper to create a batch of default parameters."""
        param_names = list(sim.model.listOfParameters.keys())
        # Map parameter names to default values
        # default_params in sim is a dict.
        # But we need values in order of param_names.
        # We assume default_params contains all keys needed or we read from model?
        # sim.default_params has values.
        # But model might have defaults in expression strings.
        # Safest is to use sim.default_params if keys match.

        values = []
        for name in param_names:
            if name in sim.default_params:
                values.append(sim.default_params[name])
            else:
                # Fallback? Should not happen in this test suite setup
                values.append(0.0)

        params_single = np.array(values)
        return np.tile(params_single, (batch_size, 1))

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

        # Create batch of default params
        params = self._get_default_param_batch(sim, num_sims)

        results = sim.simulate(params)

        self.assertIsInstance(results, np.ndarray)
        # Expected shape: (num_sims, num_species * time_points)
        num_species = len(sim.get_species_names())
        self.assertEqual(results.shape, (num_sims, num_species * time_points))

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

        params = self._get_default_param_batch(sim, num_sims)
        results = sim.simulate(params)

        self.assertIsInstance(results, dict)
        self.assertCountEqual(results.keys(), species_names)

        for species in species_names:
            self.assertIsInstance(results[species], np.ndarray)
            # Expected shape: (num_sims, time_points)
            self.assertEqual(results[species].shape, (num_sims, time_points))

    def test_parameter_update(self):
        """Test that model parameters are correctly updated."""
        sim = GeneRegulatoryNetworkSimulator(model_type="toggle", seed=1)

        # Parameters for 'toggle' need to match the order of keys in model
        # The previous test used a list [10.0, ...] assuming a specific order.
        # The order in gillespy2 parameters dict usually matches insertion order but not guaranteed across versions?
        # But let's assume the previous test order was correct or we should query order.
        # Order: k_transcription, k_translation, k_mrna_decay, k_protein_decay, K_d, n_hill
        # Let's verify against what was asserted in previous test:
        # k_transcription=0.10, K_d=10.0, n_hill=4.0, k_translation=5.0, k_mrna_decay=0.10

        # Previous test passed: [10.0, 0.1, 1.0, 0.1, 5.0, 4.0]
        # And checked:
        # k_transcription=0.10 (3rd element in list was 1.0? No wait.)

        # Let's look at previous test assertions:
        # self.assertEqual(sim.model.get_parameter("k_transcription").expression, str(0.10))
        # self.assertEqual(sim.model.get_parameter("K_d").expression, str(10.0))
        # self.assertEqual(sim.model.get_parameter("n_hill").expression, str(4.0))
        # self.assertEqual(sim.model.get_parameter("k_translation").expression, str(5.0))
        # self.assertEqual(sim.model.get_parameter("k_mrna_decay").expression, str(0.10))

        # If input was [10.0, 0.1, 1.0, 0.1, 5.0, 4.0]
        # And k_transcription became 0.10. That means k_transcription was the 2nd (index 1) or 4th (index 3) element?
        # Or maybe the order in sim.model.listOfParameters.keys() is specific.

        # To make this robust, I should rely on keys order.
        # In _create_toggle_switch, I added params in this order:
        # k_transcription, k_translation, k_mrna_decay, k_protein_decay, K_d, n_hill

        # So:
        # 0: k_transcription
        # 1: k_translation
        # 2: k_mrna_decay
        # 3: k_protein_decay
        # 4: K_d
        # 5: n_hill

        # If I want:
        # k_transcription = 0.1
        # K_d = 10.0
        # n_hill = 4.0
        # k_translation = 5.0
        # k_mrna_decay = 0.1
        # k_protein_decay = (anything, say 1.0)

        new_params_dict = {
            "k_transcription": 0.1,
            "k_translation": 5.0,
            "k_mrna_decay": 0.1,
            "k_protein_decay": 1.0,
            "K_d": 10.0,
            "n_hill": 4.0,
        }

        # Construct list in correct order
        param_names = list(sim.model.listOfParameters.keys())
        new_params_list = [new_params_dict[name] for name in param_names]

        sim.simulate(parameters=new_params_list)

        # Check if the model's internal parameters were updated
        self.assertEqual(
            sim.model.get_parameter("k_transcription").expression, str(0.1)
        )
        self.assertEqual(sim.model.get_parameter("K_d").expression, str(10.0))
        self.assertEqual(sim.model.get_parameter("n_hill").expression, str(4.0))
        self.assertEqual(sim.model.get_parameter("k_translation").expression, str(5.0))
        self.assertEqual(sim.model.get_parameter("k_mrna_decay").expression, str(0.1))

    def test_reproducibility_with_seed(self):
        """Test that simulations are reproducible when a seed is provided."""
        sim1 = GeneRegulatoryNetworkSimulator(model_type="toggle", seed=123)
        params1 = self._get_default_param_batch(sim1, 2)
        results1 = sim1.simulate(params1)

        sim2 = GeneRegulatoryNetworkSimulator(model_type="toggle", seed=123)
        params2 = self._get_default_param_batch(sim2, 2)
        results2 = sim2.simulate(params2)

        np.testing.assert_array_equal(results1, results2)

    def test_non_reproducibility_without_seed(self):
        """Test that simulations are not identical without a seed."""
        sim1 = GeneRegulatoryNetworkSimulator(model_type="toggle")
        params1 = self._get_default_param_batch(sim1, 1)
        results1 = sim1.simulate(params1)

        sim2 = GeneRegulatoryNetworkSimulator(model_type="toggle")
        params2 = self._get_default_param_batch(sim2, 1)
        results2 = sim2.simulate(params2)

        # It's statistically very unlikely for them to be identical
        self.assertFalse(np.array_equal(results1, results2))


def plot_simulation_results():
    """Generate and display plots for visual verification."""
    # ... (omitted for brevity in test suite execution, but good for manual run)
    pass


if __name__ == "__main__":
    unittest.main(exit=False)
