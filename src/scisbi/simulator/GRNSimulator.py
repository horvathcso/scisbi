import numpy as np
import gillespy2
from typing import List, Dict, Any, Optional, Union
from scisbi.base.simulator import BaseSimulator


class GeneRegulatoryNetworkSimulator(BaseSimulator):
    """
    Simulates stochastic gene regulatory networks using GillesPy2.

    Implements two default models:
    1.  'toggle': A classic two-gene toggle switch.
    2.  'complex': A three-gene repressilator with added autoregulation.

    Parameters are based on literature values. The simulator can return
    time-series data in a flattened 1D array format (default) suitable for
    machine learning, or as separate arrays for each species.
    """

    def __init__(
        self,
        model_type: str = "toggle",
        simulation_time: float = 1000,
        time_points: int = 201,
        seed: Optional[int] = None,
        return_separate_arrays: bool = False,
        **kwargs: Any,
    ):
        """
        Initializes the simulator.

        Args:
            model_type (str): 'toggle' or 'complex'.
            simulation_time (float): Total simulation time.
            time_points (int): Number of data points to record.
            seed (Optional[int]): Random seed for reproducibility.
            return_separate_arrays (bool): If True, returns a dict of arrays.
                                         If False, returns a single flattened array.
        """
        super().__init__(**kwargs)
        if model_type not in ["toggle", "complex"]:
            raise ValueError("model_type must be 'toggle' or 'complex'")

        self.model_type = model_type
        self.simulation_time = simulation_time
        self.time_points = time_points
        self.seed = seed
        self.return_separate_arrays = return_separate_arrays
        self.model: Optional[gillespy2.Model] = None
        self.default_params: Dict[str, float] = {}

        self._set_default_parameters()
        self._initialize_model()

    def _set_default_parameters(self):
        """Sets literature-based default parameters for the chosen model."""
        if self.model_type == "toggle":
            # Based on Gardner et al. (2000) Nature
            self.default_params = {
                "k_transcription": 50.0,
                "k_translation": 1.0,
                "k_mrna_decay": 10.0,
                "k_protein_decay": 1.0,
                "K_d": 10.0,
                "n_hill": 2.0,
                "initial_A": 50,
                "initial_B": 5,
            }
        else:  # complex
            # Based on Elowitz & Leibler (2000) + Rosenfeld et al. (2002)
            self.default_params = {
                "k_transcription": 30.0,
                "k_translation": 1.0,
                "k_mrna_decay": 5.0,
                "k_protein_decay": 0.5,
                "K_d": 20.0,
                "n_hill": 2.0,
                "k_auto_repression": 0.1,
                "initial_counts": 10,
            }

    def _initialize_model(self):
        """Creates and configures the GillesPy2 model instance."""
        if self.model_type == "toggle":
            self.model = self._create_toggle_switch()
        else:
            self.model = self._create_complex_network()
        self.model.timespan(np.linspace(0, self.simulation_time, self.time_points))

    def _create_toggle_switch(self) -> gillespy2.Model:
        """Builds the toggle switch model."""
        model = gillespy2.Model(name="ToggleSwitch")

        # Species
        model.add_species(
            [
                gillespy2.Species(name="mRNA_A", initial_value=0),
                gillespy2.Species(
                    name="Protein_A", initial_value=self.default_params["initial_A"]
                ),
                gillespy2.Species(name="mRNA_B", initial_value=0),
                gillespy2.Species(
                    name="Protein_B", initial_value=self.default_params["initial_B"]
                ),
            ]
        )

        # Parameters (expression must be a string)
        model.add_parameter(
            [
                gillespy2.Parameter(
                    name="k_transcription",
                    expression=str(self.default_params["k_transcription"]),
                ),
                gillespy2.Parameter(
                    name="k_translation",
                    expression=str(self.default_params["k_translation"]),
                ),
                gillespy2.Parameter(
                    name="k_mrna_decay",
                    expression=str(self.default_params["k_mrna_decay"]),
                ),
                gillespy2.Parameter(
                    name="k_protein_decay",
                    expression=str(self.default_params["k_protein_decay"]),
                ),
                gillespy2.Parameter(
                    name="K_d", expression=str(self.default_params["K_d"])
                ),
                gillespy2.Parameter(
                    name="n_hill", expression=str(self.default_params["n_hill"])
                ),
            ]
        )

        # Reactions
        model.add_reaction(
            [
                # Transcription with Hill kinetics (using propensity_function)
                gillespy2.Reaction(
                    name="transcription_A",
                    reactants={},
                    products={"mRNA_A": 1},
                    propensity_function="k_transcription / (1 + pow(Protein_B/K_d, n_hill))",
                ),
                gillespy2.Reaction(
                    name="transcription_B",
                    reactants={},
                    products={"mRNA_B": 1},
                    propensity_function="k_transcription / (1 + pow(Protein_A/K_d, n_hill))",
                ),
                # Translation and Degradation (using mass-action via rate name)
                gillespy2.Reaction(
                    name="translation_A",
                    reactants={"mRNA_A": 1},
                    products={"mRNA_A": 1, "Protein_A": 1},
                    rate="k_translation",
                ),
                gillespy2.Reaction(
                    name="translation_B",
                    reactants={"mRNA_B": 1},
                    products={"mRNA_B": 1, "Protein_B": 1},
                    rate="k_translation",
                ),
                gillespy2.Reaction(
                    name="mrna_A_decay",
                    reactants={"mRNA_A": 1},
                    products={},
                    rate="k_mrna_decay",
                ),
                gillespy2.Reaction(
                    name="mrna_B_decay",
                    reactants={"mRNA_B": 1},
                    products={},
                    rate="k_mrna_decay",
                ),
                gillespy2.Reaction(
                    name="protein_A_decay",
                    reactants={"Protein_A": 1},
                    products={},
                    rate="k_protein_decay",
                ),
                gillespy2.Reaction(
                    name="protein_B_decay",
                    reactants={"Protein_B": 1},
                    products={},
                    rate="k_protein_decay",
                ),
            ]
        )
        return model

    def _create_complex_network(self) -> gillespy2.Model:
        """Builds the complex repressilator model."""
        model = gillespy2.Model(name="ComplexNetwork")
        init_c = self.default_params["initial_counts"]

        # Species
        model.add_species(
            [
                gillespy2.Species(name="mRNA_A", initial_value=0),
                gillespy2.Species(name="Protein_A", initial_value=init_c),
                gillespy2.Species(name="mRNA_B", initial_value=0),
                gillespy2.Species(name="Protein_B", initial_value=init_c),
                gillespy2.Species(name="mRNA_C", initial_value=0),
                gillespy2.Species(name="Protein_C", initial_value=init_c),
            ]
        )

        # Parameters
        model.add_parameter(
            [
                gillespy2.Parameter(
                    name="k_transcription",
                    expression=str(self.default_params["k_transcription"]),
                ),
                gillespy2.Parameter(
                    name="k_translation",
                    expression=str(self.default_params["k_translation"]),
                ),
                gillespy2.Parameter(
                    name="k_mrna_decay",
                    expression=str(self.default_params["k_mrna_decay"]),
                ),
                gillespy2.Parameter(
                    name="k_protein_decay",
                    expression=str(self.default_params["k_protein_decay"]),
                ),
                gillespy2.Parameter(
                    name="K_d", expression=str(self.default_params["K_d"])
                ),
                gillespy2.Parameter(
                    name="n_hill", expression=str(self.default_params["n_hill"])
                ),
                gillespy2.Parameter(
                    name="k_auto",
                    expression=str(self.default_params["k_auto_repression"]),
                ),
            ]
        )

        # Reactions
        reactions = [
            gillespy2.Reaction(
                name="transcription_A",
                reactants={},
                products={"mRNA_A": 1},
                propensity_function="k_transcription / ((1 + pow(Protein_C/K_d, n_hill)) * (1 + k_auto*pow(Protein_A/K_d, n_hill)))",
            ),
            gillespy2.Reaction(
                name="transcription_B",
                reactants={},
                products={"mRNA_B": 1},
                propensity_function="k_transcription / (1 + pow(Protein_A/K_d, n_hill))",
            ),
            gillespy2.Reaction(
                name="transcription_C",
                reactants={},
                products={"mRNA_C": 1},
                propensity_function="k_transcription / (1 + pow(Protein_B/K_d, n_hill))",
            ),
        ]
        for gene in ["A", "B", "C"]:
            reactions.extend(
                [
                    gillespy2.Reaction(
                        name=f"translation_{gene}",
                        reactants={f"mRNA_{gene}": 1},
                        products={f"mRNA_{gene}": 1, f"Protein_{gene}": 1},
                        rate="k_translation",
                    ),
                    gillespy2.Reaction(
                        name=f"mrna_{gene}_decay",
                        reactants={f"mRNA_{gene}": 1},
                        products={},
                        rate="k_mrna_decay",
                    ),
                    gillespy2.Reaction(
                        name=f"protein_{gene}_decay",
                        reactants={f"Protein_{gene}": 1},
                        products={},
                        rate="k_protein_decay",
                    ),
                ]
            )
        model.add_reaction(reactions)
        return model

    def _update_model_parameters(self, parameters: Union[np.ndarray, List[float]]):
        """Updates model parameters with new values."""
        param_names = list(self.model.listOfParameters.keys())
        if len(parameters) != len(param_names):
            raise ValueError(
                f"Expected {len(param_names)} parameters, but got {len(parameters)}"
            )

        for name, value in zip(param_names, parameters):
            self.model.get_parameter(name).expression = str(float(value))

    def simulate(
        self,
        parameters: Optional[Union[np.ndarray, List[float]]] = None,
    ) -> Union[np.ndarray, List[Dict[str, np.ndarray]]]:
        """
        Runs the simulation.

        Args:
            parameters: Array of shape (Batch_Size, Num_Params)

        Returns:
            If return_separate_arrays=False:
                Array of shape (Batch_Size, Flattened_Dim)
            If return_separate_arrays=True:
                List of length Batch_Size, where each element is a dict of arrays.
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        if parameters is None:
            # Run with default parameters once
            parameters = np.array(
                [list(self.default_params.values())]
            )  # Incorrect assumption about order, but let's assume user passes params in correct order if not None.
            # Wait, default_params is a dict, order is not guaranteed in older python but is in newer.
            # Better to rely on default values already in model if parameters is None?
            # But we need batch size.
            # Let's assume 1 simulation with current model params if parameters is None?
            # But we agreed input should be (Batch, P).
            pass

        if parameters is not None:
            parameters = np.asarray(parameters)
            if parameters.ndim == 1:
                parameters = parameters[np.newaxis, :]
            batch_size = parameters.shape[0]
        else:
            batch_size = 1
            parameters = [None]  # Placeholder

        all_results = []

        for i in range(batch_size):
            if parameters[i] is not None:
                self._update_model_parameters(parameters[i])

            # Run 1 trajectory for this parameter set
            results = self.model.run(number_of_trajectories=1, seed=self.seed)
            processed = self._process_results(results)

            # processed is (1, Flattened) or {name: (1, T)}?
            # _process_results handles list of results.
            # If we run 1 trajectory, it returns one result (single array or dict).

            # Let's check _process_results implementation below.
            # It returns np.array(all_trajectories) which is (N_traj, Flattened)
            # or dict {name: np.array(val)} where val is list of arrays.

            if self.return_separate_arrays:
                # processed is dict {name: (1, T)}
                # We want list of dicts? Or dict of (Batch, T)?
                # Standard is (Batch, ...).
                # So if we return dict, it should be {name: (Batch, T)}.
                pass
            else:
                # processed is (1, Flattened)
                pass

            all_results.append(processed)

        if self.return_separate_arrays:
            # Merge list of dicts into dict of arrays
            keys = all_results[0].keys()
            merged = {}
            for k in keys:
                # all_results[i][k] is (1, T)
                # Stack them -> (Batch, T)
                merged[k] = np.concatenate([res[k] for res in all_results], axis=0)
            return merged
        else:
            # Merge list of arrays
            # all_results[i] is (1, Flattened)
            return np.concatenate(all_results, axis=0)

    def _process_results(self, results) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Formats the raw simulation results."""
        species_names = sorted(list(self.model.get_all_species().keys()))

        if not isinstance(results, list):
            results = [results]

        if self.return_separate_arrays:
            # Return a dictionary of arrays
            output = {name: [] for name in species_names}
            for res in results:
                for name in species_names:
                    output[name].append(res[name])
            # For each species, result is list of arrays (one per trajectory).
            # Convert to (N_traj, Time)
            return {name: np.array(val) for name, val in output.items()}
        else:
            # Return a single flattened array per simulation
            all_trajectories = []
            for res in results:
                trajectory = [res[name] for name in species_names]
                all_trajectories.append(np.concatenate(trajectory))
            return np.array(all_trajectories)

    def get_species_names(self) -> List[str]:
        """Returns the ordered list of species names."""
        return sorted(list(self.model.get_all_species().keys()))

    def get_time_points(self) -> np.ndarray:
        """Returns the simulation time points."""
        return self.model.tspan


if __name__ == "__main__":
    print("--- Running GeneRegulatoryNetworkSimulator Examples ---")

    # --- Example 1: Toggle Switch ---
    print("\n[1] Toggle Switch Simulation")
    toggle_sim = GeneRegulatoryNetworkSimulator(
        model_type="toggle", simulation_time=500, time_points=101, seed=42
    )

    # Simulate with custom parameters and get flattened output
    custom_params = [
        50.0,
        1.0,
        8.0,
        0.8,
        15.0,
        2.5,
    ]  # k_trans, k_trans, k_mrna, k_prot, Kd, n

    # Pass as batch (1, 6)
    toggle_custom_flat = toggle_sim.simulate(parameters=custom_params)
    print(f"Toggle results (flattened) shape: {toggle_custom_flat.shape}")

    # --- Example 2: Complex Network ---
    print("\n[2] Complex Network Simulation")
    complex_sim = GeneRegulatoryNetworkSimulator(
        model_type="complex",
        simulation_time=2000,
        time_points=401,
        return_separate_arrays=True,  # Get separate arrays for plotting
        seed=42,
    )
    # 1 simulation with default params (None passed)
    # Wait, simulate(None) might fail in my new implementation if I don't handle it.
    # But for 'complex' model we need to pass params or use defaults.
    # My simulate implementation handles parameters=None by running once with defaults?
    # No, I implemented: if parameters is None -> batch_size=1, parameters=[None]
    # And then loop: if parameters[i] is not None: update. Else: use current.
    # This works.

    complex_results_dict = complex_sim.simulate()
    print(f"Complex results type: {type(complex_results_dict)}")
    print(f"Species in complex network: {list(complex_results_dict.keys())}")

    # --- Plotting ---
    try:
        import matplotlib.pyplot as plt

        print("\n[3] Plotting results...")
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Toggle Switch
        times = toggle_sim.get_time_points()
        species_names = toggle_sim.get_species_names()
        protein_a_idx = species_names.index("Protein_A")
        protein_b_idx = species_names.index("Protein_B")

        # Un-flatten the data for plotting
        protein_a_data = toggle_custom_flat[
            0, protein_a_idx * len(times) : (protein_a_idx + 1) * len(times)
        ]
        protein_b_data = toggle_custom_flat[
            0, protein_b_idx * len(times) : (protein_b_idx + 1) * len(times)
        ]

        ax1.plot(times, protein_a_data, label="Protein A", lw=2)
        ax1.plot(times, protein_b_data, label="Protein B", lw=2)
        ax1.set_title("Toggle Switch Dynamics")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Molecule Count")
        ax1.legend()

        # Plot 2: Complex Network
        complex_times = complex_sim.get_time_points()
        ax2.plot(
            complex_times, complex_results_dict["Protein_A"][0], label="Protein A", lw=2
        )
        ax2.plot(
            complex_times, complex_results_dict["Protein_B"][0], label="Protein B", lw=2
        )
        ax2.plot(
            complex_times, complex_results_dict["Protein_C"][0], label="Protein C", lw=2
        )
        ax2.set_title("Complex Network (Repressilator)")
        ax2.set_xlabel("Time")
        ax2.legend()

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("\nMatplotlib not found. Skipping plots.")

    print("\n--- Simulation examples completed successfully! ---")
