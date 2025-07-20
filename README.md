# scisbi - Scientific Machine Learning package for Simulation Based Inference

A personal project with multiple aims:

- Creating final project for my PhD course TTSSE (Techniques and Technologies for Scientific Software Engeneering) in Uppsala University
- Rewriting codes for existing simulation based inference methods and frameworks around them for gaining knowledge about current SOTA methods for further research
- Gaining experience in creating, developing and at least shortly maintaining OSS

# Installation

The 0.0.1-alpha release can be installed through pip.

```
pip install --pre scisbi
```

# Bugs
The library is most likely not maintained in any way furter and contains a significant amount of known bugs. The only part which might be suitable for usage is the ABC based inference methods, though their implementation is not optimized in any way, so even those are not encouraged to be used.

If you still would report any bugs that you find, you can create a related issue [here](https://github.com/horvathcso/scisbi/issues). In case you want to contribute, fork the repository on GitHub and create a pull request (PR). All changes are welcomed. In case you would like to maintain this project for whatever reason, contact the project owner.

## Documentation

Due to build inconsistency in github there is two released documentation

- [github pages](https://horvathcso.github.io/scisbi/) this is from local build and working fine
- [scisbi.readthedocs.io](https://scisbi.readthedocs.io) this is from github action build on cloud and missing most sections API documentation

Other better maintained python packages might be more suitable for personal, research or applied uage, such as

- [sbi](https://github.com/sbi-dev/sbi)
- [bayesflow](https://github.com/bayesflow-org/bayesflow)
- [sciope](https://github.com/StochSS/sciope)
- [sbijax](https://github.com/dirmeier/sbijax)

*This project is using the following packages:*

- numpy
- matplotlib
- torch
- tqdm
- scipy
- gillespy2

# Example

## 1. Define Your Simulator and Observed Data

Start by defining how your model simulates data and generate (or load) the "observed" data you wish to analyze.

```python
import numpy as np
from scisbi.simulator import LotkaVolterraSimulator

# True parameters for the Lotka-Volterra model
true_params = np.array([1.0, 0.2, 1.5, 0.1, 10.0, 5.0])

# Initialize the simulator: time span, number of points, and noise level
simulator = LotkaVolterraSimulator(t_span=(0, 30), n_points=50, noise_level=0.05)

# Generate synthetic observed data using the true parameters
# In a real scenario, this would be your experimental data
observed_data = simulator.simulate(parameters=true_params, num_simulations=10)

print(f"Simulator initialized. Observed data shape: {observed_data.shape}")
```

## 2. Specify the Prior Distribution

A prior defines your initial beliefs about the possible range of your model parameters. `scisbi` makes it easy to set up common priors like the `UniformPrior`.

```python
class UniformPrior:
    def __init__(self):
        # Define [min, max] bounds for each parameter
        self.bounds = np.array([
            [0.1, 3.0], [0.01, 1.0], [0.1, 3.0], [0.01, 1.0], [1.0, 20.0], [1.0, 20.0]
        ])

    def sample(self, num_samples=None):
        lows, highs = self.bounds.T
        size = (num_samples, len(lows)) if num_samples is not None else None
        samples = np.random.uniform(lows, highs, size=size)
        return np.maximum(samples, 1e-6)  # Ensure positivity

    def log_prob(self, x):
        x = np.atleast_2d(x)
        # Check if parameters are within bounds; calculate log-probability
        in_bounds = (x >= self.bounds[:, 0]) & (x <= self.bounds[:, 1])
        log_probs = np.where(np.all(in_bounds, axis=1), 
                             -np.log(np.prod(np.diff(self.bounds, axis=1))), 
                             -np.inf)  # Log-prob is -infinity outside bounds
        return log_probs[0] if x.shape[0] == 1 else log_probs

# Instantiate your prior
prior = UniformPrior()

print(f"Prior defined with bounds:\n{prior.bounds}")
```

## 3. Choose a Distance Function

The distance function quantifies how "different" simulated data is from your observed data. A smaller distance indicates a better fit.

```python
def euclidean_distance(x, y):
    """Calculates the Euclidean distance between flattened arrays."""
    x_arr, y_arr = np.asarray(x).flatten(), np.asarray(y).flatten()
    min_len = min(len(x_arr), len(y_arr))
    return np.sqrt(np.sum((x_arr[:min_len] - y_arr[:min_len]) ** 2))

print(f"Distance function ready. Example distance: {euclidean_distance([1,2], [3,4]):.2f}")
```

## 4. Perform ABC Inference

Finally, use `scisbi`'s Approximate Bayesian Computation (ABC) module to sample from the posterior distribution of your model parameters.

```python
from scisbi.inference import ABCRejectionSampling

# Initialize ABCRejectionSampling with your simulator, prior, and distance function
abc = ABCRejectionSampling(
    simulator=simulator,
    prior=prior,
    distance_function=euclidean_distance,
    tolerance=350.0,    # Maximum acceptable distance for a sample
    max_attempts=10000, # Total simulations to run before stopping
    verbose=True,       # Print progress during the inference
)

# Run the inference to obtain a specified number of accepted samples
abc_posterior = abc.infer(observed_data, num_simulations=10)
abc_samples = abc_posterior.get_samples()

print(f"\nABC Inference complete. Accepted posterior samples shape: {abc_samples.shape}")
print(f"First accepted sample: {abc_samples[0]}")
```
