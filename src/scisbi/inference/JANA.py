"""
Implementation of the JANA algorithm from Dax et al. (2023).

Paper: https://arxiv.org/abs/2302.09125
"""

from typing import Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import trange
import tempfile
import os

from ..base.inference import BaseInferenceAlgorithm
from ..base.simulator import BaseSimulator
from ..base.summary_statistic import BaseSummaryStatistic


class JANAPosterior:
    """
    Container for JANA results, representing the learned posterior.
    """

    def __init__(
        self,
        model: nn.Module,
        prior: Any,
        summary_statistic: Optional[BaseSummaryStatistic] = None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.prior = prior
        self.summary_statistic = summary_statistic
        self.device = device
        self.model.eval()

    def sample(self, observed_data: Any, num_samples: int = 1) -> np.ndarray:
        """
        Sample from the approximate posterior for the given observed data.
        """
        if self.summary_statistic:
            observed_summary = self.summary_statistic.compute(observed_data)
        else:
            observed_summary = observed_data

        print(observed_summary.shape)
        observed_tensor = (
            torch.from_numpy(np.atleast_2d(observed_summary)).float().to(self.device)
        )
        with torch.no_grad():
            # Tile observed_tensor to match num_samples for batch processing
            x = observed_tensor
            print(x.shape)
            # Sample from prior to get z
            z_samples = self.prior.sample(num_samples)
            if z_samples.ndim == 1:
                z_samples = z_samples.reshape(1, -1)
            z = torch.from_numpy(z_samples).float().to(self.device)

            # Ensure z matches the batch size of x
            if z.shape[0] != x.shape[0]:
                if z.shape[0] == 1:
                    z = z.repeat(x.shape[0], 1)
                else:
                    raise ValueError(
                        f"Batch size mismatch: x has {x.shape[0]} samples, z has {z.shape[0]} samples"
                    )

            samples = self.model(x, z).cpu().numpy()

        return samples

    def log_prob(self, parameters: Any, observed_data: Any) -> np.ndarray:
        """
        JANA does not directly provide a log probability of the posterior.
        This method is not supported.
        """
        raise NotImplementedError("JANA does not provide posterior log probabilities.")


class JANA(BaseInferenceAlgorithm):
    """
    Jointly Amortized Neural Approximation (JANA) of complex Bayesian models.
    """

    def __init__(
        self,
        simulator: BaseSimulator,
        prior: Any,
        model: nn.Module,
        summary_statistic: Optional[BaseSummaryStatistic] = None,
        device: str = "auto",
        **kwargs: Any,
    ):
        super().__init__(simulator, prior, summary_statistic, **kwargs)

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = model.to(self.device)

    def _transform_parameters(self, parameters):
        """
        Transform parameters to ensure they meet simulator constraints.
        Generic implementation that handles different simulator types.
        """
        if isinstance(parameters, torch.Tensor):
            params = parameters.clone()
            # For Gaussian simulators, ensure std parameters are positive
            # For multivariate case, second half are std parameters
            if params.ndim > 1:
                # Batch case
                param_dim = params.shape[-1]
                if param_dim > 2:  # Multivariate case
                    # Second half are std parameters
                    std_start = param_dim // 2
                    params[:, std_start:] = torch.abs(params[:, std_start:])
                elif param_dim == 2:  # Univariate case
                    params[:, -1] = torch.abs(params[:, -1])
            else:
                # Single parameter vector
                param_dim = len(params)
                if param_dim > 2:  # Multivariate case
                    std_start = param_dim // 2
                    params[std_start:] = torch.abs(params[std_start:])
                elif param_dim == 2:  # Univariate case
                    params[-1] = torch.abs(params[-1])

            # Return as tensor/array for multivariate, dict for univariate
            if params.ndim == 1:
                if len(params) == 2:  # Univariate Gaussian
                    return {"mean": params[0].item(), "std": params[1].item()}
                else:
                    # Multivariate case - return as array
                    return params.detach().cpu().numpy()
            else:
                # Batch case - return as tensor
                return params
        else:
            params = np.copy(parameters)
            if params.ndim == 1:
                param_dim = len(params)
                if param_dim > 2:  # Multivariate case
                    std_start = param_dim // 2
                    params[std_start:] = np.abs(params[std_start:])
                elif param_dim == 2:  # Univariate case
                    params[-1] = abs(params[-1])

                # Return dict only for univariate case
                if param_dim == 2:
                    return {"mean": params[0], "std": params[1]}
                else:
                    return params
            else:
                # Batch case - ensure std parameters are positive
                param_dim = params.shape[-1]
                if param_dim > 2:
                    std_start = param_dim // 2
                    params[:, std_start:] = np.abs(params[:, std_start:])
                elif param_dim == 2:
                    params[:, -1] = np.abs(params[:, -1])
                return params

    def infer(
        self,
        observed_data: Optional[Any] = None,
        num_simulations: int = 1000,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        num_epochs: int = 50,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 10,
        verbose: bool = True,
        **kwargs: Any,
    ) -> JANAPosterior:
        """
        Run the JANA inference procedure.
        """
        if verbose:
            print(f"Using device: {self.device}")
            print("1. Simulating data...")

        # 1. Simulate data
        thetas = self.prior.sample(num_simulations)
        # Ensure thetas is always 2D for consistent indexing
        if thetas.ndim == 1:
            thetas = thetas.reshape(1, -1)

        simulated_data = []
        for i in trange(num_simulations, desc="Simulations", disable=not verbose):
            # Transform parameters to ensure they meet simulator constraints
            transformed_params = self._transform_parameters(thetas[i])
            sim_data = self.simulator.simulate(transformed_params)
            if self.summary_statistic:
                sim_data = self.summary_statistic.compute(sim_data)

            # Ensure data is flattened for neural network compatibility
            if hasattr(sim_data, "shape"):
                sim_data = sim_data.flatten()
            elif isinstance(sim_data, list):
                sim_data = np.array(sim_data).flatten()

            simulated_data.append(sim_data)

        thetas = torch.from_numpy(np.array(thetas)).float()
        simulated_data = torch.from_numpy(np.array(simulated_data)).float()

        # 2. Train the model
        if verbose:
            print("2. Training the neural network...")

        dataset = TensorDataset(simulated_data, thetas)

        num_train = int((1.0 - validation_fraction) * num_simulations)
        num_val = num_simulations - num_train
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [num_train, num_val]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_model_saved = False

        # Create temporary file for saving best model
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
        temp_model_path = temp_file.name
        temp_file.close()

        self.model.train()
        for epoch in trange(num_epochs, desc="Epochs", disable=not verbose):
            total_train_loss = 0
            for x_batch, theta_batch in train_loader:
                x_batch, theta_batch = (
                    x_batch.to(self.device),
                    theta_batch.to(self.device),
                )

                optimizer.zero_grad()

                # Sample z from prior
                z_samples = self.prior.sample(x_batch.shape[0])
                # Always ensure z_samples is 2D with correct batch size
                if z_samples.ndim == 1:
                    z_samples = z_samples.reshape(1, -1)
                    z_samples = np.repeat(z_samples, x_batch.shape[0], axis=0)
                z_batch = torch.from_numpy(z_samples).float().to(self.device)

                theta_pred = self.model(x_batch, z_batch)

                loss = nn.MSELoss()(theta_pred, theta_batch)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for x_batch, theta_batch in val_loader:
                    x_batch, theta_batch = (
                        x_batch.to(self.device),
                        theta_batch.to(self.device),
                    )
                    z_samples = self.prior.sample(x_batch.shape[0])
                    # Always ensure z_samples is 2D with correct batch size
                    if z_samples.ndim == 1:
                        z_samples = z_samples.reshape(1, -1)
                        z_samples = np.repeat(z_samples, x_batch.shape[0], axis=0)
                    z_batch = torch.from_numpy(z_samples).float().to(self.device)
                    theta_pred = self.model(x_batch, z_batch)
                    loss = nn.MSELoss()(theta_pred, theta_batch)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            scheduler.step(avg_val_loss)

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
                )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), temp_model_path)
                best_model_saved = True
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= stop_after_epochs:
                if verbose:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

            self.model.train()

        if verbose:
            print("Training finished.")

        # Load best model if saved, otherwise keep current state
        if best_model_saved:
            try:
                self.model.load_state_dict(torch.load(temp_model_path))
                if verbose:
                    print("Loaded best model from training.")
            except FileNotFoundError:
                if verbose:
                    print(
                        "Warning: Best model file not found, keeping current model state."
                    )

        # Cleanup temporary file
        try:
            os.unlink(temp_model_path)
        except OSError:
            pass  # File may not exist or already deleted

        self.model.eval()

        return JANAPosterior(
            self.model, self.prior, self.summary_statistic, self.device
        )
