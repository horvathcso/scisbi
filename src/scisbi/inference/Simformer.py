"""
Implementation of the Simformer algorithm for simulation-based inference.

Simformer uses transformer architecture for amortized Bayesian inference,
processing sequences of simulated data to learn posterior approximations.

Based on the paper: "Simulation-based inference with the transformer" (arXiv:2404.09636)
"""

from typing import Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import trange
import math

from ..base.inference import BaseInferenceAlgorithm
from ..base.simulator import BaseSimulator
from ..base.summary_statistic import BaseSummaryStatistic


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer architecture.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(0), :]


class SimformerTransformer(nn.Module):
    """
    Transformer-based neural network for Simformer inference.

    This model processes sequences of simulation data and parameters
    to learn amortized posterior approximations.
    """

    def __init__(
        self,
        data_dim: int,
        param_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
        activation: str = "relu",
    ):
        super().__init__()

        self.data_dim = data_dim
        self.param_dim = param_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Input embedding layers
        self.data_projection = nn.Linear(data_dim, d_model)
        self.param_projection = nn.Linear(param_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output layers for posterior parameters
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, param_dim * 2),  # mean and log_std
        )

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        data_sequence: torch.Tensor,
        observed_data: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Simformer model.

        Args:
            data_sequence: Simulated data sequence [batch_size, seq_len, data_dim]
            observed_data: Observed data [batch_size, data_dim]
            mask: Optional attention mask [batch_size, seq_len]

        Returns:
            Tuple of (posterior_mean, posterior_log_std)
        """
        batch_size, seq_len, _ = data_sequence.shape

        # Project data to model dimension
        data_embedded = self.data_projection(data_sequence)  # [batch, seq_len, d_model]

        # Add observed data as a special token at the beginning
        obs_embedded = self.data_projection(observed_data).unsqueeze(
            1
        )  # [batch, 1, d_model]

        # Concatenate observed data with simulation sequence
        sequence = torch.cat(
            [obs_embedded, data_embedded], dim=1
        )  # [batch, seq_len+1, d_model]

        # Add positional encoding
        sequence = sequence.transpose(0, 1)  # [seq_len+1, batch, d_model]
        sequence = self.pos_encoder(sequence)
        sequence = sequence.transpose(0, 1)  # [batch, seq_len+1, d_model]

        # Apply transformer encoder
        if mask is not None:
            # Extend mask for observed data token
            extended_mask = torch.cat(
                [
                    torch.zeros(batch_size, 1, device=mask.device, dtype=mask.dtype),
                    mask,
                ],
                dim=1,
            )
        else:
            extended_mask = None

        transformer_output = self.transformer_encoder(
            sequence, src_key_padding_mask=extended_mask
        )

        # Use the first token (observed data) for posterior prediction
        cls_output = transformer_output[:, 0, :]  # [batch, d_model]

        # Generate posterior parameters
        posterior_params = self.output_projection(cls_output)  # [batch, param_dim * 2]

        # Split into mean and log_std
        posterior_mean = posterior_params[:, : self.param_dim]
        posterior_log_std = posterior_params[:, self.param_dim :]

        return posterior_mean, posterior_log_std

    def sample_posterior(
        self,
        data_sequence: torch.Tensor,
        observed_data: torch.Tensor,
        num_samples: int = 1,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample from the learned posterior distribution.
        """
        with torch.no_grad():
            mean, log_std = self.forward(data_sequence, observed_data, mask)
            std = torch.exp(log_std)

            # Sample from Gaussian posterior
            eps = torch.randn(num_samples, *mean.shape, device=mean.device)
            samples = mean.unsqueeze(0) + std.unsqueeze(0) * eps

            # Handle 1D parameter case by squeezing last dimension if it's 1
            if self.param_dim == 1:
                samples = samples.squeeze(-1)

            return samples.squeeze() if num_samples == 1 else samples


class SimformerPosterior:
    """
    Container for Simformer results, representing the learned posterior.
    """

    def __init__(
        self,
        model: SimformerTransformer,
        prior: Any,
        training_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        summary_statistic: Optional[BaseSummaryStatistic] = None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.prior = prior
        self.training_data = training_data  # (data_sequences, params, observed_data)
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

        observed_tensor = (
            torch.from_numpy(np.atleast_2d(observed_summary)).float().to(self.device)
        )

        with torch.no_grad():
            # Use training data sequences for context (in practice, this could be
            # a representative subset or new simulations)
            if self.training_data is not None:
                data_sequences, _, _ = self.training_data
                # Use a subset of training sequences as context
                context_sequences = data_sequences[: min(10, len(data_sequences))].to(
                    self.device
                )
            else:
                # Generate some context sequences on-the-fly
                context_params = self.prior.sample(10)
                if context_params.ndim == 1:
                    context_params = context_params.reshape(1, -1)

                # This would require access to simulator - simplified for now
                context_sequences = torch.randn(
                    10, 50, observed_tensor.shape[-1], device=self.device
                )

            # Repeat observed data for batch processing
            batch_observed = observed_tensor.repeat(num_samples, 1)

            # Use the first context sequence for all samples
            first_context = context_sequences[:1].to(
                self.device
            )  # Take first sequence [1, seq_len, data_dim]
            batch_sequences = first_context.repeat(
                num_samples, 1, 1
            )  # [num_samples, seq_len, data_dim]

            samples = self.model.sample_posterior(
                batch_sequences, batch_observed, num_samples=1
            )

        return samples.cpu().numpy()

    def log_prob(self, parameters: Any, observed_data: Any) -> np.ndarray:
        """
        Compute log probability of parameters given observed data.
        """
        if self.summary_statistic:
            observed_summary = self.summary_statistic.compute(observed_data)
        else:
            observed_summary = observed_data

        observed_tensor = (
            torch.from_numpy(np.atleast_2d(observed_summary)).float().to(self.device)
        )

        param_tensor = (
            torch.from_numpy(np.atleast_2d(parameters)).float().to(self.device)
        )

        with torch.no_grad():
            # Use training context as before
            if self.training_data is not None:
                data_sequences, _, _ = self.training_data
                context_sequences = data_sequences[: min(10, len(data_sequences))].to(
                    self.device
                )
            else:
                context_sequences = torch.randn(
                    10, 50, observed_tensor.shape[-1], device=self.device
                )

            # Get posterior parameters
            # Use only the first sample from context for single observation
            single_context = context_sequences[:1]  # [1, seq_len, data_dim]
            mean, log_std = self.model.forward(single_context, observed_tensor)
            std = torch.exp(log_std)

            # Compute log probability under Gaussian posterior
            log_prob = -0.5 * torch.sum(
                ((param_tensor - mean) / std) ** 2 + 2 * log_std + np.log(2 * np.pi),
                dim=-1,
            )

        return log_prob.cpu().numpy()


class Simformer(BaseInferenceAlgorithm):
    """
    Simformer: Transformer-based amortized inference for simulation-based models.

    This implementation uses a transformer architecture to process sequences of
    simulated data and learn to approximate posterior distributions.
    """

    def __init__(
        self,
        simulator: BaseSimulator,
        prior: Any,
        data_dim: int,
        param_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
        summary_statistic: Optional[BaseSummaryStatistic] = None,
        device: str = "auto",
        **kwargs: Any,
    ):
        super().__init__(simulator, prior, summary_statistic, **kwargs)

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.data_dim = data_dim
        self.param_dim = param_dim

        # Initialize transformer model
        self.model = SimformerTransformer(
            data_dim=data_dim,
            param_dim=param_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
        ).to(self.device)

    def _generate_training_sequences(
        self, num_simulations: int, sequence_length: int, verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate training data consisting of parameter sequences and corresponding
        simulation sequences.
        """
        all_sequences = []
        all_params = []
        all_observed = []

        for i in trange(
            num_simulations, desc="Generating sequences", disable=not verbose
        ):
            # Sample parameters for this sequence
            seq_params = []
            seq_data = []
            for j in range(sequence_length):
                # Sample parameter
                param = self.prior.sample()
                if isinstance(param, (int, float)):
                    param = np.array([param])
                elif hasattr(param, "ndim"):
                    if param.ndim == 0:
                        param = np.array([param])
                    elif param.ndim == 1:
                        param = param
                    else:
                        param = param.flatten()
                else:
                    param = np.atleast_1d(param)
                # Simulate data
                sim_data = self.simulator.simulate(param)
                if self.summary_statistic:
                    sim_data = self.summary_statistic.compute(sim_data)
                # Ensure sim_data is 1D
                if isinstance(sim_data, (list, tuple)):
                    sim_data = np.array(sim_data)
                if sim_data.ndim == 0:
                    sim_data = np.array([sim_data])
                elif sim_data.ndim > 1:
                    sim_data = sim_data.flatten()

                seq_params.append(param)
                seq_data.append(sim_data)

            # Store sequence
            all_sequences.append(np.array(seq_data))
            all_params.append(np.array(seq_params))
            # Generate observed data (last simulation in sequence)
            observed_param = seq_params[-1]
            observed_data = self.simulator.simulate(observed_param)
            if self.summary_statistic:
                observed_data = self.summary_statistic.compute(observed_data)

            if isinstance(observed_data, (list, tuple)):
                observed_data = np.array(observed_data)
            if observed_data.ndim == 0:
                observed_data = np.array([observed_data])
            elif observed_data.ndim > 1:
                observed_data = observed_data.flatten()

            all_observed.append(observed_data)

        return np.array(all_sequences), np.array(all_params), np.array(all_observed)

    def infer(
        self,
        observed_data: Optional[Any] = None,
        num_simulations: int = 1000,
        sequence_length: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 15,
        verbose: bool = True,
        **kwargs: Any,
    ) -> SimformerPosterior:
        """
        Run the Simformer inference procedure.
        """
        if verbose:
            print(f"Using device: {self.device}")
            print("1. Generating training sequences...")

        # Generate training data
        data_sequences, param_sequences, observed_sequences = (
            self._generate_training_sequences(num_simulations, sequence_length, verbose)
        )

        if verbose:
            print(f"Generated {num_simulations} sequences of length {sequence_length}")
            print(f"Data shape: {data_sequences.shape}")
            print(f"Param shape: {param_sequences.shape}")
            print(f"Observed shape: {observed_sequences.shape}")

        # Convert to tensors
        data_tensor = torch.from_numpy(data_sequences).float()
        param_tensor = torch.from_numpy(param_sequences).float()
        observed_tensor = torch.from_numpy(observed_sequences).float()

        # For training, we predict the last parameter in each sequence given the data sequence
        target_params = param_tensor[:, -1, :]  # Last parameter in sequence

        # Create dataset
        dataset = TensorDataset(data_tensor, observed_tensor, target_params)

        # Split into train/validation
        num_train = int((1.0 - validation_fraction) * num_simulations)
        num_val = num_simulations - num_train
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [num_train, num_val]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Setup training
        optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_model_saved = False

        if verbose:
            print("2. Training the Simformer model...")

        self.model.train()
        for epoch in trange(num_epochs, desc="Training epochs", disable=not verbose):
            total_train_loss = 0

            for data_batch, observed_batch, target_batch in train_loader:
                data_batch = data_batch.to(self.device)
                observed_batch = observed_batch.to(self.device)
                target_batch = target_batch.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                pred_mean, pred_log_std = self.model.forward(data_batch, observed_batch)
                pred_std = torch.exp(pred_log_std)

                # Compute negative log-likelihood loss
                mse_loss = torch.mean((pred_mean - target_batch) ** 2)
                nll_loss = torch.mean(
                    0.5 * ((pred_mean - target_batch) / pred_std) ** 2
                    + pred_log_std
                    + 0.5 * np.log(2 * np.pi)
                )

                # Combine losses
                loss = nll_loss + 0.1 * mse_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for data_batch, observed_batch, target_batch in val_loader:
                    data_batch = data_batch.to(self.device)
                    observed_batch = observed_batch.to(self.device)
                    target_batch = target_batch.to(self.device)

                    pred_mean, pred_log_std = self.model.forward(
                        data_batch, observed_batch
                    )
                    pred_std = torch.exp(pred_log_std)

                    mse_loss = torch.mean((pred_mean - target_batch) ** 2)
                    nll_loss = torch.mean(
                        0.5 * ((pred_mean - target_batch) / pred_std) ** 2
                        + pred_log_std
                        + 0.5 * np.log(2 * np.pi)
                    )

                    loss = nll_loss + 0.1 * mse_loss
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            scheduler.step(avg_val_loss)

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}"
                )

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), "best_simformer_model.pth")
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

        # Load best model if saved
        if best_model_saved:
            self.model.load_state_dict(torch.load("best_simformer_model.pth"))
        self.model.eval()

        # Store training data for posterior sampling context
        training_data = (data_tensor, param_tensor, observed_tensor)

        return SimformerPosterior(
            self.model, self.prior, training_data, self.summary_statistic, self.device
        )
