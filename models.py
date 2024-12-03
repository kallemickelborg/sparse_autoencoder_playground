import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict
from torch.optim import Adam


class SparseAutoencoder(nn.Module):
    """Sparse autoencoder for analyzing word embeddings."""

    def __init__(self, input_size: int, hidden_size: int):
        """Initialize the autoencoder.

        Args:
            input_size: Dimension of input embeddings
            hidden_size: Number of features to learn
        """
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the autoencoder.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Tuple of (decoded output, encoded representation)
        """
        encoded = F.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded, encoded


def train_autoencoder(
    embeddings: np.ndarray,
    hidden_size: int,
    num_epochs: int,
    learning_rate: float = 0.001,
    sparsity_weight: float = 0.1,
    device: str = "cpu",
) -> Tuple[SparseAutoencoder, np.ndarray]:
    """Train the sparse autoencoder on word embeddings.

    Args:
        embeddings: Input word embeddings
        hidden_size: Number of features to learn
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        sparsity_weight: Weight of sparsity regularization
        device: Device to train on ("cpu" or "cuda")

    Returns:
        Tuple of (trained autoencoder, encoded embeddings)
    """
    embeddings_tensor = torch.FloatTensor(embeddings).to(device)

    model = SparseAutoencoder(embeddings.shape[1], hidden_size).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        decoded, encoded = model(embeddings_tensor)

        # Compute losses
        reconstruction_loss = F.mse_loss(decoded, embeddings_tensor)
        sparsity_loss = torch.mean(torch.abs(encoded))
        total_loss = reconstruction_loss + sparsity_weight * sparsity_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        _, encoded = model(embeddings_tensor)
        encoded_embeddings = encoded.cpu().numpy()

    return model, encoded_embeddings
