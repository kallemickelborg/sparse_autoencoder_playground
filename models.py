from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize


class SparseAutoencoder(nn.Module):
    """Sparse autoencoder for analyzing word embeddings.

    Attributes:
        encoder (nn.Linear): Encoder layer
        decoder (nn.Linear): Decoder layer
    """

    def __init__(self, input_size: int, hidden_size: int):
        """Initialize the autoencoder.

        Args:
            input_size: Dimension of input embeddings
            hidden_size: Dimension of hidden layer
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
    epochs: int = 500,
    learning_rate: float = 0.001,
    l1_weight: float = 1e-5,
) -> Tuple[SparseAutoencoder, np.ndarray]:
    """Train the sparse autoencoder on word embeddings.

    Args:
        embeddings: Input word embeddings array
        hidden_size: Size of the hidden layer
        epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        l1_weight: Weight for L1 regularization

    Returns:
        Tuple of (trained autoencoder, encoded embeddings)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = embeddings.shape[1]

    autoencoder = SparseAutoencoder(input_size, hidden_size).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    embedding_tensor = torch.from_numpy(embeddings).float().to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        decoded, encoded = autoencoder(embedding_tensor)
        mse_loss = criterion(decoded, embedding_tensor)
        l1_loss = torch.norm(autoencoder.encoder.weight, 1)
        loss = mse_loss + l1_weight * l1_loss
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        _, encoded_embeddings = autoencoder(embedding_tensor)
        return autoencoder, encoded_embeddings.cpu().numpy()


def get_top_words_for_neuron(
    encoded_embeddings: np.ndarray, neuron_index: int, labels: List[str], top_k: int = 5
) -> List[str]:
    """Get the top words that activate a specific neuron.

    Args:
        encoded_embeddings: Encoded representations from the autoencoder
        neuron_index: Index of the neuron to analyze
        labels: List of word labels corresponding to embeddings
        top_k: Number of top words to return

    Returns:
        List of top k words that activate the neuron
    """
    activations = encoded_embeddings[:, neuron_index]
    top_indices = activations.argsort()[-top_k:][::-1]
    return [labels[i] for i in top_indices]
