from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP

logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters."""

    perplexity: int = 30
    random_state: int = 42
    n_neighbors: int = 15
    min_dist: float = 0.1


def get_neuron_activations(
    encoded_embeddings: np.ndarray, neuron_index: int, labels: List[str]
) -> List[Tuple[str, float]]:
    """Get activation values for a specific neuron across all words.

    Args:
        encoded_embeddings: Encoded representations
        neuron_index: Index of neuron to analyze
        labels: Word labels

    Returns:
        List of (word, activation) pairs sorted by activation strength
    """
    activations = encoded_embeddings[:, neuron_index]
    word_activations = list(zip(labels, activations))
    return sorted(word_activations, key=lambda x: x[1], reverse=True)


def create_3d_scatter(
    data: Dict,
    method: str,
    selected_neuron: Optional[int] = None,
    encoded_embeddings: Optional[np.ndarray] = None,
) -> Figure:
    """Create an enhanced 3D scatter plot.

    Args:
        data: Visualization data dictionary
        method: Dimensionality reduction method used
        selected_neuron: Optional index of selected neuron
        encoded_embeddings: Optional encoded embeddings for coloring by activation

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    if selected_neuron is not None and encoded_embeddings is not None:
        activations = encoded_embeddings[:, selected_neuron]
        normalized_activations = (activations - activations.min()) / (
            activations.max() - activations.min()
        )

        for group in set(data["Group"]):
            mask = np.array(data["Group"]) == group
            fig.add_trace(
                go.Scatter3d(
                    x=np.array(data["X"])[mask],
                    y=np.array(data["Y"])[mask],
                    z=np.array(data["Z"])[mask],
                    mode="markers",
                    name=group,
                    marker=dict(
                        size=6,
                        color=normalized_activations[mask],
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Neuron Activation"),
                    ),
                    text=[
                        f"Word: {w}<br>Activation: {a:.3f}"
                        for w, a in zip(np.array(data["Word"])[mask], activations[mask])
                    ],
                    hoverinfo="text",
                )
            )
    else:
        fig = px.scatter_3d(
            data,
            x="X",
            y="Y",
            z="Z",
            color="Group",
            hover_name="Word",
            title=f"3D Representation ({method.upper()}) of Word Proximity",
            template="plotly_white",
        )

    axis_title = "Relative position in semantic space"
    if method == "tsne":
        axis_explanation = "(t-SNE projection preserves local similarities)"
    else:
        axis_explanation = "(UMAP projection preserves both local and global structure)"

    fig.update_layout(
        scene=dict(
            xaxis_title=f"Dimension 1<br>{axis_title}<br>{axis_explanation}",
            yaxis_title=f"Dimension 2<br>{axis_title}",
            zaxis_title=f"Dimension 3<br>{axis_title}",
            annotations=[
                dict(
                    showarrow=False,
                    x=0,
                    y=0,
                    z=0,
                    text="Points closer together have more similar meanings",
                    xanchor="left",
                    yanchor="bottom",
                )
            ],
        ),
        showlegend=True,
        legend_title_text="Concept Groups",
        margin=dict(l=0, r=0, b=0, t=30),
    )

    return fig


def plot_3d_proximity(
    encoded_embeddings: np.ndarray,
    labels: List[str],
    concept_groups: Dict[str, List[str]],
    method: str = "tsne",
    metric: str = "cosine",
    config: Optional[VisualizationConfig] = None,
    selected_neuron: Optional[int] = None,
) -> Figure:
    """Create enhanced interactive 3D visualization of word embeddings.

    Args:
        encoded_embeddings: Encoded embeddings matrix
        labels: Word labels
        concept_groups: Dictionary of concept groups
        method: Reduction method
        metric: Distance metric
        config: Optional visualization configuration
        selected_neuron: Optional index of neuron to highlight

    Returns:
        Plotly figure object

    Raises:
        ValueError: If inputs are invalid
    """
    try:
        if len(labels) != encoded_embeddings.shape[0]:
            raise ValueError("Number of labels must match number of embeddings")

        if not concept_groups:
            raise ValueError("At least one concept group must be provided")

        reducer = DimensionalityReducer(config)
        reduced_embeddings = reducer.reduce_to_3d(
            encoded_embeddings, method=method, metric=metric
        )

        word_groups = []
        for word in labels:
            found_groups = [
                group for group, words in concept_groups.items() if word in words
            ]
            word_groups.append(found_groups[0] if found_groups else "Other")

        data = {
            "Word": labels,
            "X": reduced_embeddings[:, 0],
            "Y": reduced_embeddings[:, 1],
            "Z": reduced_embeddings[:, 2],
            "Group": word_groups,
        }

        fig = create_3d_scatter(
            data,
            method,
            selected_neuron,
            encoded_embeddings if selected_neuron is not None else None,
        )

        return fig

    except Exception as e:
        logger.exception("Failed to create 3D visualization")
        raise


class DimensionalityReducer:
    """Handles dimensionality reduction for visualization."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()

    def compute_pairwise_distances(
        self, embeddings: np.ndarray, metric: str = "cosine"
    ) -> np.ndarray:
        """Compute pairwise distances between embeddings."""
        try:
            if metric == "cosine":
                return 1 - cosine_similarity(embeddings)
            elif metric == "euclidean":
                return pairwise_distances(embeddings, metric="euclidean")
            else:
                raise ValueError(f"Unsupported metric: {metric}")

        except Exception as e:
            logger.exception("Failed to compute pairwise distances")
            raise

    def reduce_to_3d(
        self, embeddings: np.ndarray, method: str = "tsne", metric: str = "cosine"
    ) -> np.ndarray:
        """Reduce embeddings to 3D for visualization."""
        try:
            if method == "tsne":
                reducer = TSNE(
                    n_components=3,
                    metric=metric,
                    random_state=self.config.random_state,
                    perplexity=self.config.perplexity,
                )
            elif method == "umap":
                reducer = UMAP(
                    n_components=3,
                    metric=metric,
                    random_state=self.config.random_state,
                    n_neighbors=self.config.n_neighbors,
                    min_dist=self.config.min_dist,
                )
            else:
                raise ValueError(f"Unsupported method: {method}")

            return reducer.fit_transform(embeddings)

        except Exception as e:
            logger.exception(f"Failed to reduce dimensionality using {method}")
            raise
