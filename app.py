import logging
from typing import Dict, List, Tuple, Optional

import streamlit as st
import torch
import numpy as np
import pandas as pd

from models import SparseAutoencoder, train_autoencoder, get_top_words_for_neuron
from visualization import plot_3d_proximity, VisualizationConfig, get_neuron_activations
from data import EmbeddingLoader, parse_concept_groups, prepare_data_for_analysis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_WORDS = "doctor, nurse, engineer, teacher, lawyer, man, woman"
DEFAULT_CONCEPT_GROUPS = """Profession: doctor, nurse, engineer, teacher
Gender: man, woman"""

# Initialize session state
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "visualization_config" not in st.session_state:
    st.session_state.visualization_config = VisualizationConfig()
if "selected_neuron" not in st.session_state:
    st.session_state.selected_neuron = None


def setup_sidebar() -> Dict:
    """Configure and return sidebar parameters."""
    st.sidebar.title("Configuration")

    # Model and data parameters
    model_params = {
        "embedding_model": st.sidebar.selectbox(
            "Select Embedding Model",
            ["BERT", "GloVe (100d)"],  # Removed Word2Vec as it requires manual download
            help="BERT works best but is slower. GloVe is faster but less accurate.",
        ),
        "custom_words": st.sidebar.text_area(
            "Input Words (comma-separated)",
            value=DEFAULT_WORDS,
            help="Enter words to analyze, separated by commas",
        ),
        "concept_groups": st.sidebar.text_area(
            "Concept Groups (Group: words)",
            value=DEFAULT_CONCEPT_GROUPS,
            help="Enter concept groups in the format 'Group: word1, word2'",
        ),
        "monosemantic_threshold": st.sidebar.slider(
            "Monosemanticity Threshold",
            0.0,
            1.0,
            0.6,
            help="Higher values mean more selective neurons",
        ),
        "hidden_size": st.sidebar.slider(
            "Hidden Layer Size",
            10,
            200,
            50,
            help="Number of neurons in the autoencoder",
        ),
        "num_epochs": st.sidebar.slider(
            "Training Epochs", 100, 1000, 500, help="Number of training iterations"
        ),
    }

    # Visualization parameters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Visualization Settings")

    viz_config = st.session_state.visualization_config
    viz_config.perplexity = st.sidebar.slider(
        "t-SNE Perplexity",
        5,
        50,
        30,
        help="Higher values consider more global structure",
    )
    viz_config.n_neighbors = st.sidebar.slider(
        "UMAP Neighbors", 2, 30, 15, help="Higher values create more global structure"
    )
    viz_config.min_dist = st.sidebar.slider(
        "UMAP Minimum Distance",
        0.0,
        1.0,
        0.1,
        help="Controls how tightly points are clustered",
    )

    model_params.update(
        {
            "dim_reduction": st.sidebar.selectbox(
                "Dimensionality Reduction Method",
                ("UMAP", "t-SNE"),
                help="UMAP better preserves global structure, t-SNE is better for local structure",
            ),
            "distance_metric": st.sidebar.selectbox(
                "Distance Metric",
                ("cosine", "euclidean"),
                help="Cosine is better for word embeddings",
            ),
        }
    )

    return model_params


def run_analysis(
    params: Dict,
) -> Optional[Tuple[np.ndarray, List[str], Dict, List[Tuple[int, float]]]]:
    """Run the main analysis pipeline."""
    try:
        # Reset selected neuron
        st.session_state.selected_neuron = None

        # Create an EmbeddingLoader instance and load embeddings
        with st.spinner(
            "Loading embedding model... This may take a while the first time."
        ):
            loader = EmbeddingLoader()
            embeddings, labels, selected_words = loader.load_embeddings(
                params["embedding_model"],
                [w.strip() for w in params["custom_words"].split(",") if w.strip()],
            )
        st.success("✅ Successfully loaded embeddings")

        # Train sparse autoencoder
        with st.spinner("Training autoencoder... This may take a few minutes."):
            autoencoder, encoded_embeddings = train_autoencoder(
                embeddings, params["hidden_size"], params["num_epochs"]
            )
        st.success("✅ Successfully trained autoencoder")

        # Analyze monosemantic neurons
        with st.spinner("Analyzing neurons..."):
            concept_groups = parse_concept_groups(params["concept_groups"])
            monosemantic_neurons = prepare_data_for_analysis(
                encoded_embeddings,
                labels,
                concept_groups,
                params["monosemantic_threshold"],
            )

        if monosemantic_neurons:
            st.write("### Monosemantic Neurons Found:", len(monosemantic_neurons))
            for neuron, score in monosemantic_neurons:
                st.write(f"- Neuron {neuron}: {score:.3f}")
        else:
            st.warning(
                "No monosemantic neurons found with current threshold. Try lowering the threshold."
            )

        # Store results in session state
        results = (encoded_embeddings, labels, concept_groups, monosemantic_neurons)
        st.session_state.analysis_results = results
        return results

    except NotImplementedError as e:
        st.error(str(e))
        logger.exception("Analysis failed - NotImplementedError")
        st.session_state.analysis_results = None
        return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.exception("Analysis pipeline failed")
        st.session_state.analysis_results = None
        return None


def display_neuron_analysis(
    encoded_embeddings: np.ndarray, labels: List[str], neuron_index: int
):
    """Display detailed analysis of a selected neuron."""
    st.markdown("### Neuron Analysis")

    # Get activation values for all words
    word_activations = get_neuron_activations(encoded_embeddings, neuron_index, labels)

    # Create a DataFrame for better display
    df = pd.DataFrame(word_activations, columns=["Word", "Activation"])

    # Display statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Activation", f"{df['Activation'].mean():.3f}")
    with col2:
        st.metric("Max Activation", f"{df['Activation'].max():.3f}")
    with col3:
        st.metric("Active Words", f"{(df['Activation'] > 0).sum()}")

    # Display activation distribution
    st.markdown("#### Word Activations")
    st.markdown("Words are colored by their activation strength:")

    # Create three columns for different activation levels
    high_act = df[df["Activation"] > df["Activation"].quantile(0.8)]
    med_act = df[
        (df["Activation"] <= df["Activation"].quantile(0.8))
        & (df["Activation"] > df["Activation"].quantile(0.2))
    ]
    low_act = df[df["Activation"] <= df["Activation"].quantile(0.2)]

    cols = st.columns(3)
    with cols[0]:
        st.markdown("🟢 **Strong Activation**")
        for _, row in high_act.iterrows():
            st.markdown(f"- {row['Word']}: {row['Activation']:.3f}")

    with cols[1]:
        st.markdown("🟡 **Medium Activation**")
        for _, row in med_act.iterrows():
            st.markdown(f"- {row['Word']}: {row['Activation']:.3f}")

    with cols[2]:
        st.markdown("🔴 **Weak Activation**")
        for _, row in low_act.iterrows():
            st.markdown(f"- {row['Word']}: {row['Activation']:.3f}")


def display_visualization_section(params: Dict):
    """Handle visualization controls and display."""
    st.markdown("---")
    st.subheader("Visualization")

    if st.session_state.analysis_results is None:
        st.warning("Please run the analysis first to visualize results.")
        return

    encoded_embeddings, labels, concept_groups, monosemantic_neurons = (
        st.session_state.analysis_results
    )

    # Create two columns for visualization and analysis
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
        ### 3D Visualization Explanation
        - The 3D space shows how words are related to each other
        - Points that are closer together have more similar meanings
        - Colors indicate concept groups or neuron activation strength
        - Hover over points to see details
        """
        )

        show_plot = st.button("Generate 3D Visualization", key="viz_button")
        if show_plot:
            with st.spinner("Generating 3D visualization..."):
                try:
                    fig = plot_3d_proximity(
                        encoded_embeddings,
                        labels,
                        concept_groups,
                        method=params["dim_reduction"].lower(),
                        metric=params["distance_metric"],
                        config=st.session_state.visualization_config,
                        selected_neuron=st.session_state.selected_neuron,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Visualization failed: {str(e)}")
                    logger.exception("Visualization failed")

    with col2:
        if monosemantic_neurons:
            st.markdown("### Neuron Selection")
            st.markdown(
                """
            Select a neuron to:
            1. View its activation pattern
            2. Color the 3D plot by activation strength
            3. Analyze word relationships
            """
            )

            selected_neuron = st.selectbox(
                "Select Neuron",
                [f"Neuron {n}" for n, s in monosemantic_neurons],
                format_func=lambda x: f"{x} (Score: {dict(monosemantic_neurons)[int(x.split()[1])]:.3f})",
            )

            if selected_neuron:
                neuron_index = int(selected_neuron.split()[1])
                st.session_state.selected_neuron = neuron_index
                display_neuron_analysis(encoded_embeddings, labels, neuron_index)


def main():
    """Main application entry point."""
    st.title("Interactive Analysis of Word Embeddings")
    st.write(
        "Analyze monosemanticity and polysemanticity in text embeddings "
        "using sparse autoencoders."
    )

    # Setup configuration
    params = setup_sidebar()

    # Run analysis when requested
    if st.button("Run Analysis", key="run_analysis"):
        with st.spinner("Running analysis..."):
            run_analysis(params)

    # Display visualization section
    display_visualization_section(params)


if __name__ == "__main__":
    main()
