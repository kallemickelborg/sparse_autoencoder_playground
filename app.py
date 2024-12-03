import logging
from typing import Dict, List, Tuple, Optional
import streamlit as st
import torch
import numpy as np
import pandas as pd
from models import SparseAutoencoder, train_autoencoder
from visualization import plot_3d_proximity, VisualizationConfig, get_neuron_activations
from data import EmbeddingLoader, parse_concept_groups, prepare_data_for_analysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_WORDS = "doctor, nurse, engineer, teacher, lawyer, man, woman"
DEFAULT_CONCEPT_GROUPS = """Profession: doctor, nurse, engineer, teacher
Gender: man, woman"""

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "visualization_config" not in st.session_state:
    st.session_state.visualization_config = VisualizationConfig()
if "selected_neuron" not in st.session_state:
    st.session_state.selected_neuron = None


def setup_sidebar() -> Dict:
    """Configure and return sidebar parameters."""
    st.sidebar.title("Configuration")

    model_params = {
        "embedding_model": st.sidebar.selectbox(
            "Select Embedding Model",
            ["BERT", "GloVe (100d)"],
            help="BERT: 768 dimensions, better quality but slower. GloVe: 100 dimensions, faster but less accurate.",
        ),
    }

    max_features = 768 if model_params["embedding_model"] == "BERT" else 100
    default_size = min(50, max_features)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Analysis Configuration")

    st.sidebar.markdown(
        """
        #### Input Words
        These are the actual words you want to analyze. They should be relevant to your concepts of interest.
    """
    )
    model_params["custom_words"] = st.sidebar.text_area(
        "Words to Analyze (comma-separated)",
        value=DEFAULT_WORDS,
        help="Enter all words you want to analyze, separated by commas",
    )

    st.sidebar.markdown(
        """
        #### Concept Groups
        Groups help identify neurons that respond to specific semantic categories.
        Example:
        ```
        Profession: doctor, nurse
        Gender: man, woman
        ```
    """
    )
    model_params["concept_groups"] = st.sidebar.text_area(
        "Semantic Groups",
        value=DEFAULT_CONCEPT_GROUPS,
        help="Format: Group_Name: word1, word2",
    )

    st.sidebar.markdown("#### Autoencoder Settings")
    model_params.update(
        {
            "hidden_size": st.sidebar.slider(
                "Autoencoder Feature Size",
                10,
                max_features,
                default_size,
                help=f"Maximum {max_features} features based on selected model. More features can capture more patterns but may be harder to interpret.",
            ),
            "num_epochs": st.sidebar.slider(
                "Training Epochs",
                100,
                1000,
                500,
                help="Number of training iterations. More epochs may improve results but take longer.",
            ),
            "monosemantic_threshold": st.sidebar.slider(
                "Monosemanticity Threshold",
                0.0,
                1.0,
                0.6,
                help="Higher values mean more selective neurons. Adjust if you're not finding meaningful patterns.",
            ),
        }
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Visualization Settings")

    viz_config = st.session_state.visualization_config

    model_params["dim_reduction"] = "umap"

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

    model_params["distance_metric"] = st.sidebar.selectbox(
        "Distance Metric",
        ("cosine", "euclidean"),
        help="Cosine similarity is recommended for word embeddings",
    )

    return model_params


def run_analysis(
    params: Dict,
) -> Optional[Tuple[np.ndarray, List[str], Dict, List[Tuple[int, float]]]]:
    """Run the main analysis pipeline."""
    try:
        st.session_state.selected_neuron = None

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


def display_visualization_section(params: Dict):
    """Handle visualization controls and display."""
    st.markdown("---")
    st.markdown("### Step 2: Select Feature")

    if st.session_state.analysis_results is None:
        st.info(
            "👆 Start by clicking 'Run Analysis' to train the autoencoder and analyze the words."
        )
        return

    encoded_embeddings, labels, concept_groups, monosemantic_neurons = (
        st.session_state.analysis_results
    )

    with st.container():

        if monosemantic_neurons:
            selected_neuron = st.selectbox(
                "Analyze Feature",
                [f"Feature {n}" for n, s in monosemantic_neurons],
                format_func=lambda x: f"{x} (Score: {dict(monosemantic_neurons)[int(x.split()[1])]:.3f})",
                help="Select a feature to see which words activate it most strongly",
            )
            if selected_neuron:
                neuron_index = int(selected_neuron.split()[1])
                st.session_state.selected_neuron = neuron_index

        # Feature Analysis
        if monosemantic_neurons and st.session_state.selected_neuron is not None:
            st.markdown("---")
            st.markdown(
                f"""
                <h3 style='text-align: center;'>Feature {st.session_state.selected_neuron} Analysis</h3>
                """,
                unsafe_allow_html=True,
            )

            stats_cols = st.columns(3)
            word_activations = get_neuron_activations(
                encoded_embeddings, st.session_state.selected_neuron, labels
            )
            df = pd.DataFrame(word_activations, columns=["Word", "Activation"])

            with stats_cols[0]:
                st.metric("Mean Activation", f"{df['Activation'].mean():.3f}")
            with stats_cols[1]:
                st.metric("Max Activation", f"{df['Activation'].max():.3f}")
            with stats_cols[2]:
                st.metric("Active Words", f"{(df['Activation'] > 0).sum()}")

            st.markdown("#### Word Activation Distribution")

            high_act = df[df["Activation"] > df["Activation"].quantile(0.8)]
            med_act = df[
                (df["Activation"] <= df["Activation"].quantile(0.8))
                & (df["Activation"] > df["Activation"].quantile(0.2))
            ]
            low_act = df[df["Activation"] <= df["Activation"].quantile(0.2)]

            cols = st.columns(3)
            with cols[0]:
                st.markdown(
                    """
                    <div style='background-color: #e6ffe6; padding: 0.5rem; border-radius: 0.5rem;'>
                        <h5>🟢 Strong Activation</h5>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                for _, row in high_act.iterrows():
                    st.markdown(f"**{row['Word']}**: {row['Activation']:.3f}")

            with cols[1]:
                st.markdown(
                    """
                    <div style='background-color: #fffff0; padding: 0.5rem; border-radius: 0.5rem;'>
                        <h5>🟡 Medium Activation</h5>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                for _, row in med_act.iterrows():
                    st.markdown(f"**{row['Word']}**: {row['Activation']:.3f}")

            with cols[2]:
                st.markdown(
                    """
                    <div style='background-color: #ffe6e6; padding: 0.5rem; border-radius: 0.5rem;'>
                        <h5>🔴 Weak Activation</h5>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                for _, row in low_act.iterrows():
                    st.markdown(f"**{row['Word']}**: {row['Activation']:.3f}")

        # Visualization
        st.markdown("### Step 3: Visualize Results")
        st.markdown(
            """
            <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
                <h4>📊 Visualization Guide</h4>
                • Words are mapped to 3D space while preserving their semantic relationships<br>
                • Words with similar meanings appear closer together<br>
                • Colors indicate either concept groups or activation strength of the selected feature<br>
                • Hover over points to see word details<br>
                • Use mouse to rotate and zoom the visualization
            </div>
            """,
            unsafe_allow_html=True,
        )

        show_plot = st.button(
            "Generate 3D Visualization",
            key="viz_button",
            use_container_width=True,
            help="Click to create an interactive 3D visualization of the word relationships",
        )

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
                    fig.update_layout(
                        height=600,
                        width=None,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Visualization failed: {str(e)}")
                    logger.exception("Visualization failed")


def main():
    """Main application entry point."""
    st.title("Interactive Analysis of Word Embeddings")

    st.markdown(
        """
    This tool helps you discover interpretable features in word embeddings using sparse autoencoders.
    
    1. **Configure**: Set up your analysis in the sidebar
    2. **Run Analysis**: Train the autoencoder to find interpretable features
    3. **Visualize**: Explore the relationships between words and features
    """
    )

    params = setup_sidebar()

    st.markdown("### Step 1: Run Analysis")
    st.markdown(
        """
    Clicking 'Run Analysis' will:
    1. Load the selected embedding model
    2. Train a sparse autoencoder to find interpretable features
    3. Analyze which features respond to specific semantic concepts
    """
    )

    if st.button("Run Analysis", key="run_analysis"):
        with st.spinner("Running analysis..."):
            run_analysis(params)

    display_visualization_section(params)


if __name__ == "__main__":
    main()
