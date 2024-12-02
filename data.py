from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import os
import urllib.request
import zipfile
import gzip
import shutil

import numpy as np
import torch
from gensim.models import KeyedVectors
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_ZIP = "glove.6B.zip"
GLOVE_FILE = "glove.6B.100d.txt"


class EmbeddingLoader:
    """Handles loading and processing of different word embedding models."""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the embedding loader."""
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._models = {}

    def download_glove(self):
        """Download and extract GloVe embeddings."""
        glove_path = self.cache_dir / GLOVE_FILE
        if not glove_path.exists():
            zip_path = self.cache_dir / GLOVE_ZIP

            # Download
            logger.info("Downloading GloVe embeddings...")
            urllib.request.urlretrieve(GLOVE_URL, zip_path)

            # Extract
            logger.info("Extracting GloVe embeddings...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extract(GLOVE_FILE, self.cache_dir)

            # Cleanup
            zip_path.unlink()

        return glove_path

    def ensure_model_available(self, model_name: str):
        """Ensure the requested model is available, downloading if necessary."""
        if model_name.startswith("GloVe"):
            return self.download_glove()
        elif model_name.startswith("Word2Vec"):
            raise NotImplementedError(
                "Word2Vec model requires manual download of GoogleNews vectors. "
                "Please use GloVe or BERT instead."
            )
        elif model_name == "BERT":
            # BERT will be downloaded automatically by transformers
            return None
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def load_embeddings(
        self, model_name: str, selected_words: List[str]
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """Load word embeddings for selected words."""
        try:
            # Ensure model is available
            self.ensure_model_available(model_name)

            if model_name.startswith("GloVe"):
                embeddings_dict = self._load_glove_embeddings()
            elif model_name.startswith("Word2Vec"):
                embeddings_dict = self._load_word2vec_embeddings()
            elif model_name == "BERT":
                embeddings_dict = self._get_bert_embeddings(selected_words)
            else:
                raise ValueError(f"Invalid embedding model: {model_name}")

            # Filter available words
            available_words = [
                word.strip()
                for word in selected_words
                if word.strip() in embeddings_dict
            ]
            missing_words = set(selected_words) - set(available_words)

            if missing_words:
                logger.warning(f"Words not found in embeddings: {missing_words}")

            if not available_words:
                raise ValueError("No valid words found in embeddings")

            embeddings = np.array([embeddings_dict[word] for word in available_words])
            return embeddings, available_words, selected_words

        except Exception as e:
            logger.exception(f"Failed to load embeddings for {model_name}")
            raise

    def _load_glove_embeddings(self) -> Dict[str, np.ndarray]:
        """Load GloVe embeddings from file."""
        if "glove" not in self._models:
            try:
                embeddings = {}
                glove_path = self.cache_dir / GLOVE_FILE

                logger.info(f"Loading GloVe embeddings from {glove_path}")
                with open(glove_path, "r", encoding="utf-8") as f:
                    for line in f:
                        values = line.split()
                        word = values[0]
                        vector = np.asarray(values[1:], dtype="float32")
                        embeddings[word] = vector

                self._models["glove"] = embeddings
                logger.info("Successfully loaded GloVe embeddings")

            except Exception as e:
                logger.exception("Failed to load GloVe embeddings")
                raise

        return self._models["glove"]

    def _load_word2vec_embeddings(self) -> Dict[str, np.ndarray]:
        """Load Word2Vec embeddings from binary file."""
        raise NotImplementedError(
            "Word2Vec model requires manual download. Please use GloVe or BERT instead."
        )

    def _get_bert_embeddings(self, words: List[str]) -> Dict[str, np.ndarray]:
        """Get BERT embeddings for a list of words."""
        if "bert" not in self._models:
            try:
                logger.info("Loading BERT model...")
                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                model = AutoModel.from_pretrained("bert-base-uncased")
                self._models["bert"] = (tokenizer, model)
                logger.info("Successfully loaded BERT model")
            except Exception as e:
                logger.exception("Failed to load BERT model")
                raise

        tokenizer, model = self._models["bert"]
        embeddings = {}

        try:
            for word in words:
                inputs = tokenizer(
                    word, return_tensors="pt", padding=True, truncation=True
                )
                with torch.no_grad():
                    outputs = model(**inputs)
                embeddings[word] = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

            return embeddings

        except Exception as e:
            logger.exception("Failed to generate BERT embeddings")
            raise


def parse_concept_groups(input_text: str) -> Dict[str, List[str]]:
    """Parse concept groups from input text.

    Args:
        input_text: Text containing concept groups in format "Group: word1, word2"

    Returns:
        Dictionary mapping group names to lists of words

    Raises:
        ValueError: If input text is invalid
    """
    try:
        groups = {}
        for line in input_text.strip().split("\n"):
            if ":" not in line:
                continue

            group, words = line.split(":", 1)
            group = group.strip()
            if not group:
                continue

            words = [w.strip() for w in words.split(",") if w.strip()]
            if words:
                groups[group] = words

        if not groups:
            raise ValueError("No valid concept groups found in input")

        return groups

    except Exception as e:
        logger.exception("Failed to parse concept groups")
        raise


def prepare_data_for_analysis(
    encoded_embeddings: np.ndarray,
    labels: List[str],
    concept_groups: Dict[str, List[str]],
    threshold: float,
) -> List[Tuple[int, float]]:
    """Prepare encoded embeddings data for analysis.

    Args:
        encoded_embeddings: Matrix of encoded embeddings
        labels: List of word labels
        concept_groups: Dictionary of concept groups
        threshold: Threshold for neuron selectivity

    Returns:
        List of tuples (neuron_index, selectivity_score)
    """
    try:
        num_neurons = encoded_embeddings.shape[1]
        group_names = list(concept_groups.keys())
        num_groups = len(group_names)

        if num_groups == 0:
            raise ValueError("No concept groups provided")

        selectivity = np.zeros((num_neurons, num_groups))

        # Calculate selectivity for each group
        for i, (group, words) in enumerate(concept_groups.items()):
            group_indices = [idx for idx, word in enumerate(labels) if word in words]

            if not group_indices:
                logger.warning(f"No words found for group '{group}'")
                continue

            group_embeddings = encoded_embeddings[group_indices]
            selectivity[:, i] = np.mean(group_embeddings, axis=0)

        # Clean and normalize selectivity scores
        selectivity = np.nan_to_num(selectivity, 0)
        selectivity_normalized = normalize(selectivity, axis=1, norm="l1")

        # Find neurons above threshold
        results = []
        for neuron in range(num_neurons):
            max_selectivity = np.max(selectivity_normalized[neuron])
            if max_selectivity > threshold:
                results.append((neuron, max_selectivity))

        return sorted(results, key=lambda x: x[1], reverse=True)

    except Exception as e:
        logger.exception("Failed to prepare data for analysis")
        raise


def load_embeddings(
    model_name: str, selected_words: List[str], cache_dir: Optional[str] = None
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Convenience function to load embeddings."""
    loader = EmbeddingLoader(cache_dir)
    return loader.load_embeddings(model_name, selected_words)
