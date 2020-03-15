import logging
import numpy as np
import pandas as pd
import traceback

from itertools import combinations
from numbers import Number
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import DBSCAN
from topicnet.cooking_machine import Dataset
from typing import (
    Callable,
    Dict,
    List,
    Tuple,
    Union
)

import topnum.search_methods.topic_bank.phi_initialization.utils as utils


_COL_DOCUMENT_TEXT = 'raw_text'

_Logger = logging.getLogger()


def compute_phi(
        dataset: Dataset,
        main_modality: str,
        local_context_words_percentile: int = 20,  # TODO: why 20?
        clusterization_distance: Callable[[np.ndarray, np.ndarray], float] = None,
        eps: float = 0.6,
        min_samples: int = 20) -> pd.DataFrame:
    """
    Vladimir Dobrynin, David Patterson, and Niall Rooney.
    "Contextual document clustering."
    European Conference on Information Retrieval.
    Springer, Berlin, Heidelberg, 2004.
    """
    if clusterization_distance is None:
        clusterization_distance = jensenshannon

    _check_clusterization_distance_func(clusterization_distance)

    phi_index = utils.get_phi_index(dataset)
    # TODO: optimize?
    word2index = {
        modality_word_pair[1]: index
        for index, modality_word_pair in enumerate(phi_index)
        if modality_word_pair[0] == main_modality
    }

    assert len(word2index) > 0

    word_in_word_frequencies, document_frequencies = _count_word_in_word_frequencies(
        dataset=dataset,
        word2index=word2index
    )
    word_in_word_probabilities = _count_word_in_word_probabilities(
        word_in_word_frequencies
    )
    word_entropies = _count_word_entropies(
        word_in_word_probabilities
    )
    local_context_words = _find_local_context_words(
        word_entropies,
        document_frequencies,
        local_context_words_percentile
    )

    centers = _clusterize_local_contexts(
        local_context_words,
        word_in_word_probabilities,
        metric=clusterization_distance,
        eps=eps,
        min_samples=min_samples
    )

    topic_names = [f'cdc_topic_{i}' for i in range(len(centers))]
    phi_values = np.zeros(shape=(len(phi_index), len(centers)))
    phi_values[list(word2index.values())] = word_in_word_probabilities[:, centers]

    return pd.DataFrame(
        index=phi_index,
        columns=topic_names,
        data=phi_values
    )


def _check_clusterization_distance_func(
        clusterization_distance: Callable[[np.ndarray, np.ndarray], float] = None
) -> None:

    error_message = 'Clusterization function seems inappropriate!'

    try:
        test_value = clusterization_distance([0.1, 0.9], [0.5, 0.5])
    except TypeError:
        raise TypeError(
            f'{error_message} {traceback.format_exc()}'
        )

    if not isinstance(test_value, Number):
        raise TypeError(error_message)


def _count_word_in_word_frequencies(
        dataset: Dataset,
        word2index: Dict[str, int],
        split_on_paragraphs: bool = True,
        max_document_size: int = None,
        window: int = 10,
        smoothing_value: float = 0.01,
        num_docs_to_log: int = 500) -> Tuple[np.ndarray, np.ndarray]:  # 2D, 1D

    frequencies = np.zeros((len(word2index), len(word2index)))
    document_frequencies = np.zeros((len(word2index),))

    def process_words(words: List[str]) -> None:
        for w in set(words):
            if w not in word2index:
                continue

            document_frequencies[word2index[w]] += 1

        for i in range(0, len(words) - window):
            words_in_window = words[i:i + window]
            words_in_window = set(words_in_window)

            for word in set(words_in_window).intersection(set(word2index)):
                frequencies[word2index[word]][word2index[word]] += 1

            for word_pair in combinations(set(words_in_window).intersection(set(word2index)), 2):
                frequencies[word2index[word_pair[0]]][word2index[word_pair[1]]] += 1
                frequencies[word2index[word_pair[1]]][word2index[word_pair[0]]] += 1

    for doc_index, doc_text in enumerate(dataset._data[_COL_DOCUMENT_TEXT].values):
        if doc_index % num_docs_to_log == 0:
            _Logger.info(f'Counting word frequencies in the document number {doc_index}')

        if split_on_paragraphs:
            text_pieces = doc_text.split('\n')
        # TODO: maybe better if instead of elif
        elif max_document_size is not None:
            text_pieces = [doc_text[i:i + max_document_size] for i in
                    range(0, max(len(doc_text) - max_document_size, 1))]
        else:
            text_pieces = [doc_text]

        for text_piece in text_pieces:
            process_words(text_piece.split())

    if smoothing_value > 0:
        frequencies += smoothing_value
        # frequencies /= np.sum(frequencies, axis=0, keepdims=True)

    return frequencies, document_frequencies


def _count_word_in_word_probabilities(frequencies: np.ndarray) -> np.ndarray:  # 2D
    probs = np.array(frequencies)

    for i in range(probs.shape[0]):
        probs[i] /= max(1, probs[i, i])

    return probs


def _count_word_entropies(word_in_word_probabilities: np.ndarray) -> np.ndarray:
    return np.sum(
        -word_in_word_probabilities * np.log2(word_in_word_probabilities),  # non-zero
        axis=1
    )


def _find_local_context_words(
        entropies: np.ndarray,  # 1D
        document_frequencies: np.ndarray,  # 1D
        percentile) -> List[int]:
    # TODO: may be improved? (quantile regression, percentile param)
    result = []

    for document_frequency in set(document_frequencies):
        candidate_indices = np.where(document_frequencies == document_frequency)[0]
        candidate_indices = sorted(candidate_indices, key=lambda i: entropies[i])
        current_indices = candidate_indices[:len(candidate_indices) * percentile // 100]

        result += current_indices

    return result


def _clusterize_local_contexts(
        local_context_word_indices: List[int],
        word_in_word_probabilities: np.ndarray,
        metric: Union[str, Callable],
        eps: float,
        min_samples: int) -> List[int]:

    # TODO: may be improved? (vary DBSCAN params)
    X = word_in_word_probabilities[local_context_word_indices]

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    dbscan.fit(X)

    return [
        local_context_word_indices[i]
        for i in dbscan.core_sample_indices_
    ]
