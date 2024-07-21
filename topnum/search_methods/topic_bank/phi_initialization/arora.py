import anchor_topic.topics
import numpy as np
import pandas as pd
import scipy
import scipy.sparse
import warnings

from collections import Counter
from typing import Dict

from topicnet.cooking_machine import Dataset
from topicnet.cooking_machine.dataset import VW_TEXT_COL

from . import (
    utils,
    WARNING_VW_TEXT_WRONG_FORMAT,
)


np.int = np.int32  # Arora uses old NumPy (current version has not "int" attribute)
                   # https://stackoverflow.com/q/74946845/8094251
                   # https://github.com/scikit-learn-contrib/boruta_py/issues/122#issuecomment-1914122968


def compute_phi(
        dataset: Dataset,
        main_modality: str,
        text_column: str = VW_TEXT_COL,
        num_topics: int = 100,
        document_occurrences_threshold_percentage: float = 0.05) -> pd.DataFrame:
    """
    Parameters
    ----------
    text_column
        Should be a name of `dataset` column.
        Text in this column is going to be used by the method.
        Is ts recommended that `text_column` is 'vw_text',
        and Vowpal Wabbit text is in natural (not bag of words) order

    References
    ----------
    Sanjeev Arora, Rong Ge, Ravindran Kannan, and Ankur Moitra
    "Computing a nonnegative matrix factorization---Provably."
    SIAM Journal on Computing 45.4 (2016): 1582-1611.
    """
    phi_index = utils.get_phi_index(dataset)
    word2index = {
        modality_word_pair[1]: index
        for index, modality_word_pair in enumerate(phi_index)
        if modality_word_pair[0] == main_modality
    }

    word_document_frequencies = _count_word_document_frequencies(
        dataset=dataset,
        vocabulary_size=len(phi_index),
        text_column=text_column,
        word2index=word2index,
    )
    word_document_frequencies = scipy.sparse.csc_matrix(word_document_frequencies)

    word_topic_matrix, word_cooccurrence_matrix, anchors = anchor_topic.topics.model_topics(
        word_document_frequencies,
        num_topics,
        document_occurrences_threshold_percentage
    )

    topic_names = [f'arora_topic_{i}' for i in range(len(anchors))]
    phi_values = np.zeros(shape=(len(phi_index), len(topic_names)))
    phi_values[:word_topic_matrix.shape[0], :] = word_topic_matrix

    return pd.DataFrame(
        index=phi_index,
        columns=topic_names,
        data=phi_values
    )


def _count_word_document_frequencies(
        dataset: Dataset,
        vocabulary_size: int,
        text_column: str,
        word2index: Dict[str, int],
        ) -> np.ndarray:

    num_documents = len(dataset._data)  # TODO: for big data may be slow here
    frequencies = np.zeros(
        shape=(vocabulary_size, num_documents)
    )

    for doc_index, doc_text in enumerate(dataset._data[text_column]):
        words = doc_text.split()
        preprocessed_words = list(utils._trim_vw(words))  # TODO: maybe require much memory
        if preprocessed_words[:100] != words[:100]:
            warnings.warn(WARNING_VW_TEXT_WRONG_FORMAT)

        words_counter = Counter(preprocessed_words)

        for w, c in words_counter.items():
            if w not in word2index:
                continue

            frequencies[word2index[w], doc_index] += c

    return frequencies
