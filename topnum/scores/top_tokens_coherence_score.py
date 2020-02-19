import itertools
import numpy as np
import pandas as pd
import warnings

from collections import defaultdict
from topicnet.cooking_machine.dataset import Dataset
from typing import (
    Dict,
    List,
    Tuple
)

from .base_custom_score import BaseCustomScore
from ._base_coherence_score import (
    _BaseCoherenceScore,
    TextType,
    WordTopicRelatednessType,
    SpecificityEstimationMethod
)


class TopTokensCoherenceScore(BaseCustomScore):
    """
    Newman's PMI.
    One may see the paper Newman D. et al. "Automatic evaluation of topic coherence"
    """
    def __init__(
            self,
            name: str,
            dataset: Dataset,
            documents: List[str] = None,
            text_type: TextType = TextType.VW_TEXT,
            word_topic_relatedness: WordTopicRelatednessType = WordTopicRelatednessType.PWT,
            specificity_estimation: SpecificityEstimationMethod = SpecificityEstimationMethod.NONE,
            window=10):
        """
        Parameters
        ----------
        dataset
            Dataset with document collection
            (any model passed to `call()` is supposed to be trained on it)
        documents
            Which documents from the dataset are to be used for computing coherence
        text_type
            What text to use when computing coherence: raw text or VW text
            Preferable to use VW (as it is usually preprocessed, stop-words removed etc.),
            and with words in *natural order*.
            Score needs "real" text to compute coherence
        computation_method
            The way to compute intra-text coherence
        word_topic_relatedness
            How to estimate word relevance to topic: using p(w | t) or p(t | w)
        specificity_estimation
            How to estimate specificity of word to topic
        window
            In case computation_method = ComputationMethod.SUM_OVER_WINDOW:
            Window width. So the window will be the words with positions
            in [current position - window / 2, current position + window / 2)
        """
        super().__init__(name)

        self._dataset = dataset
        self._documents = documents
        self._text_type = text_type
        self._word_topic_relatedness = word_topic_relatedness
        self._specificity_estimation = specificity_estimation
        self._window = window

        self._score = self._initialize()

    def _initialize(self) -> _BaseCoherenceScore:
        return _TopTokensCoherenceScore(
            dataset=self._dataset,
            documents=self._documents,
            text_type=self._text_type,
            word_topic_relatedness=self._word_topic_relatedness,
            specificity_estimation=self._specificity_estimation,
            window=self._window
        )


class _TopTokensCoherenceScore(_BaseCoherenceScore):
    def __init__(
            self,
            dataset: Dataset,
            documents: List[str] = None,
            text_type: TextType = TextType.VW_TEXT,
            word_topic_relatedness: WordTopicRelatednessType = WordTopicRelatednessType.PWT,
            specificity_estimation: SpecificityEstimationMethod = SpecificityEstimationMethod.NONE,
            num_top_words=10,
            window=10):

        super().__init__(
            dataset=dataset,
            documents=documents,
            text_type=text_type,
            word_topic_relatedness=word_topic_relatedness,
            specificity_estimation=specificity_estimation
        )

        if not isinstance(num_top_words, int):
            raise TypeError(
                f'Wrong "num_top_words": \"{num_top_words}\". '
                f'Expect to be \"int\"')

        if not isinstance(window, int):
            raise TypeError(
                f'Wrong "window": \"{window}\". '
                f'Expect to be \"int\"')

        self._num_top_words = num_top_words
        self._window = window

    def _compute_coherence(self, topic, document, word_topic_relatednesses):
        document_words = self._get_words(document)
        top_words = self._get_top_words(topic, word_topic_relatednesses)
        top_words_cooccurrences = self._get_top_words_cooccurrences(top_words, document_words)

        return self._compute_newman_coherence(
            top_words, top_words_cooccurrences, len(document_words) - self._window + 1
        )

    def _compute_newman_coherence(
            self, top_words, top_words_cooccurrences: Dict[Tuple[str, str], int], num_windows):
        pair_estimates = np.array([
            np.log2(
                max(1,
                    top_words_cooccurrences[(w1, w2)] /
                        max(np.log2(max(top_words_cooccurrences[(w1, w1)], 1)), 1) /  # noqa line alignment
                        max(np.log2(max(top_words_cooccurrences[(w2, w2)], 1)), 1) *
                        num_windows
                )
            )
            for (w1, w2) in itertools.combinations(top_words, 2)
        ])

        if len(pair_estimates[pair_estimates > 0]) == 0:
            return None

        return np.sum(pair_estimates) / len(pair_estimates)

    def _get_top_words(
            self, topic, word_topic_relatednesses: pd.DataFrame) -> List[str]:

        sorted_words = list(
            word_topic_relatednesses[topic].sort_values(ascending=False).index
        )
        top_words = sorted_words[:self._num_top_words]  # top words are sorted, but it doesn't matter # noqa: long line

        if len(top_words) < self._num_top_words:
            warnings.warn(
                f'Topic "{topic}" has less words than specified num_top_words: '
                f'{len(top_words)} < {self._num_top_words}. '
                f'So only "{len(top_words)}" words will be used')

        return top_words

    def _get_top_words_cooccurrences(
            self, top_words: List[str],
            document_words: List[str]
    ) -> Dict[Tuple[str, str], int]:

        cooccurrences = defaultdict(int)
        start_window = document_words[:self._window]
        words_num_appearances_in_window = defaultdict(int)

        for w in start_window:
            words_num_appearances_in_window[w] += 1

        self._update_cooccurrences(cooccurrences, top_words, words_num_appearances_in_window)

        last_word_in_window = start_window[0]

        for w in document_words[self._window:]:
            words_num_appearances_in_window[last_word_in_window] = max(
                0, words_num_appearances_in_window[last_word_in_window] - 1
            )

            words_num_appearances_in_window[w] += 1

            self._update_cooccurrences(cooccurrences, top_words, words_num_appearances_in_window)

        self._remove_discrepancies_for_reversed_pairs(cooccurrences, top_words)

        return cooccurrences

    @staticmethod
    def _update_cooccurrences(
            cooccurrences: Dict[Tuple[str, str], int],
            top_words: List[str],
            words_num_appearances_in_window: Dict[str, int]) -> None:

        for w, u in itertools.combinations(top_words, 2):
            cooccurrences[(w, u)] += 1 * (
                words_num_appearances_in_window[w] > 0 and
                words_num_appearances_in_window[u] > 0
            )

        for w in top_words:
            cooccurrences[(w, w)] += 1 * (
                words_num_appearances_in_window[w] > 0
            )

    @staticmethod
    def _remove_discrepancies_for_reversed_pairs(
            cooccurrences: Dict[Tuple[str, str], int],
            top_words: List[str]) -> None:

        for w, u in itertools.combinations(top_words, 2):
            coocs_num_for_pair = cooccurrences[(w, u)]
            coocs_num_for_reversed_pair = cooccurrences[(u, w)]

            cooccurrences[(w, u)] = coocs_num_for_pair + coocs_num_for_reversed_pair
            cooccurrences[(u, w)] = cooccurrences[(w, u)]
