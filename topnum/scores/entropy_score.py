import logging
import numpy as np
from topicnet.cooking_machine.models import (
    BaseScore as BaseTopicNetScore,
    TopicModel
)
from typing import (
    List,
    Tuple
)

from .base_custom_score import BaseCustomScore


RENYI = 'renyi'
SHANNON = 'shannon'


_TINY = 1e-9

_logger = logging.getLogger()


class EntropyScore(BaseCustomScore):
    def __init__(
            self,
            name: str,
            entropy: str = RENYI,
            threshold_factor: float = 1.0,
            class_ids: List[str] = None):

        super().__init__(name)

        entropy = entropy.lower()

        if entropy not in [RENYI, SHANNON]:
            raise ValueError(f'merge_method: {entropy}')

        if threshold_factor <= 0:
            raise ValueError(f'threshold_factor: {threshold_factor}')

        self._entropy = entropy
        self._threshold_factor = threshold_factor
        self._class_ids = class_ids

        self._score = self._initialize()

    def _initialize(self) -> BaseTopicNetScore:
        return _RenyiShannonEntropyScore(self._entropy, self._threshold_factor, self._class_ids)


class _RenyiShannonEntropyScore(BaseTopicNetScore):
    def __init__(self, entropy: str, threshold_factor: float, class_ids: List[str] = None):
        super().__init__()

        self._entropy = entropy
        self._threshold_factor = threshold_factor
        self._class_ids = class_ids

    def call(self, model: TopicModel):
        phi, _ = self._get_matrices(model)

        return self._calculate_entropy(phi)

    def _get_matrices(self, model: TopicModel) -> Tuple[np.array, np.array]:
        pwt = model.get_phi(class_ids=self._class_ids).values
        nwt = model._model.get_phi(model_name=model._model.model_nwt).values

        return pwt, nwt

    def _calculate_entropy(self, pwt: np.array) -> float:
        num_words, num_topics = pwt.shape
        threshold = self._threshold_factor * 1.0 / num_words

        current_entropies = list()
        current_topics = list()
        probability_sum = 0.0
        word_ratio = 0

        for topic_index in range(num_topics):
            current_probability_sum = 0
            current_word_ratio = 0

            # TODO: optimize, use numpy
            for word_index in range(num_words):
                if pwt[word_index][topic_index] > threshold:
                    current_probability_sum = (
                        current_probability_sum + pwt[word_index][topic_index]
                    )
                    current_word_ratio = current_word_ratio + 1

            probability_sum = probability_sum + current_probability_sum
            word_ratio = word_ratio + current_word_ratio

            current_probability_sum = current_probability_sum / num_topics
            current_word_ratio = current_word_ratio / (num_topics * num_words)

            current_probability_sum = max(_TINY, current_probability_sum)
            current_word_ratio = max(_TINY, current_word_ratio)

            current_energy = -1 * np.log(current_probability_sum)
            current_shannon_entropy = np.log(current_word_ratio)
            current_free_energy = current_energy - num_topics * current_shannon_entropy
            current_renyi_entropy = -1 * current_free_energy / max(_TINY, num_topics - 1)

            current_entropies.append(current_renyi_entropy)
            current_topics.append(topic_index)

        probability_sum = probability_sum / num_topics
        word_ratio = word_ratio / (num_topics * num_words)

        probability_sum = max(_TINY, probability_sum)
        word_ratio = max(_TINY, word_ratio)

        # TODO: DRY
        energy = -1 * np.log(probability_sum)
        shannon_entropy = np.log(word_ratio)
        free_energy = energy - num_topics * shannon_entropy
        renyi_entropy = free_energy / max(_TINY, num_topics - 1)

        if self._entropy == RENYI:
            return renyi_entropy
        elif self._entropy == SHANNON:
            return shannon_entropy
        else:
            raise ValueError(self._entropy)  # this is not going to happen
