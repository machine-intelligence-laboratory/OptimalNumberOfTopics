import itertools
import logging
import numpy as np
import scipy
from datetime import datetime
from topicnet.cooking_machine.models import (
    BaseScore as BaseTopicNetScore,
    TopicModel
)
from typing import (
    List,
    Tuple
)

from .base_custom_score import BaseCustomScore


TINY = 1e-9


logger = logging.getLogger('main')


class RenyiEntropyScore(BaseCustomScore):
    def __init__(self, name, merge_method='entropy', threshold_factor=1, verbose=False):
        # methods: entropy, random, kl

        super().__init__(name)

        merge_method = merge_method.lower()

        if merge_method not in ['entropy', 'random', 'kl']:
            raise ValueError(f'merge_method: {merge_method}')

        if threshold_factor <= 0:
            raise ValueError(f'threshold_factor: {threshold_factor}')

        self._method = merge_method
        self._threshold_factor = threshold_factor
        self._verbose = verbose

        self._score = self._initialize()

    def _initialize(self) -> BaseTopicNetScore:
        return _RenyiEntropyScore(self._method, self._threshold_factor, self._verbose)


class _RenyiEntropyScore(BaseTopicNetScore):
    def __init__(self, merge_method='entropy', threshold_factor=1, verbose=False):
        super().__init__()

        self._method = merge_method
        self._threshold_factor = threshold_factor
        self._verbose = verbose

    def call(self, model: TopicModel):
        return 0

    def _get_matrices(self, model: TopicModel) -> Tuple[np.array, np.array]:
        phi = model.get_phi().values
        nwt = model._model.get_phi(model_name=model._model.model_nwt).values

        return phi, nwt

    # TODO: typing
    def _renormalize_using_phi(self, p_wt, n_wt):
        start_time = datetime.now()

        nums_topics = list()
        entropies = list()
        energies = list()
        densities = list()

        minimum_entropy = None
        optimum_num_topics = None

        num_words, original_num_topics = p_wt.shape

        message = f'Original number of topics: {original_num_topics}'
        logger.info(message)

        if self._verbose:
            print(message)

        threshold = self._threshold_factor * 1.0 / num_words

        for renormalization_iteration in range(original_num_topics - 1):
            num_words, num_topics = p_wt.shape
            nums_topics.append(num_topics)

            if self._verbose is True:
                print('Current number of topics: ', num_topics)

            current_entropies = list()
            current_topics = list()
            probability_sum = 0.0
            word_ratio = 0

            for topic_index in range(num_topics):
                current_probability_sum = 0
                current_word_ratio = 0

                # TODO: optimize
                for word_index in range(num_words):
                    if p_wt[word_index][topic_index] > threshold:
                        current_probability_sum = (
                            current_probability_sum + p_wt[word_index][topic_index]
                        )
                        current_word_ratio = current_word_ratio + 1

                probability_sum = probability_sum + current_probability_sum
                word_ratio = word_ratio + current_word_ratio

                # Normalizing
                current_probability_sum = current_probability_sum / num_topics
                current_word_ratio = current_word_ratio / (num_topics * num_words)

                current_probability_sum = max(TINY, current_probability_sum)
                current_word_ratio = max(TINY, current_word_ratio)

                current_energy = -1 * np.log(current_probability_sum)
                current_shannon_entropy = np.log(current_word_ratio)
                current_free_energy = (
                    current_energy - num_topics * current_shannon_entropy
                )
                current_renyi_entropy = -1 * current_free_energy / max(TINY, num_topics - 1)

                current_entropies.append(current_renyi_entropy)
                current_topics.append(topic_index)

            probability_sum = probability_sum / num_topics
            word_ratio = word_ratio / (num_topics * num_words)

            probability_sum = max(TINY, probability_sum)
            word_ratio = max(TINY, word_ratio)

            # TODO: DRY
            energy = -1 * np.log(probability_sum)
            shannon_entropy = np.log(word_ratio)
            free_energy = energy - num_topics * shannon_entropy
            renyi_entropy = free_energy / max(TINY, num_topics - 1)

            entropies.append(renyi_entropy)
            densities.append(shannon_entropy)
            energies.append(energy)

            if minimum_entropy is None or minimum_entropy > renyi_entropy:
                minimum_entropy = renyi_entropy
                optimum_num_topics = num_topics

            message = (f'Minimum Renyi entropy: {minimum_entropy}.' +
                       f' Number of clusters: {optimum_num_topics}')
            logger.info(message)

            if self._verbose is True:
                print(message)

            if self._method == 'entropy':
                topic_a, topic_b = self._select_topics_to_merge_by_entropy(
                    current_topics, current_entropies
                )
            elif self._method == 'random':
                topic_a, topic_b = self._select_topics_to_merge_by_random(
                    current_topics
                )
            elif self._method == 'kl':
                topic_a, topic_b = self._select_topics_to_merge_by_kl_divergence(
                    current_topics, p_wt
                )
            else:
                raise ValueError(self._method)

            n_wt[:, topic_a] = n_wt[:, topic_a] + n_wt[:, topic_b]
            merged_topic = n_wt[:, topic_a]
            p_wt[:, topic_a] = merged_topic / sum(merged_topic)

            p_wt = np.delete(p_wt, topic_b, 1)
            n_wt = np.delete(n_wt, topic_b, 1)

        finish_time = datetime.now()

        if self._verbose:
            print('The calculation is over!')

            print('Minimum Renyi entropy: ', minimum_entropy)
            print('Number of clusters: ', optimum_num_topics)

            print(f'Time of execution: {(finish_time - start_time).total_seconds() / 60} (min)')

        return nums_topics, entropies, densities, energies

    @staticmethod
    def _select_topics_to_merge_by_entropy(
            topics: List[int], entropies: List[float]) -> Tuple[int, int]:

        topic_entropy_pairs = zip(topics, entropies)
        topic_entropy_pairs = sorted(topic_entropy_pairs, key=lambda pair: pair[1])

        topic_a = topic_entropy_pairs[0][0]
        topic_b = topic_entropy_pairs[1][0]

        assert topic_a != topic_b

        return topic_a, topic_b

    @staticmethod
    def _select_topics_to_merge_by_random(topics: List[int]) -> Tuple[int, int]:
        return tuple(np.random.choice(topics, 2, replace=False))

    # TODO: matrix: dimensions
    @staticmethod
    def _select_topics_to_merge_by_kl_divergence(topics: List[int], p_wt: np.array):
        kl_values = [
            (
                scipy.stats.entropy(p_wt[:, p], p_wt[:, q]) +
                scipy.stats.entropy(p_wt[:, q], p_wt[:, p])
            ) / 2
            for p, q in itertools.combinations(topics, 2)
        ]

        kl_argmin = np.argmin(kl_values)

        return list(itertools.combinations(topics, 2))[kl_argmin]
