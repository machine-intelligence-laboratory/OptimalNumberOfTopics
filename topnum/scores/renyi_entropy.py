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


# TODO: ???
# Removing zeros from matrix so as not to get infinity
# Убираем нули в массиве, что бы не получили бесконечность
TINY = 1e-9


logger = logging.getLogger('main')


class RenyiEntropyScore(BaseCustomScore):
    def __init__(self, name, method='entropy', threshold_factor=1, verbose=False):
        # methods: entropy, random, kl

        super().__init__(name)

        if method not in ['entropy', 'random', 'kl']:
            raise ValueError(method)

        self._method = method
        self._threshold_factor = threshold_factor
        self._verbose = verbose

        self._score = self._initialize()

    def _initialize(self) -> BaseTopicNetScore:
        return _RenyiEntropyScore(self._method, self._threshold_factor, self._verbose)


class _RenyiEntropyScore(BaseTopicNetScore):
    # TODO: rename threshold_factor?
    def __init__(self, method='entropy', threshold_factor=1, verbose=False):
        # methods: entropy, random, kl

        super().__init__()

        self._method = method
        self._threshold_factor = threshold_factor
        self._verbose = verbose

    def call(self, model: TopicModel):
        return 0

    def _get_matrices(self, model: TopicModel) -> Tuple[np.array, np.array]:
        phi = model.get_phi().values
        nwt = model._model.get_phi(model_name=model._model.model_nwt).values

        # TODO: is this needed?
        phi[phi < TINY] = TINY
        phi = phi / phi.sum(axis=0)

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

            if self._verbose is True:
                print('Current number of topics: ', num_topics)

            current_entropies = list()
            current_topics = list()
            probability_sum = 0.0
            word_ratio = 0

            for j in range(num_topics):
                current_probability_sum = 0
                current_word_ratio = 0

                # TODO: optimize
                for k in range(num_words):
                    if p_wt[k][j] > threshold:
                        current_probability_sum = current_probability_sum + p_wt[k][j]
                        current_word_ratio = current_word_ratio + 1

                probability_sum = probability_sum + current_probability_sum
                word_ratio = word_ratio + current_word_ratio

                # Normalizing
                current_probability_sum = current_probability_sum / num_topics
                current_word_ratio = current_word_ratio / (num_topics * num_words)

                if current_probability_sum < TINY:
                    current_probability_sum = TINY

                if current_word_ratio < TINY:
                    current_word_ratio = TINY

                current_energy = -1 * np.log(current_probability_sum)
                current_shannon_entropy = np.log(current_word_ratio)
                current_free_energy = (
                    current_energy - num_topics * current_shannon_entropy
                )
                current_renyi_entropy = -1 * current_free_energy / max(TINY, num_topics - 1)

                current_entropies.append(current_renyi_entropy)
                current_topics.append(j)

            probability_sum = probability_sum / num_topics
            word_ratio = word_ratio / (num_topics * num_words)

            energy = -1 * np.log(probability_sum)
            shannon_entropy = np.log(word_ratio)
            free_energy = energy - num_topics * shannon_entropy
            renyi_entropy = free_energy / max(TINY, num_topics - 1)

            entropies.append(renyi_entropy)
            densities.append(shannon_entropy)
            energies.append(energy)
            nums_topics.append(num_topics)

            if minimum_entropy is None or minimum_entropy > renyi_entropy:
                minimum_entropy = renyi_entropy
                optimum_num_topics = num_topics

            message = f'Renyi min: {minimum_entropy}. Num clusters: {optimum_num_topics}'
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

        stop = datetime.now()

        if self._verbose:
            print('The calculation is over')

            print('Renyi min: ', minimum_entropy)
            print('Num clusters: ', optimum_num_topics)

            print(f'Time of execution: {(stop - start_time).total_seconds()} (sec)')

        return nums_topics, entropies, densities, energies

    @staticmethod
    def _select_topics_to_merge_by_entropy(
            topics: List[int], entropies: List[float]) -> Tuple[int, int]:

        topic_entropy_pairs = zip(topics, entropies)
        topic_entropy_pairs = sorted(topic_entropy_pairs, key=lambda pair: pair[1])

        topic_a = topic_entropy_pairs[0][0]
        topic_b = topic_entropy_pairs[0][0]

        return topic_a, topic_b

    @staticmethod
    def _select_topics_to_merge_by_random(topics: List[int]) -> Tuple[int, int]:
        return tuple(np.random.choice(topics, 2, replace=False))

    # TODO: remove topics_num

    # TODO: matrix: dimensions
    @staticmethod
    def _select_topics_to_merge_by_kl_divergence(topics: List[int], topic_matrix):
        kl_values = [
            (
                scipy.stats.entropy(topic_matrix[:, p], topic_matrix[:, q])
                + scipy.stats.entropy(topic_matrix[:, q], topic_matrix[:, p])
            ) / 2
            for p, q in itertools.combinations(topics, 2)
        ]

        kl_argmin = np.argmin(kl_values)

        return list(itertools.combinations(topics, 2))[kl_argmin]
