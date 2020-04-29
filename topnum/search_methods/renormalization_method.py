import artm
import itertools
import logging
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import sys

from datetime import datetime
from tqdm import tqdm
from typing import (
    Callable,
    List,
    Tuple
)

from topicnet.cooking_machine.dataset import Dataset
from topicnet.cooking_machine.models import TopicModel
from topicnet.cooking_machine.model_constructor import init_simple_default_model

from .base_search_method import (
    BaseSearchMethod,
    _KEY_VALUES
)
from .constants import (
    DEFAULT_MAX_NUM_TOPICS,
    DEFAULT_MIN_NUM_TOPICS,
    DEFAULT_NUM_FIT_ITERATIONS
)
from ..data.vowpal_wabbit_text_collection import VowpalWabbitTextCollection


ENTROPY_MERGE_METHOD = 'entropy'
RANDOM_MERGE_METHOD = 'random'
KL_MERGE_METHOD = 'kl'

PHI_RENORMALIZATION_MATRIX = 'phi'
THETA_RENORMALIZATION_MATRIX = 'theta'

_TINY = 1e-9

_logger = logging.getLogger()


class RenormalizationMethod(BaseSearchMethod):
    # TODO: add modalities as parameter?
    def __init__(
            self,
            merge_method: str = ENTROPY_MERGE_METHOD,
            matrix_for_renormalization=PHI_RENORMALIZATION_MATRIX,
            threshold_factor: float = 1,
            verbose: bool = False,
            num_restarts: int = 3,
            min_num_topics: int = DEFAULT_MIN_NUM_TOPICS,
            max_num_topics: int = DEFAULT_MAX_NUM_TOPICS,
            num_fit_iterations: int = DEFAULT_NUM_FIT_ITERATIONS):

        super().__init__(min_num_topics, max_num_topics, num_fit_iterations)

        merge_method = merge_method.lower()
        matrix_for_renormalization = matrix_for_renormalization.lower()

        if merge_method not in [ENTROPY_MERGE_METHOD, RANDOM_MERGE_METHOD, KL_MERGE_METHOD]:
            raise ValueError(f'merge_method: {merge_method}')

        if matrix_for_renormalization not in [PHI_RENORMALIZATION_MATRIX, THETA_RENORMALIZATION_MATRIX]:
            raise ValueError(f'matrix_for_renormalization: {matrix_for_renormalization}')

        if threshold_factor <= 0:
            raise ValueError(f'threshold_factor: {threshold_factor}')

        self._method = merge_method
        self._matrix = matrix_for_renormalization
        self._threshold_factor = threshold_factor
        self._verbose = verbose
        self._num_restarts = num_restarts

        self._result = dict()

        self._key_num_topics_values = _KEY_VALUES.format('num_topics')
        self._key_renyi_entropy_values = _KEY_VALUES.format('renyi_entropy')
        self._key_shannon_entropy_values = _KEY_VALUES.format('snannon_entropy')
        self._key_energy_values = _KEY_VALUES.format('energy')

        for key in [
                self._key_num_topics_values,
                self._key_renyi_entropy_values,
                self._key_shannon_entropy_values,
                self._key_energy_values]:

            # np.mean is not actually needed for _key_num_topics_values:
            # all restarts must have the same number of topics
            # TODO: add assert or int()

            self._keys_mean_many.append(key)
            self._keys_std_many.append(key)

    def search_for_optimum(self, text_collection: VowpalWabbitTextCollection) -> None:
        _logger.info('Starting to search for optimum...')

        dataset = text_collection._to_dataset()
        restart_results = list()

        for seed in tqdm(range(self._num_restarts), total=self._num_restarts, file=sys.stdout):
            # seed -1 is somewhat similar to seed 0 ?
            # so skipping -1
            _logger.info(f'Seed is {seed}')

            restart_result = dict()
            restart_result[self._key_optimum] = None
            restart_result[self._key_num_topics_values] = list()
            restart_result[self._key_renyi_entropy_values] = list()
            restart_result[self._key_shannon_entropy_values] = list()
            restart_result[self._key_energy_values] = list()

            artm_model = init_simple_default_model(
                dataset,
                modalities_to_use=text_collection._modalities,
                main_modality=text_collection._main_modality,
                specific_topics=self._max_num_topics,
                background_topics=0  # TODO: or better add ability to specify?
            )

            artm_model.seed = seed  # TODO: seed -> init_simple_default_model

            model = TopicModel(artm_model)
            model._fit(
                dataset.get_batch_vectorizer(),
                num_iterations=self._num_fit_iterations
            )

            pwt, nwt = self._get_matrices(model)

            if self._matrix == PHI_RENORMALIZATION_MATRIX:
                (nums_topics, entropies, densities, energies) = (
                    self._renormalize_using_phi(pwt, nwt)
                )
            elif self._matrix == THETA_RENORMALIZATION_MATRIX:
                (nums_topics, entropies, densities, energies) = (
                    self._renormalize_using_theta(pwt, nwt, dataset)
                )
            else:
                raise ValueError(f'_matrix: {self._matrix}')

            restart_result[self._key_num_topics_values] = nums_topics
            restart_result[self._key_renyi_entropy_values] = entropies
            restart_result[self._key_shannon_entropy_values] = densities
            restart_result[self._key_energy_values] = energies

            restart_result[self._key_optimum] = nums_topics[
                np.argmin(restart_result[self._key_renyi_entropy_values])
            ]

            restart_results.append(restart_result)

        result = dict()

        self._compute_mean_one(restart_results, result)
        self._compute_std_one(restart_results, result)
        self._compute_mean_many(restart_results, result)
        self._compute_std_many(restart_results, result)

        self._result = result

        _logger.info('Finished searching!')

    @staticmethod
    def _get_matrices(model: TopicModel) -> Tuple[np.array, np.array]:
        pwt = model.get_phi().values
        nwt = model._model.get_phi(model_name=model._model.model_nwt).values

        return pwt, nwt

    @staticmethod
    def get_theta(phi: np.array, dataset: Dataset) -> pd.DataFrame:
        artm_model = artm.ARTM(num_topics=phi.shape[1])
        artm_model.initialize(dataset.get_dictionary())

        artm_model.fit_offline(dataset.get_batch_vectorizer(), 1)

        (_, phi_ref) = artm_model.master.attach_model(
            model=artm_model.model_pwt
        )

        np.copyto(
            phi_ref,
            phi
        )

        return artm_model.transform(dataset.get_batch_vectorizer())

    @staticmethod
    def _get_pdt(phi: np.array, dataset: Dataset, theta: np.array = None):
        # TODO: seems like here could have been a better name (instead of _get_anti_theta)...

        if theta is None:
            theta = RenormalizationMethod.get_theta(phi, dataset).values

        docs = dataset._data.index

        # TODO: raw_text -> const
        p_d = list()

        for i, d in enumerate(docs):
            vw_text = dataset._data.iloc[i]['vw_text']
            tokens = [w for w in vw_text.split() if not w.startswith('|')]
            frequencies = [
                float(t.split(':')[1]) if ':' in t else 1.0
                for t in tokens
            ]
            p_d.append(sum(frequencies))

        p_d = [d / sum(p_d) for d in p_d]
        p_d = np.array(p_d)

        new_theta = theta / theta.sum(axis=1).reshape(-1, 1) * p_d
        anti_theta = new_theta.T

        return anti_theta

    def _renormalize_using_phi(
            self,
            pwt: np.array,
            nwt: np.array
    ) -> Tuple[List[int], List[float], List[float], List[float]]:

        def update_callback(
                pwt_renormalization_matrix: np.array,
                topic_a: int,
                topic_b: int) -> np.array:

            nonlocal nwt

            nwt[:, topic_a] = nwt[:, topic_a] + nwt[:, topic_b]
            merged_topic = nwt[:, topic_a]
            pwt_renormalization_matrix[:, topic_a] = merged_topic / sum(merged_topic)

            pwt_renormalization_matrix = np.delete(pwt_renormalization_matrix, topic_b, 1)
            nwt = np.delete(nwt, topic_b, 1)

            return pwt_renormalization_matrix

        return self._renormalize(pwt, update_callback)

    def _renormalize_using_theta(
            self,
            pwt: np.array,
            nwt: np.array,
            dataset: Dataset
    ) -> Tuple[List[int], List[float], List[float], List[float]]:

        pdt = self._get_pdt(pwt, dataset)

        def update_callback(
                pdt_renormalization_matrix: np.array,
                topic_a: int,
                topic_b: int) -> np.array:

            nonlocal pwt
            nonlocal nwt

            nwt[:, topic_a] = nwt[:, topic_a] + nwt[:, topic_b]
            merged_topic = nwt[:, topic_a]
            pwt[:, topic_a] = merged_topic / sum(merged_topic)

            pwt = np.delete(pwt, topic_b, 1)
            nwt = np.delete(nwt, topic_b, 1)

            pdt_renormalization_matrix = RenormalizationMethod._get_pdt(pwt, dataset)

            return pdt_renormalization_matrix

        return self._renormalize(pdt, update_callback)

    def _renormalize(
            self,
            pwt: np.array,
            update_callback: Callable[[np.array, int, int], np.array]
    ) -> Tuple[List[int], List[float], List[float], List[float]]:

        start_time = datetime.now()

        nums_topics = list()
        entropies = list()
        energies = list()
        densities = list()

        minimum_entropy = None
        optimum_num_topics = None

        num_words, original_num_topics = pwt.shape

        message = f'Original number of topics: {original_num_topics}'
        _logger.info(message)

        if self._verbose:
            print(message)

        threshold = self._threshold_factor * 1.0 / num_words

        for renormalization_iteration in tqdm(
                range(original_num_topics - 1),
                total=original_num_topics - 1,
                file=sys.stdout):

            num_words, num_topics = pwt.shape
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
                    if pwt[word_index][topic_index] > threshold:
                        current_probability_sum = (
                                current_probability_sum + pwt[word_index][topic_index]
                        )
                        current_word_ratio = current_word_ratio + 1

                probability_sum = probability_sum + current_probability_sum
                word_ratio = word_ratio + current_word_ratio

                # Normalizing
                current_probability_sum = current_probability_sum / num_topics
                current_word_ratio = current_word_ratio / (num_topics * num_words)

                current_probability_sum = max(_TINY, current_probability_sum)
                current_word_ratio = max(_TINY, current_word_ratio)

                current_energy = -1 * np.log(current_probability_sum)
                current_shannon_entropy = np.log(current_word_ratio)
                current_free_energy = (
                    current_energy - num_topics * current_shannon_entropy
                )
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

            entropies.append(renyi_entropy)
            densities.append(shannon_entropy)
            energies.append(energy)

            if minimum_entropy is None or minimum_entropy > renyi_entropy:
                minimum_entropy = renyi_entropy
                optimum_num_topics = num_topics

            message = (f'Minimum Renyi entropy: {minimum_entropy}.' +
                       f' Number of clusters: {optimum_num_topics}')
            _logger.info(message)

            if self._verbose is True:
                print(message)

            if self._method == ENTROPY_MERGE_METHOD:
                topic_a, topic_b = self._select_topics_to_merge_by_entropy(
                    current_topics, current_entropies
                )
            elif self._method == RANDOM_MERGE_METHOD:
                topic_a, topic_b = self._select_topics_to_merge_by_random(
                    current_topics
                )
            elif self._method == KL_MERGE_METHOD:
                topic_a, topic_b = self._select_topics_to_merge_by_kl_divergence(
                    current_topics, pwt
                )
            else:
                raise ValueError(self._method)

            pwt = update_callback(pwt, topic_a, topic_b)

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
