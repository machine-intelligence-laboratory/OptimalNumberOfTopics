import logging
import numpy as np
import pytest
import warnings

from numbers import Number
from typing import (
    Dict,
    List,
)

from topicnet.cooking_machine.dataset import (
    Dataset,
    W_DIFF_BATCHES_1,
)
from topicnet.cooking_machine.models import BaseModel

from topnum.scores.base_topic_score import BaseTopicScore
from topnum.search_methods import RenormalizationMethod
from topnum.search_methods.base_search_method import BaseSearchMethod
from topnum.search_methods.renormalization_method import (
    ENTROPY_MERGE_METHOD,
    RANDOM_MERGE_METHOD,
    KL_MERGE_METHOD,
    PHI_RENORMALIZATION_MATRIX,
    THETA_RENORMALIZATION_MATRIX,
)
from topnum.tests.data_generator import TestDataGenerator


_Logger = logging.getLogger()


# TODO: remove? try to use Coherence instead of this
class _DummyTopicScore(BaseTopicScore):
    def __init__(self, name='dummy_score'):
        super().__init__(name)

    def compute(
            self,
            model: BaseModel,
            topics: List[str] = None,
            documents: List[str] = None) -> Dict[str, float]:

        if topics is None:
            topics = list(model.get_phi().columns)

        return {
            t: np.random.randint(1, 10)
            for t in topics
        }


@pytest.mark.filterwarnings(f'ignore:{W_DIFF_BATCHES_1}')
class TestRenormalization:
    data_generator = None

    main_modality = None
    other_modality = None
    text_collection = None

    optimizer = None

    @classmethod
    def setup_class(cls):
        cls.data_generator = TestDataGenerator()

        cls.data_generator.generate()

        cls.data_generator.text_collection._dataset = None

        cls.text_collection = cls.data_generator.text_collection
        cls.main_modality = cls.data_generator.main_modality
        cls.other_modality = cls.data_generator.other_modality

    def setup_method(self):
        assert self.text_collection._dataset is None

    def teardown_method(self):
        self.text_collection._set_dataset_kwargs()
        self.text_collection._dataset = None

        if self.optimizer is not None:
            self.optimizer.clear()

    @classmethod
    def teardown_class(cls):
        if cls.data_generator is not None:
            cls.data_generator.clear()

    def dataset(self, keep_in_memory: bool = True) -> Dataset:
        self.text_collection._set_dataset_kwargs(
            keep_in_memory=keep_in_memory
        )
        dataset = self.text_collection._to_dataset()

        # TODO: "workaround", TopicBank needs raw text
        dataset._data['raw_text'] = dataset._data['vw_text'].apply(
            lambda text: ' '.join(
                w.split(':')[0] for w in text.split()[1:] if not w.startswith('|'))
        )

        return dataset

    @pytest.mark.parametrize(
        'merge_method',
        [ENTROPY_MERGE_METHOD, RANDOM_MERGE_METHOD, KL_MERGE_METHOD]
    )
    @pytest.mark.parametrize(
        'threshold_factor',
        [1.0, 0.5, 1e-7, 1e7]
    )
    @pytest.mark.parametrize(
        'matrix_for_renormalization',
        [PHI_RENORMALIZATION_MATRIX, THETA_RENORMALIZATION_MATRIX]
    )
    def test_renormalize(self, merge_method, threshold_factor, matrix_for_renormalization):
        max_num_topics = 10

        optimizer = RenormalizationMethod(
            merge_method=merge_method,
            matrix_for_renormalization=matrix_for_renormalization,
            threshold_factor=threshold_factor,
            max_num_topics=max_num_topics,
            num_fit_iterations=10,
            num_restarts=3
        )
        num_search_points = len(list(range(1, max_num_topics)))

        optimizer.search_for_optimum(self.text_collection)

        self._check_search_result(optimizer._result, optimizer, num_search_points)

    @pytest.mark.parametrize('keep_in_memory', [True, False])
    def test_renormalize_small_big_data(self, keep_in_memory):
        max_num_topics = 10

        optimizer = RenormalizationMethod(
            max_num_topics=max_num_topics,
            num_fit_iterations=10,
            num_restarts=3,
        )
        num_search_points = len(list(range(1, max_num_topics)))

        self.text_collection._set_dataset_kwargs(keep_in_memory=keep_in_memory)

        optimizer.search_for_optimum(self.text_collection)

        self._check_search_result(optimizer._result, optimizer, num_search_points)

    def _check_search_result(
            self,
            search_result: Dict,
            optimizer: BaseSearchMethod,
            num_search_points: int):

        tiny = 1e-7

        for key in optimizer._keys_mean_one:
            assert key in search_result
            assert isinstance(search_result[key], Number)

        for key in optimizer._keys_std_one:
            assert key in search_result
            assert isinstance(search_result[key], Number)

        for key in optimizer._keys_mean_many:
            assert key in search_result
            assert len(search_result[key]) == num_search_points
            assert all(isinstance(v, Number) for v in search_result[key])

            # TODO: remove this check when refactor computation inside optimizer
            if (hasattr(optimizer, '_key_num_topics_values')
                    and key == optimizer._key_num_topics_values):

                assert all(
                    abs(v - int(v)) == 0
                    for v in search_result[optimizer._key_num_topics_values]
                )

            if all(abs(v) <= tiny for v in search_result[key]):
                warnings.warn(f'All score values "{key}" are zero!')

        for key in optimizer._keys_std_many:
            assert key in search_result
            assert len(search_result[key]) == num_search_points
            assert all(isinstance(v, Number) for v in search_result[key])
