import logging
import numpy as np
import pandas as pd
import pytest

from numbers import Number
from topicnet.cooking_machine.dataset import (
    Dataset,
    RAW_TEXT_COL,
    VW_TEXT_COL,
    W_DIFF_BATCHES_1
)
from topicnet.cooking_machine.models import (
    BaseModel,
    TopicModel
)
from typing import (
    Callable,
    Dict,
    List,
)

from topnum.scores.base_score import BaseScore
from topnum.scores.base_topic_score import BaseTopicScore
from topnum.search_methods import TopicBankMethod
from topnum.search_methods.topic_bank import BankUpdateMethod
from topnum.search_methods.topic_bank.one_model_train_funcs import (
    background_topics_train_func,
    default_train_func,
    regularization_train_func,
    specific_initial_phi_train_func
)
from topnum.search_methods.topic_bank.phi_initialization import arora
from topnum.search_methods.topic_bank.phi_initialization import cdc
from topnum.search_methods.topic_bank.phi_initialization.initialize_phi_funcs import (
    initialize_randomly,
    initialize_with_copying_topics
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
class TestTopicBank:
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
            self.optimizer._topic_bank.eliminate()
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

        return dataset

    @pytest.mark.parametrize('keep_in_memory', [True, False])
    def test_topic_bank_smoke(self, keep_in_memory):
        self._test_topic_bank(
            self.dataset(keep_in_memory=keep_in_memory),
            BankUpdateMethod.PROVIDE_NON_LINEARITY,
        )

    @pytest.mark.parametrize(
        'keep_in_memory, bank_update',
        [
            (True, BankUpdateMethod.JUST_ADD_GOOD_TOPICS),
            (False, BankUpdateMethod.JUST_ADD_GOOD_TOPICS),
            (True, BankUpdateMethod.PROVIDE_NON_LINEARITY),
        ]
    )
    @pytest.mark.parametrize(
        'train_funcs',
        [None, background_topics_train_func, default_train_func, regularization_train_func]
    )
    def test_topic_bank(self, keep_in_memory, bank_update, train_funcs):
        self._test_topic_bank(
            self.dataset(keep_in_memory=keep_in_memory),
            bank_update,
            train_func=train_funcs,
        )

    @pytest.mark.parametrize('keep_in_memory', [True, False])
    @pytest.mark.parametrize(
        'bank_update',
        [BankUpdateMethod.JUST_ADD_GOOD_TOPICS, BankUpdateMethod.PROVIDE_NON_LINEARITY]
    )
    def test_topic_bank_specific_phi_random(self, keep_in_memory, bank_update):
        def initialize_phi_func(
                dataset: Dataset,
                model_number: int,
                num_topics: int) -> pd.DataFrame:

            return initialize_randomly(dataset, model_number, num_topics)

        def train_func(
                dataset: Dataset,
                model_number: int,
                num_topics: int,
                num_fit_iterations: int,
                scores: List[BaseScore] = None,
                **kwargs) -> TopicModel:

            return specific_initial_phi_train_func(
                dataset, model_number, num_topics,
                num_fit_iterations, scores,
                initialize_phi_func=initialize_phi_func,
                **kwargs
            )

        self._test_topic_bank(
            self.dataset(keep_in_memory=keep_in_memory),
            bank_update,
            train_func=train_func,
        )

    @pytest.mark.parametrize('keep_in_memory', [True, False])
    @pytest.mark.parametrize(
        'bank_update',
        [BankUpdateMethod.JUST_ADD_GOOD_TOPICS, BankUpdateMethod.PROVIDE_NON_LINEARITY]
    )
    def test_topic_bank_specific_phi_cdc(self, keep_in_memory, bank_update):
        dataset = self.dataset(keep_in_memory=keep_in_memory)
        one_model_num_topics = 10
        num_topics_to_copy = one_model_num_topics // 2
        phi = cdc.compute_phi(
            dataset,
            self.main_modality,
            local_context_words_percentile=95,
            eps=0.05,
            min_samples=1
        )

        def initialize_phi_func(
                dataset: Dataset,
                model_number: int,
                num_topics: int) -> pd.DataFrame:

            return initialize_with_copying_topics(
                dataset, model_number, num_topics,
                phi=phi, num_topics_to_copy=num_topics_to_copy
            )

        def train_func(
                dataset: Dataset,
                model_number: int,
                num_topics: int,
                num_fit_iterations: int,
                scores: List[BaseScore] = None,
                **kwargs) -> TopicModel:

            return specific_initial_phi_train_func(
                dataset, model_number, num_topics,
                num_fit_iterations, scores,
                initialize_phi_func=initialize_phi_func,
                **kwargs
            )

        self._test_topic_bank(
            dataset,
            bank_update,
            one_model_num_topics=one_model_num_topics,
            train_func=train_func,
        )

    @pytest.mark.parametrize('keep_in_memory', [True, False])
    @pytest.mark.parametrize(
        'bank_update',
        [BankUpdateMethod.JUST_ADD_GOOD_TOPICS, BankUpdateMethod.PROVIDE_NON_LINEARITY]
    )
    def test_topic_bank_specific_phi_arora(self, keep_in_memory, bank_update):
        dataset = self.dataset(keep_in_memory=keep_in_memory)
        one_model_num_topics = 10
        num_topics_to_copy = one_model_num_topics // 2
        phi = arora.compute_phi(
            dataset,
            self.main_modality,
            num_topics=one_model_num_topics,
            document_occurrences_threshold_percentage=0.001
        )

        def initialize_phi_func(
                dataset: Dataset,
                model_number: int,
                num_topics: int) -> pd.DataFrame:

            return initialize_with_copying_topics(
                dataset, model_number, num_topics,
                phi=phi, num_topics_to_copy=num_topics_to_copy
            )

        def train_func(
                dataset: Dataset,
                model_number: int,
                num_topics: int,
                num_fit_iterations: int,
                scores: List[BaseScore] = None,
                **kwargs) -> TopicModel:

            return specific_initial_phi_train_func(
                dataset, model_number, num_topics,
                num_fit_iterations, scores,
                initialize_phi_func=initialize_phi_func,
                **kwargs
            )

        self._test_topic_bank(
            dataset,
            bank_update,
            one_model_num_topics=one_model_num_topics,
            train_func=train_func,
        )

    def _test_topic_bank(
            self,
            dataset: Dataset,
            bank_update: BankUpdateMethod,
            one_model_num_topics: int = 2,
            train_func: Callable = None):

        self.optimizer = TopicBankMethod(
            data=dataset,
            main_modality=self.main_modality,
            min_df_rate=0.0,
            max_df_rate=1.1,
            main_topic_score=_DummyTopicScore(),
            other_topic_scores=list(),
            max_num_models=5,
            one_model_num_topics=one_model_num_topics,
            num_fit_iterations=5,
            train_funcs=train_func,
            topic_score_threshold_percentile=2,
            bank_update=bank_update,
            save_bank=True,
            save_model_topics=True,
        )

        self.optimizer.search_for_optimum(self.text_collection)

        # TODO: improve check
        for result_key in ['optimum', 'optimum_std']:
            assert result_key in self.optimizer._result
            assert isinstance(self.optimizer._result[result_key], Number)
