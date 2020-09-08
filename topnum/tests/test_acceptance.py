import logging
import numpy as np
import os
import pytest
import shutil
import subprocess
import tempfile
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
from topicnet.cooking_machine.models import (
    BaseModel,
    TopicModel,
)

from topnum.data import VowpalWabbitTextCollection
from topnum.model_constructor import KnownModel
from topnum.scores import (
    DiversityScore,
    IntratextCoherenceScore,
    PerplexityScore,
)
from topnum.scores.base_topic_score import BaseTopicScore
from topnum.search_methods import (
    OptimizeScoresMethod,
    RenormalizationMethod,
    StabilitySearchMethod,
    TopicBankMethod,
)
from topnum.search_methods.base_search_method import BaseSearchMethod
from topnum.search_methods.optimize_scores_method import _KEY_SCORE_RESULTS
from topnum.tests.data_generator import TestDataGenerator
from topnum.utils import (
    build_every_score,
    split_into_train_test,
)


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
class TestAcceptance:
    data_generator = None

    main_modality = None
    other_modality = None
    text_collection = None

    optimizer = None

    working_folder_path = None

    @classmethod
    def setup_class(cls):
        cls.data_generator = TestDataGenerator()

        cls.data_generator.generate()

        cls.data_generator.text_collection._dataset = None

        cls.text_collection = cls.data_generator.text_collection
        cls.main_modality = cls.data_generator.main_modality
        cls.other_modality = cls.data_generator.other_modality

        cls.working_folder_path = tempfile.mkdtemp(prefix='test_acceptance__')

    def setup_method(self):
        assert self.text_collection._dataset is None

        os.makedirs(self.working_folder_path, exist_ok=True)

    def teardown_method(self):
        self.text_collection._set_dataset_kwargs()
        self.text_collection._dataset = None

        if self.optimizer is not None:
            self.optimizer.clear()

        if os.path.isdir(self.working_folder_path):
            shutil.rmtree(self.working_folder_path)

    @classmethod
    def teardown_class(cls):
        if cls.data_generator is not None:
            cls.data_generator.clear()

        if os.path.isdir(cls.working_folder_path):
            shutil.rmtree(cls.working_folder_path)

    def dataset(self, keep_in_memory: bool = True) -> Dataset:
        self.text_collection._set_dataset_kwargs(
            keep_in_memory=keep_in_memory
        )
        dataset = self.text_collection._to_dataset()

        return dataset

    @pytest.mark.parametrize('keep_in_memory', [True, False])
    def test_optimize_perplexity(self, keep_in_memory):
        score = PerplexityScore(
            'perplexity_score',
            class_ids=[self.main_modality, self.other_modality]
        )
        self.text_collection._set_dataset_kwargs(
            keep_in_memory=keep_in_memory
        )

        self._test_optimize_score(score)

    @pytest.mark.parametrize('keep_in_memory', [True, False])
    def test_optimize_diversity(self, keep_in_memory):
        score = DiversityScore(
            'diversity_score',
            class_ids=self.main_modality
        )
        self.text_collection._set_dataset_kwargs(
            keep_in_memory=keep_in_memory
        )

        self._test_optimize_score(score)

    @pytest.mark.parametrize('keep_in_memory', [True, False])
    def test_optimize_intratext(self, keep_in_memory):
        dataset = self.dataset(keep_in_memory=keep_in_memory)
        score = IntratextCoherenceScore(
            name='intratext_coherence',
            data=dataset,
            documents=dataset.documents[:1],
            window=2,
        )
        self.text_collection._set_dataset_kwargs(
            keep_in_memory=keep_in_memory
        )

        # a bit slow -> just 2 restarts
        self._test_optimize_score(score, num_restarts=2)

    @pytest.mark.parametrize('keep_in_memory', [True, False])
    def test_optimize_all_scores(self, keep_in_memory):
        batches_prefix = 'all_scores'
        train_dataset, test_dataset = split_into_train_test(
            self.dataset(keep_in_memory=keep_in_memory),
            config={'batches_prefix': batches_prefix},  # TODO: fragile
            save_folder=self.working_folder_path,
        )

        assert train_dataset._small_data == keep_in_memory
        assert test_dataset._small_data == keep_in_memory

        text_collection = VowpalWabbitTextCollection.from_dataset(
            train_dataset,
            main_modality=self.main_modality,
        )

        assert text_collection._to_dataset()._small_data == keep_in_memory

        built_scores = build_every_score(
            dataset=train_dataset,
            test_dataset=test_dataset,
            config={'word': self.main_modality},  # TODO: fragile
        )

        min_num_topics = 1
        max_num_topics = 2
        num_topics_interval = 1
        num_search_points = len(
            list(range(min_num_topics, max_num_topics + 1, num_topics_interval))
        )
        num_restarts = 3
        experiment_name = 'all_scores'
        experiment_folder = os.path.join(self.working_folder_path, 'experiment_all_scores')

        optimizer = OptimizeScoresMethod(
            scores=built_scores,
            min_num_topics=min_num_topics,
            max_num_topics=max_num_topics,
            num_topics_interval=num_topics_interval,
            num_fit_iterations=3,
            num_restarts=num_restarts,
            one_model_num_processors=1,
            separate_thread=False,
            experiment_name=experiment_name,
            experiment_directory=experiment_folder,
        )

        optimizer.search_for_optimum(text_collection=text_collection)
        restart_folder_names = os.listdir(experiment_folder)

        assert len(restart_folder_names) == num_restarts

        for restart_folder_name in restart_folder_names:
            assert restart_folder_name.startswith(experiment_name)

            model_folder_names = os.listdir(os.path.join(experiment_folder, restart_folder_name))

            assert len(model_folder_names) == num_search_points

    def test_optimize_for_given_nums_topics(self, keep_in_memory=True):
        batches_prefix = 'given_nums_topics'
        train_dataset, test_dataset = split_into_train_test(
            self.dataset(keep_in_memory=keep_in_memory),
            config={'batches_prefix': batches_prefix},  # TODO: fragile
            save_folder=self.working_folder_path,
        )

        assert train_dataset._small_data == keep_in_memory
        assert test_dataset._small_data == keep_in_memory

        text_collection = VowpalWabbitTextCollection.from_dataset(
            train_dataset,
            main_modality=self.main_modality,
        )

        assert text_collection._to_dataset()._small_data == keep_in_memory

        score = PerplexityScore(
            'perplexity_score',
            class_ids=[self.main_modality, self.other_modality]
        )

        nums_topics = [1, 2, 5]
        num_search_points = len(nums_topics)
        num_restarts = 3
        experiment_name = 'given_nums_topics'
        experiment_folder = os.path.join(self.working_folder_path, 'experiment_given_nums_topics')

        optimizer = OptimizeScoresMethod(
            scores=[score],
            min_num_topics=0,  # TODO: need to set some placeholders
            max_num_topics=0,
            nums_topics=nums_topics,
            num_fit_iterations=3,
            num_restarts=num_restarts,
            one_model_num_processors=1,
            separate_thread=False,
            experiment_name=experiment_name,
            experiment_directory=experiment_folder,
        )

        optimizer.search_for_optimum(text_collection=text_collection)
        restart_folder_names = os.listdir(experiment_folder)

        assert len(restart_folder_names) == num_restarts

        for restart_folder_name in restart_folder_names:
            assert restart_folder_name.startswith(experiment_name)

            model_folder_names = os.listdir(os.path.join(experiment_folder, restart_folder_name))

            assert len(model_folder_names) == num_search_points

    @pytest.mark.parametrize('keep_in_memory', [True, False])
    @pytest.mark.parametrize('model_family', list(KnownModel))
    def test_optimize_for_model(self, keep_in_memory, model_family):
        # Thetaless currently fails
        # see https://github.com/machine-intelligence-laboratory/TopicNet/issues/79

        artm_score_name = 'perplexity_score'
        artm_score = PerplexityScore(
            name=artm_score_name,
            class_ids=[self.main_modality, self.other_modality]
        )

        custom_score_name = 'diversity_score'
        custom_score = DiversityScore(
            custom_score_name,
            class_ids=self.main_modality
        )

        self.text_collection._set_dataset_kwargs(
            keep_in_memory=keep_in_memory
        )

        min_num_topics = 1
        max_num_topics = 2
        num_topics_interval = 1
        num_fit_iterations = 3
        num_search_points = len(
            list(range(min_num_topics, max_num_topics + 1, num_topics_interval))
        )
        num_restarts = 3
        experiment_name = model_family.value
        experiment_folder = self.working_folder_path

        optimizer = OptimizeScoresMethod(
            scores=[artm_score, custom_score],
            model_family=model_family,
            min_num_topics=min_num_topics,
            max_num_topics=max_num_topics,
            num_topics_interval=num_topics_interval,
            num_fit_iterations=num_fit_iterations,
            num_restarts=num_restarts,
            one_model_num_processors=1,
            separate_thread=False,
            experiment_name=experiment_name,
            experiment_directory=experiment_folder,
        )

        optimizer.search_for_optimum(text_collection=self.text_collection)
        restart_folder_names = os.listdir(experiment_folder)

        assert len(restart_folder_names) == num_restarts

        for restart_folder_name in restart_folder_names:
            assert restart_folder_name.startswith(experiment_name)

            restart_folder_path = os.path.join(experiment_folder, restart_folder_name)
            model_folder_names = os.listdir(restart_folder_path)

            assert len(model_folder_names) == num_search_points

            for model_folder_name in model_folder_names:
                topic_model = TopicModel.load(os.path.join(restart_folder_path, model_folder_name))

                assert artm_score_name in topic_model.scores
                assert custom_score_name in topic_model.scores

                assert len(topic_model.scores[artm_score_name]) == num_fit_iterations
                assert len(topic_model.scores[custom_score_name]) == 1

                assert all(isinstance(v, Number) for v in topic_model.scores[artm_score_name])
                assert all(isinstance(v, Number) for v in topic_model.scores[custom_score_name])

    @pytest.mark.parametrize('keep_in_memory', [True, False])
    def test_topic_bank(self, keep_in_memory):
        dataset = self.dataset(keep_in_memory=keep_in_memory)
        self.optimizer = TopicBankMethod(
            data=dataset,
            main_modality=self.main_modality,
            min_df_rate=0.0,
            max_df_rate=1.1,
            main_topic_score=_DummyTopicScore(),
            other_topic_scores=list(),
            max_num_models=5,
            one_model_num_topics=2,
            num_fit_iterations=5,
            topic_score_threshold_percentile=2,
        )

        self.optimizer.search_for_optimum()

        # TODO: improve check
        for result_key in ['optimum', 'optimum_std']:
            assert result_key in self.optimizer._result
            assert isinstance(self.optimizer._result[result_key], Number)

    @pytest.mark.parametrize('keep_in_memory', [True, False])
    def test_stability(self, keep_in_memory):
        min_num_topics = 1
        max_num_topics = 5
        num_topics_interval = 1

        self.optimizer = StabilitySearchMethod(
            min_num_topics=min_num_topics,
            max_num_topics=max_num_topics,
            num_topics_interval=num_topics_interval,
        )
        self.text_collection._set_dataset_kwargs(
            keep_in_memory=keep_in_memory
        )

        self.optimizer.search_for_optimum(self.text_collection)

        assert len(list(self.optimizer._result.keys())) > 0

        result_value = list(self.optimizer._result.values())[0]
        num_topics_values = list(range(min_num_topics, max_num_topics + 1, num_topics_interval))

        assert len(list(result_value.values())) == len(num_topics_values)

    @pytest.mark.parametrize('keep_in_memory', [True, False])
    def test_renormalize(self, keep_in_memory):
        max_num_topics = 10

        optimizer = RenormalizationMethod(
            max_num_topics=max_num_topics,
            num_fit_iterations=10,
            num_restarts=3
        )
        num_search_points = len(list(range(1, max_num_topics)))
        self.text_collection._set_dataset_kwargs(
            keep_in_memory=keep_in_memory
        )

        optimizer.search_for_optimum(self.text_collection)

        self._check_search_result(optimizer._result, optimizer, num_search_points)

    def test_sample_script(self):
        tests_folder_path = os.path.dirname(os.path.abspath(__file__))
        samples_folder_path = os.path.join(tests_folder_path, '..', '..', 'sample')
        script_file_path = os.path.join(samples_folder_path, 'optimize_scores.sh')

        assert os.path.isfile(script_file_path)

        process = subprocess.Popen(script_file_path, cwd=samples_folder_path)
        process.wait()

        assert process.returncode == 0

        process = subprocess.Popen(script_file_path, cwd='.')
        process.wait()

        assert process.returncode != 0

    def _test_optimize_score(self, score, num_restarts: int = 3) -> None:
        min_num_topics = 1
        max_num_topics = 2
        num_topics_interval = 1

        num_fit_iterations = 3
        num_processors = 1

        optimizer = OptimizeScoresMethod(
            scores=[score],
            min_num_topics=min_num_topics,
            max_num_topics=max_num_topics,
            num_topics_interval=num_topics_interval,
            num_fit_iterations=num_fit_iterations,
            num_restarts=num_restarts,
            one_model_num_processors=num_processors,
            separate_thread=False,
            experiment_name=score.name,  # otherwise will be using same folder
            experiment_directory=self.working_folder_path,
        )
        num_search_points = len(
            list(range(min_num_topics, max_num_topics + 1, num_topics_interval))
        )

        optimizer.search_for_optimum(self.text_collection)

        assert len(optimizer._result) == 1
        assert _KEY_SCORE_RESULTS in optimizer._result

        # TODO: ptobably remove the assert below, because there are many default scores now
        # assert len(optimizer._result[_KEY_SCORE_RESULTS]) == 1
        assert score.name in optimizer._result[_KEY_SCORE_RESULTS]

        self._check_search_result(
            optimizer._result[_KEY_SCORE_RESULTS][score.name],
            optimizer,
            num_search_points
        )

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
