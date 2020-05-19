import logging
import numpy as np
import os
import pytest
import subprocess
import warnings

from numbers import Number
from typing import (
    Dict,
    List,
)

from topicnet.cooking_machine.dataset import W_DIFF_BATCHES_1
from topicnet.cooking_machine.models import BaseModel

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
from topnum.search_methods.constants import DEFAULT_EXPERIMENT_DIR
from topnum.search_methods.optimize_scores_method import _KEY_SCORE_RESULTS
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
class TestAcceptance:
    data_generator = None

    dataset = None
    main_modality = None
    other_modality = None
    text_collection = None

    optimizer = None

    @classmethod
    def setup_class(cls):
        cls.data_generator = TestDataGenerator()

        cls.data_generator.generate()

        cls.text_collection = cls.data_generator.text_collection
        cls.main_modality = cls.data_generator.main_modality
        cls.other_modality = cls.data_generator.other_modality

        cls.dataset = cls.text_collection._to_dataset()

        # TODO: "workaround", TopicBank needs raw text
        cls.dataset._data['raw_text'] = cls.dataset._data['vw_text'].apply(
            lambda text: ' '.join(
                w.split(':')[0] for w in text.split()[1:] if not w.startswith('|'))
        )

    def teardown_method(self):
        if self.optimizer is not None:
            self.optimizer.clear()

    @classmethod
    def teardown_class(cls):
        if cls.data_generator is not None:
            cls.data_generator.clear()

    def test_optimize_perplexity(self):
        score = PerplexityScore(
            'perplexity_score',
            class_ids=[self.main_modality, self.other_modality]
        )

        self._test_optimize_score(score)

    def test_optimize_diversity(self):
        score = DiversityScore(
            'diversity_score',
            class_ids=self.main_modality
        )

        self._test_optimize_score(score)

    def test_optimize_intratext(self):
        score = IntratextCoherenceScore(
            name='intratext_coherence',
            data=self.dataset,
            documents=self.dataset._data.index[:1],
            window=2
        )

        # a bit slow -> just 2 restarts
        self._test_optimize_score(score, num_restarts=2)

    def test_topic_bank(self):
        self.optimizer = TopicBankMethod(
            data=self.dataset,
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

        self.optimizer.search_for_optimum(self.text_collection)

        # TODO: improve check
        for result_key in ['optimum', 'optimum_std']:
            assert result_key in self.optimizer._result
            assert isinstance(self.optimizer._result[result_key], Number)

    def test_stability(self):
        min_num_topics = 1
        max_num_topics = 5
        num_topics_interval = 1

        self.optimizer = StabilitySearchMethod(
            min_num_topics=min_num_topics,
            max_num_topics=max_num_topics,
            num_topics_interval=num_topics_interval,
        )

        self.optimizer.search_for_optimum(self.text_collection)

        assert len(list(self.optimizer._result.keys())) > 0

        result_value = list(self.optimizer._result.values())[0]
        num_topics_values = list(range(min_num_topics, max_num_topics + 1, num_topics_interval))

        assert len(list(result_value.values())) == len(num_topics_values)

    def test_renormalize(self):
        max_num_topics = 10

        optimizer = RenormalizationMethod(
            max_num_topics=max_num_topics,
            num_fit_iterations=10,
            num_restarts=3
        )
        num_search_points = len(list(range(1, max_num_topics)))

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
            experiment_directory=DEFAULT_EXPERIMENT_DIR
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
