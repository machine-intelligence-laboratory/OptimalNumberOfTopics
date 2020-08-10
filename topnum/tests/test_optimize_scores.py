import logging
import numpy as np
import pytest
import shutil
import tempfile
import warnings

from itertools import combinations
from numbers import Number
from time import sleep
from typing import (
    Dict,
    List,
)

import artm

from topicnet.cooking_machine import Experiment
from topicnet.cooking_machine.cubes import (
    CubeCreator,
    RegularizersModifierCube,
)
from topicnet.cooking_machine.dataset import (
    Dataset,
    W_DIFF_BATCHES_1,
)
from topicnet.cooking_machine.models import (
    BaseModel,
    TopicModel,
)
from topicnet.cooking_machine.model_constructor import init_simple_default_model

from topnum.scores import (
    CalinskiHarabaszScore,
    DiversityScore,
    EntropyScore,
    HoldoutPerplexityScore,
    IntratextCoherenceScore,
    LikelihoodBasedScore,
    MeanLiftScore,
    PerplexityScore,
    SilhouetteScore,
    SimpleTopTokensCoherenceScore,
    SophisticatedTopTokensCoherenceScore,
    SparsityPhiScore,
    SparsityThetaScore,
    SpectralDivergenceScore,
)
from topnum.scores.base_topic_score import BaseTopicScore
from topnum.search_methods import OptimizeScoresMethod
from topnum.search_methods.base_search_method import BaseSearchMethod
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
class TestOptimizeScores:
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

        cls.working_folder_path = tempfile.mktemp(prefix='test_optimize_scores__')

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

        shutil.rmtree(cls.working_folder_path)

    def dataset(self, keep_in_memory: bool = True) -> Dataset:
        self.text_collection._set_dataset_kwargs(
            keep_in_memory=keep_in_memory
        )
        dataset = self.text_collection._to_dataset()

        return dataset

    @pytest.mark.parametrize('keep_in_memory', [True, False])
    def test_optimize_calinski_harabasz(self, keep_in_memory):
        dataset = self.dataset(keep_in_memory)
        self.text_collection._set_dataset_kwargs(keep_in_memory=keep_in_memory)

        score = CalinskiHarabaszScore(
            'calinski_harabasz_score',
            validation_dataset=dataset,
        )

        self._test_optimize_score(score)

    def test_optimize_diversity(self):
        score = DiversityScore(
            'diversity_score',
            class_ids=self.main_modality
        )

        self._test_optimize_score(score)

    @pytest.mark.parametrize('keep_in_memory', [True, False])
    def test_optimize_silhouette(self, keep_in_memory):
        dataset = self.dataset(keep_in_memory=keep_in_memory)
        score = SilhouetteScore(
            'silhouette_score',
            validation_dataset=dataset,
        )

        self._test_optimize_score(score)

    @pytest.mark.parametrize('keep_in_memory', [True, False])
    def test_optimize_lift(self, keep_in_memory):
        dataset = self.dataset(keep_in_memory=keep_in_memory)
        score = MeanLiftScore(
            'lift_score',
            validation_dataset=dataset,
            modalities=[self.main_modality, self.other_modality],
        )

        self._test_optimize_score(score)

    @pytest.mark.parametrize('keep_in_memory', [True, False])
    def test_optimize_likelihood(self, keep_in_memory):
        dataset = self.dataset(keep_in_memory=keep_in_memory)
        score = LikelihoodBasedScore(
            'likelihood_score',
            validation_dataset=dataset,
            modality=self.main_modality,
        )

        self._test_optimize_score(score)

    @pytest.mark.parametrize('keep_in_memory', [True, False])
    def test_optimize_divergence(self, keep_in_memory):
        dataset = self.dataset(keep_in_memory=keep_in_memory)
        score = SpectralDivergenceScore(
            'divergence_score',
            validation_dataset=dataset,
            modalities=[self.main_modality],
        )

        self._test_optimize_score(score)

    def test_optimize_perplexity(self):
        score = PerplexityScore(
            'perplexity_score',
            class_ids=[self.main_modality, self.other_modality]
        )

        self._test_optimize_score(score)

    @pytest.mark.parametrize('keep_in_memory', [True, False])
    def test_optimize_holdout_perplexity(self, keep_in_memory):
        dataset = self.dataset(keep_in_memory)
        score = HoldoutPerplexityScore(
            'holdout_perplexity_score',
            test_dataset=dataset,
            class_ids=[self.main_modality, self.other_modality]
        )

        self._test_optimize_score(score)

    @pytest.mark.parametrize('entropy', ['renyi', 'shannon'])
    @pytest.mark.parametrize('threshold_factor', [1.0, 0.5, 1e-7, 1e7])
    def test_optimize_entropy(self, entropy, threshold_factor):
        sleep(3)  # TODO: remove

        score = EntropyScore(
            name='renyi_entropy',
            entropy=entropy,
            threshold_factor=threshold_factor
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

        # a bit slow -> just 2 restarts
        self._test_optimize_score(score, num_restarts=2)

    @pytest.mark.parametrize('keep_in_memory', [True, False])
    def test_optimize_sophisticated_toptokens(self, keep_in_memory):
        dataset = self.dataset(keep_in_memory=keep_in_memory)
        score = SophisticatedTopTokensCoherenceScore(
            name='sophisticated_toptokens_coherence',
            data=dataset,
            documents=dataset.documents[:1]
        )

        self._test_optimize_score(score, num_restarts=2)

    @pytest.mark.parametrize('keep_in_memory', [True, False])
    @pytest.mark.parametrize('what_modalities', ['None', 'one', 'many'])
    def test_optimize_simple_toptokens(self, keep_in_memory, what_modalities):
        if what_modalities == 'None':
            modalities = None
        elif what_modalities == 'one':
            modalities = self.main_modality
        elif what_modalities == 'many':
            modalities = [self.main_modality]
        else:
            assert False

        cooccurrence_values = dict()
        cooccurrence_values[('play__m', 'boy__m')] = 2

        num_unique_words = 5

        for i, (w1, w2) in enumerate(
                combinations(self.data_generator.vocabulary[:num_unique_words], 2)):

            cooccurrence_values[(w1, w2)] = i

        dataset = self.dataset(keep_in_memory=keep_in_memory)
        score = SimpleTopTokensCoherenceScore(
            name='simple_toptokens_coherence',
            cooccurrence_values=cooccurrence_values,
            data=dataset,
            modalities=modalities
        )

        self._test_optimize_score(score)

    def test_optimize_sparsity_phi(self):
        score = SparsityPhiScore(
            'sparsity_phi_score',
            class_id=self.main_modality,
        )

        self._test_optimize_score(score)

    def test_optimize_sparsity_theta(self):
        score = SparsityThetaScore(
            name='sparsity_theta_score'
        )

        self._test_optimize_score(score)

    @pytest.mark.parametrize('keep_in_memory', [True, False])
    def test_two_stage_experiment_with_all_scores(self, keep_in_memory):
        dataset = self.dataset(keep_in_memory=keep_in_memory)
        artm_model = init_simple_default_model(
            dataset=dataset,
            modalities_to_use=[self.main_modality, self.other_modality],
            main_modality=self.main_modality,
            specific_topics=5,
            background_topics=1,
        )
        artm_model.num_processors = 2
        topic_model = TopicModel(artm_model)
        experiment = Experiment(
            topic_model=topic_model,
            experiment_id=f'test_all_scores_workable__{keep_in_memory}',
            save_path=self.working_folder_path,
        )

        scores = [
            SpectralDivergenceScore(
                'divergence_score',
                validation_dataset=dataset,
                modalities=[self.main_modality],
            ),
            LikelihoodBasedScore(
                'likelihood_score',
                validation_dataset=dataset,
                modality=self.main_modality,
            ),
            SilhouetteScore(
                'silhouette_score',
                validation_dataset=dataset,
            ),
            DiversityScore(
                'diversity_score',
                class_ids=self.main_modality,
            ),
            CalinskiHarabaszScore(
                'calinski_harabasz_score',
                validation_dataset=dataset,
            ),
            EntropyScore(
                name='renyi_entropy',
            ),
            HoldoutPerplexityScore(
                'holdout_perplexity_score',
                test_dataset=dataset,
                class_ids=[self.main_modality, self.other_modality]
            ),
            IntratextCoherenceScore(
                name='intratext_coherence',
                data=dataset,
                documents=dataset.documents[:1],
                window=2,
            ),
            SophisticatedTopTokensCoherenceScore(
                name='sophisticated_toptokens_coherence',
                data=dataset,
                documents=dataset.documents[:1],
            ),
            PerplexityScore(
                'perplexity_score',
                class_ids=[self.main_modality, self.other_modality],
            ),
            SparsityPhiScore(
                'sparsity_phi_score',
                class_id=self.main_modality,
            ),
        ]

        for score in scores:
            score._attach(topic_model)

        num_iters_first_cube = 5
        cube = CubeCreator(
            num_iter=num_iters_first_cube,
            parameters=[
                {
                    'name': 'num_document_passes',
                    'values': [1, 2]
                },
            ],
            reg_search='grid',
            tracked_score_function=f'PerplexityScore{self.main_modality}',
            separate_thread=False,
            verbose=False,
        )

        _ = cube(experiment.root, dataset)
        selection_criterion = f'PerplexityScore{self.main_modality} -> min'
        best_model_first_cube = experiment.select(selection_criterion)[0]

        num_iters_second_cube = 7
        cube = RegularizersModifierCube(
            num_iter=num_iters_second_cube,
            regularizer_parameters=[
                {
                    'regularizer': artm.DecorrelatorPhiRegularizer(
                        name='decorr',
                        tau=0,
                    ),
                    'tau_grid': [0.01, 0.1]
                },
                {
                    'regularizer': artm.SmoothSparsePhiRegularizer(
                        name='smooth',
                        tau=0,
                    ),
                    'tau_grid': [0.001]
                }
            ],
            use_relative_coefficients=True,
            reg_search='grid',
            separate_thread=False,
            verbose=False,
        )

        _ = cube(best_model_first_cube, dataset)
        best_model_second_cube = experiment.select(selection_criterion)[0]

        for score in scores:
            for model, model_num_iters in zip(
                    [best_model_first_cube, best_model_second_cube],
                    [num_iters_first_cube, num_iters_first_cube + num_iters_second_cube]):

                assert score.name in model.scores
                assert len(model.scores[score.name]) == model_num_iters
                assert all([isinstance(v, Number) for v in model.scores[score.name]])

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

        # TODO: probably remove the assert below, because there are many default scores now
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
