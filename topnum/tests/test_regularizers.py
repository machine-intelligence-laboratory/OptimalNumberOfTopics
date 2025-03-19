import logging
import os
import shutil
import tempfile

from copy import deepcopy
from typing import List

import numpy as np
import pytest

from pandas import DataFrame

from topicnet.cooking_machine.dataset import (
    Dataset,
    W_DIFF_BATCHES_1,
)
from topicnet.cooking_machine.models import TopicModel
from topicnet.cooking_machine.model_constructor import init_simple_default_model


from topnum.regularizers import (
    FastFixPhiRegularizer,
    DecorrelateWithOtherPhiRegularizer,
    DecorrelateWithOtherPhiRegularizer2,
)
from topnum.scores import PerplexityScore
from topnum.tests.data_generator import TestDataGenerator


_Logger = logging.getLogger()


@pytest.mark.filterwarnings(f'ignore:{W_DIFF_BATCHES_1}')
class TestOptimizeScores:
    PPL_SCORE_NAME = 'ppl'
    ONE_FIT_NUM_ITERS = 10

    NUM_TOPICS = 10

    # Ideally, these topics should be found by looking at the scores
    # (Here we are assigning the labels just out of thin air)
    GOOD_TOPIC_INDICES = [0, 1, 2]
    BAD_TOPIC_INDICES = [-1, -2, -3]

    data_generator = None

    main_modality = None
    other_modality = None
    text_collection = None

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

        os.mkdir(self.working_folder_path)

    def teardown_method(self):
        self.text_collection._set_dataset_kwargs()
        self.text_collection._dataset = None

        if os.path.isdir(self.working_folder_path):
            shutil.rmtree(self.working_folder_path)

    @classmethod
    def teardown_class(cls):
        if cls.data_generator is not None:
            cls.data_generator.clear()

        if os.path.isdir(cls.working_folder_path):
            shutil.rmtree(cls.working_folder_path)

    def _get_dataset(self, keep_in_memory: bool = True) -> Dataset:
        self.text_collection._set_dataset_kwargs(
            keep_in_memory=keep_in_memory
        )
        dataset = self.text_collection._to_dataset()

        return dataset

    def _get_topic_model_and_topics(
            self,
            dataset: Dataset,
            num_specific_topics=5,
            num_background_topics=1,
            num_processors=2,
            ):
        artm_model = init_simple_default_model(
            dataset=dataset,
            modalities_to_use=[self.main_modality, self.other_modality],
            main_modality=self.main_modality,
            specific_topics=num_specific_topics,
            background_topics=num_background_topics,
        )
        artm_model.num_processors = num_processors

        topic_model = TopicModel(artm_model)
        score = PerplexityScore(self.PPL_SCORE_NAME)
        score._attach(topic_model)

        topic_model._fit(
            dataset.get_batch_vectorizer(),
            num_iterations=self.ONE_FIT_NUM_ITERS,
        )

        phi = topic_model.get_phi()
        good_topic_names = [phi.columns[t] for t in self.GOOD_TOPIC_INDICES]
        bad_topic_names = [phi.columns[t] for t in self.BAD_TOPIC_INDICES]
        not_good_topic_names = [
            phi.columns[t]
            for t in range(len(phi.columns))
            if t not in self.GOOD_TOPIC_INDICES
        ]

        return (
            topic_model,
            good_topic_names,
            bad_topic_names,
            not_good_topic_names,
        )

    def _get_fix_regularizer(
            self,
            name: str,
            target_topic_names: List[str],
            parent_topic_model: TopicModel = None,
            parent_phi: DataFrame = None,
            ):
        fix_regularizer = FastFixPhiRegularizer(
            name=name,
            topic_names=target_topic_names,
            parent_model=parent_topic_model,
            parent_phi=parent_phi,
        )

        return fix_regularizer

    def _get_decorr_regularizer_base(
            self,
            name: str,
            tau: float,
            target_topic_names: List[str],
            other_topic_model: TopicModel,
            other_topic_names: List[str],
            decorrelate_regularizer_class,
            ):
        other_phi = other_topic_model._model.get_phi()[other_topic_names]
        other_phi = deepcopy(other_phi)
        decorr_regularizer = decorrelate_regularizer_class(
            name=name,
            tau=tau,
            topic_names=target_topic_names,
            other_phi=other_phi,
        )

        return decorr_regularizer, other_phi

    def _get_decorr_regularizer(
            self,
            name: str,
            tau: float,
            target_topic_names: List[str],
            other_topic_model: TopicModel,
            other_topic_names: List[str],
            ):
        return self._get_decorr_regularizer_base(
            name=name, tau=tau,
            target_topic_names=target_topic_names,
            other_topic_model=other_topic_model,
            other_topic_names=other_topic_names,
            decorrelate_regularizer_class=DecorrelateWithOtherPhiRegularizer,
        )

    def _get_decorr_regularizer2(
            self,
            name: str,
            tau: float,
            target_topic_names: List[str],
            other_topic_model: TopicModel,
            other_topic_names: List[str],
            ):
        return self._get_decorr_regularizer_base(
            name=name, tau=tau,
            target_topic_names=target_topic_names,
            other_topic_model=other_topic_model,
            other_topic_names=other_topic_names,
            decorrelate_regularizer_class=DecorrelateWithOtherPhiRegularizer2,
        )

    @pytest.mark.parametrize('keep_in_memory', [True, False])
    def test_fix_good(self, keep_in_memory):
        dataset = self._get_dataset(keep_in_memory=keep_in_memory)
        (topic_model,
         good_topic_names,
         bad_topic_names,
         not_good_topic_names) = self._get_topic_model_and_topics(dataset=dataset)

        good_phi = deepcopy(
            topic_model._model.get_phi()[good_topic_names]
        )

        fix_regularizer = self._get_fix_regularizer(
            name='fix',
            target_topic_names=good_topic_names,
            parent_phi=good_phi,
        )

        topic_model._fit(
            dataset.get_batch_vectorizer(),
            num_iterations=self.ONE_FIT_NUM_ITERS,
            custom_regularizers={
                fix_regularizer.name: fix_regularizer,
            }
        )

        new_phi = topic_model._model.get_phi()

        assert np.allclose(
            new_phi[good_topic_names], good_phi
        )

    @pytest.mark.parametrize('decorr_v2', [False, True])
    @pytest.mark.parametrize('keep_in_memory', [True, False])
    def test_decorr_bad(self, decorr_v2, keep_in_memory):
        dataset = self._get_dataset(keep_in_memory=keep_in_memory)
        (topic_model,
         good_topic_names,
         bad_topic_names,
         not_good_topic_names) = self._get_topic_model_and_topics(dataset=dataset)

        good_phi = deepcopy(
            topic_model._model.get_phi()[good_topic_names]
        )
        base_topic_decorr_kwargs = dict(
            target_topic_names=not_good_topic_names,
            other_topic_model=topic_model,
            other_topic_names=bad_topic_names,
        )

        if not decorr_v2:
            decorr_bad_regularizer, bad_phi = self._get_decorr_regularizer(
                name='ext_decorr_bad',
                tau=1e5,
                **base_topic_decorr_kwargs,
            )
        else:
            decorr_bad_regularizer, bad_phi = self._get_decorr_regularizer2(
                name='ext_decorr_bad2',
                tau=1e8,
                **base_topic_decorr_kwargs,
            )

        topic_model._fit(
            dataset.get_batch_vectorizer(),
            num_iterations=self.ONE_FIT_NUM_ITERS,
            custom_regularizers={
                decorr_bad_regularizer.name: decorr_bad_regularizer,
            }
        )

        new_phi = topic_model._model.get_phi()

        # TODO: good topics also change (as they are not fixed)
        #   so, the meaningfulness of this test is questionable
        #   (other than the fact that it simply tests runnability)
        # assert np.allclose(
        #     new_phi[good_topic_names], good_phi, rtol=0.05
        # )
        assert not np.allclose(
            new_phi[not_good_topic_names], bad_phi, rtol=0.5
        )

    @pytest.mark.parametrize('decorr_v2', [False, True])
    def test_fix_good_and_decorr_good_bad(self, decorr_v2):
        dataset = self._get_dataset(keep_in_memory=True)
        (topic_model,
         good_topic_names,
         bad_topic_names,
         not_good_topic_names) = self._get_topic_model_and_topics(dataset=dataset)

        fix_regularizer = self._get_fix_regularizer(
            name='fix',
            target_topic_names=good_topic_names,
            parent_topic_model=topic_model._model,
        )
        # TODO: test breaks if pass just `topic_model` for `parent_topic_model`
        #   aah, I guess, there are some score saving issues (_score_caches=None)

        base_topic_decorr_kwargs = dict(
            target_topic_names=not_good_topic_names,
            other_topic_model=topic_model,
        )

        if not decorr_v2:
            decorr_bad_regularizer, bad_phi = self._get_decorr_regularizer(
                name='ext_decorr_bad', tau=1e5,
                other_topic_names=bad_topic_names,
                **base_topic_decorr_kwargs,
            )
            decorr_good_regularizer, good_phi = self._get_decorr_regularizer(
                name='ext_decorr_good', tau=1e5,
                other_topic_names=good_topic_names,
                **base_topic_decorr_kwargs
            )
        else:
            decorr_bad_regularizer, bad_phi = self._get_decorr_regularizer2(
                name='ext_decorr_bad2', tau=1e8,
                other_topic_names=bad_topic_names,
                **base_topic_decorr_kwargs
            )
            decorr_good_regularizer, good_phi = self._get_decorr_regularizer2(
                name='ext_decorr_good2', tau=1e8,
                other_topic_names=good_topic_names,
                **base_topic_decorr_kwargs
            )

        topic_model._fit(
            dataset.get_batch_vectorizer(),
            num_iterations=self.ONE_FIT_NUM_ITERS,
            custom_regularizers={
                fix_regularizer.name: fix_regularizer,
                decorr_bad_regularizer.name: decorr_bad_regularizer,
                decorr_good_regularizer.name: decorr_good_regularizer,
            }
        )

        new_phi = topic_model._model.get_phi()

        assert np.allclose(
            new_phi[good_topic_names], good_phi
        )

        assert not np.allclose(
            new_phi[not_good_topic_names], good_phi, rtol=0.5
        )
        assert not np.allclose(
            new_phi[not_good_topic_names], bad_phi, rtol=0.5
        )
