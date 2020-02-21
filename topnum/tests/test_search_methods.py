import numpy as np
import os
import pytest
import shutil
import tempfile
import warnings
from topicnet.cooking_machine.dataset import W_DIFF_BATCHES_1
from typing import (
    Dict,
    List
)

from itertools import combinations
from topnum.data.vowpal_wabbit_text_collection import VowpalWabbitTextCollection
from topnum.scores import (
    EntropyScore,
    IntratextCoherenceScore,
    PerplexityScore,
    SimpleTopTokensCoherenceScore,
    SophisticatedTopTokensCoherenceScore
)
from topnum.search_methods import (
    OptimizeScoresMethod,
    RenormalizationMethod
)
from topnum.search_methods.base_search_method import BaseSearchMethod
from topnum.search_methods.optimize_scores_method import _KEY_SCORE_RESULTS
from topnum.search_methods.renormalization_method import (
    ENTROPY_MERGE_METHOD,
    RANDOM_MERGE_METHOD,
    KL_MERGE_METHOD,
    PHI_RENORMALIZATION_MATRIX,
    THETA_RENORMALIZATION_MATRIX,
)


@pytest.mark.filterwarnings(f'ignore:{W_DIFF_BATCHES_1}')
class TestSearchMethods:
    text_collection_folder = None
    vw_file_name = 'collection_vw.txt'
    main_modality = '@main'
    other_modality = '@other'
    num_documents = 10
    num_words_in_document = 100
    vocabulary = None
    text_collection = None

    @classmethod
    def setup_class(cls):
        cls.text_collection_folder = tempfile.mkdtemp()

        vw_texts = cls.generate_vowpal_wabbit_texts()
        vw_file_path = os.path.join(cls.text_collection_folder, cls.vw_file_name)

        with open(vw_file_path, 'w') as f:
            f.write('\n'.join(vw_texts))

        cls.text_collection = VowpalWabbitTextCollection(
            vw_file_path,
            main_modality=cls.main_modality,
            modalities=[cls.main_modality, cls.other_modality]
        )

        cls.dataset = cls.text_collection._to_dataset()

    @classmethod
    def teardown_class(cls):
        cls.text_collection._remove_dataset()
        shutil.rmtree(cls.text_collection_folder)

    @classmethod
    def generate_vowpal_wabbit_texts(cls) -> List[str]:
        words = list(set([
            w.lower()
            for w in 'All work and no play makes Jack a dull boy'.split()
        ]))
        frequencies = list(range(1, 13))
        main_modality_num_words = int(0.7 * cls.num_words_in_document)
        other_modality_num_words = int(0.3 * cls.num_words_in_document)

        texts = list()

        cls.vocabulary = list()

        for document_index in range(cls.num_documents):
            text = ''
            text = text + f'doc_{document_index}'

            for modality_suffix, modality, num_words in zip(
                    ['m', 'o'],
                    [cls.main_modality, cls.other_modality],
                    [main_modality_num_words, other_modality_num_words]):

                text = text + f' |{modality}'

                for _ in range(num_words):
                    word = np.random.choice(words)
                    frequency = np.random.choice(frequencies)
                    token = f'{word}__{modality_suffix}'

                    cls.vocabulary.append(token)

                    text = text + f' {token}:{frequency}'

            texts.append(text)

        return texts

    def test_optimize_perplexity(self):
        score = PerplexityScore(
            'perplexity_score',
            class_ids=[self.main_modality, self.other_modality]
        )

        self._test_optimize_score(score)

    @pytest.mark.parametrize('entropy', ['renyi', 'shannon'])
    @pytest.mark.parametrize('threshold_factor', [1.0, 0.5, 1e-7, 1e7])
    def test_optimize_entropy(self, entropy, threshold_factor):
        score = EntropyScore(
            name='renyi_entropy',
            entropy=entropy,
            threshold_factor=threshold_factor
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

    def test_optimize_sophisticated_toptokens(self):
        score = SophisticatedTopTokensCoherenceScore(
            name='sophisticated_toptokens_coherence',
            data=self.dataset,
            documents=self.dataset._data.index[:1]
        )

        self._test_optimize_score(score, num_restarts=2)

    @pytest.mark.parametrize('what_modalities', ['None', 'one', 'many'])
    def test_optimize_simple_toptokens(self, what_modalities):
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
                combinations(self.vocabulary[:num_unique_words], 2)):

            cooccurrence_values[(w1, w2)] = i

        score = SimpleTopTokensCoherenceScore(
            name='simple_toptokens_coherence',
            cooccurrence_values=cooccurrence_values,
            data=self.dataset,
            modalities=modalities
        )

        self._test_optimize_score(score)

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

    def _test_optimize_score(self, score, num_restarts: int = 3) -> None:
        min_num_topics = 1
        max_num_topics = 5
        num_topics_interval = 2
        num_fit_iterations = 3

        optimizer = OptimizeScoresMethod(
            scores=[score],
            min_num_topics=min_num_topics,
            max_num_topics=max_num_topics,
            num_topics_interval=num_topics_interval,
            num_fit_iterations=num_fit_iterations,
            num_restarts=num_restarts
        )
        num_search_points = len(
            list(range(min_num_topics, max_num_topics + 1, num_topics_interval))
        )

        optimizer.search_for_optimum(self.text_collection)

        assert len(optimizer._result) == 1
        assert _KEY_SCORE_RESULTS in optimizer._result
        assert len(optimizer._result[_KEY_SCORE_RESULTS]) == 1
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
            assert isinstance(search_result[key], float)

        for key in optimizer._keys_std_one:
            assert key in search_result
            assert isinstance(search_result[key], float)

        for key in optimizer._keys_mean_many:
            assert key in search_result
            assert len(search_result[key]) == num_search_points
            assert all(isinstance(v, float) for v in search_result[key])

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
            assert all(isinstance(v, float) for v in search_result[key])
