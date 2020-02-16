import numpy as np
import os
import pytest
import shutil
import tempfile
import warnings
from topicnet.cooking_machine.dataset import W_DIFF_BATCHES_1
from typing import List

from topnum.data.vowpal_wabbit_text_collection import VowpalWabbitTextCollection
from topnum.scores import (
    PerplexityScore,
    EntropyScore
)
from topnum.search_methods import (
    OptimizeScoreMethod,
    RenormalizationMethod
)


@pytest.mark.filterwarnings(f'ignore:{W_DIFF_BATCHES_1}')
class TestSearchMethods:
    text_collection_folder = None
    vw_file_name = 'collection_vw.txt'
    main_modality = '@main'
    other_modality = '@other'
    num_documents = 10
    num_words_in_document = 100
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
                    text = text + f' {word}__{modality_suffix}:{frequency}'

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

    @pytest.mark.parametrize('merge_method', ['entropy', 'random', 'kl'])
    @pytest.mark.parametrize('threshold_factor', [1.0, 0.5, 1e-7, 1e7])
    def test_renormalize(self, merge_method, threshold_factor):
        max_num_topics = 10
        tiny = 1e-7

        optimizer = RenormalizationMethod(
            merge_method=merge_method,
            threshold_factor=threshold_factor,
            max_num_topics=max_num_topics,
            num_collection_passes=10,
            num_restarts=3
        )

        # TODO: make clearer
        num_points = max_num_topics - 1

        optimizer.search_for_optimum(self.text_collection)
        result = optimizer._result

        assert optimizer._key_optimum in result
        assert isinstance(result[optimizer._key_optimum], int)

        assert optimizer._key_optimum_std in result
        assert isinstance(result[optimizer._key_optimum_std], float)

        assert optimizer._key_nums_topics in result
        assert isinstance(result[optimizer._key_nums_topics], list)
        assert all(
            abs(v - int(v)) == 0
            for v in result[optimizer._key_nums_topics]
        )

        for result_key in [
                optimizer._key_renyi_entropy_values,
                optimizer._key_renyi_entropy_values_std,
                optimizer._key_shannon_entropy_values,
                optimizer._key_shannon_entropy_values_std,
                optimizer._key_energy_values,
                optimizer._key_energy_values_std]:

            assert result_key in result
            assert len(result[result_key]) == num_points
            assert all(isinstance(v, float) for v in result[result_key])

        for result_key in [
            optimizer._key_renyi_entropy_values,
            optimizer._key_shannon_entropy_values,
            optimizer._key_energy_values]:

            if not any(abs(v) > tiny for v in result[result_key]):
                warnings.warn(f'All score values "{result_key}" are zero!')

    def _test_optimize_score(self, score):
        min_num_topics = 1
        max_num_topics = 10
        num_topics_interval = 2
        tiny = 1e-7

        optimizer = OptimizeScoreMethod(
            score=score,
            min_num_topics=min_num_topics,
            max_num_topics=max_num_topics,
            num_topics_interval=num_topics_interval,
            num_collection_passes=10,
            num_restarts=3
        )

        num_points = len(list(range(min_num_topics, max_num_topics + 1, num_topics_interval)))

        optimizer.search_for_optimum(self.text_collection)
        result = optimizer._result

        assert optimizer._key_optimum in result
        assert isinstance(result[optimizer._key_optimum], int)

        assert optimizer._key_optimum_std in result
        assert isinstance(result[optimizer._key_optimum_std], float)

        for result_key in [
                optimizer._key_score_values,
                optimizer._key_score_values_std]:

            assert result_key in result
            assert len(result[result_key]) == num_points
            assert all(isinstance(v, float) for v in result[result_key])

            if (result_key == optimizer._key_score_values
                and not any(abs(v) > tiny for v in result[result_key])):

                warnings.warn(f'All score values "{result_key}" are zero!')
