import numpy as np
import os
import shutil
import tempfile
from typing import List

from topnum.data.vowpal_wabbit_text_collection import VowpalWabbitTextCollection
from topnum.scores.perplexity_score import PerplexityScore
from topnum.search_methods.optimize_score_method import OptimizeScoreMethod


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

    # TODO: eliminate warning about different batches
    def test_optimize_perplexity(self):
        score = PerplexityScore(
            'perplexity_score',
            class_ids=[self.main_modality, self.other_modality]
        )

        min_num_topics = 1
        max_num_topics = 10
        num_topics_interval = 2

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

        # TODO: constants
        assert 'optimum' in result
        assert isinstance(result['optimum'], int)

        assert 'optimum_std' in result
        assert isinstance(result['optimum_std'], float)

        for result_key in ['score_values', 'score_values_std']:
            assert result_key in result
            assert len(result[result_key]) == num_points
            assert all(isinstance(v, float) for v in result[result_key])
