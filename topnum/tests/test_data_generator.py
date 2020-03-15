import logging
import numpy as np
import os
import shutil
import tempfile

from typing import List

from topnum.data.vowpal_wabbit_text_collection import VowpalWabbitTextCollection
from topnum.search_methods.constants import DEFAULT_EXPERIMENT_DIR


class TestDataGenerator:
    text_collection_folder = None
    vw_file_name = 'collection_vw.txt'
    main_modality = '@main'
    other_modality = '@other'
    num_documents = 10
    num_words_in_document = 100
    vocabulary = None
    text_collection = None

    @classmethod
    def generate(cls):
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
    def clear(cls):
        cls.text_collection._remove_dataset()
        shutil.rmtree(cls.text_collection_folder)

        if os.path.isdir(DEFAULT_EXPERIMENT_DIR):
            shutil.rmtree(DEFAULT_EXPERIMENT_DIR)

    # TODO: DRY (same code in tests for optimize scores)
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
