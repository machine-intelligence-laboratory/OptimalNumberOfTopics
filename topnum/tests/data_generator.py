import numpy as np
import os
import shutil
import tempfile

from typing import List

from topnum.data.vowpal_wabbit_text_collection import VowpalWabbitTextCollection
from topnum.search_methods.constants import DEFAULT_EXPERIMENT_DIR


class TestDataGenerator:
    def __init__(self):
        self.text_collection_folder = None
        self.vw_file_name = 'collection_vw.txt'
        self.main_modality = '@main'
        self.other_modality = '@other'
        self.num_documents = 10
        self.num_words_in_document = 100
        self.vocabulary = None
        self.text_collection: VowpalWabbitTextCollection = None

    def generate(self):
        self.text_collection_folder = tempfile.mkdtemp()

        vw_texts = self.generate_vowpal_wabbit_texts()
        vw_file_path = os.path.join(
            self.text_collection_folder,
            self.vw_file_name
        )

        with open(vw_file_path, 'w') as f:
            f.write('\n'.join(vw_texts))

        self.text_collection = VowpalWabbitTextCollection(
            vw_file_path,
            main_modality=self.main_modality,
            modalities=[self.main_modality, self.other_modality]
        )

        self.dataset = self.text_collection._to_dataset()

    def clear(self):
        self.text_collection._remove_dataset()
        shutil.rmtree(self.text_collection_folder)

        if os.path.isdir(DEFAULT_EXPERIMENT_DIR):
            shutil.rmtree(DEFAULT_EXPERIMENT_DIR)

    # TODO: DRY (same code in tests for optimize scores)
    def generate_vowpal_wabbit_texts(self) -> List[str]:
        words = list(set([
            w.lower()
            for w in 'All work and no play makes Jack a dull boy'.split()
        ]))
        frequencies = list(range(1, 13))
        main_modality_num_words = int(0.7 * self.num_words_in_document)
        other_modality_num_words = int(0.3 * self.num_words_in_document)

        texts = list()

        self.vocabulary = list()

        use_empty_text = True

        for document_index in range(self.num_documents):
            text = ''
            text = text + f'doc_{document_index}'

            if document_index == 0 and use_empty_text:
                text = text + f' |{self.main_modality}'
                texts.append(text)

                continue

            for modality_suffix, modality, num_words in zip(
                    ['m', 'o'],
                    [self.main_modality, self.other_modality],
                    [main_modality_num_words, other_modality_num_words]):

                text = text + f' |{modality}'

                for _ in range(num_words):
                    word = np.random.choice(words)
                    frequency = np.random.choice(frequencies)
                    token = f'{word}__{modality_suffix}'

                    self.vocabulary.append(token)

                    text = text + f' {token}:{frequency}'

            texts.append(text)

        return texts
