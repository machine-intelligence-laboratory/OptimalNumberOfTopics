from topicnet.cooking_machine import Dataset

from topnum.data import VowpalWabbitTextCollection
from topnum.tests.data_generator import TestDataGenerator


class TestData:
    data_generator = None

    dataset = None
    main_modality = None
    other_modality = None
    text_collection = None

    @classmethod
    def setup_class(cls):
        cls.data_generator = TestDataGenerator()

        cls.data_generator.generate()

        cls.text_collection = cls.data_generator.text_collection
        cls.main_modality = cls.data_generator.main_modality
        cls.other_modality = cls.data_generator.other_modality

        cls.dataset = cls.text_collection._to_dataset()

    @classmethod
    def teardown_class(cls):
        if cls.data_generator is not None:
            cls.data_generator.clear()

    def test_vw_collection(self):
        text_collection = VowpalWabbitTextCollection(
            file_path=self.text_collection._file_path,
            main_modality=self.main_modality,
            modalities=[self.main_modality],
        )

        assert isinstance(text_collection._to_dataset(), Dataset)
        assert text_collection._main_modality == self.main_modality
        assert len(text_collection._modalities) == 1
        assert self.main_modality in text_collection._modalities

    def test_vw_collection_from_dataset(self):
        text_collection = VowpalWabbitTextCollection.from_dataset(
            self.dataset,
            main_modality=self.main_modality,
            modalities=[self.main_modality, self.other_modality],
        )

        assert text_collection._to_dataset() is self.dataset
        assert text_collection._main_modality == self.main_modality
        assert len(text_collection._modalities) == 2
        assert self.main_modality in text_collection._modalities
        assert self.other_modality in text_collection._modalities
