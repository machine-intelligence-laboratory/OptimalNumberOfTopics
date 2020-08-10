import dill

from artm.scores import PerplexityScore
from topicnet.cooking_machine import Dataset
from topicnet.cooking_machine.models import (
    BaseScore as BaseTopicNetScore,
    TopicModel
)
from typing import List

from .base_custom_score import BaseCustomScore


class HoldoutPerplexityScore(BaseCustomScore):
    def __init__(self, name: str, test_dataset: Dataset, class_ids: List[str] = None):
        super().__init__(name, higher_better=False)

        self._test_dataset = test_dataset
        self._class_ids = class_ids

        self._score = self._initialize()

    def _initialize(self) -> BaseTopicNetScore:
        return _HoldoutPerplexityScore(
            base_score_name=f'_{self._name}',
            test_dataset=self._test_dataset,
            class_ids=self._class_ids,
        )


class _HoldoutPerplexityScore(BaseTopicNetScore):
    def __init__(
            self,
            base_score_name: str,
            test_dataset: Dataset,
            class_ids: List[str] = None):

        super().__init__()

        self._base_score_name = base_score_name
        self._dataset = test_dataset
        self._class_ids = class_ids

        self._call_number = 0

    def call(self, model: TopicModel) -> float:
        self._call_number = self._call_number + 1

        # TODO: maybe possible with only one score?
        score_name = f'{self._base_score_name}_{self._call_number}'

        assert score_name not in model._model.scores.data

        model._model.scores.add(
            PerplexityScore(
                name=score_name,
                class_ids=self._class_ids,
            )
        )

        model._model.transform(
            batch_vectorizer=self._dataset.get_batch_vectorizer(),
            theta_matrix_type=None
        )

        perplexity = model._model.get_score(score_name)

        return perplexity.value

    # TODO: this piece is copy-pastd among four different scores
    def save(self, path: str) -> None:
        dataset = self._dataset
        self._dataset = None

        with open(path, 'wb') as f:
            dill.dump(self, f)

        self._dataset = dataset

    @classmethod
    def load(cls, path: str):
        """

        Parameters
        ----------
        path

        Returns
        -------
        an instance of this class

        """

        with open(path, 'rb') as f:
            score = dill.load(f)

        score._dataset = Dataset(
            score._dataset_file_path,
            internals_folder_path=score._dataset_internals_folder_path,
            keep_in_memory=score._keep_dataset_in_memory,
        )

        return score
