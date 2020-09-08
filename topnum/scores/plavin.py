import numpy as np
import scipy.stats as stats
import dill


from topicnet.cooking_machine import Dataset
from topicnet.cooking_machine.models import (
    BaseScore as BaseTopicNetScore,
    TopicModel
)
from typing import (
    List
)

from .base_custom_score import BaseCustomScore
from .base_custom_score import __NO_LOADING_DATASET__

from .dataset_utils import col_total_len, compute_document_details


def _compute_kl(T, theta, doc_lengths):
    uniform_distrib = np.ones(T) / T

    doc_lengths = doc_lengths / sum(doc_lengths)
    theta_distrib = theta.dot(doc_lengths)

    # TODO: dtype was 'object'? how could it be?
    theta_distrib = np.array(theta_distrib.values, dtype=np.float)

    return stats.entropy(uniform_distrib, theta_distrib)


class UniformThetaDivergenceScore(BaseCustomScore):
    """
    svn.code.sf.net/p/mlalgorithms/code/Group174/Plavin2015TopicSelection/doc/Plavin2015Diploma.pdf
    """

    def __init__(
            self,
            name: str,
            validation_dataset: Dataset,
            modalities: List
            ):

        super().__init__(name)

        self._score = _UniformThetaDivergenceScore(validation_dataset, modalities)


class _UniformThetaDivergenceScore(BaseTopicNetScore):
    def __init__(self, validation_dataset, modalities):
        super().__init__()

        self._dataset = validation_dataset
        document_length_stats = compute_document_details(validation_dataset, modalities)

        self.document_lengths = sum(document_length_stats[col_total_len(m)] for m in modalities)
        self.modalities = modalities
        self._keep_dataset_in_memory = validation_dataset._small_data
        self._dataset_internals_folder_path = validation_dataset._internals_folder_path
        self._dataset_file_path = validation_dataset._data_path

    def call(self, model: TopicModel):
        theta = model.get_theta(dataset=self._dataset)
        T = theta.shape[0]

        return _compute_kl(T, theta, self.document_lengths)

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

        if __NO_LOADING_DATASET__[0]:
            score._dataset = None
        else:
            score._dataset = Dataset(
                score._dataset_file_path,
                internals_folder_path=score._dataset_internals_folder_path,
                keep_in_memory=score._keep_dataset_in_memory,
            )

        return score
