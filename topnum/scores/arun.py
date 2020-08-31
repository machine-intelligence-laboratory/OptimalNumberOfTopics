import dill
import numpy as np
import scipy.stats as stats
import warnings


from topicnet.cooking_machine import Dataset
from topicnet.cooking_machine.models import (
    BaseScore as BaseTopicNetScore,
    TopicModel,
)
from typing import (
    List
)

from .base_custom_score import BaseCustomScore
from .dataset_utils import col_total_len, compute_document_details


def _symmetric_kl(distrib_p, distrib_q):
    return 0.5 * np.sum([stats.entropy(distrib_p, distrib_q), stats.entropy(distrib_p, distrib_q)])


class SpectralDivergenceScore(BaseCustomScore):
    """
    Implements Arun metric to estimate the optimal number of topics:
    Arun, R., V. Suresh, C. V. Madhavan, and M. N. Murthy
    On finding the natural number of topics with latent dirichlet allocation: Some observations.
    In PAKDD (2010), pp. 391–402.


    The code is based on analagous code from TOM:
    https://github.com/AdrienGuille/TOM/blob/388c71ef/tom_lib/nlp/topic_model.py
    """

    def __init__(
            self,
            name: str,
            validation_dataset: Dataset,
            modalities: List
            ):

        super().__init__(name)

        self._score = _SpectralDivergenceScore(validation_dataset, modalities)


class _SpectralDivergenceScore(BaseTopicNetScore):
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
        phi = model.get_phi(class_ids=self.modalities)

        c_m1 = np.linalg.svd(phi, compute_uv=False)
        c_m2 = self.document_lengths.dot(theta.T)
        c_m2 += 0.0001  # we need this to prevent components equal to zero

        if len(c_m1) != phi.shape[1]:
            warnings.warn(
                f'Phi has {phi.shape[1]} topics'
                f' but its SVD resulted in a vector of size {len(c_m1)}!'
                f' To work correctly, SpectralDivergenceScore expects to get a vector'
                f' of exactly {phi.shape[1]} singular values.'
            )

            return 1.0

        # we do not need to normalize these vectors
        return _symmetric_kl(c_m1, c_m2)

    # TODO: this piece is copy-pastd among three different scores
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
