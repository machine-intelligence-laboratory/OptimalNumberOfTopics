import logging
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score

from topicnet.cooking_machine import Dataset
from topicnet.cooking_machine.models import (
    BaseScore as BaseTopicNetScore,
    TopicModel
)
from typing import (
    List,
    Tuple
)

from .base_custom_score import BaseCustomScore


logger = logging.getLogger()

'''

Uses of Silhouette:

http://www.cs.wm.edu/~denys/pubs/ICSE%2713-LDA-CRC.pdf
https://ieeexplore.ieee.org/document/7008665
https://arxiv.org/pdf/1808.08098.pdf

'''


def silhouette_score_by_sampling(X, labels, sample_size=10000, batches_number=20, **kwargs):
    """
    Calculate silhouette_score as mean value for several sampled batches

    Parameters
    ----------
    X : array [n_samples_a, n_samples_a]
        Array of pairwise distances between samples, or a feature array.

    labels : array, shape = [n_samples]
        Predicted labels for each sample.

    sample_size : int
        The size of the sample to use when computing the Silhouette Coefficient on
        a random subset of the data. If sample_size is None, no sampling is used.

    batches_number : int
        Number of batches for averaging

    Returns
    -------
    : float
        Silhouette_score
    """
    scores = []
    for i in range(batches_number):
        cur_score = silhouette_score(X, labels,
                                     sample_size=sample_size,
                                     random_state=i, **kwargs)
        if not np.isnan(cur_score):
            scores.append(cur_score)

    if np.sum(scores) + 1 < 1e-10:
        result = -1
    else:
        result = np.mean(scores)

    return result


class SilhouetteScore(BaseCustomScore):
    def __init__(
            self,
            name: str,
            validation_dataset: Dataset,
            sample_size: int = 10000,
            batches_number: int = 20
            ):

        super().__init__(name)

        self._score = _SilhouetteScore(validation_dataset, sample_size, batches_number)



class _SilhouetteScore(BaseTopicNetScore):
    def __init__(self, validation_dataset, sample_size=10000, batches_number=20):
        super().__init__()

        self.sample_size = sample_size
        self.batches_number = batches_number
        self.validation_dataset = validation_dataset

    def call(self, model: TopicModel):
        theta = model.get_theta(dataset=self.validation_dataset)

        theta.columns = range(len(theta.columns))
        objects_clusters = theta.values.argmax(axis=0)

        result = silhouette_score_by_sampling(
            theta.T.values, objects_clusters,
            sample_size=self.sample_size, batches_number=self.batches_number
        )

        return result


class CalinskiHarabaszScore(BaseCustomScore):
    def __init__(
            self,
            name: str,
            validation_dataset: Dataset
            ):

        super().__init__(name)

        self._score = _CalinskiHarabaszScore(validation_dataset)


class _CalinskiHarabaszScore(BaseTopicNetScore):
    def __init__(self, validation_dataset):
        super().__init__()
        self.validation_dataset = validation_dataset

    def call(self, model: TopicModel):
        theta = model.get_theta(dataset=self.validation_dataset)

        theta.columns = range(len(theta.columns))
        objects_clusters = theta.values.argmax(axis=0)

        calinski_harabasz_score_value = calinski_harabasz_score(theta.T.values, objects_clusters)

        return calinski_harabasz_score_value

