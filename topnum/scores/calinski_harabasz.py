import logging
import numpy as np
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


class CalinskiHarabaszScore(BaseCustomScore):
    '''
    Uses of Calinski-Harabasz:

    https://link.springer.com/article/10.1007/s40815-017-0327-9
    '''
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

        return calinski_harabasz_score(theta.T.values, objects_clusters)

