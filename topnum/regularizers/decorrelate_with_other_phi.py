from typing import List, Optional

import numpy as np
from numpy import ndarray
from pandas import DataFrame
from scipy.spatial.distance import cdist

from artm import ARTM
from topicnet.cooking_machine.models.base_regularizer import BaseRegularizer


class DecorrelateWithOtherPhiRegularizer(BaseRegularizer):
    def __init__(
            self,
            name: str,
            tau: float,
            topic_names: List[str],
            other_phi: DataFrame,
            ):
        super().__init__(name, tau=tau)

        self._topic_names = topic_names
        self._other_phi = other_phi
        self._other_topic_sum = self._other_phi.values.sum(
            axis=1, keepdims=True
        )

        self._topic_indices = None

    def grad(self, pwt: DataFrame, nwt: DataFrame) -> ndarray:
        rwt = np.zeros_like(pwt)
        rwt[:, self._topic_indices] += (
            pwt.values[:, self._topic_indices] * self._other_topic_sum
        )

        return -1 * self.tau * rwt

    def attach(self, model: ARTM) -> None:
        super().attach(model)

        phi = model.get_phi()
        self._topic_indices = [
            phi.columns.get_loc(topic_name)
            for topic_name in self._topic_names
        ]


class DecorrelateWithOtherPhiRegularizer2(BaseRegularizer):
    def __init__(
            self,
            name: str,
            tau: float,
            topic_names: List[str],
            other_phi: DataFrame,
            num_iters: Optional[int] = None,
            ):
        super().__init__(name, tau=tau)

        self._topic_names = topic_names
        self._other_phi = other_phi
        self._num_iters = num_iters
        self._cur_iter = 0

        self._topic_indices = None

    def grad(self, pwt: DataFrame, nwt: DataFrame) -> ndarray:
        rwt = np.zeros_like(pwt)

        if self._num_iters is not None and self._cur_iter >= self._num_iters:
            return rwt

        correlations = cdist(
            self._other_phi.values.T,
            pwt.values[:, self._topic_indices].T,
            lambda u, v: (u * v).sum()
        )
        weighted_other_topics = self._other_phi.values.dot(correlations)

        rwt[:, self._topic_indices] += (
                pwt.values[:, self._topic_indices] * weighted_other_topics
        )
        self._cur_iter += 1

        return -1 * self.tau * rwt

    def attach(self, model: ARTM) -> None:
        super().attach(model)

        phi = model.get_phi()
        self._topic_indices = [
            phi.columns.get_loc(topic_name)
            for topic_name in self._topic_names
        ]
