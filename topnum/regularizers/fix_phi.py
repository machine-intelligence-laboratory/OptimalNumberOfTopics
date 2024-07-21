from typing import List, Optional

import numpy as np
from numpy import ndarray
from pandas import DataFrame

from artm import ARTM
from topicnet.cooking_machine.models.topic_model import TopicModel
from topicnet.cooking_machine.models.base_regularizer import BaseRegularizer


class FastFixPhiRegularizer(BaseRegularizer):
    _VERY_BIG_TAU = 10 ** 9

    def __init__(
            self,
            name: str,
            topic_names: List[str],
            parent_model: Optional[TopicModel] = None,
            parent_phi: DataFrame = None,
            tau: float = _VERY_BIG_TAU,
            ):
        super().__init__(name, tau=tau)

        if parent_phi is None and parent_model is None:
            raise ValueError('Both parent Phi and parent model not specified.')

        self._topic_names = topic_names
        self._topic_indices = None
        self._parent_model = parent_model
        self._parent_phi = parent_phi

    def grad(self, pwt: DataFrame, nwt: DataFrame) -> ndarray:
        rwt = np.zeros_like(pwt)

        if self._parent_phi is not None:
            parent_phi = self._parent_phi
            vals = parent_phi.values
        else:
            parent_phi = self._parent_model.get_phi()
            vals = parent_phi.values[:, self._topic_indices]

        assert vals.shape[0] == rwt.shape[0]
        assert vals.shape[1] == len(self._topic_indices), (vals.shape[1], len(self._topic_indices))

        rwt[:, self._topic_indices] += vals

        return self.tau * rwt

    def attach(self, model: ARTM) -> None:
        super().attach(model)

        phi = self._model.get_phi()
        self._topic_indices = [
            phi.columns.get_loc(topic_name)
            for topic_name in self._topic_names
        ]
