from artm.scores import SparsityThetaScore as SparsityThetaArtmScore
from typing import List

from .base_default_score import BaseDefaultScore


class SparsityThetaScore(BaseDefaultScore):
    def __init__(
            self,
            name: str,
            topic_names: List[str] = None,
            eps: float = None):

        super().__init__(name)

        self._topic_names = topic_names
        self._eps = eps

        self._score = self._initialize()

    def _initialize(self) -> SparsityThetaArtmScore:
        return SparsityThetaArtmScore(
            name=self._name,
            topic_names=self._topic_names,
            eps=self._eps
        )
