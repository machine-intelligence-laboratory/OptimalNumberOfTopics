from artm.scores import SparsityPhiScore as SparsityPhiArtmScore
from typing import List

from .base_default_score import BaseDefaultScore


class SparsityPhiScore(BaseDefaultScore):
    def __init__(
            self,
            name: str,
            class_id: str = None,
            topic_names: List[str] = None,
            eps: float = None):

        super().__init__(name)

        self._class_id = class_id
        self._topic_names = topic_names
        self._eps = eps

        self._score = self._initialize()

    def _initialize(self) -> SparsityPhiArtmScore:
        return SparsityPhiArtmScore(
            name=self._fullname,
            class_id=self._class_id,
            topic_names=self._topic_names,
            eps=self._eps
        )
