from artm.scores import (
    BaseScore as BaseArtmScore,
    PerplexityScore as ArtmPerplexityScore,
)
from typing import List

from .base_default_score import BaseDefaultScore


class PerplexityScore(BaseDefaultScore):
    def __init__(self, name: str, class_ids: List[str] = None):
        super().__init__(name)

        self._class_ids = class_ids
        self._score = self._initialize()

    def _initialize(self) -> BaseArtmScore:
        return ArtmPerplexityScore(
            name=self._name,
            class_ids=self._class_ids
        )
