from artm.scores import PerplexityScore as ArtmPerplexityScore
from .base_default_score import BaseDefaultScore


class PerplexityScore(BaseDefaultScore):
    def __init__(self, name, class_ids):
        super().__init__(name)

        self._class_ids = class_ids
        self._score = self._initialize_score()

    def _initialize_score(self) -> BaseDefaultScore:
        # self._score = artm.scores.<SCORE>(...)

        return ArtmPerplexityScore(
            name=self._name,
            class_ids=self._class_ids
        )