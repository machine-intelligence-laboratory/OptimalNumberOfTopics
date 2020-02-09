from artm.scores import PerplexityScore as _PerplexityScore
from .base_artm_score import BaseArtmScore


class PerplexityScore(BaseArtmScore):
    def __init__(self, name, class_ids):
        super().__init__(name)

        self._class_ids = class_ids
        self._score = self._initialize_score()

    def _initialize_score(self) -> BaseArtmScore:
        # self._score = artm.scores.<SCORE>(...)

        return _PerplexityScore(
            name=self._name,
            class_ids=self._class_ids
        )