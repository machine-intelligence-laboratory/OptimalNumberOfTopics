import numpy as np

from typing import List

from topicnet.cooking_machine.models import (
    BaseScore as BaseTopicNetScore,
    TopicModel
)

from .base_custom_score import BaseCustomScore


class RateOfPerplexityChangeScore(BaseCustomScore):
    def __init__(self, name: str, perplexity_score_name: str):
        super().__init__(name, higher_better=False)

        self._perplexity_score_name = perplexity_score_name
        self._score = self._initialize()

    def _initialize(self) -> BaseTopicNetScore:
        return _RateOfPerplexityChangeScore(
            perplexity_score_name=self._perplexity_score_name
        )


class _RateOfPerplexityChangeScore(BaseTopicNetScore):
    def __init__(self, perplexity_score_name: str):
        super().__init__()

        self._perplexity_score_name = perplexity_score_name

    def call(self, model: TopicModel) -> float:
        """
        Assuming that perplexity score is computed on every iteration!
        """
        if self._perplexity_score_name not in model._model.scores.data:
            raise KeyError(
                f'Model model doesn\'t have'
                f' the perplexity score "{self._perplexity_score_name}"!'
            )

        perplexity_score_values = model.scores[self._perplexity_score_name]

        if len(perplexity_score_values) == 0:
            raise RuntimeError(
                f'Perplexity score "{self._perplexity_score_name}"'
                f' should be computed at least once!'
            )
        elif len(perplexity_score_values) == 1:
            return perplexity_score_values[-1] - float('+inf')
        else:
            # TODO: is there any way to get iteration number for each perplexity value?
            return perplexity_score_values[-1] - perplexity_score_values[-2]

    @classmethod
    def compute_by_range(
            cls,
            perplexity_score_values: List[float],
            iterations: List[int] = None) -> List[float]:
        """
        If `iterations` is `None`, each steps is assumed to be equal `1`.
        """

        if iterations is None:
            iterations = list(range(1, len(perplexity_score_values) + 1))

        assert iterations[0] > 0

        # TODO: optimize? can't append inf of nan in np.ediff1d
        result = [float('+inf')] + list(np.ediff1d(perplexity_score_values))
        result = np.array(result)

        iterations = [iterations[0]] + list(np.ediff1d(iterations))

        return list(
            result / iterations
        )
