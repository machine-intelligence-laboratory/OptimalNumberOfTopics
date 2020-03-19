import logging
from topicnet.cooking_machine.models import (
    BaseScore as BaseTopicNetScore,
    TopicModel
)

from .base_score import BaseScore


_logger = logging.getLogger()


class BaseCustomScore(BaseScore):
    def __init__(self, name, higher_better=True):
        super().__init__(name, higher_better)

    def _attach(self, model: TopicModel):
        if self._name in model.custom_scores:
            _logger.warning(
                f'Score with such name "{self._name}" already attached to model!'
                f' So rewriting it...'
                f' All model\'s custom scores: {list(model.custom_scores.keys())}'
            )

        # TODO: TopicModel should provide ability to add custom scores
        model.custom_scores[self.name] = self._score

    def _initialize(self) -> BaseTopicNetScore:
        raise NotImplementedError()
