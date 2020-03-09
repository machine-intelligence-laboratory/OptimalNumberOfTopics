from artm.scores import BaseScore as BaseArtmScore
from topicnet.cooking_machine.models import TopicModel

from .base_score import BaseScore


class BaseDefaultScore(BaseScore):
    """Default aka implemented in BigARTM library
    """
    def __init__(self, name, higher_better=True):
        super().__init__(name, higher_better)

    def _attach(self, model: TopicModel):
        model._model.scores.add(self._score)

    def _initialize(self) -> BaseArtmScore:
        # return artm.scores.<SCORE>(...)

        raise NotImplementedError()
