from artm.scores import BaseScore as BaseArtmScore
from topicnet.cooking_machine.models import (
    BaseScore as BaseTopicNetScore,
    TopicModel
)
from typing import Union


class BaseScore:
    def __init__(self, name, higher_better=True):
        self._name = name
        self._higher_better = higher_better
        self._score = None

    @property
    def name(self):
        return self._name

    def _initialize(self) -> Union[BaseArtmScore, BaseTopicNetScore]:
        raise NotImplementedError()

    def _attach(self, model: TopicModel) -> None:
        raise NotImplementedError()
