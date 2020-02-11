from artm.scores import BaseScore as BaseArtmScore
from topicnet.cooking_machine.models import (
    BaseScore as BaseTopicNetScore,
    TopicModel
)
from typing import Union


class BaseScore:
    def __init__(self, name):
        self._name = name
        self._score = None

    def _initialize(self) -> Union[BaseArtmScore, BaseTopicNetScore]:
        raise NotImplementedError()

    def _attach(self, model: TopicModel) -> None:
        raise NotImplementedError()
