import logging

from artm.scores import BaseScore as BaseArtmScore
from topicnet.cooking_machine.models import (
    BaseScore as BaseTopicNetScore,
    TopicModel
)
from typing import Union


_logger = logging.getLogger()


class BaseScore:
    def __init__(self, name, higher_better: bool = True):
        self._name = name
        self._higher_better = higher_better
        self._score = None

    @property
    def name(self) -> str:
        return self._name

    def _initialize(self) -> Union[BaseArtmScore, BaseTopicNetScore]:
        raise NotImplementedError()

    def _attach(self, model: TopicModel) -> None:
        raise NotImplementedError()
