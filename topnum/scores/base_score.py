import logging

from artm.scores import BaseScore as BaseArtmScore
from topicnet.cooking_machine.models import (
    BaseScore as BaseTopicNetScore,
    TopicModel
)
from typing import Union


_logger = logging.getLogger()


class BaseScore:
    def __init__(self, name, higher_better=True):
        self._name = name
        self._higher_better = higher_better
        self._score = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def _fullname(self) -> str:
        return f'{self.name}__{self._higher_better}'

    @staticmethod
    def _extract_name(fullname: str) -> str:
        return fullname.split('__')[0]

    @staticmethod
    def _is_higher_better(fullname: str) -> bool:
        if '__' not in fullname:
            _logger.warning(
                f'Seems there is no info about "higher-better"'
                f' in the score name "{fullname}".'
                f' Assuming, that higher-better is True'
            )

            return True

        higher_better_as_string = fullname.split('__')[1]

        return higher_better_as_string == f'{True}'

    def _initialize(self) -> Union[BaseArtmScore, BaseTopicNetScore]:
        raise NotImplementedError()

    def _attach(self, model: TopicModel) -> None:
        raise NotImplementedError()
