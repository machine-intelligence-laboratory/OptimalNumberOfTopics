from topicnet.cooking_machine.models.base_model import BaseModel
from topicnet.cooking_machine.models.base_score import BaseScore as TopicNetBaseScore
from typing import (
    Dict,
    List
)

from .base_custom_score import BaseCustomScore


class BaseTopicScore(BaseCustomScore):
    def __init__(self, name, higher_better: bool = True):
        super().__init__(name, higher_better)

    def _initialize(self) -> TopicNetBaseScore:
        raise NotImplementedError()

    def compute(
            self,
            model: BaseModel,
            topics: List[str] = None,
            documents: List[str] = None) -> Dict[str, float]:

        raise NotImplementedError()
