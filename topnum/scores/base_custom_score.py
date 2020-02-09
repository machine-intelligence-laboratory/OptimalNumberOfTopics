from topicnet.cooking_machine.models import (
    BaseScore as BaseTopicNetScore,
    TopicModel
)
from .base_score import BaseScore


class BaseCustomScore(BaseScore):
    def __init__(self, name):
        super().__init__(name)

    def _attach_to_model(self, model: TopicModel):
        if self._name in model.custom_scores:
            print('Noo')
            # TODO: log

        # TODO: TopicModel should provide ability to add custom scores
        model.custom_scores[self._name] = self._score

    def _initialize_score(self) -> BaseTopicNetScore:
        raise NotImplementedError()
