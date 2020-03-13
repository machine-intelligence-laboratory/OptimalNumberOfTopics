import warnings

from typing import (
    Dict,
    List,
    Tuple,
    Union
)

TokenType = Union[str, Tuple[str, str]]


class TopicBank:
    def __init__(self):
        self._topics: List[Union[Dict[TokenType, float], None]] = list()
        self._topic_scores: List[Union[Dict[str, float], None]] = list()

    @property
    def topics(self):
        return [t for t in self._topics if t is not None]

    @property
    def topic_scores(self):
        return [s for s in self._topic_scores if s is not None]

    def add_topic(
            self,
            topic: Dict[TokenType, float],
            scores: Dict[str, float]) -> None:

        self._topics.append(topic)
        self._topic_scores.append(scores)

    def delete_topic(self, index: int) -> None:
        if index < 0:
            raise ValueError(f'index: {index}')

        if index >= len(self._topics):
            warnings.warn(
                f'Index {index} is greater than the number of topics in the bank!'
            )

            return

        self._topics[index] = None
        self._topic_scores[index] = None
