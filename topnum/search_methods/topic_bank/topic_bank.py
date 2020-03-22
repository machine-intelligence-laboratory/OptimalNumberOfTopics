import dill
import logging
import os
import pandas as pd
import shutil
import tempfile
import warnings

from topicnet.cooking_machine.models import TopicModel
from typing import (
    Dict,
    List,
    Tuple,
    Union
)


_logger = logging.getLogger()


TokenType = Union[str, Tuple[str, str]]


class TopicBank:
    def __init__(
            self,
            save: bool = True,
            save_folder_path: str = None,
            num_changes_for_save: int = 1):

        self._save = save

        if save_folder_path is None:
            self._path = tempfile.mkdtemp(suffix='TopicBank_')
        elif os.path.isdir(save_folder_path):
            self._path = save_folder_path
            self.load()
        else:
            raise NotADirectoryError(f'save_folder_path: {save_folder_path}')

        self._topics: List[Union[Dict[TokenType, float], None]] = list()
        self._topic_scores: List[Union[Dict[str, float], None]] = list()

        self._num_changes_for_save = num_changes_for_save
        self._num_changes = 0

    @property
    def topics(self) -> List[Union[Dict[TokenType, float], None]]:
        return [t for t in self._topics if t is not None]

    @property
    def topic_scores(self) -> List[Union[Dict[str, float], None]]:
        return [s for s in self._topic_scores if s is not None]

    def add_topic(
            self,
            topic: Dict[TokenType, float],
            scores: Dict[str, float]) -> None:

        self._topics.append(topic)
        self._topic_scores.append(scores)

        self._save_if_its_time()

    def delete_topic(self, index: int) -> None:
        _logger.debug(
            f'Deleting topic number {index}.'
            f' Number of topics in bank: {len(self._topics)}'
        )

        if index < 0:
            raise ValueError(f'index: {index}')

        if index >= len(self._topics):
            warnings.warn(
                f'Index {index} is greater than the number of topics in the bank!'
            )

            return

        self._topics[index] = None
        self._topic_scores[index] = None

        self._save_if_its_time()

    def view_topics(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            {
                f'topic_{i}': word_probs
                for i, word_probs in enumerate(self.topics)
            }
        )

    def view_topic_scores(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            {
                f'topic_{i}': topic_scores
                for i, topic_scores in enumerate(self.topic_scores)
            }
        )

    def save_model_topics(
            self,
            name: str,
            model: TopicModel,
            topic_scores: List[Dict[str, float]] = None,
            phi: pd.DataFrame = None) -> None:

        if phi is None:
            phi = model.get_phi()

        with open(os.path.join(self._path, f'{name}__phi.bin'), 'wb') as f:
            f.write(dill.dumps(phi))

        if topic_scores is None:
            topic_scores = dict()

        with open(os.path.join(self._path, f'{name}__topic_scores.bin'), 'wb') as f:
            f.write(dill.dumps(topic_scores))

    def save(self) -> None:
        # TODO: maybe too slow if bank is big and num_changes is one!
        with open(os.path.join(self._path, 'topics.bin'), 'wb') as f:
            f.write(dill.dumps(self._topics))
        with open(os.path.join(self._path, 'topic_scores.bin'), 'wb') as f:
            f.write(dill.dumps(self._topic_scores))

    def load(self) -> None:
        file_path = os.path.join(self._path, 'topics.bin')

        if os.path.isfile(file_path):
            with open(file_path, 'rb') as f:
                self._topics = dill.loads(f.read())

        file_path = os.path.join(self._path, 'topic_scores.bin')

        if os.path.isfile(file_path):
            with open(file_path, 'rb') as f:
                self._topic_scores = dill.loads(f.read())

    def clear(self) -> None:
        del self._topics
        del self._topic_scores

        self._topics = list()
        self._topic_scores = list()

        self.save()

    def eliminate(self) -> None:
        self.clear()

        if os.path.isdir(self._path):
            shutil.rmtree(self._path)
            self._path = None

    def _save_if_its_time(self) -> None:
        if not self._save:
            return

        self._num_changes += 1

        if self._num_changes % self._num_changes_for_save == 0:
            self.save()
