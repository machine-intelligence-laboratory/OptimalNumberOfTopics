import logging
import numpy as np

from itertools import combinations
from topicnet.cooking_machine.dataset import Dataset
from topicnet.cooking_machine.models import TopicModel
from topicnet.cooking_machine.models.base_score import BaseScore as BaseTopicNetScore
from typing import (
    Dict,
    List,
    Tuple
)

from .base_custom_score import BaseCustomScore


AVERAGE_TYPE_MEAN = 'mean'
AVERAGE_TYPE_MEDIAN = 'median'


_logger = logging.getLogger()


class SimpleTopTokensCoherenceScore(BaseCustomScore):
    def __init__(
            self,
            name: str,
            cooccurrence_values: Dict[Tuple[str, str], float],
            dataset: Dataset,
            modality: str,
            topics: List[str],
            num_top_tokens: int = 10,
            kernel: bool = False,
            average: str = AVERAGE_TYPE_MEAN,
            active_topic_threshold: bool = None):
        # TODO: expand docstring
        """
        kernel — use only tokens from topic kernel
        active_topic_threshold — if defined, non active topics won't be considered

        """
        super().__init__(name)

        if average not in [AVERAGE_TYPE_MEAN, AVERAGE_TYPE_MEDIAN]:
            raise ValueError(f'average: {average}')

        self._cooc_values = cooccurrence_values
        self._dataset = dataset
        self._modality = modality
        self._topics = topics
        self._num_top_tokens = num_top_tokens
        self._kernel = kernel
        self._average = average
        self._active_topic_threshold = active_topic_threshold

        self._score = self._initialize()

    def _initialize(self) -> BaseTopicNetScore:
        return _TopTokensCoherenceScore(
            cooccurrence_values=self._cooc_values,
            dataset=self._dataset,
            modality=self._modality,
            topics=self._topics,
            kernel=self._kernel,
            average=self._average,
            active_topic_threshold=self._active_topic_threshold
        )


class _TopTokensCoherenceScore(BaseTopicNetScore):
    def __init__(
            self,
            cooccurrence_values: Dict[Tuple[str, str], float],
            dataset: Dataset,
            modality: str,
            topics: List[str],
            num_top_tokens: int = 10,
            kernel: bool = False,
            average: str = AVERAGE_TYPE_MEAN,
            active_topic_threshold: bool = None):

        super().__init__()

        self._cooc_values = cooccurrence_values
        self._dataset = dataset
        self._modality = modality
        self._topics = topics
        self._num_top_tokens = num_top_tokens
        self._kernel = kernel
        self._average = average
        self._active_topic_threshold = active_topic_threshold

    def call(self, model: TopicModel) -> float:
        subphi = model.get_phi().loc[self._modality, self._topics]
        vocabulary_size = subphi.shape[0]

        topic_coherences = list()

        if self._active_topic_threshold is None:
            topics = self._topics
        else:
            # TODO: can't do without transform here, cache theta didn't help
            theta = model._model.transform(self._dataset.get_batch_vectorizer())
            subtheta_values = theta.loc[self._topics, :].values
            max_probs = np.max(subtheta_values, axis=1)
            active_topic_indices = np.where(max_probs > self._active_topic_threshold)[0]
            topics = [t for i, t in enumerate(self._topics) if i in active_topic_indices]

        for topic in topics:
            topic_column = subphi.loc[:, topic]

            if not self._kernel:
                tokens = topic_column.sort_values(ascending=False)[
                         :self._num_top_tokens].index.to_list()
            else:
                # if self._num_top_tokens is None — also Ok
                tokens = topic_column[topic_column > 1.0 / vocabulary_size][
                         :self._num_top_tokens].index.to_list()

            current_cooc_values = list()

            for token_a, token_b in combinations(tokens, 2):
                if (token_a, token_b) in self._cooc_values:
                    current_cooc_values.append(self._cooc_values[(token_a, token_b)])
                elif (token_b, token_a) in self._cooc_values:
                    current_cooc_values.append(self._cooc_values[(token_b, token_a)])
                else:
                    _logger.warning(
                        f'Cooc pair "{token_a}, {token_b}" not found in the provided data!'
                        f' Using zero 0 for this pair as cooc value'
                    )

                    current_cooc_values.append(0)

            if len(current_cooc_values) > 0:
                topic_coherences.append(np.mean(current_cooc_values))
            else:
                topic_coherences.append(0)

        if len(topic_coherences) == 0:
            return 0
        elif self._average == AVERAGE_TYPE_MEAN:
            return float(np.mean(topic_coherences))
        elif self._average == AVERAGE_TYPE_MEDIAN:
            return float(np.median(topic_coherences))
        else:
            # Unlikely to ever happen
            raise ValueError(f'Don\'t know how to average like {self._average}')
