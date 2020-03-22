import logging
import numpy as np

from itertools import combinations
from topicnet.cooking_machine.dataset import Dataset
from topicnet.cooking_machine.models import TopicModel
from topicnet.cooking_machine.models.base_score import BaseScore as BaseTopicNetScore
from typing import (
    Dict,
    List,
    Tuple,
    Union
)

from .base_topic_score import BaseTopicScore
from ..data.vowpal_wabbit_text_collection import VowpalWabbitTextCollection


AVERAGE_TYPE_MEAN = 'mean'
AVERAGE_TYPE_MEDIAN = 'median'


_logger = logging.getLogger()


class SimpleTopTokensCoherenceScore(BaseTopicScore):
    def __init__(
            self,
            name: str,
            cooccurrence_values: Dict[Tuple[str, str], float],
            data: Union[Dataset, VowpalWabbitTextCollection],
            modalities: Union[str, List[str], None] = None,
            topics: List[str] = None,
            num_top_tokens: int = 10,
            kernel: bool = True,
            average: str = AVERAGE_TYPE_MEAN,
            active_topic_threshold: bool = None):
        # TODO: expand docstring
        """
        Parameters
        ----------
        cooccurrence_values
            Word cooccurence values:
            the bigger the number corresponding to words (w1, w2),
            the more the confidence that the occurrence of these words
            together in text is non-random.
            The cooc values should be in a certain format, for example:
            >>> {
            >>>     ('dipper', 'mabel'): 112.7,
            >>>     ('mabel', 'monotony'): 0.2,
            >>>     ...
            >>> }
        data
            Text document collection
        modalities
            Modalities for which words the coherence is to be calculated
            If not specified (value None), all the modalities will be used
        topics
            Topics to calculate coherence for
        num_top_tokens
            Number of top tokens for coherence calculation
        kernel
            If true, only tokens from topic kernel will be used
        active_topic_threshold
            If defined, non-active topics won't be considered
        average
            How to average coherences over topics to get the final score result
        """
        super().__init__(name)

        if average not in [AVERAGE_TYPE_MEAN, AVERAGE_TYPE_MEDIAN]:
            raise ValueError(f'average: {average}')

        self._cooc_values = cooccurrence_values
        self._data = data

        if modalities is None:
            pass
        elif isinstance(modalities, str):
            modalities = [modalities]
        elif isinstance(modalities, list) and len(modalities) == 0:
            modalities = None
        elif not isinstance(modalities, list):
            raise TypeError(f'modalities: {modalities}')

        self._modalities = modalities
        self._topics = topics
        self._num_top_tokens = num_top_tokens
        self._kernel = kernel
        self._average = average
        self._active_topic_threshold = active_topic_threshold

        self._score = self._initialize()

    def _initialize(self) -> BaseTopicNetScore:
        if isinstance(self._data, Dataset):
            dataset = self._data
        else:
            dataset = self._data._to_dataset()

        return _TopTokensCoherenceScore(
            cooccurrence_values=self._cooc_values,
            dataset=dataset,
            modalities=self._modalities,
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
            modalities: Union[str, List[str], None] = None,
            topics: List[str] = None,
            num_top_tokens: int = 10,
            kernel: bool = False,
            average: str = AVERAGE_TYPE_MEAN,
            active_topic_threshold: bool = None):

        super().__init__()

        self._cooc_values = cooccurrence_values
        self._dataset = dataset
        self._modalities = modalities
        self._topics = topics
        self._num_top_tokens = num_top_tokens
        self._kernel = kernel
        self._average = average
        self._active_topic_threshold = active_topic_threshold

    def call(self, model: TopicModel) -> float:  # not BaseModel
        topic_coherences = self.compute(model)
        coherence_values = list(topic_coherences.values())

        if len(coherence_values) == 0:
            return 0
        elif self._average == AVERAGE_TYPE_MEAN:
            return float(np.mean(coherence_values))
        elif self._average == AVERAGE_TYPE_MEDIAN:
            return float(np.median(coherence_values))
        else:
            # Unlikely to ever happen
            raise ValueError(f'Don\'t know how to average like {self._average}')

    def compute(
            self,
            model: TopicModel) -> Dict[str, float]:  # not BaseModel, because need access to ._model

        phi = model.get_phi()

        if self._topics is not None:
            topics = self._topics
        else:
            topics = list(phi.columns)

        if self._modalities is not None:
            # As self._modalities is list, here always will be df with multiIndex
            subphi = model.get_phi().loc[self._modalities, topics]
        else:
            subphi = model.get_phi().loc[:, topics]

        vocabulary_size = subphi.shape[0]

        topic_coherences = dict()

        if self._active_topic_threshold is None:
            pass
        else:
            # TODO: can't do without transform here, cache theta didn't help
            theta = model._model.transform(self._dataset.get_batch_vectorizer())
            subtheta_values = theta.loc[topics, :].values
            max_probs = np.max(subtheta_values, axis=1)
            active_topic_indices = np.where(max_probs > self._active_topic_threshold)[0]
            topics = [t for i, t in enumerate(topics) if i in active_topic_indices]

        for topic in topics:
            topic_column = subphi.loc[:, topic]

            if not self._kernel:
                tokens = topic_column\
                    .sort_values(ascending=False)[:self._num_top_tokens]\
                    .index\
                    .get_level_values(1)\
                    .to_list()
            else:
                # if self._num_top_tokens is None — also Ok
                tokens = topic_column[topic_column > 1.0 / vocabulary_size][:self._num_top_tokens]\
                    .index\
                    .get_level_values(1)\
                    .to_list()

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
                topic_coherences[topic] = float(np.mean(current_cooc_values))
            else:
                # TODO: warn?
                topic_coherences[topic] = 0.0

        return topic_coherences
