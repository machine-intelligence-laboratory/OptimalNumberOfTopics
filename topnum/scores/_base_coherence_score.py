import dill
import logging
import numpy as np
import pandas as pd
import sys
import tqdm

from collections import defaultdict
from enum import (
    auto,
    Enum,
    IntEnum
)
from topicnet.cooking_machine.dataset import (
    Dataset,
    VW_TEXT_COL,
    RAW_TEXT_COL,
    DEFAULT_ARTM_MODALITY,
    MODALITY_START_SYMBOL
)
from topicnet.cooking_machine.models.base_model import BaseModel
from topicnet.cooking_machine.models.base_score import BaseScore as TopicNetBaseScore
from typing import (
    Dict,
    List,
    Tuple,
    Union
)

from .base_custom_score import BaseCustomScore
from ..data.vowpal_wabbit_text_collection import VowpalWabbitTextCollection


_logger = logging.getLogger()


# TODO: some parts below are borrowed from TopicNet

# This is all for TopTokensCoherenceScore (which currently is not in TopicNet)
# The score as long as IntratextCoherenceScore inherits from the BaseCoherenceScore defined here


WordType = Tuple[str, str]  # modality + word


class TextType(Enum):
    VW_TEXT = VW_TEXT_COL
    RAW_TEXT = RAW_TEXT_COL


class WordTopicRelatednessType(IntEnum):
    """
    Word-topic relatedness estimate

    Attributes
    ----------
        PWT
            p(w | t)
        PTW
            p(t | w)
    """
    PWT = auto()
    PTW = auto()


class SpecificityEstimationMethod(IntEnum):
    """
    Way to estimate how particular word is specific for particular topic.
    Unlike probability, eg. p(w | t), specificity_estimation takes into account
    values for all topics, eg. p(w | t_1), p(w | t_2), ..., p(w | t_n):
    the higher the value p(w | t) comparing other p(w | t_i),
    the higher the specificity_estimation of word "w" for the topic "t"

    Attributes
    ----------
        NONE
            Don't try to estimate specificity_estimation, return the probability as is
        MAXIMUM
            From probability, corresponding to word and topic,
            extract *maximum* among probabilities for the word and other topics
        AVERAGE
            From probability, corresponding to word and topic,
            extract *average* among probabilities for the word and other topics
    """
    NONE = auto()
    MAXIMUM = auto()
    AVERAGE = auto()


class _BaseCoherenceScore(TopicNetBaseScore):
    def __init__(
            self,
            dataset: Dataset,
            documents: List[str] = None,
            text_type: TextType = TextType.VW_TEXT,
            word_topic_relatedness: WordTopicRelatednessType = WordTopicRelatednessType.PWT,
            specificity_estimation: SpecificityEstimationMethod = SpecificityEstimationMethod.NONE,
            verbose: bool = False,
    ):
        super().__init__()

        if not isinstance(dataset, Dataset):
            raise TypeError(
                f'Got "{type(dataset)}" as \"dataset\". Expect it to derive from "Dataset"')

        if not isinstance(text_type, TextType):
            raise TypeError(
                f'Wrong "text_type": \"{text_type}\". '
                f'Expect to be \"{TextType}\"')

        if not isinstance(word_topic_relatedness, WordTopicRelatednessType):
            raise TypeError(
                f'Wrong "word_topic_relatedness": \"{word_topic_relatedness}\". '
                f'Expect to be \"{WordTopicRelatednessType}\"')

        if not isinstance(specificity_estimation, SpecificityEstimationMethod):
            raise TypeError(
                f'Wrong "specificity_estimation": \"{specificity_estimation}\". '
                f'Expect to be \"{SpecificityEstimationMethod}\"')

        self._dataset = dataset
        self._text_type = text_type
        self._word_topic_relatedness = word_topic_relatedness
        self._specificity_estimation_method = specificity_estimation

        self._verbose = verbose

        if documents is not None:
            self._documents = documents
        else:
            self._documents = list(self._dataset.get_dataset().index)

        self._dataset_file_path = dataset._data_path
        self._dataset_internals_folder_path = dataset._internals_folder_path
        self._keep_dataset_in_memory = dataset._small_data

    def call(self, model: BaseModel) -> float:
        topic_coherences = self.compute(model, None)

        coherence_values = list(
            v if v is not None else 0.0  # TODO: state the behavior clearer somehow
            for v in topic_coherences.values()
        )

        return float(np.median(coherence_values))  # TODO: or mean?

    def compute(
            self,
            model: BaseModel,
            topics: List[str] = None,
            documents: List[str] = None) -> Dict[str, float]:

        if not isinstance(model, BaseModel):
            raise TypeError(
                f'Got "{type(model)}" as "model". '
                f'Expect it to derive from "BaseModel"')

        if topics is None:
            topics = _BaseCoherenceScore._get_topics(model)

        if documents is None:
            documents = list(self._documents)

        if not isinstance(topics, list):
            raise TypeError(
                f'Got "{type(topics)}" as "topics". Expect list of topic names')

        if not isinstance(documents, list):
            raise TypeError(
                f'Got "{type(documents)}" as "documents". Expect list of document ids')

        word_topic_relatednesses = self._get_word_topic_relatednesses(model)

        topic_document_coherences = np.zeros((len(topics), len(documents)))
        document_indices_with_topic_coherence = defaultdict(list)

        if not self._verbose:
            document_enumeration = enumerate(documents)
        else:
            document_enumeration = tqdm.tqdm(
                enumerate(documents), total=len(documents), file=sys.stdout
            )

        for document_index, document in document_enumeration:
            for topic_index, topic in enumerate(topics):
                # TODO: read document text only once for all topics
                topic_coherence = self._compute_coherence(
                    topic, document, word_topic_relatednesses)

                if topic_coherence is not None:
                    topic_document_coherences[topic_index, document_index] = topic_coherence
                    document_indices_with_topic_coherence[topic].append(document_index)

        topic_coherences = [
            topic_document_coherences[topic_index, document_indices_with_topic_coherence[topic]]
            if len(document_indices_with_topic_coherence) > 0 else list()
            for topic_index, topic in enumerate(topics)
        ]

        return dict(zip(
            topics,
            [
                float(np.mean(coherence_values))
                if len(coherence_values) > 0 else None
                for coherence_values in topic_coherences
            ]
        ))

    def _compute_coherence(
            self,
            topic: str,
            document: str,
            word_topic_relatednesses: pd.DataFrame) -> Union[float, None]:

        raise NotImplementedError()

    @staticmethod
    def _get_topics(model: BaseModel) -> List[str]:
        return list(model.get_phi().columns)

    def _get_word_topic_relatednesses(self, model: BaseModel) -> pd.DataFrame:
        phi = model.get_phi()

        word_topic_probs = self._get_word_topic_probs(phi)

        if self._specificity_estimation_method == SpecificityEstimationMethod.NONE:
            pass

        elif self._specificity_estimation_method == SpecificityEstimationMethod.AVERAGE:
            word_topic_probs[:] = (
                word_topic_probs.values -
                    np.sum(word_topic_probs.values, axis=1, keepdims=True) /  # noqa: line alignment
                        max(word_topic_probs.shape[1], 1)
            )

        elif self._specificity_estimation_method == SpecificityEstimationMethod.MAXIMUM:
            new_columns = []

            for t in word_topic_probs.columns:
                new_column = (
                    word_topic_probs[t].values -
                    np.max(
                        word_topic_probs[word_topic_probs.columns.difference([t])].values, axis=1)
                )
                new_columns.append(list(new_column))

            word_topic_probs[:] = np.array(new_columns).T

        return word_topic_probs

    def _get_word_topic_probs(self, phi: pd.DataFrame) -> pd.DataFrame:

        if self._word_topic_relatedness == WordTopicRelatednessType.PWT:
            return phi

        elif self._word_topic_relatedness == WordTopicRelatednessType.PTW:
            # Treat all topics as equally probable
            eps = np.finfo(float).tiny

            pwt = phi
            pwt_values = pwt.values

            return pd.DataFrame(
                index=pwt.index,
                columns=pwt.columns,
                data=pwt_values / (pwt_values.sum(axis=1).reshape(-1, 1) + eps)
            )

        assert False

    def _get_words(self, document: str) -> List[Tuple[str, str]]:

        if self._text_type == TextType.RAW_TEXT:
            text = self._get_source_document(document)
            modality = self._get_biggest_modality_or_default()

            return list(map(lambda w: (modality, w), text.split()))

        if self._text_type == TextType.VW_TEXT:
            text = self._get_vw_document(document)

            words = []
            modality = None

            # TODO: there was similar bunch of code somewhere...
            for word in text.split()[1:]:  # skip document id
                if word.startswith(MODALITY_START_SYMBOL):
                    modality = word[1:]

                    continue

                word = word.split(':')[0]

                if modality is not None:
                    word = (modality, word)  # phi multiIndex
                else:
                    word = (DEFAULT_ARTM_MODALITY, word)

                words.append(word)

            return words

        assert False

    # TODO: can't Dataset do something like this?
    def _get_biggest_modality_or_default(self) -> str:

        modalities = list(self._dataset.get_possible_modalities())

        if len(modalities) == 0:
            return DEFAULT_ARTM_MODALITY

        modalities_vocabulary_sizes = list(map(
            lambda m: self._dataset.get_dataset().loc[m].shape[0],
            modalities
        ))

        return modalities[np.argmax(modalities_vocabulary_sizes)]

    # TODO: try again self._dataset.get_source_document()?
    def _get_source_document(self, document_id: str) -> str:
        return self._dataset.get_source_document(document_id).loc[document_id, RAW_TEXT_COL]

    def _get_vw_document(self, document_id: str) -> str:
        return self._dataset.get_vw_document(document_id).loc[document_id, VW_TEXT_COL]

    @staticmethod
    def _get_relatedness(
            word: Tuple[str, str],
            topic: str,
            word_topic_relatednesses: pd.DataFrame) -> float:

        if word in word_topic_relatednesses.index:
            return word_topic_relatednesses.loc[word, topic]

        _logger.warning(
            f'The word "{word}" not found in Word-Topic relatedness matrix!'
            f' Returning mean value over all word relatednesses for topic "{topic}"'
        )

        return float(np.mean(word_topic_relatednesses.values))

    # TODO: DRY
    def save(self, path: str) -> None:
        dataset = self._dataset
        self._dataset = None

        with open(path, 'wb') as f:
            dill.dump(self, f)

        self._dataset = dataset

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            score = dill.load(f)

        score._dataset = Dataset(
            score._dataset_file_path,
            internals_folder_path=score._dataset_internals_folder_path,
            keep_in_memory=score._keep_dataset_in_memory,
        )

        return score
