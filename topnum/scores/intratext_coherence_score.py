import numpy as np
import pandas as pd
import warnings

from enum import (
    auto,
    IntEnum
)
from topicnet.cooking_machine import Dataset
from topicnet.cooking_machine.models.base_model import BaseModel
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union
)

from ..data.vowpal_wabbit_text_collection import VowpalWabbitTextCollection
from ._base_coherence_score import (
    _BaseCoherenceScore,
    SpecificityEstimationMethod,
    TextType,
    WordTopicRelatednessType,
    WordType
)
from .base_topic_score import BaseTopicScore


class ComputationMethod(IntEnum):
    """
    Ways to compute intra-text coherence
    (see more about coherence below in IntratextCoherenceScore)

    Attributes
    ----------
        SEGMENT_LENGTH
            Estimate the length of topic segments (TopLen)
        SEGMENT_WEIGHT
            Estimate the weight of topic segment
            (weight as sum of specificities for the topic over words in segment)
        SUM_OVER_WINDOW
            Sum of specificities for the topic over words in given window.
            The process is as follows:
            word of the topic is found in text, it is the center of the first window;
            next word of the topic is found (outside of the previous window),
            it is the center of the new window; etc
        VARIANCE_IN_WINDOW
            Estimate the variance between segment word vector components
            corresponding to the topic (SemantiC_Var)
        FOCUS_CONSISTENCY
            Estimate how much text adjacent words differ,
            summing the pairs of differences between max components
            of corresponding word vectors (FoCon)
    """
    SEGMENT_LENGTH = auto()
    SEGMENT_WEIGHT = auto()
    SUM_OVER_WINDOW = auto()
    VARIANCE_IN_WINDOW = auto()
    FOCUS_CONSISTENCY = auto()


_RESEARCH_COMPUTATION_METHODS = [
    ComputationMethod.VARIANCE_IN_WINDOW,
    ComputationMethod.FOCUS_CONSISTENCY,
]


class IntratextCoherenceScore(BaseTopicScore):
    """
    Computes intratext coherence

    For each topic of topic model its distribution throughout document collection is observed.
    Hypothetically, the better the topic, the more often it is represented by
    long segments of words highly related to the topic.
    The score tries to bring to life this idea.

    For more details one may see the article http://www.dialog-21.ru/media/4281/alekseevva.pdf
    """

    # TODO: ability to specify modalities in the constructor
    def __init__(
            self,
            name: str,
            data: Union[Dataset, VowpalWabbitTextCollection],
            documents: List[str] = None,
            text_type: TextType = TextType.VW_TEXT,
            computation_method: ComputationMethod = ComputationMethod.SEGMENT_LENGTH,
            word_topic_relatedness: WordTopicRelatednessType = WordTopicRelatednessType.PWT,
            specificity_estimation: SpecificityEstimationMethod = SpecificityEstimationMethod.NONE,
            max_num_out_of_topic_words=10,
            window=10,
            verbose: bool = False,
            should_compute: Optional[
                Union[Callable[[int], bool], bool]] = True,  # TODO: very slow on full collection
    ):
        """
        Parameters
        ----------
        data
            Document collection
            (any model passed to `call()` is supposed to be trained on it)
        documents
            Which documents from the dataset are to be used for computing coherence
        text_type
            What text to use when computing coherence: raw text or VW text
            Preferable to use VW (as it is usually preprocessed, stop-words removed etc.),
            and with words in *natural order*.
            Score needs "real" text to compute coherence
        computation_method
            The way to compute intra-text coherence
        word_topic_relatedness
            How to estimate word relevance to topic: using p(w | t) or p(t | w)
        specificity_estimation
            How to estimate specificity of word to topic
        max_num_out_of_topic_words
            In case computation_method = ComputationMethod.SEGMENT_LENGTH or
            ComputationMethod.SEGMENT_WEIGHT:
            Maximum number of words not of the topic which can be encountered without stopping
            the process of adding words to the current segment
        window
            In case computation_method = ComputationMethod.SUM_OVER_WINDOW:
            Window width. So the window will be the words with positions
            in [current position - window / 2, current position + window / 2)
        verbose
            Whether to show progress bar for documents or not.
            As the score is not very fast, it might be helpful to see how many documents
            are yet to be processed
        """
        super().__init__(name)

        self._data = data
        self._documents = documents
        self._text_type = text_type
        self._word_topic_relatedness = word_topic_relatedness
        self._specificity_estimation = specificity_estimation

        self._computation_method = computation_method
        self._max_num_out_of_topic_words = max_num_out_of_topic_words
        self._window = window

        self._verbose = verbose
        self._should_compute = should_compute

        self._score = self._initialize()

    def _initialize(self) -> _BaseCoherenceScore:
        if isinstance(self._data, Dataset):
            dataset = self._data
        else:
            dataset = self._data._to_dataset()

        return _IntratextCoherenceScore(
            dataset=dataset,
            documents=self._documents,
            text_type=self._text_type,
            computation_method=self._computation_method,
            word_topic_relatedness=self._word_topic_relatedness,
            specificity_estimation=self._specificity_estimation,
            max_num_out_of_topic_words=self._max_num_out_of_topic_words,
            window=self._window,
            verbose=self._verbose,
            should_compute=self._should_compute,
        )

    def compute(
            self,
            model: BaseModel,
            topics: List[str] = None,
            documents: List[str] = None) -> Dict[str, float]:

        return self._score.compute(model, topics, documents)


# TODO: same score also is in TopicNet
#  but needed this implementation because of TopTokensCoherenceScore and BaseCoherenceScore
#  Maybe it would be better to move this whole stuff to TopicNet
class _IntratextCoherenceScore(_BaseCoherenceScore):
    def __init__(
            self,
            dataset: Dataset,
            documents: List[str] = None,
            text_type: TextType = TextType.VW_TEXT,
            computation_method: ComputationMethod = ComputationMethod.SEGMENT_LENGTH,
            word_topic_relatedness: WordTopicRelatednessType = WordTopicRelatednessType.PWT,
            specificity_estimation: SpecificityEstimationMethod = SpecificityEstimationMethod.NONE,
            max_num_out_of_topic_words: int = 10,
            window: int = 10,
            verbose: bool = False,
            should_compute: Optional[
                Union[Callable[[int], bool], bool]] = None,
            ):

        # TODO: word_topic_relatedness seems to be connected with TopTokensViewer stuff
        super().__init__(
            dataset=dataset,
            documents=documents,
            text_type=text_type,
            word_topic_relatedness=word_topic_relatedness,
            specificity_estimation=specificity_estimation,
            verbose=verbose,
            should_compute=should_compute,
        )

        if not isinstance(computation_method, ComputationMethod):
            raise TypeError(
                f'Wrong "computation_method": \"{computation_method}\". '
                f'Expect to be \"{ComputationMethod}\"')

        if computation_method in _RESEARCH_COMPUTATION_METHODS:
            warnings.warn(
                f"Coherences {_RESEARCH_COMPUTATION_METHODS} were also presented in the original paper"
                f" but preference should be given to other (TopLen-based) methods."
                f" Still, coherences {_RESEARCH_COMPUTATION_METHODS} are also implemented,"
                f" partly as a tribute, partly for research purposes."
                f" Once again, coherence {computation_method} is not intended for \"production\" use."
                f" But you do you, it's not like there's a coherence police or something."
            )

        if not isinstance(max_num_out_of_topic_words, int):
            raise TypeError(
                f'Wrong "max_num_out_of_topic_words": \"{max_num_out_of_topic_words}\". '
                f'Expect to be \"int\"')

        if not isinstance(window, int):
            raise TypeError(
                f'Wrong "window": \"{window}\". '
                f'Expect to be \"int\"')

        if window < 0 or (window == 0 and computation_method in [ComputationMethod.SUM_OVER_WINDOW,
                                                                 ComputationMethod.VARIANCE_IN_WINDOW]):
            raise ValueError(
                f'Wrong value for "window": \"{window}\". '
                f'Expect to be non-negative. And greater than zero in case '
                f'computation_method is SUM_OVER_WINDOW or VARIANCE_IN_WINDOW.')

        self._computation_method = computation_method
        self._max_num_out_of_topic_words = max_num_out_of_topic_words
        self._window = window

    def _compute_coherence(
            self,
            topic: str,
            document: str,
            word_topic_relatednesses: pd.DataFrame) -> Union[float, None]:

        assert isinstance(self._computation_method, ComputationMethod)

        words = self._get_words(document)

        if self._computation_method == ComputationMethod.SUM_OVER_WINDOW:
            average_sum_over_window = self._sum_relatednesses_over_window(
                topic, words, word_topic_relatednesses
            )

            return average_sum_over_window

        elif self._computation_method == ComputationMethod.VARIANCE_IN_WINDOW:
            average_variance_in_window = self._compute_variance_in_window(
                topic, words, word_topic_relatednesses
            )

            return average_variance_in_window

        elif self._computation_method == ComputationMethod.FOCUS_CONSISTENCY:
            average_focus_consistency = self._compute_focus_consistency(
                topic, words, word_topic_relatednesses
            )

            return average_focus_consistency

        topic_segment_length, topic_segment_weight = self._compute_segment_characteristics(
            topic, words, word_topic_relatednesses
        )

        if self._computation_method == ComputationMethod.SEGMENT_LENGTH:
            return topic_segment_length

        elif self._computation_method == ComputationMethod.SEGMENT_WEIGHT:
            return topic_segment_weight

    def _get_word_topic_index(
            self,
            word: WordType,
            word_topic_relatednesses: pd.DataFrame,
            word_topic_indices: np.array,
            ) -> int:
        # if word not in word_topic_relatednesses.index:
        #     return -1
        # else:
        #     return word_topic_indices[
        #         word_topic_relatednesses.index.get_loc(word)
        #     ]
        if word not in self._word2index:
            return -1
        else:
            return word_topic_indices[self._word2index[word]]

    def _compute_segment_characteristics(
            self,
            topic: str,
            words: List[WordType],
            word_topic_relatednesses: pd.DataFrame
    ) -> Tuple[Union[float, None], Union[float, None]]:

        topic_segment_lengths = []
        topic_segment_weights = []

        topic_index = self._topic2index[topic]  # word_topic_relatednesses.columns.get_loc(topic)
        # word_topic_indices = np.argmax(word_topic_relatednesses.values, axis=1)

        def get_word_topic_index(word: WordType) -> int:
            return self._get_word_topic_index(
                word=word,
                word_topic_relatednesses=word_topic_relatednesses,
                word_topic_indices=self._word_topic_indices,
            )

        index = 0

        while index < len(words):
            original_index = index

            if get_word_topic_index(words[index]) != topic_index:
                index += 1

                continue

            segment_length = 1
            segment_weight = self._get_relatedness(
                words[index], topic, word_topic_relatednesses
            )

            num_out_of_topic_words = 0

            index += 1

            while index < len(words) and num_out_of_topic_words < self._max_num_out_of_topic_words:
                if get_word_topic_index(words[index]) != topic_index:
                    num_out_of_topic_words += 1
                else:
                    segment_length += 1
                    segment_weight += self._get_relatedness(
                        words[index], topic, word_topic_relatednesses
                    )

                    num_out_of_topic_words = 0

                index += 1

            topic_segment_lengths.append(segment_length)
            topic_segment_weights.append(segment_weight)

            assert index > original_index

        if len(topic_segment_lengths) == 0:
            return None, None
        else:
            return (
                float(np.mean(topic_segment_lengths)),
                float(np.mean(topic_segment_weights))
            )

    def _sum_relatednesses_over_window(
            self,
            topic: str,
            words: List[WordType],
            word_topic_relatednesses: pd.DataFrame) -> Union[float, None]:

        topic_index = self._topic2index[topic]  # word_topic_relatednesses.columns.get_loc(topic)
        # word_topic_indices = np.argmax(word_topic_relatednesses.values, axis=1)

        def get_word_topic_index(word: WordType) -> int:
            return self._get_word_topic_index(
                word=word,
                word_topic_relatednesses=word_topic_relatednesses,
                word_topic_indices=self._word_topic_indices,
            )

        def find_next_topic_word(starting_index: int) -> int:
            index = starting_index

            while index < len(words) and\
                    get_word_topic_index(words[index]) != topic_index:
                index += 1

            if index == len(words):
                return -1  # failed to find next topic word

            return index

        word_index = find_next_topic_word(0)

        if word_index == -1:
            return None

        sums = list()

        while word_index < len(words) and word_index != -1:
            original_word_index = word_index

            window_lower_bound = word_index - int(np.floor(self._window // 2))
            window_upper_bound = word_index + int(np.floor(self._window // 2)) + 1

            sum_in_window = np.sum(
                [
                    self._get_relatedness(w, topic, word_topic_relatednesses)
                    for w in words[window_lower_bound:window_upper_bound]
                ]
            )

            sums.append(sum_in_window)

            word_index = find_next_topic_word(window_upper_bound)

            assert word_index > original_word_index or word_index == -1

        return float(np.mean(sums))

    def _compute_variance_in_window(
            self,
            topic: str,
            words: List[WordType],
            word_topic_relatednesses: pd.DataFrame) -> Union[float, None]:

        topic_relatednesses = [
            self._get_relatedness(word, topic, word_topic_relatednesses)
            for word in words
        ]

        variances = list()
        index = 0

        while index == 0 or index + self._window - 1 < len(words):
            relatedness_window = topic_relatednesses[index:index + self._window]
            # TODO: better differentiate good and bad topics?..
            #  (low variance is not necessarily a good "goodness" sign:
            #  for example, sequences [100, 100, 100]
            #  and [-17.5, -17.5, -17.5] both have zero variance)
            variances.append(np.var(relatedness_window))

            index += 1

        if len(variances) == 0:
            return None
        else:
            return -1 * float(np.mean(variances))  # the higher the better

    def _compute_focus_consistency(
            self,
            topic: str,
            words: List[WordType],
            word_topic_relatednesses: pd.DataFrame) -> Union[float, None]:

        if len(words) == 0:
            return None

        # word_topic_indices = np.argmax(word_topic_relatednesses.values, axis=1)

        def get_word_topic_index(word: WordType) -> int:
            return self._get_word_topic_index(
                word=word,
                word_topic_relatednesses=word_topic_relatednesses,
                word_topic_indices=self._word_topic_indices,
            )

        word_topics = [
            word_topic_relatednesses.columns[get_word_topic_index(word)]
            for word in words
        ]

        differences = list()
        index = 0

        while index + 1 < len(words):  # like window = 2
            cur_word, next_word = words[index], words[index + 1]
            cur_topic, next_topic = word_topics[index], word_topics[index + 1]

            r_cw_ct = self._get_relatedness(
                cur_word, cur_topic, word_topic_relatednesses
            )
            r_cw_nt = self._get_relatedness(
                cur_word, next_topic, word_topic_relatednesses
            )
            r_nw_ct = self._get_relatedness(
                next_word, cur_topic, word_topic_relatednesses
            )
            r_nw_nt = self._get_relatedness(
                next_word, next_topic, word_topic_relatednesses
            )

            diff1 = abs(r_cw_ct - r_nw_ct)
            diff2 = abs(r_cw_nt - r_nw_nt)
            differences.append(diff1 + diff2)

            index += 1

        if len(differences) == 0:
            return None
        else:
            return -1 * float(np.mean(differences))  # the higher the better
