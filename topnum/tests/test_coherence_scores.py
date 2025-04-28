import numpy as np
import os
import pandas as pd
import pytest
import shutil
import tempfile
import warnings

from collections import defaultdict
from functools import reduce
from itertools import product
from topicnet.cooking_machine.dataset import (
    Dataset,
    RAW_TEXT_COL,
    VW_TEXT_COL,
    DEFAULT_ARTM_MODALITY
)
from topicnet.cooking_machine.models.base_model import BaseModel
from typing import (
    Dict,
    List
)

from ..scores.intratext_coherence_score import (
    _IntratextCoherenceScore,
    ComputationMethod,
)
from ..scores.sophisticated_toptok_coherence_score import _TopTokensCoherenceScore
from ..scores._base_coherence_score import (
    _BaseCoherenceScore,
    TextType,
    WordTopicRelatednessType,
    SpecificityEstimationMethod
)


DOC_ID_COL = 'id'
NUM_TOP_WORDS = 10
BIG_SEGMENT_LENGTH = 8
SMALL_SEGMENT_LENGTHS = [1, 2, 4]
SMALL_SEGMENT_LENGTH_PROBABILITIES = [0.3, 0.45, 0.25]
DOCUMENT_LENGTH = 100
TOP_WORD_PROBABILITY_TIMES_BIGGER = 4

PHI_FILE_NAME = 'phi.csv'
DATASET_FILE_NAME = 'dataset.csv'

TEXT_TYPES = [
    TextType.VW_TEXT,
    TextType.RAW_TEXT,
]
COMPUTATION_METHODS = [
    ComputationMethod.SEGMENT_LENGTH,
    ComputationMethod.SEGMENT_WEIGHT,
    ComputationMethod.SUM_OVER_WINDOW,
    ComputationMethod.VARIANCE_IN_WINDOW,
    ComputationMethod.FOCUS_CONSISTENCY,
]
RESEARCH_COMPUTATION_METHODS = [
    ComputationMethod.VARIANCE_IN_WINDOW,
    ComputationMethod.FOCUS_CONSISTENCY,
]
WORD_TOPIC_RELATEDNESS_TYPES = [
    WordTopicRelatednessType.PWT,
    WordTopicRelatednessType.PTW,
]
SPECIFICITY_ESTIMATION_METHODS = [
    SpecificityEstimationMethod.NONE,
    SpecificityEstimationMethod.MAXIMUM,
    SpecificityEstimationMethod.AVERAGE,
]


RESEARCH_INTRATEXT_MESSAGE = (
    f"Coherences {RESEARCH_COMPUTATION_METHODS} were presented in the original paper"
    f" and are implemented partly as a tribute,"
    f" partly for research purposes."
    f" For real use, preference should be given to {COMPUTATION_METHODS} methods."
)


class _MockModel(BaseModel):
    def __init__(self, phi: pd.DataFrame):
        self._phi = phi

    def get_phi(self) -> pd.DataFrame:
        return self._phi.copy()


class TestIntratextCoherenceScore:
    topics = ['topic_1', 'topic_2', 'topic_3']
    documents = ['doc_1', 'doc_2', 'doc_3']
    topic_documents = {
        'topic_1': ['doc_1', 'doc_2'],
        'topic_2': ['doc_3'],
        'topic_3': []
    }
    best_topic = 'topic_1'
    out_of_documents_topic = 'topic_3'
    document_topics = {
        'doc_1': ['topic_1', 'topic_2'],
        'doc_2': ['topic_1'],
        'doc_3': ['topic_1', 'topic_2']
    }
    top_words = {
        topic: [f'{topic}_word_{i}' for i in range(1, NUM_TOP_WORDS + 1)]
        for topic in topics
    }
    vocabulary = list(reduce(lambda res, cur: res + cur, top_words.values(), []))
    out_of_topics_word = 'unknown_word'

    data_folder_path = None
    dataset_file_path = None
    model = None
    dataset = None

    @classmethod
    def setup_class(cls):
        cls.model = _MockModel(cls.create_phi())

        document_words = cls.create_documents()
        dataset_table = cls.create_dataset_table(document_words)

        cls.data_folder_path = tempfile.mkdtemp()
        cls.dataset_file_path = os.path.join(
            cls.data_folder_path,
            DATASET_FILE_NAME,
        )

        dataset_table.to_csv(cls.dataset_file_path, index=False)

        cls.dataset = Dataset(cls.dataset_file_path)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.data_folder_path)

    @classmethod
    def create_phi(cls) -> pd.DataFrame:
        phi = pd.DataFrame(
            index=[(DEFAULT_ARTM_MODALITY, w) for w in cls.vocabulary],
            columns=cls.topics,
            data=np.random.random((len(cls.vocabulary), len(cls.topics)))
        )

        for t in cls.topics:
            phi.loc[[(DEFAULT_ARTM_MODALITY, w)
                     for w in cls.top_words[t]], t] = 1.0

            phi.loc[[(DEFAULT_ARTM_MODALITY, w)
                     for w in cls.vocabulary
                     if w not in cls.top_words[t]], t] = 1.0 / TOP_WORD_PROBABILITY_TIMES_BIGGER

        phi[:] = phi.values / np.sum(phi.values, axis=0, keepdims=True)

        phi.index = pd.MultiIndex.from_tuples(
            phi.index, names=('modality', 'token'))  # TODO: copy-paste from TopicModel

        return phi

    @classmethod
    def create_documents(cls) -> Dict[str, List[str]]:
        document_words = defaultdict(list)

        for t, docs in cls.topic_documents.items():
            all_other_topics = list(set(cls.topic_documents.keys()).difference([t]))

            for doc in docs:
                other_topics = list(set(all_other_topics).intersection(
                    cls.document_topics[doc]
                ))

                document_words[doc] = list(reduce(
                    lambda res, cur: res + cur,
                    cls.get_segments(t, other_topics, cls.top_words),
                    []
                ))

        return document_words

    @classmethod
    def get_segments(
            cls,
            topic: str,
            other_topics: List[str],
            top_words: Dict[str, List[str]]) -> List[List[str]]:

        num_words = 0
        segments = []
        is_main_topic = True
        is_out_of_topics_word_included = False

        while num_words < DOCUMENT_LENGTH:
            if len(other_topics) == 0:
                is_main_topic = True

            if is_main_topic:
                current_topic = topic
                current_segment_length = BIG_SEGMENT_LENGTH

            else:
                current_topic = np.random.choice(other_topics)
                current_segment_length = np.random.choice(
                    SMALL_SEGMENT_LENGTHS,
                    p=SMALL_SEGMENT_LENGTH_PROBABILITIES
                )

            segment = np.random.choice(
                top_words[current_topic],
                current_segment_length
            )
            segment = segment.tolist()

            if not is_out_of_topics_word_included:
                segment += [cls.out_of_topics_word]
                is_out_of_topics_word_included = True

            is_main_topic = not is_main_topic

            num_words += len(segment)
            segments.append(segment)

        return segments

    @classmethod
    def create_dataset_table(cls, document_words: Dict[str, List[str]]) -> pd.DataFrame:

        return pd.DataFrame(
            index=cls.documents,
            columns=[DOC_ID_COL, RAW_TEXT_COL, VW_TEXT_COL],
            data=[
                [doc, cls.get_raw_text(doc, document_words), cls.get_vw_text(doc, document_words)]
                for doc in cls.documents
            ]
        )

    @classmethod
    def get_raw_text(cls, doc: str, document_words: Dict[str, List[str]]) -> str:
        return ' '.join(document_words[doc])

    @classmethod
    def get_vw_text(cls, doc: str, document_words: Dict[str, List[str]]) -> str:
        return doc + ' ' + ' '.join(document_words[doc])

    @pytest.mark.parametrize(
        'text_type, computation_method, word_topic_relatedness, specificity_estimation',
        list(product(
            TEXT_TYPES,
            COMPUTATION_METHODS,
            WORD_TOPIC_RELATEDNESS_TYPES,
            SPECIFICITY_ESTIMATION_METHODS
        ))
    )
    def test_compute_intratext(
            self,
            text_type: TextType,
            computation_method: ComputationMethod,
            word_topic_relatedness: WordTopicRelatednessType,
            specificity_estimation: SpecificityEstimationMethod) -> None:

        if computation_method in RESEARCH_COMPUTATION_METHODS:
            pytest.xfail(RESEARCH_INTRATEXT_MESSAGE)

        score = _IntratextCoherenceScore(
            self.dataset,
            text_type=text_type,
            computation_method=computation_method,
            word_topic_relatedness=word_topic_relatedness,
            specificity_estimation=specificity_estimation
        )

        self._check_compute(score)

    @pytest.mark.parametrize(
        'window',
        [2, 4, 10]  # TODO: window = 1 -> fail (sometimes?)
    )
    def test_compute_topden(self, window) -> None:
        score = _IntratextCoherenceScore(
            self.dataset,
            text_type=TextType.VW_TEXT,
            computation_method=ComputationMethod.SUM_OVER_WINDOW,
            word_topic_relatedness=WordTopicRelatednessType.PTW,
            specificity_estimation=SpecificityEstimationMethod.NONE,
            window=window,
        )

        self._check_compute(score)

    @pytest.mark.parametrize('keep_in_memory', [True, False])
    def test_compute_intratext_small_big_data(self, keep_in_memory) -> None:
        dataset = Dataset(self.dataset_file_path, keep_in_memory=keep_in_memory)
        score = _IntratextCoherenceScore(dataset)

        self._check_compute(score)

    @pytest.mark.parametrize(
        'text_type, computation_method, word_topic_relatedness, specificity_estimation',
        list(product(
            TEXT_TYPES,
            COMPUTATION_METHODS,
            WORD_TOPIC_RELATEDNESS_TYPES,
            SPECIFICITY_ESTIMATION_METHODS
        ))
    )
    def test_call_intratext(
            self,
            text_type: TextType,
            computation_method: ComputationMethod,
            word_topic_relatedness: WordTopicRelatednessType,
            specificity_estimation: SpecificityEstimationMethod) -> None:

        if computation_method in RESEARCH_COMPUTATION_METHODS:
            pytest.xfail(RESEARCH_INTRATEXT_MESSAGE)

        score = _IntratextCoherenceScore(
            self.dataset,
            text_type=text_type,
            computation_method=computation_method,
            word_topic_relatedness=word_topic_relatedness,
            specificity_estimation=specificity_estimation
        )

        self._check_call(score)

    @pytest.mark.parametrize('keep_in_memory', [True, False])
    def test_call_intratext_small_big_data(self, keep_in_memory) -> None:
        dataset = Dataset(self.dataset_file_path, keep_in_memory=keep_in_memory)
        score = _IntratextCoherenceScore(dataset)

        self._check_call(score)

    @pytest.mark.parametrize(
        'text_type, computation_method, word_topic_relatedness, specificity_estimation',
        list(product(
            TEXT_TYPES,
            COMPUTATION_METHODS,
            WORD_TOPIC_RELATEDNESS_TYPES,
            SPECIFICITY_ESTIMATION_METHODS
        ))
    )
    @pytest.mark.parametrize(
        'what_documents',
        ['first', 'all', 'none']
    )
    def test_call_intratext_with_specified_documents(
            self,
            text_type: TextType,
            computation_method: ComputationMethod,
            word_topic_relatedness: WordTopicRelatednessType,
            specificity_estimation: SpecificityEstimationMethod,
            what_documents: str) -> None:

        if computation_method in RESEARCH_COMPUTATION_METHODS:
            pytest.xfail(RESEARCH_INTRATEXT_MESSAGE)

        if what_documents == 'first':
            documents = [self.documents[0]]
        elif what_documents == 'all':
            documents = self.documents
        elif what_documents == 'none':
            documents = list()
        else:
            raise ValueError(f'{what_documents}')

        score = _IntratextCoherenceScore(
            self.dataset,
            documents=documents,
            text_type=text_type,
            computation_method=computation_method,
            word_topic_relatedness=word_topic_relatedness,
            specificity_estimation=specificity_estimation
        )

        self._check_call(score)

    @pytest.mark.parametrize(
        'text_type, word_topic_relatedness, specificity_estimation',
        list(product(
            TEXT_TYPES,
            WORD_TOPIC_RELATEDNESS_TYPES,
            SPECIFICITY_ESTIMATION_METHODS
        ))
    )
    def test_compute_toptokens(
            self,
            text_type: TextType,
            word_topic_relatedness: WordTopicRelatednessType,
            specificity_estimation: SpecificityEstimationMethod):

        score = _TopTokensCoherenceScore(
            self.dataset,
            text_type=text_type,
            word_topic_relatedness=word_topic_relatedness,
            specificity_estimation=specificity_estimation
        )

        self._check_compute(score, strict=False)

    @pytest.mark.parametrize('keep_in_memory', [True, False])
    def test_compute_toptokens_small_big_data(self, keep_in_memory) -> None:
        dataset = Dataset(self.dataset_file_path, keep_in_memory=keep_in_memory)
        score = _TopTokensCoherenceScore(dataset)

        self._check_compute(score, strict=False)

    @pytest.mark.parametrize(
        'text_type, word_topic_relatedness, specificity_estimation',
        list(product(
            TEXT_TYPES,
            WORD_TOPIC_RELATEDNESS_TYPES,
            SPECIFICITY_ESTIMATION_METHODS
        ))
    )
    def test_call_toptokens(
            self,
            text_type: TextType,
            word_topic_relatedness: WordTopicRelatednessType,
            specificity_estimation: SpecificityEstimationMethod) -> None:

        score = _TopTokensCoherenceScore(
            self.dataset,
            text_type=text_type,
            word_topic_relatedness=word_topic_relatedness,
            specificity_estimation=specificity_estimation
        )

        self._check_call(score)

    @pytest.mark.parametrize('keep_in_memory', [True, False])
    def test_call_toptokens_small_big_data(self, keep_in_memory) -> None:
        dataset = Dataset(self.dataset_file_path, keep_in_memory=keep_in_memory)
        score = _TopTokensCoherenceScore(dataset)

        self._check_call(score)

    @pytest.mark.parametrize(
        'text_type, word_topic_relatedness, specificity_estimation',
        list(product(
            TEXT_TYPES,
            WORD_TOPIC_RELATEDNESS_TYPES,
            SPECIFICITY_ESTIMATION_METHODS
        ))
    )
    @pytest.mark.parametrize(
        'what_documents',
        ['first', 'all', 'none']
    )
    def test_call_toptokens_with_specified_documents(
            self,
            text_type: TextType,
            word_topic_relatedness: WordTopicRelatednessType,
            specificity_estimation: SpecificityEstimationMethod,
            what_documents: str) -> None:

        if what_documents == 'first':
            documents = [self.documents[0]]
        elif what_documents == 'all':
            documents = self.documents
        elif what_documents == 'none':
            documents = list()
        else:
            raise ValueError(f'{what_documents}')

        score = _TopTokensCoherenceScore(
            self.dataset,
            documents=documents,
            text_type=text_type,
            word_topic_relatedness=word_topic_relatedness,
            specificity_estimation=specificity_estimation
        )

        self._check_call(score)

    def _check_compute(self, score: _BaseCoherenceScore, strict: bool = True) -> None:
        coherences = score.compute(self.model)
        coherence_values = list(coherences.values())
        maximum_coherence = max(c for c in coherence_values if c is not None)

        if coherences[self.best_topic] != maximum_coherence:
            message = (
                f'Topic that expected to be best doesn\'t have max coherence:'
                f' {coherences[self.best_topic]} != {maximum_coherence}!'
            )

            if strict:
                assert False, message
            else:
                warnings.warn(message)

        assert coherences[self.out_of_documents_topic] is None,\
            'Topic that is not in any document has coherence other than None'

    def _check_call(self, score: _BaseCoherenceScore) -> None:

        value = score.call(self.model)

        assert isinstance(value, float), f'Wrong score value type {type(value)}'
