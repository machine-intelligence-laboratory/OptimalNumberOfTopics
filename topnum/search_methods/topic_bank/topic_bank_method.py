import artm
import json
import logging
import numpy as np
import os
import pandas as pd
import sys
import tempfile
import tqdm
import warnings

from collections import defaultdict
from distutils.util import strtobool
from topicnet.cooking_machine.dataset import Dataset
from topicnet.cooking_machine.models import TopicModel
from typing import (
    Callable,
    Dict,
    List,
    Tuple,
    Union
)

from topnum.data.vowpal_wabbit_text_collection import VowpalWabbitTextCollection
from topnum.scores._base_coherence_score import (
    SpecificityEstimationMethod,
    TextType,
    WordTopicRelatednessType
)
from topnum.scores.base_score import BaseScore
from topnum.scores.base_topic_score import BaseTopicScore
from topnum.scores import (
    IntratextCoherenceScore,
    PerplexityScore,
    SophisticatedTopTokensCoherenceScore,
    SparsityPhiScore,
    SparsityThetaScore
)
from topnum.scores.intratext_coherence_score import ComputationMethod
from topnum.search_methods.base_search_method import BaseSearchMethod
from topnum.search_methods.constants import (
    DEFAULT_MAX_NUM_TOPICS,
    DEFAULT_MIN_NUM_TOPICS,
    DEFAULT_NUM_FIT_ITERATIONS
)
from topnum.search_methods.base_search_method import (
    _KEY_OPTIMUM,
    _STD_KEY_SUFFIX
)
from topnum.search_methods.topic_bank.bank_update_method import BankUpdateMethod
from topnum.search_methods.topic_bank.topic_bank import (
    TopicBank,
    TokenType
)
from topnum.search_methods.topic_bank.one_model_train_funcs import (
    default_train_func,
    _get_topic_model
)
from topnum.search_methods.topic_bank.phi_initialization.utils import _safe_copy_phi


_KEY_BANK_SCORES = 'bank_scores'
_KEY_BANK_TOPIC_SCORES = 'bank_topic_scores'
_KEY_MODEL_SCORES = 'model_scores'
_KEY_MODEL_TOPIC_SCORES = 'model_topic_scores'
_KEY_NUM_BANK_TOPICS = 'num_bank_topics'
_KEY_NUM_MODEL_TOPICS = 'num_model_topics'
_KEY_TOPIC_SCORE_DISTANCE_TO_NEAREST = 'distance_to_nearest'
_KEY_TOPIC_SCORE_KERNEL_SIZE = 'kernel_size'

_DEFAULT_WINDOW = 20


_logger = logging.getLogger()


class TopicBankMethod(BaseSearchMethod):
    _MINIMUM_TOPIC_DISTANCE = 0.0
    _MAXIMUM_TOPIC_DISTANCE = 1.0

    def __init__(
            self,
            data: Union[Dataset, VowpalWabbitTextCollection],
            main_modality: str = None,
            main_topic_score: BaseTopicScore = None,
            other_topic_scores: List[BaseTopicScore] = None,
            stop_bank_score: BaseScore = None,
            other_scores: List[BaseScore] = None,
            documents_fraction_for_topic_scores: float = 0.2,
            max_num_documents_for_topic_scores: int = 100,
            max_num_models: int = 100,
            one_model_num_topics: Union[int, List[int]] = 100,
            num_fit_iterations: int = DEFAULT_NUM_FIT_ITERATIONS,
            train_func: Union[
                Callable[[Dataset, int, int, int], TopicModel],
                List[Callable[[Dataset, int, int, int], TopicModel]],
                None] = None,
            topic_score_threshold_percentile: int = 95,
            distance_threshold: float = 0.5,
            bank_update: BankUpdateMethod = BankUpdateMethod.PROVIDE_NON_LINEARITY,
            child_parent_relationship_threshold: float = None,
            save_file_path: str = None,
            seed: int = None):

        super().__init__(
            min_num_topics=DEFAULT_MIN_NUM_TOPICS,  # not needed
            max_num_topics=DEFAULT_MAX_NUM_TOPICS,  # not needed
            num_fit_iterations=num_fit_iterations
        )

        if isinstance(data, Dataset):
            self._dataset = data
        elif isinstance(data, VowpalWabbitTextCollection):
            self._dataset = data._to_dataset()
        else:
            raise TypeError(f'data: "{data}", its type: "{type(data)}"')

        self._main_modality = main_modality

        if main_topic_score is not None:
            self._main_topic_score = main_topic_score
        else:
            self._main_topic_score = IntratextCoherenceScore(
                name='intratext_coherence_score',
                data=self._dataset,
                text_type=TextType.VW_TEXT,
                computation_method=ComputationMethod.SEGMENT_WEIGHT,
                word_topic_relatedness=WordTopicRelatednessType.PWT,
                specificity_estimation=SpecificityEstimationMethod.NONE,
                max_num_out_of_topic_words=5,
                window=_DEFAULT_WINDOW
            )

        if other_topic_scores is not None:
            self._other_topic_scores = other_topic_scores
        else:
            self._other_topic_scores = [
                SophisticatedTopTokensCoherenceScore(
                    name='top_tokens_coherence_score',
                    data=self._dataset,
                    text_type=TextType.VW_TEXT,
                    word_topic_relatedness=WordTopicRelatednessType.PWT,
                    specificity_estimation=SpecificityEstimationMethod.NONE,
                    num_top_words=10,
                    window=_DEFAULT_WINDOW
                )
            ]

        self._all_topic_scores = [self._main_topic_score] + self._other_topic_scores

        if stop_bank_score is not None:
            self._stop_bank_score = stop_bank_score
        else:
            self._stop_bank_score = PerplexityScore(name='perplexity_score')

        if other_scores is not None:
            self._other_scores = other_scores
        else:
            self._other_scores = [
                SparsityPhiScore(name='sparsity_phi_score'),
                SparsityThetaScore(name='sparsity_theta_score')
            ]

        self._all_model_scores = [self._stop_bank_score] + self._other_scores

        self._documents_fraction_for_topic_scores = documents_fraction_for_topic_scores
        self._max_num_documents_for_topic_scores = max_num_documents_for_topic_scores
        self._max_num_models = max_num_models

        if not isinstance(one_model_num_topics, list):
            one_model_num_topics = [
                one_model_num_topics for _ in range(self._max_num_models)
            ]

        if train_func is None:
            train_func = default_train_func

        if not isinstance(train_func, list):
            train_func = [
                train_func for _ in range(self._max_num_models)
            ]

        self._one_model_num_topics: List[int] = one_model_num_topics
        self._train_func: List[Callable[[Dataset, int, int, int], TopicModel]] = train_func

        if topic_score_threshold_percentile < 1:
            warnings.warn(
                f'topic_score_threshold_percentile {topic_score_threshold_percentile}'
                f' is less than one! It is expected to be in [0, 100].'
                f' Are you sure you want to proceed (yes/no)?'
            )

            answer = input()

            if strtobool(answer) is False:
                warnings.warn('Exiting')

                exit(0)

        self._topic_score_threshold_percentile = topic_score_threshold_percentile

        if distance_threshold > 1 or distance_threshold < 0:
            raise ValueError(f'distance_threshold should be in [0, 1], not {distance_threshold}')

        self._distance_threshold = distance_threshold
        self._bank_update = bank_update
        self._child_parent_relationship_threshold = child_parent_relationship_threshold

        if save_file_path is not None:
            if not os.path.isdir(os.path.dirname(save_file_path)):
                raise NotADirectoryError(f'Directory not found "{save_file_path}"')

            if os.path.isfile(save_file_path):
                warnings.warn(f'File "{save_file_path}" already exists! Overwriting')
        else:
            file_descriptor, save_file_path = tempfile.mkstemp(prefix='topic_bank_result__')
            os.close(file_descriptor)

        self._save_file_path = save_file_path

        self._random = np.random.RandomState(seed=seed)

        self._result = dict()

        self._result[_KEY_OPTIMUM] = None
        self._result[_KEY_OPTIMUM + _STD_KEY_SUFFIX] = None
        self._result[_KEY_BANK_SCORES] = list()
        self._result[_KEY_BANK_TOPIC_SCORES] = list()
        self._result[_KEY_MODEL_SCORES] = list()
        self._result[_KEY_MODEL_TOPIC_SCORES] = list()
        self._result[_KEY_NUM_BANK_TOPICS] = list()
        self._result[_KEY_NUM_MODEL_TOPICS] = list()

        self._topic_bank: TopicBank = None

    @property
    def save_path(self) -> str:
        return self._save_file_path

    def save(self) -> None:
        with open(self._save_file_path, 'w') as f:
            f.write(json.dumps(self._result))

    def clear(self) -> None:
        if os.path.isfile(self._save_file_path):
            os.remove(self._save_file_path)

    def search_for_optimum(self, text_collection: VowpalWabbitTextCollection = None) -> None:
        """
        Parameters
        ----------
        text_collection:
            Not needed, kept only for compatibility with the base search method
        """
        # TODO: simplify

        word2index = None

        documents_for_coherence = self._select_documents_for_topic_scores()
        self._topic_bank = TopicBank()

        for i in tqdm.tqdm(range(self._max_num_models), total=self._max_num_models, file=sys.stdout):
            # TODO: stop when perplexity stabilizes

            _logger.info(f'Building topic model number {i}...')

            topic_model = self._train_func[i](
                dataset=self._dataset,
                model_number=i,
                num_topics=self._one_model_num_topics[i],
                num_fit_iterations=self._num_fit_iterations,
                scores=self._all_model_scores
            )

            scores = dict()

            _logger.info('Computing scores for one topic model...')

            scores.update(self._get_default_scores(topic_model))

            raw_topic_scores = self._compute_raw_topic_scores(
                topic_model,
                documents_for_coherence
            )

            for score_name, score_values in raw_topic_scores.items():
                scores[score_name] = self._aggregate_scores_for_models(
                    raw_topic_scores[score_name], 50
                )

            self._result[_KEY_MODEL_SCORES].append(scores)
            self._result[_KEY_NUM_MODEL_TOPICS].append(topic_model.get_phi().shape[1])

            self.save()

            threshold = self._aggregate_scores_for_models(
                raw_topic_scores[self._main_topic_score.name],
                self._topic_score_threshold_percentile
            )

            _logger.info('Finding new topics...')

            phi = topic_model.get_phi()

            if self._main_modality is None:
                phi = phi
            else:
                phi = phi.iloc[phi.index.get_level_values(0).isin([self._main_modality])]

            if word2index is None:
                word2index = {
                    word: index for index, word in enumerate(phi.index)
                }

            _logger.info('Finding topics for append and update...')

            if self._bank_update == BankUpdateMethod.JUST_ADD_GOOD_TOPICS:
                topics_for_append = list(range(len(phi.columns)))
                topics_for_update = dict()
            elif self._bank_update == BankUpdateMethod.PROVIDE_NON_LINEARITY:
                topics_for_append, topics_for_update = self._extract_hierarchical_relationship(
                    bank_phi=self._get_phi(self._topic_bank.topics, word2index),
                    new_model_phi=phi,
                    psi_threshold=self._child_parent_relationship_threshold
                )
            else:
                raise NotImplementedError(f'BankUpdateMethod: "{self._bank_update}"')

            _logger.info('Finding good new topics, updating topics for append and update')

            good_new_topics = [
                topic_index for topic_index, topic_name in enumerate(phi.columns)
                if raw_topic_scores[self._main_topic_score.name][topic_name] is not None and
                raw_topic_scores[self._main_topic_score.name][topic_name] >= threshold
            ]
            topics_for_append, topics_for_update, topics_for_update_reverse = (
                self._keep_good_new_topics_only(
                    topics_for_append, topics_for_update, good_new_topics
                )
            )

            model_topic_current_scores = list()

            _logger.info('Calculating model topic scores...')

            for topic_index, topic_name in enumerate(topic_model.get_phi().columns):
                topic_scores = dict()

                v = topic_model.get_phi()[topic_name].values
                topic_scores[_KEY_TOPIC_SCORE_KERNEL_SIZE] = len(v[v > 1.0 / topic_model.get_phi().shape[0]])

                for score_name in raw_topic_scores:
                    topic_scores[score_name] = raw_topic_scores[score_name][topic_name]

                model_topic_current_scores.append(topic_scores)

                if (topic_index not in topics_for_append and
                        topic_index not in topics_for_update_reverse):

                    continue

                if topic_index in topics_for_update_reverse:
                    old_topic_index = topics_for_update_reverse[topic_index]
                    new_topic_candidates = topics_for_update[old_topic_index]
                    current_topic_score = topic_scores[self._main_topic_score.name]
                    current_old_topic_score = self._topic_bank.topic_scores[old_topic_index][self._main_topic_score.name]

                    if (len(new_topic_candidates) == 1 and
                            current_topic_score <= current_old_topic_score):

                        continue

                if len(self._topic_bank.topics) == 0:
                    d = self._MINIMUM_TOPIC_DISTANCE
                else:
                    d = (
                        min(self._jaccard_distance(phi.loc[:, topic_name].to_dict(), bt)
                            for bt in self._topic_bank.topics)
                    )

                    if d < self._distance_threshold:
                        continue

                topic_scores[_KEY_TOPIC_SCORE_DISTANCE_TO_NEAREST] = d

                self._topic_bank.add_topic(phi.loc[:, topic_name].to_dict(), topic_scores)

                if topic_index in topics_for_update_reverse:
                    # TODO: check this
                    self._topic_bank.delete_topic(topics_for_update_reverse[topic_index])

            self._result[_KEY_MODEL_TOPIC_SCORES].append(model_topic_current_scores)
            self._result[_KEY_BANK_TOPIC_SCORES] = self._topic_bank.topic_scores  # TODO: append

            self.save()

            _logger.info('Scoring bank model...')

            scores = dict()

            if len(self._topic_bank.topics) == 0:
                _logger.info('No topics in bank — returning empty default scores for bank model')
            else:
                bank_phi = self._get_phi(self._topic_bank.topics, word2index)

                bank_model = _get_topic_model(
                    self._dataset,
                    phi=bank_phi,
                    scores=self._all_model_scores,
                    num_safe_fit_iterations=1
                )
                bank_model._fit(self._dataset.get_batch_vectorizer(), 1)

                _logger.info('Computing default scores for bank model...')

                scores.update(self._get_default_scores(bank_model))

            # Topic scores already calculated

            self._result[_KEY_BANK_SCORES].append(scores)
            self._result[_KEY_NUM_BANK_TOPICS].append(len(self._topic_bank.topics))

            _logger.info(f'Num topics in bank: {len(self._topic_bank.topics)}')

            self.save()

        self._result[_KEY_OPTIMUM] = self._result[_KEY_NUM_BANK_TOPICS][-1]

        # TODO: refine computing when do early stop
        if len(self._result[_KEY_NUM_BANK_TOPICS]) <= 1:  # TODO: can be zero?
            self._result[_KEY_OPTIMUM + _STD_KEY_SUFFIX] = self._result[_KEY_OPTIMUM]
        else:
            differences = list()
            max_num_last_values = 5

            i = len(self._result[_KEY_NUM_BANK_TOPICS]) - 1

            while i > 0 and len(differences) < max_num_last_values:
                differences.append(abs(
                    self._result[_KEY_NUM_BANK_TOPICS][-i] -
                    self._result[_KEY_NUM_BANK_TOPICS][-i - 1]
                ))

            self._result[_KEY_OPTIMUM + _STD_KEY_SUFFIX] = float(np.sum(differences))

    def _select_documents_for_topic_scores(self) -> List[str]:
        document_ids = self._dataset._data['id'].values
        num_documents = len(document_ids)

        selected_documents = self._random.choice(
            document_ids,
            size=min(
                self._max_num_documents_for_topic_scores,
                int(self._documents_fraction_for_topic_scores * num_documents)
            ),
            replace=False
        )
        selected_documents = list(selected_documents)

        return selected_documents

    def _extract_hierarchical_relationship(
            self,
            bank_phi: pd.DataFrame,
            new_model_phi: pd.DataFrame,
            psi_threshold: float = None
    ) -> Tuple[List[int], Dict[int, List[int]]]:

        if bank_phi.shape[1] == 0:
            return list(range(new_model_phi.shape[1])), dict()

        assert bank_phi.shape[0] == new_model_phi.shape[0]

        # TODO: think about bank_phi.shape[1] == 1: alright to proceed?

        hierarchy = artm.hARTM(num_processors=1)

        level0 = hierarchy.add_level(
            num_topics=bank_phi.shape[1]
        )
        level0.initialize(dictionary=self._dataset.get_dictionary())
        _safe_copy_phi(
            level0, bank_phi, self._dataset,
            small_num_fit_iterations=1
        )

        level1 = hierarchy.add_level(
            num_topics=new_model_phi.shape[1],
            parent_level_weight=1
        )
        level1.initialize(dictionary=self._dataset.get_dictionary())

        # Regularizer may help to refine new topics a bit
        # in search of parent-child relationship
        # However, the regularizer won't affect the topics themselves,
        # only the ARTM hierarchy defined here.

        # TODO: or smaller tau? or without regularizer at all? or change the real topics?
        level1.regularizers.add(
            artm.HierarchySparsingThetaRegularizer(
                name='sparse_hierarchy',
                tau=1.0
            )
        )
        _safe_copy_phi(
            level1, new_model_phi, self._dataset,
            small_num_fit_iterations=3
        )

        psi = level1.get_psi()

        assert psi.shape[0] == new_model_phi.shape[1]
        assert psi.shape[1] == bank_phi.shape[1]

        if psi_threshold is None:
            psi_threshold = 1.0 / psi.shape[0]

        topics_for_append: List[int] = list()
        topics_for_update: Dict[int, List[int]] = defaultdict(list)

        for new_topic in range(level1.get_phi().shape[1]):
            psi_row = psi.iloc[new_topic, :]
            parents = np.where(psi_row > psi_threshold)[0]

            if len(parents) > 1:
                pass  # linearly dependent -> skip
            elif len(parents) == 0:
                topics_for_append.append(new_topic)
            elif len(parents) == 1:
                topics_for_update[parents[0]].append(new_topic)
            else:
                assert False

        hierarchy.del_level(1)
        hierarchy.del_level(0)

        del hierarchy

        return topics_for_append, topics_for_update

    @staticmethod
    def _keep_good_new_topics_only(
            topics_for_append: List[int],
            topics_for_update: Dict[int, List[int]],
            good_new_topics: List[int]) -> Tuple[List[int], Dict[int, List[int]], Dict[int, int]]:

        topics_for_append = [t for t in topics_for_append if t in good_new_topics]

        topics_for_update_new = dict()

        for old_topic, new_topic_candidates in topics_for_update.items():
            if all([t in good_new_topics for t in new_topic_candidates]):
                topics_for_update_new[old_topic] = new_topic_candidates

        topics_for_update = topics_for_update_new

        topics_for_update_reverse = dict()

        for old_topic, new_topics in topics_for_update.items():
            for new_topic in new_topics:
                assert new_topic not in topics_for_update_reverse  # only one parent

                topics_for_update_reverse[new_topic] = old_topic

        return (
            topics_for_append,
            topics_for_update,
            topics_for_update_reverse
        )

    @staticmethod
    def _jaccard_distance(
            p: Dict[str, float],
            q: Dict[str, float],
            kernel_only: bool = True) -> float:

        numerator = 0
        denominator = 0

        if not kernel_only:
            vocabulary_a = set([w for w in p.keys()])
            vocabulary_b = set([w for w in q.keys()])
        else:
            vocabulary_a = set([w for w in p.keys() if p[w] > 1.0 / len(p)])
            vocabulary_b = set([w for w in q.keys() if q[w] > 1.0 / len(q)])

        common_vocabulary = vocabulary_a.intersection(vocabulary_b)
        only_a_vocabulary = vocabulary_a.difference(vocabulary_b)
        only_b_vocabulary = vocabulary_b.difference(vocabulary_a)

        numerator = numerator + sum(min(p[w], q[w])
                                    for w in common_vocabulary)
        denominator = denominator + (
                sum(p[w] for w in only_a_vocabulary) +
                sum(q[w] for w in only_b_vocabulary) +
                sum(max(p[w], q[w])
                    for w in common_vocabulary)
        )

        if denominator == 0:  # both zero topics
            return TopicBankMethod._MINIMUM_TOPIC_DISTANCE

        distance = TopicBankMethod._MAXIMUM_TOPIC_DISTANCE - numerator / denominator

        distance = max(TopicBankMethod._MINIMUM_TOPIC_DISTANCE, distance)
        distance = min(TopicBankMethod._MAXIMUM_TOPIC_DISTANCE, distance)

        return distance

    @staticmethod
    def _get_phi(
            topics: List[Dict[TokenType, float]],
            word2index: Dict[str, int]) -> pd.DataFrame:

        phi = pd.DataFrame.from_dict({
            f'topic_{i}': words for i, words in enumerate(topics)
        })

        phi = phi.reindex(list(word2index.keys()), fill_value=0.0)
        phi.fillna(0.0, inplace=True)

        return phi

    def _get_default_scores(self, topic_model: TopicModel) -> Dict[str, float]:
        score_values = dict()

        for score in self._all_model_scores:
            # TODO: check here

            score_values[score.name] = (
                topic_model.scores[score.name][-1]
            )

        return score_values

    def _compute_raw_topic_scores(
            self,
            topic_model: TopicModel,
            documents: List[str] = None) -> Dict[str, Dict[str, float]]:

        score_values = dict()

        for score in self._all_topic_scores:
            score_name = score.name
            score_values[score_name] = score.compute(topic_model, documents=documents)

        return score_values

    def _compute_topic_scores(
            self,
            topic_model: TopicModel,
            documents: List[str]) -> Dict[str, float]:

        score_values = dict()

        raw_score_values = self._compute_raw_topic_scores(
            topic_model, documents=documents
        )

        for score_name, raw_values in raw_score_values.items():
            score_values[score_name] = TopicBankMethod._aggregate_scores_for_models(
                raw_values
            )

        return score_values

    @staticmethod
    def _aggregate_scores_for_models(topic_scores: Dict[str, float], p: int = 50) -> float:
        values = list(v for k, v in topic_scores.items() if v is not None)

        if len(values) == 0:
            return 0  # TODO: 0 -- so as not to think about it much

        return np.percentile(values, p)

    @staticmethod
    def _average_scores_over_measurements(scores: List[Dict[str, float]]) -> Dict[str, float]:
        result = dict()

        if len(scores) == 0:
            return result

        for s in scores[0]:
            result[s] = float(np.mean(list(v[s] for v in scores)))

        return result
