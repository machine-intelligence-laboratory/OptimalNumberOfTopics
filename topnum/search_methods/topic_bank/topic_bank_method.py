import json
import logging
import numpy as np
import os
import pandas as pd
import tempfile
import warnings

from collections import Counter
from topicnet.cooking_machine.dataset import Dataset
from topicnet.cooking_machine.models import TopicModel
from typing import (
    Callable,
    Dict,
    List,
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
from topnum.search_methods.topic_bank.train_funcs_zoo import (
    default_train_func,
    _get_topic_model
)


_KEY_BANK_SCORES = 'bank_scores'
_KEY_BANK_TOPIC_SCORES = 'bank_topic_scores'
_KEY_MODEL_SCORES = 'model_scores'
_KEY_MODEL_TOPIC_SCORES = 'model_topic_scores'
_KEY_NUM_BANK_TOPICS = 'num_bank_topics'
_KEY_NUM_MODEL_TOPICS = 'num_model_topics'

_DEFAULT_WINDOW = 20


_logger = logging.getLogger()


class TopicBankMethod(BaseSearchMethod):
    def __init__(
            self,
            data: Union[Dataset, VowpalWabbitTextCollection],
            main_modality: str,
            minimum_word_frequency: int = 30,  # TODO: think about this param
            main_topic_score: BaseTopicScore = None,
            other_topic_scores: List[BaseTopicScore] = None,
            stop_bank_score: BaseScore = None,
            other_scores: List[BaseScore] = None,
            max_num_models: int = 100,
            one_model_num_topics: Union[int, List[int]] = 100,
            num_fit_iterations: int = DEFAULT_NUM_FIT_ITERATIONS,
            train_func: Union[
                Callable[[Dataset, int, int, int], TopicModel],
                List[Callable[[Dataset, int, int, int], TopicModel]],
                None] = None,
            topic_score_threshold_percentile: int = 95,
            distance_threshold: float = 0.5,
            save_file_path: str = None):  # TODO: say that distance between 0 and 1

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
        self._minimum_word_frequency = minimum_word_frequency

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

        self._topic_score_threshold_percentile = topic_score_threshold_percentile
        self._distance_threshold = distance_threshold

        if save_file_path is not None:
            if not os.path.isdir(os.path.dirname(save_file_path)):
                raise FileNotFoundError(f'Directory not found "{save_file_path}"')  # TODO: right error?

            if os.path.isfile(save_file_path):
                warnings.warn(f'File "{save_file_path}" already exists! Overwriting')
        else:
            file_descriptor, save_file_path = tempfile.mkstemp(prefix='topic_bank_result__')
            os.close(file_descriptor)

        self._save_file_path = save_file_path

        self._result = dict()

        self._result[_KEY_OPTIMUM] = None
        self._result[_KEY_OPTIMUM + _STD_KEY_SUFFIX] = None
        self._result[_KEY_BANK_SCORES] = list()
        self._result[_KEY_BANK_TOPIC_SCORES] = list()
        self._result[_KEY_MODEL_SCORES] = list()
        self._result[_KEY_MODEL_TOPIC_SCORES] = list()
        self._result[_KEY_NUM_BANK_TOPICS] = list()
        self._result[_KEY_NUM_MODEL_TOPICS] = list()

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

        # TODO: refactor the stuff right below!
        documents = self._dataset._data['raw_text'].values.tolist()
        all_words_multiset = Counter(
            [w for d in documents for w in d.split()]  # if w not in stopwords
        )  # TODO: filtering stopwords?
        vocabulary = set(
            w for w, c in all_words_multiset.items() if c >= self._minimum_word_frequency
        )
        word2index = {w: i for i, w in enumerate(vocabulary)}

        # TODO: select better
        documents_for_coherence = np.random.choice(
            self._dataset._data['id'].values,
            size=min(100, int(0.2 * len(self._dataset._data['id'].values))),  # TODO: add as param
            replace=False
        ).tolist()

        bank_topics = list()
        bank_topic_scores = list()

        i = 0

        while i < self._max_num_models:
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

            # TODO: or all words somehow?
            phi = topic_model.get_phi().xs(self._main_modality, level=0)

            good_topic_names = [
                t for t in phi.columns
                if raw_topic_scores[self._main_topic_score.name][t] >= threshold
            ]

            if len(bank_topics) == 0 and len(good_topic_names) > 0:
                # TODO: almost copy-paste!!
                t = good_topic_names[0]

                t_scores = dict()

                v = topic_model.get_phi()[t].values
                t_scores['kernel'] = len(v[v > 1.0 / topic_model.get_phi()[t].values.shape[0]])

                # TODO: refine
                for score_name in raw_topic_scores:
                    t_scores[score_name] = raw_topic_scores[score_name][t]

                t_scores['rho'] = 0.0

                bank_topics.append(phi.loc[:, t].to_dict())
                bank_topic_scores.append(t_scores)

            model_topic_current_scores = list()

            for t in topic_model.get_phi().columns:
                t_scores = dict()

                v = topic_model.get_phi()[t].values
                t_scores['kernel'] = len(v[v > 1.0 / topic_model.get_phi()[t].values.shape[0]])

                # TODO: refine
                for score_name in raw_topic_scores:
                    t_scores[score_name] = raw_topic_scores[score_name][t]

                model_topic_current_scores.append(t_scores)

                if t not in good_topic_names:
                    continue

                d = (min(self._jaccard_distance(phi.loc[:, t].to_dict(), bt) for bt in bank_topics))

                if d < self._distance_threshold:
                    continue

                t_scores['rho'] = d

                bank_topics.append(phi.loc[:, t].to_dict())
                bank_topic_scores.append(t_scores)

            self._result[_KEY_MODEL_TOPIC_SCORES].append(model_topic_current_scores)
            self._result[_KEY_BANK_TOPIC_SCORES] = bank_topic_scores  # TODO: append

            self.save()

            _logger.info('Scoring bank model...')

            bank_phi = self._get_phi(bank_topics, word2index)
            bank_model = _get_topic_model(
                self._dataset,
                phi=bank_phi,
                scores=self._all_model_scores
            )

            # TODO: refine this shamanizm
            bank_model.fit_offline(self._dataset.get_batch_vectorizer(), 1)

            (_, phi_ref) = bank_model._model.master.attach_model(
                model=bank_model._model.model_pwt
            )

            phi_new = np.copy(phi_ref)
            phi_new[:, :bank_phi.shape[1]] = bank_phi.values

            np.copyto(
                phi_ref,
                phi_new
            )

            bank_model.fit_offline(self._dataset.get_batch_vectorizer(), 1)

            scores = dict()

            _logger.info('Computing default scores for bank model...')

            scores.update(self._get_default_scores(bank_model))

            # Topic scores already calculated

            self._result[_KEY_BANK_SCORES].append(scores)
            self._result[_KEY_NUM_BANK_TOPICS].append(bank_phi.shape[1])

            _logger.info(f'Num topics in bank: {len(bank_topics)}')

            self.save()

            i = i + 1

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

    @staticmethod
    def _jaccard_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
        numerator = 0
        denominator = 0

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
            return 0

        distance = 1.0 - numerator / denominator

        distance = max(0.0, distance)
        distance = min(1.0, distance)

        return distance

    @staticmethod
    def _get_phi(topics: List[Dict[str, float]], word2index: Dict[str, int]) -> pd.DataFrame:
        phi = pd.DataFrame.from_dict({
            f'topic_{i}': words for i, words in enumerate(topics)
        })

        phi = phi.reindex(list(word2index.keys()), fill_value=0)
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
