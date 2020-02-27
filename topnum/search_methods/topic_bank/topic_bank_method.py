import artm
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
            main_topic_score: BaseTopicScore = None,
            other_topic_scores: List[BaseTopicScore] = None,
            stop_bank_score: BaseScore = None,
            other_scores: List[BaseScore] = None,
            one_model_num_topics: int = 100,  # TODO: or list of ints
            num_fit_iterations: int = DEFAULT_NUM_FIT_ITERATIONS,
            max_num_models: int = 100,
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

        if other_scores is None:
            self._other_scores = other_scores
        else:
            self._other_scores = [
                SparsityPhiScore(name='sparsity_phi_score'),
                SparsityThetaScore(name='sparsity_theta_score')
            ]

        self._all_model_scores = [self._stop_bank_score] + self._other_scores

        self._one_model_num_topics = one_model_num_topics
        self._max_num_models = max_num_models
        self._topic_score_threshold_percentile = topic_score_threshold_percentile
        self._distance_threshold = distance_threshold

        if save_file_path is not None:
            if not os.path.isdir(os.path.dirname(save_file_path)):
                raise FileNotFoundError(f'Directory not found "{save_file_path}"')  # TODO: right error?

            if os.path.isfile(save_file_path):
                warnings.warn(f'File "{save_file_path}" already exists! Overwriting')
        else:
            save_file_path = tempfile.mkstemp(prefix='topic_bank_result__')

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
    def save_path(self):
        return self._save_file_path

    def save(self):
        with open(self._save_file_path, 'w') as f:
            f.write(json.dumps(self._result))

    def clear(self):
        if os.path.isfile(self._save_file_path):
            os.remove(self._save_file_path)

    def search_for_optimum(self, text_collection: VowpalWabbitTextCollection) -> None:
        dataset = text_collection._to_dataset()

        # TODO: refine, debug!
        documents = dataset._data['raw_text'].values.tolist()
        all_words_multiset = Counter(
            [w for d in documents for w in d.split()]  # if w not in stopwords
        )  # TODO: filtering stopwords?
        vocabulary = set(w for w, c in all_words_multiset.items() if c >= 30)
        word2index = {w: i for i, w in enumerate(vocabulary)}

        # TODO: select better
        documents_for_coherence = np.random.choice(
            dataset._data['id'].values, size=100, replace=False
        ).tolist()

        bank_topics = list()

        bank_scores = list()

        bank_topic_scores = list()

        i = 0

        while i < self._max_num_models:
            # TODO: stop when perplexity stabilizes
            seed = i - 1  # to use -1 also

            _logger.info(f'Seed: {seed}')
            _logger.info('Building topic model')

            tm = self._get_topic_model(
                dataset=dataset,
                num_topics=self._one_model_num_topics,
                dictionary=dataset.get_dictionary(),
                seed=seed
            )

            # TODO: remove or implement
            # ***Non-random initialization***
            #
            # external_phi = cdc_phi  # arora_phi / cdc_phi
            #
            # (_, phi_ref) = tm._model.master.attach_model(
            #     model=bank_model._model.model_pwt
            # )
            #
            # phi_new = np.copy(phi_ref)
            # phi_new[np.array(list(word2index[w] for w in external_phi.index)), :external_phi.shape[1]] = external_phi.values
            #
            # np.copyto(
            #     phi_ref,
            #     phi_new
            # )
            #
            # ***End of non-random initialization***

            # TODO: regularizers: remove or implement
            #
            #     tm._model.regularizers.add(
            #         artm.regularizers.DecorrelatorPhiRegularizer(tau=10**6)
            #     )
            #
            # for t in list(tm.get_phi().columns)[:NUM_CDC_TOPICS]:  # NUM_ARORA_TOPICS / NUM_CDC_TOPICS
            #     tm._model.regularizers.add(
            #         artm.regularizers.SmoothSparsePhiRegularizer(
            #             tau=1e-5,
            #             topic_names=t
            #         )
            #     )
            #
            #     tm._model.regularizers.add(
            #         artm.regularizers.SmoothSparsePhiRegularizer(tau=0.01, topic_names=list(tm.get_phi().columns)[-1])
            #     )
            #     tm._model.regularizers.add(
            #         artm.regularizers.SmoothSparsePhiRegularizer(tau=0.01, topic_names=list(tm.get_phi().columns)[-2])
            #     )
            #
            #     tm._model.regularizers.add(
            #         artm.regularizers.DecorrelatorPhiRegularizer(name='reg', tau=10**5)
            #     )
            #
            #     tm._model.fit_offline(
            #         dataset.get_batch_vectorizer(),
            #         num_fit_iterations=NUM_ITERATIONS // 2  # TODO !!!!!!
            #     )
            #
            #     tm._model.regularizers['reg'].tau = 0
            #
            #     tm._model.regularizers.add(
            #         artm.regularizers.SmoothSparsePhiRegularizer(tau=1e-5)
            #     )

            tm._model.fit_offline(
                dataset.get_batch_vectorizer(),
                num_collection_passes=self._num_fit_iterations
            )

            # **Removing Background Topics***
            #     _phi = tm.get_phi().iloc[:, :-2]
            #
            #     tm = get_topic_model(dataset, phi=_phi, dictionary=dictionary)
            #
            #     tm.fit_offline(dataset.get_batch_vectorizer(), 1)
            #
            #     (_, phi_ref) = tm._model.master.attach_model(
            #         model=tm._model.model_pwt
            #     )
            #
            #     phi_new = np.copy(phi_ref)
            #     phi_new[:, :_phi.shape[1]] = _phi.values
            #
            #     np.copyto(
            #         phi_ref,
            #         phi_new
            #     )
            #
            #     tm.fit_offline(dataset.get_batch_vectorizer(), 1)
            #
            #     del _phi
            #
            # ***End of removing***

            scores = dict()

            _logger.info('Computing scores for one topic model...')

            scores.update(self._get_default_scores(tm))

            raw_topic_scores = self._compute_raw_topic_scores(
                tm,
                documents_for_coherence
            )

            for score_name, score_values in raw_topic_scores.items():
                scores[score_name] = self._aggregate_scores_for_models(
                    raw_topic_scores[score_name], 50
                )

            self._result[_KEY_MODEL_SCORES].append(scores)
            self._result[_KEY_NUM_MODEL_TOPICS].append(tm.get_phi().shape[1])

            self.save()


            threshold = self._aggregate_scores_for_models(
                raw_topic_scores[self._main_topic_score.name],  # TODO: check
                self._topic_score_threshold_percentile
            )



            print('Finding new topics')

            # TODO: or all words somehow?
            phi = tm.get_phi().xs(self._main_modality, level=0)

            good_topic_names = [
                t for t in phi.columns
                if raw_topic_scores[self._main_topic_score.name][t] >= threshold
            ]




            # if len(bank_topics) == 0 and len(good_topic_names) > 0:
            #     # Adding first topic of there are no one in the bank
            #     bank_topics.append(phi.loc[:, good_topic_names[0]].to_dict())
            #
            # # TODO: better to compare also with each other
            # new_topic_names = [
            #     t for t in good_topic_names
            #     if (min(self._jaccard_distance(phi.loc[:, t].to_dict(), bt)
            #             for bt in bank_topics) >= self._distance_threshold)
            # ]
            #
            # for t in new_topic_names:
            #     bank_topics.append(phi.loc[:, t].to_dict())





            if len(bank_topics) == 0 and len(good_topic_names) > 0:
                # TODO: almost copy-paste!!
                t = good_topic_names[0]

                t_scores = dict()

                v = tm.get_phi()[t].values
                t_scores['kernel'] = len(v[v > 1.0 / tm.get_phi()[t].values.shape[0]])

                # TODO: refine
                for score in raw_topic_scores:
                    t_scores[score.name] = raw_topic_scores[score.name][t]

                t_scores['rho'] = 0.0

                bank_topics.append(phi.loc[:, t].to_dict())
                bank_topic_scores.append(t_scores)

            model_topic_current_scores = list()

            for t in tm.get_phi().columns:
                t_scores = dict()

                v = tm.get_phi()[t].values
                t_scores['kernel'] = len(v[v > 1.0 / tm.get_phi()[t].values.shape[0]])

                # TODO: refine
                for score in raw_topic_scores:
                    t_scores[score.name] = raw_topic_scores[score.name][t]

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
            bank_model = self._get_topic_model(
                dataset,
                phi=bank_phi,
                dictionary=dataset.get_dictionary()
            )

            # TODO: refine this shamanizm
            bank_model.fit_offline(dataset.get_batch_vectorizer(), 1)

            (_, phi_ref) = bank_model._model.master.attach_model(
                model=bank_model._model.model_pwt
            )

            phi_new = np.copy(phi_ref)
            phi_new[:, :bank_phi.shape[1]] = bank_phi.values

            np.copyto(
                phi_ref,
                phi_new
            )

            bank_model.fit_offline(dataset.get_batch_vectorizer(), 1)

            scores = dict()

            _logger.info('Computing default scores for bank model...')

            scores.update(self._get_default_scores(bank_model))

            # Not needed as per topics are calculated
            #
            # _logger.info('Computing top tokens scores for bank model...')
            # scores.update(self._get_top_tokens_scores(bank_model, dataset, documents_for_coherence))
            #
            # _logger.info('Computing intratext scores for bank model...')
            # scores.update(self._get_intratext_scores(bank_model, dataset, documents_for_coherence))

            self._result[_KEY_BANK_SCORES].append(scores)
            self._result[_KEY_NUM_BANK_TOPICS] = bank_phi.shape[1]

            _logger.info(f'Num topics in bank: {len(bank_topics)}')

            #bank_scores.append(scores)

            # TODO: save folder
            self.save()

            i = i + 1

    @staticmethod
    def _jaccard_distance(p, q) -> float:
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

    def _get_topic_model(
            self,
            dataset,
            phi=None,
            dictionary=None,
            num_topics=None,
            seed=None,
            additional_phi=None):  # TODO: workaround

        if dictionary is None:
            dictionary = dataset.get_dictionary()

        if num_topics is not None and phi is not None:
            assert num_topics >= phi.shape[1]
        elif num_topics is None and phi is not None:
            num_topics = phi.shape[1]
        elif num_topics is None and phi is None:
            raise ValueError()

        topic_names = [f'topic_{i}' for i in range(num_topics)]

        if seed is None:
            artm_model = artm.ARTM(topic_names=topic_names)
        else:
            artm_model = artm.ARTM(topic_names=topic_names, seed=seed)

        artm_model.initialize(dictionary)

        # add_topic_kernel_score(artm_model, topic_names)

        if phi is not None:
            (_, phi_ref) = artm_model.master.attach_model(
                model=artm_model.model_pwt
            )

            phi_new = np.copy(phi_ref)
            phi_new[:, :phi.shape[1]] = phi.values

            #     if additional_phi is not None:
            #         phi_new[:, phi.shape[1]:phi.shape[1] + additional_phi.shape[1]] = additional_phi.values

            np.copyto(
                phi_ref,
                phi_new
            )

        #     initial_phi = np.array(phi_new)

        #     num_times_to_fit_for_scores = 3

        #     for _ in range(num_times_to_fit_for_scores):
        #         (_, phi_ref) = artm_model.master.attach_model(
        #             model=artm_model.model_pwt
        #         )

        #         np.copyto(
        #             phi_ref,
        #             initial_phi
        #         )

        #         artm_model.fit_offline(dataset.get_batch_vectorizer(), 1)

        topic_model = TopicModel(
            artm_model=artm_model,
            model_id='0',
            cache_theta=True,
            theta_columns_naming='id'
        )

        for score in self._all_model_scores:
            score._attach(topic_model)

        return topic_model

    @staticmethod
    def _get_phi(topics, word2index):
        phi = pd.DataFrame.from_dict({
            f'topic_{i}': words for i, words in enumerate(topics)
        })

        phi = phi.reindex(list(word2index.keys()), fill_value=0)

        phi.fillna(0.0, inplace=True)

        return phi

    def _get_default_scores(self, topic_model):
        score_values = dict()

        for score in self._all_model_scores:
            # TODO: check here
            score_values[score] = (
                topic_model.scores[score.name].last_value
            )

        return score_values

    def _compute_raw_topic_scores(self, topic_model, documents=None):
        score_values = dict()

        for score in self._all_topic_scores:
            score_name = score.name
            score_values[score_name] = score.compute(topic_model, documents=documents)

        return score_values

    def _compute_topic_scores(self, topic_model, documents):
        score_values = dict()

        raw_score_values = self._compute_raw_topic_scores(
            topic_model, documents=documents
        )

        for score_name, raw_values in raw_score_values.items():
            score_values[score_name] = TopicBankMethod._aggregate_scores_for_models(raw_values)

        return score_values

    @staticmethod
    def _aggregate_scores_for_models(topic_scores, p=50):
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
