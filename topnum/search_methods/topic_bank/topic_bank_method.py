import artm
import json
import logging
import numpy as np
import pandas as pd

from collections import Counter
from topicnet.cooking_machine.models import TopicModel
from typing import (
    Dict,
    List
)

from topnum.scores._base_coherence_score import (
    SpecificityEstimationMethod,
    TextType,
    WordTopicRelatednessType
)
from topnum.scores.intratext_coherence_score import (
    _IntratextCoherenceScore,
    ComputationMethod
)
from topnum.scores.sophisticated_toptok_coherence_score import _TopTokensCoherenceScore
from topnum.search_methods.base_search_method import BaseSearchMethod
from topnum.search_methods.constants import (
    DEFAULT_MAX_NUM_TOPICS,
    DEFAULT_MIN_NUM_TOPICS,
    DEFAULT_NUM_FIT_ITERATIONS
)
from topnum.data.vowpal_wabbit_text_collection import VowpalWabbitTextCollection


_logger = logging.getLogger()


# TODO: remove this!
INTRATEXT_SCORE_NAME = 'TextType.VW_TEXT__2'
TOPTOKENS_SCORE_NAME = 'TextType.VW_TEXT__10__10'


class TopicBankMethod(BaseSearchMethod):
    def __init__(
            self,
            main_modality: str,
            one_model_num_topics: int = 100,
            num_fit_iterations: int = DEFAULT_NUM_FIT_ITERATIONS,
            max_num_models: int = 100,
            topic_score_threshold_percentile: int = 95,
            distance_threshold: float = 0.5):  # TODO: say that distance between 0 and 1

        super().__init__(
            min_num_topics=DEFAULT_MIN_NUM_TOPICS,  # not needed
            max_num_topics=DEFAULT_MAX_NUM_TOPICS,  # not needed
            num_fit_iterations=num_fit_iterations
        )

        self._main_modality = main_modality
        self._one_model_num_topics = one_model_num_topics
        self._max_num_models = max_num_models
        self._topic_score_threshold_percentile = topic_score_threshold_percentile
        self._distance_threshold = distance_threshold

        self._result = dict()

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
        model_scores = list()
        bank_scores = list()

        i = 0

        while i < self._max_num_models:
            # TODO: break when perplexity stabilizes
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

            _logger.info('Computing default scores for one topic model...')
            scores.update(self._get_default_scores(tm))

            _logger.info('Computing top tokens scores for one topic model...')

            raw_top_tokens_scores = self._get_raw_toptokens_scores(
                tm, dataset, documents_for_coherence)
            scores.update(
                {TOPTOKENS_SCORE_NAME: self._aggregate_scores_for_models(
                    raw_top_tokens_scores[TOPTOKENS_SCORE_NAME], 50)
                }
            )

            _logger.info('Computing intratext scores for one topic model...')

            raw_intratext_scores = self._get_raw_intratext_scores(
                tm, dataset, documents_for_coherence)
            scores.update(
                {INTRATEXT_SCORE_NAME: self._aggregate_scores_for_models(
                    raw_intratext_scores[INTRATEXT_SCORE_NAME], 50)
                }
            )

            model_scores.append(scores)

            # TODO: some save folder
            with open(f'model_scores.json', 'w') as f:
                f.write(json.dumps(model_scores))

            threshold = self._aggregate_scores_for_models(
                raw_intratext_scores[INTRATEXT_SCORE_NAME],
                self._topic_score_threshold_percentile
            )

            print('Finding new topics')

            # TODO: or all words somehow?
            phi = tm.get_phi().xs(self._main_modality, level=0)

            good_topic_names = [
                t for t in phi.columns
                if raw_intratext_scores[INTRATEXT_SCORE_NAME][t] >= threshold
            ]

            if len(bank_topics) == 0 and len(good_topic_names) > 0:
                bank_topics.append(phi.loc[:, good_topic_names[0]].to_dict())

            # TODO: better to compare also with each other
            new_topic_names = [
                t for t in good_topic_names
                if (min(self._jaccard_distance(phi.loc[:, t].to_dict(), bt)
                        for bt in bank_topics) >= self._distance_threshold)
            ]

            for t in new_topic_names:
                bank_topics.append(phi.loc[:, t].to_dict())

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

            _logger.info('Computing top tokens scores for bank model...')
            scores.update(self._get_top_tokens_scores(bank_model, dataset, documents_for_coherence))

            _logger.info('Computing intratext scores for bank model...')
            scores.update(self._get_intratext_scores(bank_model, dataset, documents_for_coherence))

            scores.update({'num_topics': bank_phi.shape[1]})

            _logger.info(f'Num topics in bank: {len(bank_topics)}')

            bank_scores.append(scores)

            # TODO: save folder
            with open(f'bank_scores.json', 'w') as f:
                f.write(json.dumps(bank_scores))

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

    @staticmethod
    def _get_topic_model(
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
        artm_model.scores.add(
            artm.scores.PerplexityScore(
                name=f'perplexity'
            )
        )
        artm_model.scores.add(
            artm.scores.SparsityPhiScore(
                name=f'sparsity_phi'
            )
        )
        artm_model.scores.add(
            artm.scores.SparsityThetaScore(
                name=f'sparsity_theta'
            )
        )

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

        return topic_model

    @staticmethod
    def _get_phi(topics, word2index):
        phi = pd.DataFrame.from_dict({
            f'topic_{i}': words for i, words in enumerate(topics)
        })

        phi = phi.reindex(list(word2index.keys()), fill_value=0)

        phi.fillna(0.0, inplace=True)

        return phi

    @staticmethod
    def _get_default_scores(topic_model):
        score_values = dict()

        for score in topic_model._model.score_tracker:
            if 'perplexity' in score or 'sparsity' in score:
                score_values[score] = (
                    topic_model._model.score_tracker[score].last_value
                )

                continue

            score_values[score + '__last_purity'] = TopicBankMethod._aggregate_scores_for_models(
                topic_model._model.score_tracker[score].last_purity
            )
            score_values[score + '__last_contrast'] = TopicBankMethod._aggregate_scores_for_models(
                topic_model._model.score_tracker[score].last_contrast
            )

        return score_values

    @staticmethod
    def _get_raw_intratext_scores(topic_model, dataset, documents_for_coherence):
        num_out = 5
        window = 20

        intratext_coherence_scores = {
            f'TextType.VW_TEXT__{cm}': _IntratextCoherenceScore(
                dataset,
                documents_for_coherence,
                TextType.VW_TEXT,
                cm,
                WordTopicRelatednessType.PWT,
                SpecificityEstimationMethod.NONE,
                max_num_out_of_topic_words=num_out,
                window=window
            )
            for cm in [ComputationMethod.SEGMENT_WEIGHT]
        }

        score_values = {}

        for score_name, score in intratext_coherence_scores.items():
            score_values[score_name] = score.compute(topic_model)

        return score_values

    @staticmethod
    def _get_intratext_scores(topic_model, dataset, documents_for_coherence):
        score_values = {}

        raw_score_values = TopicBankMethod._get_raw_intratext_scores(
            topic_model, dataset, documents_for_coherence)

        for score_name, raw_values in raw_score_values.items():
            score_values[score_name] = TopicBankMethod._aggregate_scores_for_models(raw_values)

        return score_values

    @staticmethod
    def _get_raw_toptokens_scores(topic_model, dataset, documents_for_coherence):
        num_top = 10
        window = 10

        top_tokens_coherence_scores = {
            f'TextType.VW_TEXT__{num_top}__{window}': _TopTokensCoherenceScore(
                dataset,
                documents_for_coherence,
                TextType.VW_TEXT,
                WordTopicRelatednessType.PWT,
                SpecificityEstimationMethod.NONE,
                num_top_words=num_top,
                window=window
            )
        }

        score_values = {}

        for score_name, score in top_tokens_coherence_scores.items():
            score_values[score_name] = score.compute(topic_model)

        return score_values

    @staticmethod
    def _get_top_tokens_scores(topic_model, dataset, documents_for_coherence):
        score_values = {}

        raw_score_values = TopicBankMethod._get_raw_toptokens_scores(
            topic_model, dataset, documents_for_coherence)

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
