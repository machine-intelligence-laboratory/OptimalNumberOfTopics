import artm
import json
import logging
import numpy as np
import pandas as pd
from topicnet.cooking_machine.models import TopicModel
from topicnet.cooking_machine.model_constructor import init_simple_default_model

from .base_search_method import BaseSearchMethod
from .constants import (
    DEFAULT_MAX_NUM_TOPICS,
    DEFAULT_MIN_NUM_TOPICS,
    DEFAULT_NUM_COLLECTION_PASSES
)
from ..data.vowpal_wabbit_text_collection import VowpalWabbitTextCollection
from ..scores.base_score import BaseScore


_DDOF = 1

_OPTIMUM = 'optimum'
_OPTIMUM_STD = 'optimum_std'
_SCORE_VALUES = '{}_values'
_SCORE_VALUES_STD = '{}_values_std'


logger = logging.getLogger()


class TopicBankMethod(BaseSearchMethod):
    def __init__(
            self,
            score: BaseScore,
            max_num_topics: int = DEFAULT_MAX_NUM_TOPICS,
            num_collection_passes: int = DEFAULT_NUM_COLLECTION_PASSES):

        min_num_topics = 1  # not needed

        super().__init__(min_num_topics, max_num_topics, num_collection_passes)

        self._score = score

        self._result = dict()

        self._key_optimum = _OPTIMUM
        self._key_optimum_std = _OPTIMUM_STD
        self._key_score_values = _SCORE_VALUES.format(self._score.name)
        self._key_score_values_std = _SCORE_VALUES_STD.format(self._score.name)

    def search_for_optimum(self, text_collection: VowpalWabbitTextCollection) -> None:
        logger.info('Starting to search for optimum...')

        dataset = text_collection._to_dataset()
        restart_results = list()

        for i in range(self._num_restarts):
            seed = i - 1  # so as to use also seed = -1 (whoever knows what this means in ARTM)
            need_set_seed = seed >= 0

            logger.info(f'Seed is {seed}')

            restart_result = dict()
            restart_result[self._key_optimum] = None
            restart_result[self._key_score_values] = list()


            artm_model = init_simple_default_model(
                dataset,
                modalities_to_use=list(text_collection._modalities.keys()),
                modalities_weights=text_collection._modalities,  # TODO: remove after release
                main_modality=text_collection._main_modality,
                specific_topics=num_topics,
                background_topics=0  # TODO: or better add ability to specify?
            )

            if need_set_seed:
                artm_model.seed = seed  # TODO: seed -> init_simple_default_model

            model = TopicModel(artm_model)

            # TODO: Find out, why in Renyi entropy test the score already in model here
            logger.info(
                f'Model\'s custom scores before attaching: {list(model.custom_scores.keys())}'
            )

            self._score._attach(model)

            model._fit(
                dataset.get_batch_vectorizer(),
                num_iterations=self._num_collection_passes
            )

            # Assume score name won't change
            score_values = model.scores[self._score._name]

            restart_result[self._key_score_values].append(
                score_values[-1]
            )


            restart_results.append(restart_result)

        result = dict()

        result[self._key_optimum] = int(np.mean([
            r[self._key_optimum] for r in restart_results
        ]))
        result[self._key_optimum_std] = np.std(
            [r[self._key_optimum] for r in restart_results],
            ddof=_DDOF
        ).tolist()

        result[self._key_score_values] = np.mean(
            np.stack([r[self._key_score_values] for r in restart_results]),
            axis=0
        ).tolist()
        result[self._key_score_values_std] = np.std(
            np.stack([r[self._key_score_values] for r in restart_results]),
            ddof=_DDOF,
            axis=0
        ).tolist()

        self._result = result

        logger.info('Finished searching!')

    @staticmethod
    def jaccard(p, q) -> float:
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

    def some_stuff(self):
        bank_topics = []

        NUM_ITERATIONS = 20
        NUM_TOPICS = 100
        DISTANCE = self.jaccard
        MAX_NUM_MODELS = 50

        model_scores = []
        bank_scores = []

        postfix = '__cdc_thresh90'

        seed = 0

        while seed < MAX_NUM_MODELS:
            print('Seed:', seed)

            print('Building topic model')

            tm = self._get_topic_model(dataset, num_topics=NUM_TOPICS, dictionary=dictionary, seed=seed)

            # ***Non-random initialization***

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

            # ***End of non-random initialization***

            #     tm._model.regularizers.add(
            #         artm.regularizers.DecorrelatorPhiRegularizer(tau=10**6)
            #     )

            # for t in list(tm.get_phi().columns)[:NUM_CDC_TOPICS]:  # NUM_ARORA_TOPICS / NUM_CDC_TOPICS
            #     tm._model.regularizers.add(
            #         artm.regularizers.SmoothSparsePhiRegularizer(
            #             tau=1e-5,
            #             topic_names=t
            #         )
            #     )

            #     tm._model.regularizers.add(
            #         artm.regularizers.SmoothSparsePhiRegularizer(tau=0.01, topic_names=list(tm.get_phi().columns)[-1])
            #     )
            #     tm._model.regularizers.add(
            #         artm.regularizers.SmoothSparsePhiRegularizer(tau=0.01, topic_names=list(tm.get_phi().columns)[-2])
            #     )

            #     tm._model.regularizers.add(
            #         artm.regularizers.DecorrelatorPhiRegularizer(name='reg', tau=10**5)
            #     )

            #     tm._model.fit_offline(
            #         dataset.get_batch_vectorizer(),
            #         num_collection_passes=NUM_ITERATIONS // 2  # TODO !!!!!!
            #     )

            #     tm._model.regularizers['reg'].tau = 0

            #     tm._model.regularizers.add(
            #         artm.regularizers.SmoothSparsePhiRegularizer(tau=1e-5)
            #     )

            tm._model.fit_offline(
                dataset.get_batch_vectorizer(),
                num_collection_passes=NUM_ITERATIONS
            )

            # !!!! Removing Background Topics !!!
            #     _phi = tm.get_phi().iloc[:, :-2]

            #     tm = get_topic_model(dataset, phi=_phi, dictionary=dictionary)

            #     tm.fit_offline(dataset.get_batch_vectorizer(), 1)

            #     (_, phi_ref) = tm._model.master.attach_model(
            #         model=tm._model.model_pwt
            #     )

            #     phi_new = np.copy(phi_ref)
            #     phi_new[:, :_phi.shape[1]] = _phi.values

            #     np.copyto(
            #         phi_ref,
            #         phi_new
            #     )

            #     tm.fit_offline(dataset.get_batch_vectorizer(), 1)

            #     del _phi

            # !!!

            scores = dict()

            print('Compute default scores')
            scores.update(self._get_default_scores(tm))

            #     print('Compute top tokens scores')
            #     scores.update(get_top_tokens_scores(tm))

            print('Compute intratext scores (and threshold)')
            raw_intratext_scores = self._get_raw_intratext_scores(tm)

            #     scores.update(
            #         {INTRATEXT_SCORE_NAME: aggregate_scores_for_models(
            #             raw_intratext_scores[INTRATEXT_SCORE_NAME], 50)
            #         }
            #     )

            model_scores.append(scores)

            with open(f'model_scores{postfix}.json', 'w') as f:
                f.write(json.dumps(model_scores))

            threshold = self._aggregate_scores_for_models(
                raw_intratext_scores[INTRATEXT_SCORE_NAME], THRESHOLD_PERCENTILE
            )

            print('Finding new topics')

            phi = tm.get_phi().xs('@default_class', level=0)

            good_topic_names = [
                t for t in phi.columns
                if raw_intratext_scores[INTRATEXT_SCORE_NAME][t] >= threshold
            ]

            if len(bank_topics) == 0 and len(good_topic_names) > 0:
                bank_topics.append(phi.loc[:, good_topic_names[0]].to_dict())

            # TODO: better to compare also with each other
            new_topic_names = [
                t for t in good_topic_names
                if (min(DISTANCE(phi.loc[:, t].to_dict(), bt) for bt in
                        bank_topics) >= DISTANCE_THRESHOLD)
            ]

            for t in new_topic_names:
                bank_topics.append(phi.loc[:, t].to_dict())

            print('Scoring bank model')

            bank_phi = self._get_phi(bank_topics)
            bank_model = self._get_topic_model(dataset, phi=bank_phi, dictionary=dictionary)

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

            print('Compute default scores')
            scores.update(get_default_scores(bank_model))

            #     print('Compute top tokens scores')
            #     scores.update(get_top_tokens_scores(bank_model))

            #     print('Compute intratext scores')
            #     scores.update(get_intratext_scores(bank_model))

            scores.update({'num_topics': bank_phi.shape[1]})

            print('Num topics in bank:', len(bank_topics))

            bank_scores.append(scores)

            with open(f'bank_scores{postfix}.json', 'w') as f:
                f.write(json.dumps(bank_scores))

            seed += 1

            print()

    @staticmethod
    def _get_topic_model(
            phi,
            dataset,
            dictionary=None,
            num_topics=None,
            seed=None,
            additional_phi=None):  # TODO: workaround

        if dictionary is None:
            dictionary = dataset.get_dictionary()

        if num_topics is not None:
            assert num_topics >= phi.shape[1]
        else:
            num_topics = phi.shape[1]

        topic_names = [f'topic_{i}' for i in range(num_topics)]

        if seed is None:
            artm_model = artm.ARTM(topic_names=topic_names)
        else:
            artm_model = artm.ARTM(topic_names=topic_names, seed=seed)

        artm_model.initialize(dictionary)

        (_, phi_ref) = artm_model.master.attach_model(
            model=artm_model.model_pwt
        )

        phi_new = np.copy(phi_ref)
        phi_new[:, :phi.shape[1]] = phi.values

        if additional_phi is not None:
            phi_new[:, phi.shape[1]:phi.shape[1] + additional_phi.shape[1]] = additional_phi.values

        np.copyto(
            phi_ref,
            phi_new
        )

        initial_phi = np.array(phi_new)

        num_times_to_fit_for_scores = 3

        for _ in range(num_times_to_fit_for_scores):
            (_, phi_ref) = artm_model.master.attach_model(
                model=artm_model.model_pwt
            )

            np.copyto(
                phi_ref,
                initial_phi
            )

            artm_model.fit_offline(dataset.get_batch_vectorizer(), 1)

        topic_model = TopicModel(
            artm_model=artm_model,
            model_id='0',
            cache_theta=True,
            theta_columns_naming='id'
        )

        return topic_model

    @staticmethod
    def _get_phi(topics):
        topic_names = [f'topic_{i}' for i in range(len(topics))]

        phi = pd.DataFrame.from_dict({
            f'topic_{i}': words for i, words in enumerate(topics)
        })

        phi = phi.reindex(list(word2index.keys()), fill_value=0)

        phi.fillna(0.0, inplace=True)

        return phi

    @staticmethod
    def _get_default_scores(topic_model):
        score_values = {}

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
    def _get_raw_intratext_scores(topic_model):
        score_values = {}

        for score_name, score in intratext_coherence_scores.items():
            score_values[score_name] = score.compute(topic_model)

        return score_values

    @staticmethod
    def _get_intratext_scores(topic_model):
        score_values = {}

        raw_score_values = TopicBankMethod._get_raw_intratext_scores(topic_model)

        for score_name, raw_values in raw_score_values.items():
            score_values[score_name] = TopicBankMethod._aggregate_scores_for_models(raw_values)

        return score_values

    @staticmethod
    def _aggregate_scores_for_models(topic_scores, p=50):
        values = list(v for k, v in topic_scores.items() if v is not None)

        if len(values) == 0:
            return 0  # TODO: 0 -- so as not to think about it much

        return np.percentile(values, p)
