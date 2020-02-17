import logging
import numpy as np
from topicnet.cooking_machine.models import TopicModel
from topicnet.cooking_machine.model_constructor import init_simple_default_model

from .base_search_method import (
    BaseSearchMethod,
    _KEY_VALUES
)
from .constants import (
    DEFAULT_MAX_NUM_TOPICS,
    DEFAULT_MIN_NUM_TOPICS,
    DEFAULT_NUM_COLLECTION_PASSES
)
from ..data.vowpal_wabbit_text_collection import VowpalWabbitTextCollection
from ..scores.base_score import BaseScore


_DDOF = 1


logger = logging.getLogger()


class OptimizeScoreMethod(BaseSearchMethod):
    def __init__(
            self,
            score: BaseScore,
            num_restarts: int = 3,
            num_topics_interval: int = 10,
            min_num_topics: int = DEFAULT_MIN_NUM_TOPICS,
            max_num_topics: int = DEFAULT_MAX_NUM_TOPICS,
            num_collection_passes: int = DEFAULT_NUM_COLLECTION_PASSES):

        super().__init__(min_num_topics, max_num_topics, num_collection_passes)

        self._score = score
        self._num_restarts = num_restarts
        self._num_topics_interval = num_topics_interval

        self._result = dict()

        self._key_num_topics_values = _KEY_VALUES.format('num_topics')
        self._key_score_values = _KEY_VALUES.format(self._score.name)

        for key in [self._key_num_topics_values, self._key_score_values]:
            # TODO: no need to take mean for num_topics: it should be the same for all restarts
            self._keys_mean_many.append(key)
            self._keys_std_many.append(key)

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

            nums_topics = list(range(
                self._min_num_topics,
                self._max_num_topics + 1,
                self._num_topics_interval))

            restart_result[self._key_num_topics_values] = nums_topics

            for num_topics in nums_topics:

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

            restart_result[self._key_optimum] = nums_topics[
                np.argmin(restart_result[self._key_score_values])
            ]
            restart_results.append(restart_result)

        result = dict()

        self._compute_mean_one(restart_results, result)
        self._compute_std_one(restart_results, result)
        self._compute_mean_many(restart_results, result)
        self._compute_std_many(restart_results, result)

        self._result = result

        logger.info('Finished searching!')
