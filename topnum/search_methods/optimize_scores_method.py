import logging
import numpy as np
from topicnet.cooking_machine.models import TopicModel
from topicnet.cooking_machine.model_constructor import init_simple_default_model
from typing import List

from .base_search_method import (
    BaseSearchMethod,
    _KEY_VALUES
)
from .constants import (
    DEFAULT_MAX_NUM_TOPICS,
    DEFAULT_MIN_NUM_TOPICS,
    DEFAULT_NUM_FIT_ITERATIONS,
    DEFAULT_DIR
)
from ..data.vowpal_wabbit_text_collection import VowpalWabbitTextCollection
from ..scores.base_score import BaseScore

from topicnet.cooking_machine.cubes import CubeCreator
from topicnet.cooking_machine import Experiment
import pandas as pd

_KEY_SCORE_RESULTS = 'score_results'
_KEY_SCORE_VALUES = 'score_values'

_logger = logging.getLogger()


class OptimizeScoresMethod(BaseSearchMethod):
    def __init__(
            self,
            scores: List[BaseScore],  # TODO: Union[BaseScore, List[BaseScore]]
            num_restarts: int = 3,
            num_topics_interval: int = 10,
            min_num_topics: int = DEFAULT_MIN_NUM_TOPICS,
            max_num_topics: int = DEFAULT_MAX_NUM_TOPICS,
            num_fit_iterations: int = DEFAULT_NUM_FIT_ITERATIONS):
            experiment_dir: str = DEFAULT_DIR):

        super().__init__(min_num_topics, max_num_topics, num_fit_iterations)

        self._scores = scores
        self._num_restarts = num_restarts
        self._num_topics_interval = num_topics_interval

        self._result = dict()
        self._experiment_dir = experiment_dir

        self._key_num_topics_values = _KEY_VALUES.format('num_topics')
        self._key_score_values = _KEY_SCORE_VALUES

        for key in [self._key_num_topics_values, self._key_score_values]:
            # TODO: no need to take mean for num_topics: it should be the same for all restarts
            self._keys_mean_many.append(key)
            self._keys_std_many.append(key)

    def search_for_optimum(self, text_collection: VowpalWabbitTextCollection) -> None:
        _logger.info('Starting to search for optimum...')

        dataset = text_collection._to_dataset()

        seeds = [i-1 for i in range(self._num_restarts)]
        nums_topics = list(range(
            self._min_num_topics,
            self._max_num_topics + 1,
            self._num_topics_interval)
        )

        
        n_bcg_topics = 0  # TODO: or better add ability to specify?
        artm_model = init_simple_default_model(
            dataset,
            modalities_to_use=list(text_collection._modalities.keys()),
            modalities_weights=text_collection._modalities,  # TODO: remove after release
            main_modality=text_collection._main_modality,
            specific_topics=nums_topics[0],   # doesn't matter, will be overwritten in experiment
            background_topics=n_bcg_topics
        )

        if n_bcg_topics:
            del artm_model.regularizers._data['smooth_theta_bcg']
            del artm_model.regularizers._data['smooth_phi_bcg']
        artm_model.num_processors = 5

        model = TopicModel(artm_model)

        # TODO: Find out, why in Renyi entropy test the score already in model here
        _logger.info(
            f'Model\'s custom scores before attaching: {list(model.custom_scores.keys())}'
        )

        for score in self._scores:
            score._attach(model)

        cube = CubeCreator(
            num_iter=self._num_collection_passes,
            parameters={
                "seed": seeds,
                "num_topics": nums_topics
            },
            verbose=False
        )
        exp = Experiment(model, "num_topics_search", self._experiment_dir)
        cube(model, dataset)

        result_models = exp.select()
        restarts = "seed=" + pd.Series(seeds, name="restart_id").astype(str)

        detailed_resut = dict()
        result = {}
        result[_KEY_SCORE_RESULTS] = dict()
        for score in self._scores:
            score_df = pd.DataFrame(index=restarts, columns=nums_topics)
            for model in result_models:
                score_values = model.scores[score.name][-1]
                score_df.loc[f"seed={model.seed}", len(model.topic_names)] = score_values
            detailed_resut[score.name] = score_df.astype(float)

        self._detailed_result = detailed_resut

        for score in self._scores:
            score_df = detailed_resut[score.name]
            score_result = {}
            optimum_series = score_df.idxmin(axis=1)
            score_result['optimum'] = optimum_series.median()
            score_result['optimum_std'] = optimum_series.std()
            score_result['num_topics_values'] = list(score_df.columns)

            score_result['score_values'] = score_df.mean(axis=0)
            score_result['score_values_std'] = score_df.std(axis=0)

            result[_KEY_SCORE_RESULTS][score.name] = score_result

        self._result = result

        _logger.info('Finished searching!')
