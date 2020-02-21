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
import uuid
import os
from tqdm import tqdm


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
            num_fit_iterations: int = DEFAULT_NUM_FIT_ITERATIONS,
            experiment_name: str or None = None,
            experiment_directory: str = DEFAULT_DIR):

        super().__init__(min_num_topics, max_num_topics, num_fit_iterations)

        self._scores = scores
        self._num_restarts = num_restarts
        self._num_topics_interval = num_topics_interval

        self._result = dict()
        self._detailed_result = dict()
        if experiment_name is None:
            experiment_name = str(uuid.uuid4())[:8] + '_experiment'
        self._experiment_name = experiment_name
        self._experiment_directory = experiment_directory

        self._key_num_topics_values = _KEY_VALUES.format('num_topics')
        self._key_score_values = _KEY_SCORE_VALUES

        for key in [self._key_num_topics_values, self._key_score_values]:
            # TODO: no need to take mean for num_topics: it should be the same for all restarts
            self._keys_mean_many.append(key)
            self._keys_std_many.append(key)

    def search_for_optimum(self, text_collection: VowpalWabbitTextCollection) -> None:
        _logger.info('Starting to search for optimum...')

        dataset = text_collection._to_dataset()

        # seed == -1 is too similar to seed == 0
        seeds = [-1] + list(range(1, self._num_restarts))

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

        # remove regularizers created by default
        # ifregularizers are needed, we will add them explicitly
        if n_bcg_topics:
            del artm_model.regularizers._data['smooth_theta_bcg']
            del artm_model.regularizers._data['smooth_phi_bcg']
        artm_model.num_processors = 3

        model = TopicModel(artm_model)

        # TODO: Find out, why in Renyi entropy test the score already in model here
        _logger.info(
            f'Model\'s custom scores before attaching: {list(model.custom_scores.keys())}'
        )

        for score in self._scores:
            score._attach(model)

        result_models = []
        for seed in tqdm(seeds):  # dirty workaround for 'too many models' issue
            exp_model = model.clone()
            cube = CubeCreator(
                num_iter=self._num_collection_passes,
                parameters={
                    "seed": [seed],
                    "num_topics": nums_topics
                },
                verbose=False,
                separate_thread=True
            )
            exp = Experiment(exp_model, f"{self._experiment_name}_{seed}", self._experiment_directory)
            print(exp.save_path)
            cube(exp_model, dataset)

            result_models += exp.select()
            del exp

        restarts = "seed=" + pd.Series(seeds, name="restart_id").astype(str)
        result, detailed_result = summarize_models(
            result_models,
            [s.name for s in self._scores],
            restarts
        )
        self._detailed_result = detailed_result
        self._result = result

        _logger.info('Finished searching!')

def summarize_models(result_models, score_names=None, restarts=None):
    detailed_resut = dict()
    result = {}
    result[_KEY_SCORE_RESULTS] = dict()
    if score_names is None:
        any_model = result_models[-1]
        score_names = any_model.describe_scores().reset_index().score_name.values

    nums_topics = sorted(list({len(tm.topic_names) for tm in result_models}))
    if restarts is None:
        seeds = list({tm.seed for tm in result_models})
        restarts = "seed=" + pd.Series(seeds, name="restart_id").astype(str)

    for score in score_names:
        score_df = pd.DataFrame(index=restarts, columns=nums_topics)
        for model in result_models:
            score_values = model.scores[score][-1]
            if isinstance(score_values, dict):
                continue
            score_df.loc[f"seed={model.seed}", len(model.topic_names)] = score_values
        detailed_resut[score] = score_df.astype(float)


    for score in score_names:
        score_df = detailed_resut[score]
        score_result = {}
        optimum_series = score_df.idxmin(axis=1)
        score_result['optimum'] = float(optimum_series.median())
        score_result['optimum_std'] = float(optimum_series.std())
        score_result['num_topics_values'] = list(score_df.columns)

        score_result['score_values'] = score_df.mean(axis=0).tolist()
        score_result['score_values_std'] = score_df.std(axis=0).tolist()

        result[_KEY_SCORE_RESULTS][score] = score_result
    return result, detailed_resut


def restore_failed_experiment(experiment_directory, experiment_name, scores=None):
    from topicnet.cooking_machine.experiment import START

    folder = os.path.join(experiment_directory, experiment_name)
    model_pathes = [
        f.path for f in os.scandir(folder)
        if f.is_dir() and f.name != START
    ]
    result_models = [TopicModel.load(path) for path in model_pathes]

    return summarize_models(result_models)
