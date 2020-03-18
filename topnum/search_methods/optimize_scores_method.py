import logging
import os
import pandas as pd
import uuid

from topicnet.cooking_machine.models import TopicModel
from tqdm import tqdm
from typing import List

from .base_search_method import (
    BaseSearchMethod,
    _KEY_OPTIMUM,
    _KEY_VALUES,
    _STD_KEY_SUFFIX
)
from .constants import (
    DEFAULT_MAX_NUM_TOPICS,
    DEFAULT_MIN_NUM_TOPICS,
    DEFAULT_NUM_FIT_ITERATIONS,
    DEFAULT_EXPERIMENT_DIR
)
from ..data.vowpal_wabbit_text_collection import VowpalWabbitTextCollection
from ..scores.base_score import BaseScore

from ..model_constructor import init_model_from_family

_KEY_SCORE_RESULTS = 'score_results'
_KEY_SCORE_VALUES = 'score_values'

_logger = logging.getLogger()


class OptimizeScoresMethod(BaseSearchMethod):
    def __init__(
            self,
            scores: List[BaseScore],  # TODO: Union[BaseScore, List[BaseScore]]
            model_family: str = "LDA",
            num_restarts: int = 3,
            num_topics_interval: int = 10,
            min_num_topics: int = DEFAULT_MIN_NUM_TOPICS,
            max_num_topics: int = DEFAULT_MAX_NUM_TOPICS,
            num_fit_iterations: int = DEFAULT_NUM_FIT_ITERATIONS,
            one_model_num_processors: int = 3,
            separate_thread: bool = True,
            experiment_name: str or None = None,
            save_experiment: bool = False,
            experiment_directory: str = DEFAULT_EXPERIMENT_DIR):

        super().__init__(min_num_topics, max_num_topics, num_fit_iterations)

        self._scores = scores
        self._family = model_family
        self._num_restarts = num_restarts
        self._num_topics_interval = num_topics_interval

        self._result = dict()
        self._detailed_result = dict()

        if experiment_name is None:
            experiment_name = str(uuid.uuid4())[:8] + '_experiment'

        self._one_model_num_processors = one_model_num_processors
        self._separate_thread = separate_thread

        self._experiment_name = experiment_name
        self._save_experiment = save_experiment
        self._experiment_directory = experiment_directory

        _logger.info(
            f'Experiment name: {self._experiment_name}.'
            f' Experiment directory: {self._experiment_directory}'
        )

        self._key_num_topics_values = _KEY_VALUES.format('num_topics')
        self._key_score_values = _KEY_SCORE_VALUES

        for key in [self._key_num_topics_values, self._key_score_values]:
            # TODO: no need to take mean for num_topics: it should be the same for all restarts
            self._keys_mean_many.append(key)
            self._keys_std_many.append(key)

    def search_for_optimum(self, text_collection: VowpalWabbitTextCollection) -> None:
        _logger.info('Starting to search for optimum...')

        dataset = text_collection._to_dataset()

        # seed == None is too similar to seed == 0
        seeds = [None] + list(range(0, self._num_restarts - 1))

        nums_topics = list(range(
            self._min_num_topics,
            self._max_num_topics + 1,
            self._num_topics_interval)
        )

        dataset_trainable = dataset._transform_data_for_training()

        for seed in tqdm(seeds):  # dirty workaround for 'too many models' issue
            for num_topics in nums_topics:
                artm_model = init_model_from_family(
                    self._family,
                    dataset,
                    modalities_to_use=list(text_collection._modalities.keys()),
                    main_modality=text_collection._main_modality,
                    num_topics=num_topics,
                    seed=seed,
                    num_processors=self._one_model_num_processors
                )

                model = TopicModel(artm_model)

                _logger.info(
                    f'Model\'s custom scores before attaching: {list(model.custom_scores.keys())}'
                )

                # TODO: remove this when TopicNet fixed
                _logger.info('Making custom scores empty dict')

                model.custom_scores = dict()

                model.set_model_id_as_timestamp()
                model.model_id = str(uuid.uuid4())

                path_components = [
                    self._experiment_directory,
                    f"{self._experiment_name}_{seed}",
                    model.model_id
                ]

                for score in self._scores:
                    score._attach(model)

                model._fit(
                    dataset_trainable=dataset_trainable,
                    num_iterations=self._num_fit_iterations,
                )
                model.save(model_save_path=os.path.join(*path_components))

                del model

        result, detailed_result = load_models_from_disk(
            self._experiment_directory,
            self._experiment_name
        )
        self._detailed_result = detailed_result
        self._result = result

        _logger.info('Finished searching!')


def _summarize_models(result_models, score_names=None, restarts=None):
    detailed_result = dict()
    result = dict()
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
                _logger.warning(
                    f'Score "{score}" has values as dict. Skipping the score'
                )

                continue

            score_df.loc[f"seed={model.seed}", len(model.topic_names)] = score_values

        detailed_result[score] = score_df.astype(float)

    for score in score_names:
        score_df = detailed_result[score]
        optimum_series = score_df.idxmin(axis=1)  # TODO: some scores need to be minimized, some - maximized

        score_result = dict()

        score_result[_KEY_OPTIMUM] = float(optimum_series.median())
        score_result[_KEY_OPTIMUM + _STD_KEY_SUFFIX] = float(optimum_series.std())
        score_result[_KEY_VALUES.format('num_topics')] = list(score_df.columns)
        score_result[_KEY_VALUES.format('score')] = score_df.mean(axis=0).tolist()
        score_result[_KEY_VALUES.format('score') + _STD_KEY_SUFFIX] = score_df.std(axis=0).tolist()

        result[_KEY_SCORE_RESULTS][score] = score_result

    return result, detailed_result


def load_models_from_disk(experiment_directory, base_experiment_name, scores=None):
    from topicnet.cooking_machine.experiment import START
    import glob

    result_models = []

    for folder in glob.glob(f"{experiment_directory}/{base_experiment_name}_*"):
        model_pathes = [
            f.path for f in os.scandir(folder)
            if f.is_dir() and f.name != START
        ]
        result_models += [TopicModel.load(path) for path in model_pathes]

    return _summarize_models(result_models)
