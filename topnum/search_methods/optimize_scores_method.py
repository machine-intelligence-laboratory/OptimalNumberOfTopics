import logging
import os
import pandas as pd
from  numpy.random import RandomState
import uuid
import warnings

from tqdm import tqdm
from typing import List

from topicnet.cooking_machine.models import TopicModel, DummyTopicModel

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

from ..model_constructor import (
    init_model_from_family,
    KnownModel,
)

_KEY_SCORE_RESULTS = 'score_results'
_KEY_SCORE_VALUES = 'score_values'

_logger = logging.getLogger()


class OptimizeScoresMethod(BaseSearchMethod):
    def __init__(
            self,
            scores: List[BaseScore],  # TODO: Union[BaseScore, List[BaseScore]]
            model_family: str or KnownModel = KnownModel.PLSA,
            model_params: dict = None,
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
        self._model_params = model_params
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

    # TODO: accept either VowpalWabbitTextCollection or Dataset with modalities
    def search_for_optimum(self, text_collection: VowpalWabbitTextCollection) -> None:
        _logger.info('Starting to search for optimum...')

        dataset = text_collection._to_dataset()

        seeds = [None] + [abs(RandomState(i).tomaxint()) for i in range(1, self._num_restarts)]

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
                    num_processors=self._one_model_num_processors,
                    model_params=self._model_params,
                )

                model = TopicModel(artm_model)

                _logger.info(
                    f'Model\'s custom scores before attaching: {list(model.custom_scores.keys())}'
                )

                # TODO: remove this when TopicNet fixed
                _logger.info('Making custom scores empty dict')

                model.custom_scores = dict()
                model.model_id = str(uuid.uuid4())

                path_components = [
                    self._experiment_directory,
                    f"{self._experiment_name}_{seed}",
                    model.model_id
                ]

                model._fit(
                    dataset_trainable=dataset_trainable,
                    num_iterations=self._num_fit_iterations - 1,
                )

                for score in self._scores:
                    score._attach(model)

                model._fit(
                    dataset_trainable=dataset_trainable,
                    num_iterations=1,
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


def _summarize_models(
        result_models: List[TopicModel],
        score_names: List[str] = None,
        restarts=None):

    detailed_result = dict()
    result = dict()
    result[_KEY_SCORE_RESULTS] = dict()

    any_model = result_models[-1]

    if score_names is None:
        score_names = any_model.describe_scores().reset_index().score_name.values

    nums_topics = sorted(list({len(tm.topic_names) for tm in result_models}))

    if restarts is None:
        seeds = list({tm.seed for tm in result_models})
        restarts = "seed=" + pd.Series(seeds, name="restart_id").astype(str)

    for score_name in score_names:
        score_df = pd.DataFrame(index=restarts, columns=nums_topics)

        for model in result_models:
            score_values = model.scores[score_name][-1]

            if isinstance(score_values, dict):
                _logger.warning(
                    f'Score "{score_name}" has values as dict. Skipping the score'
                )

                continue

            score_df.loc[f"seed={model.seed}", len(model.topic_names)] = score_values

        detailed_result[score_name] = score_df.astype(float)

    any_model_all_scores = any_model.scores
    any_model_all_given_score_names = list(any_model_all_scores.keys())

    for score_name in score_names:
        # last_value workaround stuff for some scores:
        # score_name       = TopicKernel@main.average_coherence  # not score, strictly speaking
        # given_score_name = TopicKernel@main                    # real score

        given_score_name = next(
            name for name in any_model_all_given_score_names
            if score_name.startswith(name)
        )

        if hasattr(any_model_all_scores[given_score_name], '_higher_better'):
            higher_better = any_model_all_scores[given_score_name]._higher_better
        else:
            warnings.warn(
                f'Score "{score_name}" of type {type(any_model_all_scores[given_score_name])} doesn\'t have "_higher_better" attribute!'
                f' Assuming that higher_better = True'
            )

            higher_better = True

        score_df = detailed_result[score_name]

        if higher_better is True:
            optimum_series = score_df.idxmax(axis=1)
        else:
            optimum_series = score_df.idxmin(axis=1)

        score_result = dict()

        score_result[_KEY_OPTIMUM] = float(optimum_series.median())
        score_result[_KEY_OPTIMUM + _STD_KEY_SUFFIX] = float(optimum_series.std())
        score_result[_KEY_VALUES.format('num_topics')] = list(score_df.columns)
        score_result[_KEY_VALUES.format('score')] = score_df.mean(axis=0).tolist()
        score_result[_KEY_VALUES.format('score') + _STD_KEY_SUFFIX] = score_df.std(axis=0).tolist()

        result[_KEY_SCORE_RESULTS][score_name] = score_result

    return result, detailed_result


def load_models_from_disk(experiment_directory, base_experiment_name, scores=None):
    from topicnet.cooking_machine.experiment import START
    import glob

    result_models = []

    masks = [
        f"{experiment_directory}/{base_experiment_name}_*",
        f"{experiment_directory}/{base_experiment_name}/*"
    ]
    for new_exp_format, mask in enumerate(masks):
        if not len(glob.glob(mask)):
            continue
        msg = (f'Trying to load models from {mask}.'
               f' {len(glob.glob(mask))} models found.')
        _logger.info(msg)
        for folder in glob.glob(mask):
            if new_exp_format:
                model_pathes = [folder]
            else:
                model_pathes = [
                    f.path for f in os.scandir(folder)
                    if f.is_dir() and f.name != START
                ]
            result_models += [DummyTopicModel.load(path) for path in model_pathes]
            # result_models += [TopicModel.load(path).to_dummy() for path in model_pathes]

        return _summarize_models(result_models)
    raise ValueError(f"No models found in {masks}")
