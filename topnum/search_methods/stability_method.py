import csv
import glob
import itertools
import logging
import numpy as np
import os
import pandas as pd
import scipy as sp
import scipy.stats
import shutil
import sys
import tempfile
import tqdm

from lapsolver import solve_dense
from typing import (
    Dict,
    List,
)

from topicnet.cooking_machine import Dataset
from topicnet.cooking_machine.models import TopicModel

from .constants import (
    DEFAULT_MAX_NUM_TOPICS,
    DEFAULT_MIN_NUM_TOPICS,
    DEFAULT_NUM_FIT_ITERATIONS,
)
from .base_search_method import (
    BaseSearchMethod,
    _DDOF,
)
from .topic_bank import TopicBankMethod  # TODO: or not ok to import this?
from ..data import (
    BaseTextCollection,
    VowpalWabbitTextCollection,
)
from ..model_constructor import (
    init_model_from_family,
    KnownModel,
)


_LOGGER = logging.getLogger()
_DATASET_FILE_EXTENSION = '.csv'


class StabilitySearchMethod(BaseSearchMethod):
    def __init__(
            self,
            min_num_topics: int = DEFAULT_MIN_NUM_TOPICS,
            max_num_topics: int = DEFAULT_MAX_NUM_TOPICS,
            num_topics_interval: int = 1,
            num_fit_iterations: int = DEFAULT_NUM_FIT_ITERATIONS,
            model_num_processors: int = 3,
            model_seed: int = 0,
            model_family: str or KnownModel = KnownModel.PLSA,
            datasets_folder_path: str = None):

        super().__init__(
            min_num_topics=min_num_topics,
            max_num_topics=max_num_topics,
            num_fit_iterations=num_fit_iterations,
        )

        self._num_topics_interval = num_topics_interval
        self._model_family = model_family
        self._model_num_processors = model_num_processors
        self._model_seed = model_seed

        self._models_folder_path = tempfile.mkdtemp()

        if datasets_folder_path is None:
            datasets_folder_path = tempfile.mkdtemp(prefix='stability_approach__')

        self._datasets_folder_path = datasets_folder_path

        os.makedirs(datasets_folder_path, exist_ok=True)

    def _get_dataset_subsample_file_paths(self) -> List[str]:
        return glob.glob(
            os.path.join(self._datasets_folder_path, f'*{_DATASET_FILE_EXTENSION}')
        )

    def _folder_path_num_topics(self, num_topics: int) -> str:
        return os.path.join(
            self._models_folder_path,
            f'num_topics_{num_topics:03}'
        )

    def _folder_path_model(self, num_topics: int, subsample_number: int) -> str:
        return os.path.join(
            self._folder_path_num_topics(num_topics),
            f'model_{subsample_number}',
        )

    def search_for_optimum(
            self,
            text_collection: VowpalWabbitTextCollection,
            num_dataset_subsamples: int = 10,
            dataset_subsample_size: int or float = 0.5,
            seed_for_sampling: int = 11221963) -> None:
        """

        Parameters
        ----------
        text_collection
            Not used for training: just provides some info about data
        num_dataset_subsamples
            Number of subsamplings to be done
        dataset_subsample_size
            If `int`, then is treated like a number of documents.
            If `float`, then is considered to be a fraction of total number of documents.
        seed_for_sampling
            Randomness in subsample process
        """
        _LOGGER.info('Starting to search for optimum...')

        if len(self._get_dataset_subsample_file_paths()) == 0:
            print(
                f'Folder "{self._datasets_folder_path}"'
                f' has no sub-datasets for training! Subsampling data...'
            )
            self._subsample_datasets(
                text_collection,
                num_dataset_subsamples=num_dataset_subsamples,
                dataset_subsample_size=dataset_subsample_size,
                seed=seed_for_sampling,
            )

        assert len(self._get_dataset_subsample_file_paths()) > 0

        self._train_models(text_collection)
        self._estimate_stability()

    def _subsample_datasets(
            self,
            text_collection: BaseTextCollection,
            num_dataset_subsamples: int,
            dataset_subsample_size: int or float,
            seed: int) -> None:

        dataset = text_collection._to_dataset()
        total_num_documents = dataset._data.shape[0]

        if isinstance(dataset_subsample_size, float):
            dataset_subsample_size = int(total_num_documents * dataset_subsample_size)

        document_indices = list(range(total_num_documents))
        random = np.random.RandomState(seed)

        _LOGGER.info('Subsampling documents...')

        for i in tqdm.tqdm(
                range(num_dataset_subsamples),
                total=num_dataset_subsamples,
                file=sys.stdout):

            current_document_indices = random.choice(
                document_indices,
                size=dataset_subsample_size,
                replace=False,
            )
            subsample_dataset_file_path = os.path.join(
                self._datasets_folder_path,
                f'dataset_{i}{_DATASET_FILE_EXTENSION}',
            )

            with open(subsample_dataset_file_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(dataset._data.columns)

                for document_index in current_document_indices:
                    writer.writerow(dataset._data.iloc[document_index].to_list())

        _LOGGER.info('Subsampling finished')

    def _train_models(self, text_collection: VowpalWabbitTextCollection) -> None:
        modalities_to_use = list(text_collection._modalities.keys())
        main_modality = text_collection._main_modality

        numbers_of_topics = list(range(
            self._min_num_topics,
            self._max_num_topics + 1,
            self._num_topics_interval))

        print('Training models for different number of topics...')

        for num_topics in tqdm.tqdm(
                numbers_of_topics,
                total=len(numbers_of_topics),
                file=sys.stdout):

            os.makedirs(
                self._folder_path_num_topics(num_topics)
            )

            for subsample_number, data_path in enumerate(self._get_dataset_subsample_file_paths()):
                dataset = Dataset(data_path=data_path)

                dataset_trainable = dataset.get_batch_vectorizer()

                artm_model = init_model_from_family(
                    family=self._model_family,
                    dataset=dataset,
                    modalities_to_use=modalities_to_use,
                    main_modality=main_modality,
                    num_topics=num_topics,
                    seed=self._model_seed,
                    num_processors=self._model_num_processors,
                )
                topic_model = TopicModel(artm_model)

                topic_model._fit(
                    dataset_trainable=dataset_trainable,
                    num_iterations=self._num_fit_iterations,
                )

                model_save_path = self._folder_path_model(num_topics, subsample_number)
                topic_model.save(
                    model_save_path=model_save_path,
                    phi=True,
                    theta=False,
                )

    def _estimate_stability(self):
        numbers_of_topics = list(range(
            self._min_num_topics,
            self._max_num_topics + 1,
            self._num_topics_interval))
        subsample_numbers = list(range(len(self._get_dataset_subsample_file_paths())))

        stabilities = dict()

        print('Estimating stability for different number of topics...')

        for num_topics in tqdm.tqdm(
                numbers_of_topics,
                total=len(numbers_of_topics),
                file=sys.stdout):

            distances = list()

            for subsample_number_a, subsample_number_b in itertools.combinations(
                    subsample_numbers, 2):
                topic_model_a = TopicModel.load(
                    self._folder_path_model(
                        num_topics,subsample_number=subsample_number_a
                    )
                )
                topic_model_b = TopicModel.load(
                    self._folder_path_model(
                        num_topics, subsample_number=subsample_number_a
                    )
                )
                distances.append(
                    StabilitySearchMethod._compute_distance(topic_model_a, topic_model_b)
                )

            stability_metrics = dict()
            stability_metrics['std'] = np.std(distances, ddof=_DDOF)
            stability_metrics['var'] = np.var(distances, ddof=_DDOF)
            stability_metrics['range'] = np.ptp(distances)
            stability_metrics['interquartile_range'] = sp.stats.iqr(distances)

            stabilities[num_topics] = stability_metrics

        self._result['stability_metrics_for_num_topics'] = stabilities

    def clear(self):
        shutil.rmtree(self._datasets_folder_path)
        shutil.rmtree(self._models_folder_path)

    @staticmethod
    def _compute_distance(topic_model_a: TopicModel, topic_model_b: TopicModel) -> float:
        phi_a = topic_model_a.get_phi()
        phi_b = topic_model_b.get_phi()

        assert phi_a.shape == phi_b.shape

        num_topics = phi_a.shape[1]
        topic_distances = np.zeros(shape=(num_topics, num_topics))
        topic_indices = list(range(num_topics))

        for topic_index_a, topic_index_b in zip(topic_indices, topic_indices):
            topic_distance = StabilitySearchMethod._compute_topic_distance(
                phi_a.iloc[:, topic_index_a],
                phi_b.iloc[:, topic_index_b],
            )
            topic_distances[topic_index_a, topic_index_b] = topic_distance

        row_ids, column_ids = solve_dense(topic_distances)

        return float(np.sum(
            topic_distances[row_ids, column_ids]
        ))

    @staticmethod
    def _compute_topic_distance(topic_a: pd.Series, topic_b: pd.Series) -> float:
        return TopicBankMethod._jaccard_distance(
            topic_a.to_dict(),
            topic_b.to_dict(),
        )
