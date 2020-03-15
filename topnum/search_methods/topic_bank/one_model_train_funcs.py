import artm
import pandas as pd

from topicnet.cooking_machine.dataset import Dataset
from topicnet.cooking_machine.models import TopicModel
from typing import (
    Callable,
    List
)

from topnum.scores.base_score import BaseScore
from topnum.search_methods.topic_bank.phi_initialization import initialize_phi_funcs as initialize_phi_funcs
from topnum.search_methods.topic_bank.phi_initialization import utils as init_phi_utils


def default_train_func(
        dataset: Dataset,
        model_number: int,
        num_topics: int,
        num_fit_iterations: int,
        scores: List[BaseScore] = None) -> TopicModel:

    topic_model = _get_topic_model(
        dataset,
        num_topics=num_topics,
        seed=model_number,
        scores=scores
    )

    topic_model._fit(
        dataset.get_batch_vectorizer(),
        num_iterations=num_fit_iterations
    )

    return topic_model


def specific_initial_phi_train_func(
        dataset: Dataset,
        model_number: int,
        num_topics: int,
        num_fit_iterations: int,
        scores: List[BaseScore] = None,
        initialize_phi_func: Callable[[Dataset, int, int], pd.DataFrame] = None) -> TopicModel:

    topic_model = _get_topic_model(
        dataset,
        num_topics=num_topics,
        seed=model_number,
        scores=scores
    )

    if initialize_phi_func is None:
        initialize_phi_func = initialize_phi_funcs.initialize_randomly

    initial_phi = initialize_phi_func(dataset, model_number, num_topics)
    init_phi_utils._copy_phi(topic_model._model, initial_phi)

    topic_model._fit(
        dataset.get_batch_vectorizer(),
        num_iterations=num_fit_iterations
    )

    return topic_model


def regularization_train_func(
        dataset: Dataset,
        model_number: int,
        num_topics: int,
        num_fit_iterations: int,
        scores: List[BaseScore] = None) -> TopicModel:

    topic_model = _get_topic_model(
        dataset,
        num_topics=num_topics,
        seed=model_number,
        scores=scores
    )

    topic_model._model.regularizers.add(
        artm.regularizers.DecorrelatorPhiRegularizer(tau=10**5)
    )

    for topic_name in list(topic_model.get_phi().columns):
        topic_model._model.regularizers.add(
            artm.regularizers.SmoothSparsePhiRegularizer(
                tau=1e-5,
                topic_names=topic_name
            )
        )

    first_num_fit_iterations = int(0.75 * num_fit_iterations)
    second_num_fit_iterations = num_fit_iterations - first_num_fit_iterations

    topic_model._fit(
        dataset.get_batch_vectorizer(),
        num_iterations=first_num_fit_iterations
    )

    for regularizer_name in topic_model._model.regularizers.data:
        topic_model._model.regularizers[regularizer_name].tau = 0

    topic_model._model.regularizers.add(
        artm.regularizers.SmoothSparsePhiRegularizer(tau=1e-5)
    )
    topic_model._fit(
        dataset.get_batch_vectorizer(),
        num_iterations=second_num_fit_iterations
    )

    return topic_model


def background_topics_train_func(
        dataset: Dataset,
        model_number: int,
        num_topics: int,
        num_fit_iterations: int,
        scores: List[BaseScore] = None,
        num_background_topics: int = 2) -> TopicModel:

    topic_model = _get_topic_model(
        dataset,
        num_topics=num_topics + num_background_topics,
        seed=model_number,
        scores=scores
    )

    for background_topic_name in list(topic_model.get_phi().columns)[-num_background_topics:]:
        topic_model._model.regularizers.add(
            artm.regularizers.SmoothSparsePhiRegularizer(
                tau=0.01,
                topic_names=background_topic_name  # TODO: why not list?
            )
        )

    topic_model._fit(
        dataset.get_batch_vectorizer(),
        num_iterations=num_fit_iterations
    )

    specific_topics_phi = topic_model.get_phi().iloc[:, :-num_background_topics]

    del topic_model

    topic_model = _get_topic_model(
        dataset,
        num_topics=num_topics,
        seed=model_number,
        scores=scores
    )

    init_phi_utils._copy_phi(topic_model._model, specific_topics_phi)

    return topic_model


def _get_topic_model(
        dataset: Dataset,
        phi: pd.DataFrame = None,
        num_topics: int = None,
        seed: int = None,
        scores: List[BaseScore] = None,
        num_safe_fit_iterations: int = 3) -> TopicModel:

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

    if phi is None:
        pass
    elif num_safe_fit_iterations is not None and num_safe_fit_iterations > 0:
        init_phi_utils._safe_copy_phi(artm_model, phi, dataset, num_safe_fit_iterations)
    else:
        init_phi_utils._copy_phi(artm_model, phi)

    topic_model = TopicModel(
        artm_model=artm_model,
        model_id='0',
        cache_theta=True,
        theta_columns_naming='title'
    )

    if scores is not None:
        for score in scores:
            score._attach(topic_model)

    return topic_model
