import artm
import numpy as np
import pandas as pd

from topicnet.cooking_machine.dataset import Dataset
from topicnet.cooking_machine.models import TopicModel
from typing import (
    Callable,
    List,
    Optional,
)

from topnum.scores.base_score import BaseScore
from topnum.search_methods.topic_bank.phi_initialization import initialize_phi_funcs as initialize_phi_funcs
from topnum.search_methods.topic_bank.phi_initialization import utils as init_phi_utils


def default_train_func(
        dataset: Dataset,
        main_modality: Optional[str],
        model_number: int,
        num_topics: int,
        num_fit_iterations: int,
        scores: List[BaseScore] = None,
        **kwargs) -> TopicModel:
    """

    Additional Parameters
    ---------------------
    kwargs
        Some params for `_get_topic_model`, such as `cache_theta` and `num_processors`
    """

    topic_model = _get_topic_model(
        dataset,
        main_modality=main_modality,
        num_topics=num_topics,
        seed=model_number,
        **kwargs,
    )

    num_fit_iterations_with_scores = 1

    topic_model._fit(
        dataset.get_batch_vectorizer(),
        num_iterations=max(0, num_fit_iterations - num_fit_iterations_with_scores)
    )
    _fit_model_with_scores(
        topic_model,
        dataset,
        scores,
        num_fit_iterations=num_fit_iterations_with_scores
    )

    return topic_model


def specific_initial_phi_train_func(
        dataset: Dataset,
        main_modality: Optional[str],
        model_number: int,
        num_topics: int,
        num_fit_iterations: int,
        scores: List[BaseScore] = None,
        initialize_phi_func: Callable[[Dataset, int, int], pd.DataFrame] = None,
        **kwargs) -> TopicModel:

    topic_model = _get_topic_model(
        dataset,
        main_modality=main_modality,
        num_topics=num_topics,
        seed=model_number,
        **kwargs,
    )

    if initialize_phi_func is None:
        initialize_phi_func = initialize_phi_funcs.initialize_randomly

    initial_phi = initialize_phi_func(dataset, model_number, num_topics)

    if main_modality is not None:
        initial_phi = init_phi_utils.get_modality_phi(
            initial_phi, modality=main_modality
        )

    # TODO: However strange it may seem,
    #  it is really crucial to initialize `phi_ref` variable here.
    #  Otherwise, all this init-copy manipulation won't work.
    #  (Yes, at first glance `phi_ref` is not used anywhere,
    #  but apparently it is used somewhere...)
    #  The owls are not what they seem.
    phi_ref = init_phi_utils._copy_phi(topic_model._model, initial_phi)

    assert np.allclose(phi_ref, topic_model.get_phi().to_numpy())

    num_fit_iterations_with_scores = 1

    topic_model._fit(
        dataset.get_batch_vectorizer(),
        num_iterations=max(0, num_fit_iterations - num_fit_iterations_with_scores)
    )
    _fit_model_with_scores(
        topic_model,
        dataset,
        scores,
        num_fit_iterations=num_fit_iterations_with_scores
    )

    return topic_model


def regularization_train_func(
        dataset: Dataset,
        main_modality: Optional[str],
        model_number: int,
        num_topics: int,
        num_fit_iterations: int,
        scores: List[BaseScore] = None,
        decorrelating_tau: float = 10**5,
        smoothing_tau: float = 1e-5,
        sparsing_tau: float = -0.01,
        **kwargs) -> TopicModel:

    topic_model = _get_topic_model(
        dataset,
        main_modality=main_modality,
        num_topics=num_topics,
        seed=model_number,
        **kwargs,
    )

    topic_model._model.regularizers.add(
        artm.regularizers.DecorrelatorPhiRegularizer(tau=decorrelating_tau)
    )

    for topic_name in list(topic_model.get_phi().columns):
        topic_model._model.regularizers.add(
            artm.regularizers.SmoothSparsePhiRegularizer(
                tau=smoothing_tau,
                topic_names=topic_name
            )
        )

    num_fit_iterations_with_scores = 1
    first_num_fit_iterations = int(
        0.75 * (num_fit_iterations - num_fit_iterations_with_scores)
    )
    second_num_fit_iterations = (
        num_fit_iterations - num_fit_iterations_with_scores - first_num_fit_iterations
    )

    topic_model._fit(
        dataset.get_batch_vectorizer(),
        num_iterations=first_num_fit_iterations
    )

    for regularizer_name in topic_model._model.regularizers.data:
        topic_model._model.regularizers[regularizer_name].tau = 0

    topic_model._model.regularizers.add(
        artm.regularizers.SmoothSparsePhiRegularizer(tau=sparsing_tau)
    )

    topic_model._fit(
        dataset.get_batch_vectorizer(),
        num_iterations=max(0, second_num_fit_iterations - num_fit_iterations_with_scores)
    )
    _fit_model_with_scores(
        topic_model,
        dataset,
        scores,
        num_fit_iterations=num_fit_iterations_with_scores
    )

    return topic_model


def background_topics_train_func(
        dataset: Dataset,
        main_modality: Optional[str],
        model_number: int,
        num_topics: int,
        num_fit_iterations: int,
        scores: List[BaseScore] = None,
        num_background_topics: int = 2,
        smoothing_tau: float = 0.01,
        **kwargs) -> TopicModel:

    topic_model = _get_topic_model(
        dataset,
        main_modality=main_modality,
        num_topics=num_topics + num_background_topics,
        seed=model_number,
        **kwargs,
    )

    for background_topic_name in list(topic_model.get_phi().columns)[-num_background_topics:]:
        topic_model._model.regularizers.add(
            artm.regularizers.SmoothSparsePhiRegularizer(
                tau=smoothing_tau,
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
        main_modality=main_modality,
        num_topics=num_topics,
        seed=model_number,
        **kwargs,
    )

    num_fit_iterations_with_scores = 1
    num_fit_iterations_without_scores = num_fit_iterations - num_fit_iterations_with_scores

    phi_ref = None

    for fit_iteration in range(num_fit_iterations_without_scores):
        phi_ref = init_phi_utils._copy_phi(
            topic_model._model,
            specific_topics_phi,
            phi_ref=phi_ref
        )
        topic_model._fit(
            dataset.get_batch_vectorizer(),
            num_iterations=1
        )

    phi_ref = init_phi_utils._copy_phi(
        topic_model._model,
        specific_topics_phi,
        phi_ref=phi_ref
    )
    _fit_model_with_scores(
        topic_model,
        dataset,
        scores,
        num_fit_iterations=num_fit_iterations_with_scores
    )

    # TODO: not very safe here? (if cache_theta us True, Theta not updated here)
    phi_ref = init_phi_utils._copy_phi(
        topic_model._model,
        specific_topics_phi,
        phi_ref=phi_ref
    )

    return topic_model


def _get_topic_model(
        dataset: Dataset,
        main_modality: Optional[str],
        phi: pd.DataFrame = None,
        num_topics: int = None,
        seed: int = None,
        scores: List[BaseScore] = None,
        num_safe_fit_iterations: int = 3,  # TODO: remove param (only FastFixPhiRegularizer to be used for safe copy)
        num_processors: int = 3,
        cache_theta: bool = False) -> TopicModel:

    if phi is not None:
        raise ValueError(
            "Do not use `phi` parameter, use `num_topics` instead!"
            " Currently, this method is not responsible for copying Phi matrix."
            " We have temporarily turned off this functionality,"
            " because the realization appeared not perfectly reliable."
            " In the future, Phi copying will be improved and returned"
            " (it will be based on FastFixPhiRegularizer)."
        )

    dictionary = dataset.get_dictionary()

    # for modality in dataset.get_possible_modalities():
    #     if modality not in modalities_to_use:
    #         dictionary.filter(class_id=modality, max_df=0, inplace=True)

    if num_topics is not None and phi is not None:
        assert num_topics >= phi.shape[1]
    elif num_topics is None and phi is not None:
        num_topics = phi.shape[1]
    elif num_topics is None and phi is None:
        raise ValueError()

    topic_names = [f'topic_{i}' for i in range(num_topics)]

    # if seed is None:
    #     artm_model = artm.ARTM(topic_names=topic_names)
    # else:
    #     artm_model = artm.ARTM(topic_names=topic_names, seed=seed)

    if main_modality is not None:
        class_ids = {main_modality: 1}
    else:
        class_ids = None

    if seed is None:
        seed = -1  # for ARTM, it means "no seed"

    artm_model = artm.ARTM(topic_names=topic_names, seed=seed, class_ids=class_ids)  # TODO: not list, but dict!!!

    # artm_model = init_model(topic_names, class_ids=[MAIN_MODALITY])

    # artm_model = init_plsa(DATASET, [MAIN_MODALITY], MAIN_MODALITY, 5)

    artm_model.num_processors = num_processors
    artm_model.initialize(dictionary)

    """
    if phi is None:
        pass
    elif num_safe_fit_iterations is not None and num_safe_fit_iterations > 0:
        init_phi_utils._safe_copy_phi(artm_model, phi, dataset, num_safe_fit_iterations)
    else:
        init_phi_utils._copy_phi(artm_model, phi)
    """
    # this breaks smth in ARTM
    # test_ppl@word [1827.4515380859375, 2707.63623046875, 2707.67919921875, 2707.679443359375, 2707.679443359375]
    # test_ppl@word_with_d [4073.36328125, 6035.2822265625, 6035.3779296875, 6035.37841796875, 6035.37841796875]
    # test_ppl@all [1827.4515380859375, 2707.63623046875, 2707.67919921875, 2707.679443359375, 2707.679443359375]
    # test_ppl@all_2 [1827.4515380859375, 2707.63623046875, 2707.67919921875, 2707.679443359375, 2707.679443359375]
    # test_ppl@all_2_with_d [4073.36328125, 6035.2822265625, 6035.3779296875, 6035.37841796875, 6035.37841796875]
    
    topic_model = TopicModel(
        artm_model=artm_model,
        model_id='0',
        cache_theta=cache_theta,
        theta_columns_naming='title'
    )

    if scores is not None:
        for score in scores:
            score._attach(topic_model)

    return topic_model


def _fit_model_with_scores(
        topic_model: TopicModel,
        dataset: Dataset,
        scores: List[BaseScore] = None,
        num_fit_iterations: int = 1):

    if scores is not None:
        for score in scores:
            score._attach(topic_model)

    topic_model._fit(
        dataset.get_batch_vectorizer(),
        num_iterations=num_fit_iterations
    )
