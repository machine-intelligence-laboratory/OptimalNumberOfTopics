import artm
import numpy as np
import pandas as pd

from topicnet.cooking_machine.dataset import Dataset
from topicnet.cooking_machine.models import TopicModel
from typing import List

from topnum.scores.base_score import BaseScore


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


def train_func_nonrandom_initialization(
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

    # TODO
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

    topic_model._fit(
        dataset.get_batch_vectorizer(),
        num_iterations=num_fit_iterations
    )

    return topic_model


def train_func_regularizers(
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

    # TODO

    topic_model._model.regularizers.add(
        artm.regularizers.DecorrelatorPhiRegularizer(tau=10**3)
    )

    # for t in list(tm.get_phi().columns)[:NUM_CDC_TOPICS]:  # NUM_ARORA_TOPICS / NUM_CDC_TOPICS
    #     tm._model.regularizers.add(
    #         artm.regularizers.SmoothSparsePhiRegularizer(
    #             tau=1e-5,
    #             topic_names=t
    #         )
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

    topic_model._fit(
        dataset.get_batch_vectorizer(),
        num_iterations=num_fit_iterations
    )

    return topic_model


def train_func_background_topics(
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

    # TODO

    #     tm._model.regularizers.add(
    #         artm.regularizers.SmoothSparsePhiRegularizer(tau=0.01, topic_names=list(tm.get_phi().columns)[-1])
    #     )
    #     tm._model.regularizers.add(
    #         artm.regularizers.SmoothSparsePhiRegularizer(tau=0.01, topic_names=list(tm.get_phi().columns)[-2])
    #     )

    topic_model._fit(
        dataset.get_batch_vectorizer(),
        num_iterations=num_fit_iterations
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
        _safe_copy_phi(artm_model, phi, dataset, num_safe_fit_iterations)
    else:
        _copy_phi(artm_model, phi)

    topic_model = TopicModel(
        artm_model=artm_model,
        model_id='0',
        cache_theta=True,
        theta_columns_naming='id'
    )

    if scores is not None:
        for score in scores:
            score._attach(topic_model)

    return topic_model


def _copy_phi(model: artm.ARTM, phi: pd.DataFrame) -> None:
    # TODO: assuming, that vocabularies are the same
    #  maybe better to check
    (_, phi_ref) = model.master.attach_model(
        model=model.model_pwt
    )

    phi_new = np.copy(phi_ref)
    phi_new[:, :phi.shape[1]] = phi.values

    np.copyto(
        phi_ref,
        phi_new
    )


def _safe_copy_phi(
        model: artm.ARTM,
        phi: pd.DataFrame,
        dataset: Dataset,
        small_num_fit_iterations: int = 3) -> None:

    if small_num_fit_iterations == 0:
        _copy_phi(model, phi)

        return

    for _ in range(small_num_fit_iterations):
        _copy_phi(model, phi)
        model.fit_offline(dataset.get_batch_vectorizer(), 1)
