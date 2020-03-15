import artm
import numpy as np
import pandas as pd
import warnings

from pandas.core.indexes.base import Index
from topicnet.cooking_machine import Dataset
from topicnet.cooking_machine.models import TopicModel


# TODO: seems like method suitable for Dataset?
def get_phi_index(dataset: Dataset) -> Index:
    artm_model_template = artm.ARTM(num_topics=1, num_processors=1)
    artm_model_template.initialize(dictionary=dataset.get_dictionary())
    model_template = TopicModel(artm_model=artm_model_template)
    phi_template = model_template.get_phi()
    phi_index = phi_template.index

    del model_template
    del artm_model_template

    return phi_index


def _copy_phi(model: artm.ARTM, phi: pd.DataFrame) -> None:
    model_wrapper = TopicModel(artm_model=model)
    base_phi_index = model_wrapper.get_phi().index

    # TODO: faster?
    source_indices = list(phi.index)
    target_indices = list()
    not_found_indices = list()
    not_found_indices_fraction_threshold = 0.5

    for index in source_indices:
        try:
            target_index = base_phi_index.get_loc(index)
            target_indices.append(target_index)
        except KeyError:
            not_found_indices.append(index)

    if len(not_found_indices) == 0:
        pass
    elif len(not_found_indices) < not_found_indices_fraction_threshold * len(source_indices):
        warnings.warn(
            f'There are {not_found_indices_fraction_threshold * 100}% of words'
            f' (i.e. {not_found_indices_fraction_threshold * len(source_indices)} words)'
            f' in the given Phi matrix'
            f' which were not found in the model\'s Phi matrix'
        )
    else:
        raise RuntimeError(
            f'Not less than {not_found_indices_fraction_threshold * 100}% of words'
            f' in the given Phi matrix with {len(source_indices)} words were not found'
            f' in the model\'s Phi matrix with {len(base_phi_index)} words!'
            f' Seems like doing initialization in such circumstances is not good'
        )

    (_, phi_ref) = model.master.attach_model(
        model=model.model_pwt
    )

    phi_new = np.copy(phi_ref)
    phi_new[target_indices, :phi.shape[1]] = phi.values

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
