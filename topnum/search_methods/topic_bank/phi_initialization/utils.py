import numpy as np
import pandas as pd
import logging
import warnings

from pandas.core.indexes.base import Index
from typing import (
    Iterable,
    List,
)

import artm

from topicnet.cooking_machine import Dataset
from topicnet.cooking_machine.models import TopicModel


_logger = logging.getLogger()


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


def _copy_phi(model: artm.ARTM, phi: pd.DataFrame, phi_ref: np.ndarray = None) -> np.ndarray:
    model_wrapper = TopicModel(artm_model=model)
    base_phi_index = model_wrapper.get_phi().index

    # TODO: faster?
    source_indices = list(phi.index)
    target_indices = list()
    found_indices = list()
    not_found_indices = list()
    not_found_indices_fraction_threshold = 0.5

    for index in source_indices:
        try:
            target_index = base_phi_index.get_loc(index)
        except KeyError:
            not_found_indices.append(index)
        else:
            target_indices.append(target_index)
            found_indices.append(index)

    if len(not_found_indices) == 0:
        pass
    elif len(not_found_indices) < not_found_indices_fraction_threshold * len(source_indices):
        warnings.warn(
            f'There are {len(not_found_indices) / (1e-7 + len(source_indices)) * 100}% of words'
            f' (i.e. {len(not_found_indices)} words)'
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

    _logger.debug(f'Attaching pwt and copying')

    if phi_ref is None:
        (_, phi_ref) = model.master.attach_model(
            model=model.model_pwt
        )

    phi_ref[target_indices, :phi.shape[1]] = phi.loc[found_indices, :].values

    return phi_ref


def _safe_copy_phi(
        model: artm.ARTM,
        phi: pd.DataFrame,
        dataset: Dataset,
        small_num_fit_iterations: int = 3) -> np.ndarray:

    if small_num_fit_iterations == 0:
        phi_ref = _copy_phi(model, phi)

        return phi_ref

    phi_ref = None

    # TODO: small_num_fit_iterations bigger than 1 seems not working for big matrices
    for _ in range(small_num_fit_iterations):
        phi_ref = _copy_phi(model, phi, phi_ref=phi_ref)
        model.fit_offline(dataset.get_batch_vectorizer(), 1)

    return phi_ref


def _trim_vw(tokens: List[str]) -> Iterable[str]:
    modality_start_symbol = '|'

    for token in tokens:
        if token.startswith(modality_start_symbol):
            continue

        if ':' not in token:
            word = token
        else:
            word, frequency = token.split(':')

        yield word
