import numpy as np
import scipy.stats as stats


from topicnet.cooking_machine import Dataset
from topicnet.cooking_machine.models import (
    BaseScore as BaseTopicNetScore,
    TopicModel
)
from typing import (
    List
)

from .base_custom_score import BaseCustomScore


import pandas as pd


def get_log_likelihood(model, modality):
    raw_perplexity = model.score_tracker[f'PerplexityScore{modality}'].last_class_id_info[modality].raw
    assert raw_perplexity != 0.0
    return raw_perplexity


class LikelihoodBasedScore(BaseCustomScore):
    '''
    '''

    def __init__(
            self,
            name: str,
            validation_dataset: Dataset = None,
            modality: str,
            mode: str = 'AIC',
            consider_sparsity: bool = False,
            ):

        super().__init__(name)

        self._score = _LikelihoodBasedScore(
            validation_dataset, modality,
            mode, consider_sparsity
        )



class _LikelihoodBasedScore(BaseTopicNetScore):

    def __init__(
            self,
            validation_dataset: Dataset,
            modality: str,
            mode: str = 'AIC',
            consider_sparsity: bool = False,
            ):

        super().__init__()

        self.num_docs = validation_dataset._data.shape[0]

        self.consider_sparsity = consider_sparsity
        self.mode = mode.upper()
        self.modality = modality

    def call(self, model: TopicModel):

        phi = model.get_phi(class_ids=[self.modality])
        V, T = phi.shape
        D = self.num_docs

        # TODO: consider the case of having vector of taus instead
        hyperparams = len(model.regularizers)

        # than2012 (https://link.springer.com/content/pdf/10.1007/978-3-642-33460-3_37.pdf)
        # argues that number of free parameters in LDA and sparse models (such as PLSA)
        # should should be calculated differently
        if self.consider_sparsity:
            N_p = phi.astype(bool).sum().sum() + hyperparams
        else:
            N_p = (V - 1) * T + hyperparams

        ll = get_log_likelihood(model._model, self.modality)

        if self.mode == "MDL":
            return 0.5 * N_p * np.log(T * D) - ll
        if self.mode == "AIC":
            return 2 * N_p - 2 * ll
        if self.mode == "BIC":
            return N_p * np.log(D) - 2 * ll

        raise ValueError(f"Unsupported score type {self.mode}; Supported ones are: AIC/BIC/MDL")
