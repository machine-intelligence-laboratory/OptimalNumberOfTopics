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
    return model.score_tracker[f'PerplexityScore{modality}'].last_class_id_info[modality].raw


class LikelihoodBasedScore(BaseCustomScore):
    '''
    '''

    def __init__(
            self,
            name: str,
            validation_dataset: Dataset = None,
            modalities: List = None,
            mode: str = 'AIC',
            consider_sparsity: bool = False,
            ):

        super().__init__(name)

        self._score = _LikelihoodBasedScore(
            validation_dataset, modalities,
            mode, consider_sparsity
        )



class _LikelihoodBasedScore(BaseTopicNetScore):

    def __init__(
            self,
            validation_dataset: Dataset = None,
            modalities: List = None,
            mode: str = 'AIC',
            consider_sparsity: bool = False,
            ):

        super().__init__()

        if mode == "MDL" and validation_dataset is None:
            raise ValueError("MDL requires the corpus")

        self.num_docs = validation_dataset._data.shape[0]

        self.consider_sparsity = consider_sparsity
        self.mode = mode
        if modalities is None or len(modalities) != 1:
            raise ValueError("not supported")
        self.modalities = modalities

    def call(self, model: TopicModel):

        phi = model.get_phi(class_ids=self.modalities)
        V, T = phi.shape
        D = self.num_docs

        # TODO: consider the case of having vector of taus instead
        hyperparams = len(model.regularizers) 

        if self.consider_sparsity:
            N_p = phi.astype(bool).sum().sum() + hyperparams
        else:
            N_p = (V - 1) * T + hyperparams

        ll = get_log_likelihood(model, self.modalities[0])

        if self.mode == "MDL":
            return 0.5 * N_p * np.log(T * D) - ll
        if self.mode == "AIC":
            return 2 * N_p - 2 * ll
        if self.mode == "BIC":
            return N_p * np.log(D) - 2 * ll

        raise ValueError("Unsupported score type")
