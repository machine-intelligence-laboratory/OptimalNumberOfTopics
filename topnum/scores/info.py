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
from topicnet.cooking_machine.dataset import get_modality_vw


def get_log_likelihood(model, modality):
    return model.score_tracker[f'PerplexityScore{modality}'].last_class_id_info[modality].raw

class ApproximateMDLScore(BaseCustomScore):
    '''
    '''

    def __init__(
            self,
            name: str,
            validation_dataset: Dataset,
            modalities: List
            ):

        super().__init__(name)

        self._score = _ApproximateMDLScore(validation_dataset, modalities)



class _ApproximateMDLScore(BaseTopicNetScore):
    def __init__(self, validation_dataset, modalities):
        super().__init__()

        self.validation_dataset = validation_dataset

        self.modalities = modalities

    def call(self, model: TopicModel):
        if self.consider_sparsity:
            N_p = #TODO
        else:
            N_p = #TODO but different
        ll = get_log_likelihood(model)
        mdl = 0.5 * N_p * log(T * D) - ll
        aic = 2 * N_p - 2 * ll
        bic = N_p * log(D) - 2 * ll
        return aic, bic, mdl
