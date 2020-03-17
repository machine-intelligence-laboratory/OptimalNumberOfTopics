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


# TODO: move this to TopicNet Dataset
# ==================================

def col_total_len(modality):
    return f'len_total{modality}'


def col_uniq_len(modality):
    return f'len_uniq{modality}'


def count_tokens_unigram(text):
    result_uniq, result_total = 0, 0
    for raw_token in text.split():
        token, _, count = raw_token.partition(":")
        count = int(count or 1)
        result_uniq += 1
        result_total += count

    return result_total, result_uniq


def count_tokens_raw_tokenized(text):
    data_split = text.split()
    return len(data_split), len(set(data_split))


def compute_document_details(demo_data, all_mods):
    columns = [col_total_len(m) for m in all_mods] + [col_uniq_len(m) for m in all_mods]
    token_count_df = pd.DataFrame(index=demo_data._data.index, columns=columns)

    is_raw_tokenized = not demo_data._data.vw_text.str.contains(":").any()

    for m in all_mods:
        local_columns = col_total_len(m), col_uniq_len(m)
        vw_copy = demo_data._data.vw_text.apply(lambda vw_string: get_modality_vw(vw_string, m))
        if is_raw_tokenized:
            data = vw_copy.apply(count_tokens_raw_tokenized)
        else:
            data = vw_copy.apply(count_tokens_unigram)

        token_count_df.loc[:, local_columns] = pd.DataFrame(data.tolist(), index=data.index, columns=local_columns)

    return token_count_df

# ==================================


def _symmetric_kl(distrib_p, distrib_q):
    return 0.5 * np.sum([stats.entropy(distrib_p, distrib_q), stats.entropy(distrib_p, distrib_q)])


class SpectralDivergenceScore(BaseCustomScore):
    '''
        Implements Arun metric to estimate the optimal number of topics:
        Arun, R., V. Suresh, C. V. Madhavan, and M. N. Murthy
        On finding the natural number of topics with latent dirichlet allocation: Some observations.
        In PAKDD (2010), pp. 391–402.


        The code is based on analagous code from TOM:
        https://github.com/AdrienGuille/TOM/blob/388c71ef/tom_lib/nlp/topic_model.py
    '''

    def __init__(
            self,
            name: str,
            validation_dataset: Dataset,
            modalities: List
            ):

        super().__init__(name)

        self._score = _SpectralDivergenceScore(validation_dataset, modalities)



class _SpectralDivergenceScore(BaseTopicNetScore):
    def __init__(self, validation_dataset, modalities):
        super().__init__()

        self.validation_dataset = validation_dataset
        document_length_stats = compute_document_details(validation_dataset, modalities)

        self.document_lengths = sum(document_length_stats[col_total_len(m)] for m in modalities)
        self.modalities = modalities

    def call(self, model: TopicModel):
        theta = model.get_theta(dataset=self.validation_dataset)
        phi = model.get_phi(class_ids=self.modalities)

        c_m1 = np.linalg.svd(phi, compute_uv=False)
        c_m2 = self.document_lengths.dot(theta.T)
        c_m2 += 0.0001  # we need this to prevent components equal to zero

        # we do not need to normalize these vectors
        return _symmetric_kl(c_m1, c_m2)
