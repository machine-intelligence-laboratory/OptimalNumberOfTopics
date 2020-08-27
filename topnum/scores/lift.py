import os
import pandas as pd
import numpy as np

from functools import reduce
from typing import (
    List
)

import artm

from topicnet.cooking_machine.models.thetaless_regularizer import artm_dict2df
from topicnet.cooking_machine import Dataset
from topicnet.cooking_machine.models import (
    BaseScore as BaseTopicNetScore,
    TopicModel
)

from .base_custom_score import BaseCustomScore


class MeanLiftScore(BaseCustomScore):
    """Implements log lift.

    Fan, A., Doshi‐Velez, F., & Miratrix, L.
    Assessing topic model relevance: Evaluation and informative priors.
    Statistical Analysis and Data Mining: The ASA Data Science Journal (2019), 12(3), 210-222.
    """

    def __init__(
            self,
            name: str,
            validation_dataset: Dataset,
            modalities: List
            ):

        super().__init__(name)

        self._score = _MeanLiftScore(validation_dataset, modalities)


class _MeanLiftScore(BaseTopicNetScore):
    def __init__(self, validation_dataset, modalities, topic_names=None):
        super().__init__()

        self._dict_path = os.path.join(validation_dataset._internals_folder_path, 'dict.dict')
        self.topic_names = topic_names
        self.modalities = modalities
        self.num_toptokens = 30

    def _compute_lift(
            self,
            phi: pd.DataFrame,
            chosen_words_array: List[pd.Index] = None,
            ):
        # inspired by gist.github.com/jrnold/daa039f02486009a24cf3e83403dabf0
        artm_dict = artm.Dictionary(dictionary_path=self._dict_path)
        dict_df = artm_dict2df(artm_dict).query("class_id in @self.modalities")

        # TODO: this is possible to do using aggregate / join and stuff
        for m in self.modalities:
            subdf = dict_df.query("class_id == @m")
            idx = subdf.index
            # theretically, token_freq is unnecasry duplicate of token_value
            # in practice, we have float32 errors and also user could run dictionary filtering
            # without setting recalculate_value=True
            dict_df.loc[idx, 'token_freq'] = dict_df.loc[idx, 'token_tf'] / subdf.token_tf.sum()

        dict_df.set_index(["class_id", "token"], inplace=True)
        dict_df.index.names = ['modality', 'token']
        dict_df.sort_index(inplace=True)
        phi.sort_index(inplace=True)

        known_chosen_words_array = [
            words.intersection(dict_df.index)
            for words in chosen_words_array
        ]

        if known_chosen_words_array:
            merged_index = reduce(
                lambda idx1, idx2: idx1.union(idx2),
                known_chosen_words_array
            )
            chosen_words = merged_index.drop_duplicates()
            dict_df = dict_df.loc[chosen_words]
            phi = phi.loc[chosen_words]

        data = np.log(phi.values) - np.log(dict_df[['token_freq']].values)
        log_lift = pd.DataFrame(data=np.log(data), index=phi.index, columns=phi.columns)

        if not known_chosen_words_array:
            return log_lift

        result = []

        for t, words in zip(phi.columns, known_chosen_words_array):
            result.append(log_lift.loc[words, t].sum())

        log_lift_total = pd.Series(data=result, index=phi.columns)

        return log_lift_total

    def _select_topwords(self, phi: pd.DataFrame) -> List[pd.Index]:
        relevant_words = list()

        for t in phi.columns:
            top30 = phi[t].sort_values().tail(self.num_toptokens)
            relevant_words.append(top30.index)

        return relevant_words

    def call(self, model: TopicModel):
        phi = model.get_phi(class_ids=self.modalities)

        relevant_words = self._select_topwords(phi)

        loglift = self._compute_lift(phi, relevant_words)

        if self.topic_names is not None:
            topic_names = self.topic_names
        else:
            topic_names = model.topic_names

        total_loglift = loglift[topic_names]

        return float(total_loglift.mean())
