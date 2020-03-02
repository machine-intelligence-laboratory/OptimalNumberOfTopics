import numpy as np
from typing import (
    Dict,
    List
)

from ..data.base_text_collection import BaseTextCollection


_DDOF = 1

_KEY_OPTIMUM = 'optimum'
_KEY_VALUES = '{}_values'
_STD_KEY_SUFFIX = '_std'


class BaseSearchMethod:
    def __init__(self, min_num_topics, max_num_topics, num_fit_iterations):
        self._min_num_topics = min_num_topics
        self._max_num_topics = max_num_topics
        self._num_fit_iterations = num_fit_iterations

        self._result = dict()

        self._key_optimum = _KEY_OPTIMUM

        self._keys_mean_one = [self._key_optimum]
        self._keys_std_one = [self._key_optimum]
        self._keys_mean_many = list()
        self._keys_std_many = list()

    def search_for_optimum(self, text_collection: BaseTextCollection) -> None:
        # self._result = ...

        raise NotImplementedError()

    def _compute_mean_one(
            self,
            intermediate_results: List[Dict],
            final_result: Dict) -> None:

        assert len(intermediate_results) > 0

        for key in set(self._keys_mean_one).intersection(
                set(intermediate_results[0].keys())):

            final_result[key] = float(np.mean([
                r[key] for r in intermediate_results
            ]))

    def _compute_std_one(
            self,
            intermediate_results: List[Dict],
            final_result: Dict) -> None:

        assert len(intermediate_results) > 0

        for key in set(self._keys_std_one).intersection(
                set(intermediate_results[0].keys())):

            final_result[key + _STD_KEY_SUFFIX] = np.std(
                [r[key] for r in intermediate_results],
                ddof=_DDOF
            ).tolist()

    def _compute_mean_many(
            self,
            intermediate_results: List[Dict],
            final_result: Dict) -> None:

        assert len(intermediate_results) > 0

        for key in set(self._keys_mean_many).intersection(
                set(intermediate_results[0].keys())):

            final_result[key] = np.mean(
                np.stack([r[key] for r in intermediate_results]),
                axis=0
            ).tolist()

    def _compute_std_many(
            self,
            intermediate_results: List[Dict],
            final_result: Dict) -> None:

        assert len(intermediate_results) > 0

        for key in set(self._keys_std_many).intersection(
                set(intermediate_results[0].keys())):

            final_result[key + _STD_KEY_SUFFIX] = np.std(
                np.stack([r[key] for r in intermediate_results]),
                ddof=_DDOF,
                axis=0
            ).tolist()
