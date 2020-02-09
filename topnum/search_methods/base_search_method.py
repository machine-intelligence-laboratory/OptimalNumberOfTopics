from ..data.base_text_collection import BaseTextCollection


class BaseSearchMethod:
    def __init__(self, min_num_topics, max_num_topics, num_collection_passes):
        self._min_num_topics = min_num_topics
        self._max_num_topics = max_num_topics
        self._num_collection_passes = num_collection_passes

        self._result = dict()

    def search_for_optimum(self, text_collection: BaseTextCollection) -> None:
        # self._result = ...

        raise NotImplementedError()
