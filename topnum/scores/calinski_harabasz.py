import logging
import dill

from sklearn.metrics import calinski_harabasz_score
from topicnet.cooking_machine import Dataset
from topicnet.cooking_machine.models import (
    BaseScore as BaseTopicNetScore,
    TopicModel
)


from .base_custom_score import BaseCustomScore


_Logger = logging.getLogger()


class CalinskiHarabaszScore(BaseCustomScore):
    """
    Uses of Calinski-Harabasz:

    https://link.springer.com/article/10.1007/s40815-017-0327-9
    """
    def __init__(
            self,
            name: str,
            validation_dataset: Dataset
            ):

        super().__init__(name)

        self._score = _CalinskiHarabaszScore(validation_dataset)


class _CalinskiHarabaszScore(BaseTopicNetScore):
    def __init__(self, validation_dataset):
        super().__init__()

        self._dataset = validation_dataset
        self._keep_dataset_in_memory = validation_dataset._small_data
        self._dataset_internals_folder_path = validation_dataset._internals_folder_path
        self._dataset_file_path = validation_dataset._data_path

    def call(self, model: TopicModel):
        theta = model.get_theta(dataset=self._dataset)

        theta.columns = range(len(theta.columns))
        objects_clusters = theta.values.argmax(axis=0)

        # TODO: or return some numeric?
        if len(set(objects_clusters)) == 1:
            _Logger.warning(
                'Only one unique cluster! Returning None as score value'
            )

            return float('nan')

        return calinski_harabasz_score(theta.T.values, objects_clusters)

    # TODO: this piece is copy-pastd among three different scores
    def save(self, path: str) -> None:
        dataset = self._dataset
        self._dataset = None

        with open(path, 'wb') as f:
            dill.dump(self, f)

        self._dataset = dataset

    @classmethod
    def load(cls, path: str):
        """

        Parameters
        ----------
        path

        Returns
        -------
        an instance of this class

        """

        with open(path, 'rb') as f:
            score = dill.load(f)

        score._dataset = Dataset(
            score._dataset_file_path,
            internals_folder_path=score._dataset_internals_folder_path,
            keep_in_memory=score._keep_dataset_in_memory,
        )

        return score
