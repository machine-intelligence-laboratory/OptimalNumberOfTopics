import os
import shutil
import warnings
from topicnet.cooking_machine import Dataset


class BaseTextCollection:
    def __init__(self):
        self._dataset_folder = None

    def _to_dataset(self) -> Dataset:
        raise NotImplementedError()

    def _remove_dataset(self):
        if self._dataset_folder is None:
            return

        if not os.path.isdir(self._dataset_folder):
            warnings.warn(
                f'There is no directory by path "{self._dataset_folder}"!'
                f' Maybe something is wrong, so nothing is going to be removed'
            )

            return

        shutil.rmtree(self._dataset_folder)
