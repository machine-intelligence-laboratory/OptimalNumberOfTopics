import csv
import os
import tempfile
import warnings

from topicnet.cooking_machine import Dataset
from topicnet.cooking_machine.dataset import (
    VW_TEXT_COL,
    RAW_TEXT_COL,
)
from typing import (
    Dict,
    List,
    Optional,
    Union,
)

from .base_text_collection import BaseTextCollection


DOC_ID_COL = 'id'
WARNING_MAIN_MODALITY_NOT_IN_MODALITIES = (
    'Main modality "{}" is not in the list of modalities.'
    'Assuming it is an inadvertent mistake, so appending it to all modalities.'
)


class VowpalWabbitTextCollection(BaseTextCollection):
    def __init__(
            self,
            file_path: str,
            main_modality: str,
            modalities: Union[None, List[str], Dict[str, float]] = None,
            **dataset_kwargs):

        super().__init__()

        self._file_path = file_path

        if not os.path.isfile(self._file_path):
            raise FileNotFoundError(self._file_path)

        if main_modality is None:
            raise ValueError(f'Main modality shoul be specified')

        self._main_modality = main_modality

        if self._is_any_line_blank():
            raise ValueError(
                f'Some lines are blank in file "{self._file_path}".'
                ' Each line in a vowpal wabbit file should represent a document.'
                ' So lines can\'t be blank.'
            )

        if modalities is None:
            modalities = {self._main_modality: 1.0}

        elif isinstance(modalities, list):
            if self._main_modality not in modalities:
                warnings.warn(
                    WARNING_MAIN_MODALITY_NOT_IN_MODALITIES.format(self._main_modality)
                )

                modalities = modalities + [self._main_modality]

            modalities = {m: 1.0 for m in modalities}

        elif isinstance(modalities, dict):
            if self._main_modality not in modalities:
                warnings.warn(
                    WARNING_MAIN_MODALITY_NOT_IN_MODALITIES.format(self._main_modality)
                )

                modalities[self._main_modality] = 1.0

        else:
            raise TypeError(f'modalities: {type(modalities)}')

        self._modalities = modalities
        self._dataset: Optional[Dataset] = None
        self._dataset_kwargs = dataset_kwargs

    def _set_dataset_kwargs(self, **kwargs) -> None:
        self._dataset_kwargs = kwargs

    def _to_dataset(self) -> Dataset:
        """

        Additional Parameters
        ---------------------
        kwargs
            Optional init parameters of `Dataset`
        """
        if self._dataset is not None:
            return self._dataset

        if self._dataset_folder is not None:
            assert os.path.isdir(self._dataset_folder)
        else:
            self._dataset_folder = tempfile.mkdtemp(
                prefix='_dataset_',
                dir=os.path.dirname(self._file_path)
            )

        dataset_table_path = os.path.join(
            self._dataset_folder, 'dataset.csv'
        )

        with open(self._file_path, 'r') as f_in, open(dataset_table_path, 'w') as f_out:
            writer = csv.writer(f_out)

            writer.writerow([DOC_ID_COL, VW_TEXT_COL, RAW_TEXT_COL])

            for raw_vw_text in f_in:
                vw_text = raw_vw_text.strip()

                if len(vw_text) == 0:
                    continue

                doc_id = vw_text.split()[0]
                raw_text = None  # TODO: check if this OK

                writer.writerow([doc_id, vw_text, raw_text])

        self._dataset = Dataset(dataset_table_path, **self._dataset_kwargs)

        # TODO: remove this after TopicNet new release
        setattr(self._dataset, 'documents', list(self._dataset._data.index))

        return self._dataset

    @classmethod
    def from_dataset(
            cls,
            dataset: Dataset,
            main_modality: str,
            modalities: Union[None, List[str], Dict[str, float]] = None):
        """

        Returns
        -------
        VowpalWabbitTextCollection

        """
        file_descriptor = None
        text_collection: VowpalWabbitTextCollection = None

        try:
            file_descriptor, file_path = tempfile.mkstemp()
            text_collection = VowpalWabbitTextCollection(
                file_path,
                main_modality=main_modality,
                modalities=modalities,
            )
            text_collection._dataset = dataset

        finally:
            if file_descriptor is not None:
                os.close(file_descriptor)

        return text_collection

    def _is_any_line_blank(self) -> bool:
        # TODO: or better to not read all lines at once?
        return any((
            len(line.strip()) == 0
            for line in open(self._file_path).readlines()
        ))