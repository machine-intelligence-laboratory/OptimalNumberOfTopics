import artm
import numpy as np
import pandas as pd

from topicnet.cooking_machine import Dataset
from topicnet.cooking_machine.models import TopicModel
from typing import List

from topnum.search_methods.topic_bank.phi_initialization import utils as init_phi_utils


def initialize_randomly(
        dataset: Dataset,
        model_number: int,
        num_topics: int) -> pd.DataFrame:

    phi_template = _get_phi_template(dataset, num_topics)

    random = np.random.RandomState(seed=model_number)
    phi_values = random.random(phi_template.shape)

    return pd.DataFrame(
        index=phi_template.index,
        columns=phi_template.columns,
        data=phi_values
    )


def initialize_with_copying_topics(
        dataset: Dataset,
        model_number: int,
        num_topics: int,
        phi: pd.DataFrame,
        num_topics_to_copy: int = None,
        topic_indices_to_copy: List[int] = None) -> pd.DataFrame:

    random = np.random.RandomState(seed=model_number)

    if num_topics_to_copy is None and topic_indices_to_copy is None:
        raise ValueError(
            'Either `num_topics_to_copy` or `topic_indices_to_copy` should be specified!'
        )
    elif topic_indices_to_copy is None:
        topic_indices_to_copy = list(phi.index)
    elif num_topics_to_copy is None:
        num_topics_to_copy = len(topic_indices_to_copy)
    elif num_topics_to_copy != len(topic_indices_to_copy):
        raise ValueError(
            'If both `num_topics_to_copy` and `topic_indices_to_copy` are specified,'
            ' they shouldn\'t contradict each other!'
        )
    else:
        assert False

    topics_to_copy = random.choice(
        topic_indices_to_copy,
        size=num_topics_to_copy,
        replace=False
    )
    artm_model_template = _get_artm_model_template(
        dataset,
        num_topics
    )
    init_phi_utils._copy_phi(
        artm_model_template,
        phi.loc[:, topics_to_copy]
    )
    model_template = TopicModel(
        artm_model=artm_model_template
    )

    return model_template.get_phi()


def _get_phi_template(dataset: Dataset, num_topics: int) -> pd.DataFrame:
    artm_model = _get_artm_model_template(dataset, num_topics)
    model = TopicModel(artm_model=artm_model)
    phi_template = model.get_phi()

    del model
    del artm_model

    return phi_template


def _get_artm_model_template(dataset: Dataset, num_topics: int) -> artm.ARTM:
    artm_model = artm.ARTM(num_topics=num_topics, num_processors=1)
    artm_model.initialize(dictionary=dataset.get_dictionary())

    return artm_model
