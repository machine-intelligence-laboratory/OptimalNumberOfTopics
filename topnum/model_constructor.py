import artm

from enum import Enum
from typing import List

from topicnet.cooking_machine import Dataset
from topicnet.cooking_machine.rel_toolbox_lite import count_vocab_size, transform_regularizer
from topicnet.cooking_machine.model_constructor import (
    create_default_topics, add_standard_scores, init_model
)
import numpy as np
import pandas as pd


class KnownModel(Enum):
    LDA = 'LDA'
    PLSA = 'PLSA'
    SPARSE = 'sparse'
    DECORRELATION = 'decorrelation'
    ARTM = 'ARTM'


PARAMS_EXPLORED = {
    KnownModel.LDA: {'prior': ['symmetric', #'asymmetric',
                               'small', 'heuristic']},
    KnownModel.PLSA: {},
    KnownModel.SPARSE: {'smooth_bcg_tau': [0.05, 0.1],
                        'sparse_sp_tau':  [-0.05, -0.1]},
    KnownModel.DECORRELATION: {'decorrelation_tau': [0.02, 0.05, 0.1]},
    KnownModel.ARTM: {'smooth_bcg_tau':    [0.05, 0.1],
                      'sparse_sp_tau':     [-0.05, -0.1],
                      'decorrelation_tau': [0.02, 0.05, 0.1]}
}

# TODO: move this to BigARTM dictionary
# ==================================

FIELDS = 'token class_id token_value token_tf token_df'.split()


def artm_dict2df(artm_dict):
    dictionary_data = artm_dict._master.get_dictionary(artm_dict._name)
    dict_pandas = {field: getattr(dictionary_data, field)
                   for field in FIELDS}
    return pd.DataFrame(dict_pandas)


# ==================================


def init_model_from_family(
            family: str or KnownModel,
            dataset: Dataset,
            main_modality: str,
            num_topics: int,
            seed: int,
            modalities_to_use: List[str] = None,
            num_processors: int = 3,
            model_params: dict = None
):
    """
    """
    if isinstance(family, KnownModel):
        family = family.value

    if modalities_to_use is None:
        modalities_to_use = [main_modality]

    if family == "LDA":
        model = init_lda(dataset, modalities_to_use, main_modality, num_topics, model_params)
    elif family == "PLSA":
        model = init_plsa(dataset, modalities_to_use, main_modality, num_topics)
    elif family == "sparse":
        # TODO: TARTM
        model = init_bcg_sparse_model(dataset, modalities_to_use, main_modality, num_topics, 1, model_params)
    elif family == "decorrelation":
        model = init_decorrelated_plsa(dataset, modalities_to_use, main_modality, num_topics, model_params)
    elif family == "ARTM":
        model = init_baseline_artm(dataset, modalities_to_use, main_modality, num_topics, 1, model_params)
    else:
        raise ValueError(f'family: {family}')

    model.num_processors = num_processors

    if seed is not None:
        model.seed = seed

    return model


def init_plsa(
        dataset, modalities_to_use, main_modality, num_topics, num_bcg_topics=0
):
    """
    Creates simple artm model with standard scores.

    Parameters
    ----------
    dataset : Dataset
    modalities_to_use : list of str
    main_modality : str
    num_topics : int
    num_bcg_topics : int

    Returns
    -------
    model: artm.ARTM() instance
    """
    baseline_class_ids = {class_id: 1 for class_id in modalities_to_use}

    specific_topic_names, background_topic_names = create_default_topics(
        num_topics, num_bcg_topics
    )
    dictionary = dataset.get_dictionary()

    model = init_model(
        topic_names=specific_topic_names + background_topic_names,
        class_ids=baseline_class_ids,
    )

    dictionary = dataset.get_dictionary()
    model.initialize(dictionary)
    add_standard_scores(model, dictionary, main_modality=main_modality,
                        all_modalities=modalities_to_use)

    return model


def init_decorrelated_plsa(
        dataset, modalities_to_use, main_modality, num_topics, model_params
):
    """
    Creates simple artm model with standard scores.

    Parameters
    ----------
    dataset : Dataset
    modalities_to_use : list of str
    main_modality : str
    num_topics : int
    model_params : dict

    Returns
    -------
    model: artm.ARTM() instance
    """

    model = init_plsa(
        dataset, modalities_to_use, main_modality, num_topics
    )
    tau = model_params.get('decorrelation_tau', 0.01)

    specific_topic_names = model.topic_names  # let's decorrelate everything
    model.regularizers.add(
        artm.DecorrelatorPhiRegularizer(
            gamma=0,
            tau=tau,
            name='decorrelation',
            topic_names=specific_topic_names,
            class_ids=modalities_to_use,
        )
    )

    return model


def _init_dirichlet_prior(name, num_topics, num_terms):
    prior_shape = num_topics if name == 'alpha' else num_terms

    init_prior = np.fromiter(
        (1.0 / (i + np.sqrt(prior_shape)) for i in range(prior_shape)),
        dtype=np.float32, count=prior_shape
    )
    init_prior /= init_prior.sum()
    return init_prior


def init_lda(
        dataset, modalities_to_use, main_modality,
        num_topics, model_params
):
    """
    Creates simple artm model with standard scores.

    Parameters
    ----------
    dataset : Dataset
    modalities_to_use : list of str
    main_modality : str
    num_topics : int
    prior : str

    Returns
    -------
    model: artm.ARTM() instance
    """
    model = init_plsa(
        dataset, modalities_to_use, main_modality, num_topics
    )

    if model_params is None:
        model_params = {}
    prior = model_params.get('prior', 'symmetric')

    # what GenSim returns by default (everything is 'symmetric')
    # see https://github.com/RaRe-Technologies/gensim/blob/master/gensim/models/ldamodel.py#L521
    # note that you can specify prior shape for alpha and beta separately
    # but we do not do that here
    if prior == "symmetric":
        alpha = 1.0 / num_topics
        eta = 1.0 / num_topics
    elif prior == "asymmetric":
        # TODO: turns out, BigARTM does not support tau as a list of floats
        # so we need to use custom regularzir instead (TopicPrior perhaps?)
        # this won't be happening today :(
        artm_dict = dataset.get_dictionary()
        temp_df = artm_dict2df(artm_dict)
        num_terms = temp_df.query("class_id in @modalities_to_use").shape[0]
        alpha = _init_dirichlet_prior("alpha", num_topics, num_terms)
        eta = _init_dirichlet_prior("eta", num_topics, num_terms)
        raise NotImplementedError
    elif prior == "small":
        # used in BigARTM
        alpha = 0.01
        eta = 0.01
    elif prior == "heuristic":
        # found in doi.org/10.1007/s10664-015-9379-3 (2016)
        #
        # "We use the defacto standard heuristics of α=50/K and β=0.01
        # (Biggers et al. 2014) for our hyperparameter values"
        alpha = 50.0 / num_topics
        eta = 0.01
    else:
        raise TypeError(f"prior type '{prior}' is not supported")

    model.regularizers.add(
        artm.SmoothSparsePhiRegularizer(
             name='smooth_phi',
             tau=eta,
             class_ids=[main_modality],
        ),
    )
    model.regularizers.add(
        artm.SmoothSparseThetaRegularizer(
             name='smooth_theta',
             tau=alpha,
        ),
    )

    return model


def init_bcg_sparse_model(
        dataset, modalities_to_use, main_modality,
        specific_topics, bcg_topics,
        model_params
):
    """
    Creates simple artm model with standard scores.

    Parameters
    ----------
    dataset : Dataset
    modalities_to_use : list of str or dict
    main_modality : str
    specific_topics : int
    bcg_topics : int

    Returns
    -------
    model: artm.ARTM() instance
    """
    model = init_plsa(
        dataset, modalities_to_use, main_modality, specific_topics, bcg_topics
    )
    background_topic_names = model.topic_names[-bcg_topics:]
    specific_topic_names = model.topic_names[:-bcg_topics]

    dictionary = dataset.get_dictionary()
    baseline_class_ids = {class_id: 1 for class_id in modalities_to_use}
    data_stats = count_vocab_size(dictionary, baseline_class_ids)

    # all coefficients are relative
    regularizers = [
        artm.SmoothSparsePhiRegularizer(
             name='smooth_phi_bcg',
             topic_names=background_topic_names,
             tau=model_params.get("smooth_bcg_tau", 0.1),
             class_ids=[main_modality],
        ),
        artm.SmoothSparseThetaRegularizer(
             name='smooth_theta_bcg',
             topic_names=background_topic_names,
             tau=model_params.get("smooth_bcg_tau", 0.1),
        ),
        artm.SmoothSparsePhiRegularizer(
             name='sparse_phi_sp',
             topic_names=specific_topic_names,
             tau=model_params.get("sparse_sp_tau", -0.05),
             class_ids=[main_modality],
            ),
        artm.SmoothSparseThetaRegularizer(
             name='sparse_theta_sp',
             topic_names=specific_topic_names,
             tau=model_params.get("sparse_sp_tau", -0.05),
        ),
    ]
    for reg in regularizers:
        model.regularizers.add(transform_regularizer(
            data_stats,
            reg,
            model.class_ids,
            n_topics=len(reg.topic_names)
        ))

    return model


def init_baseline_artm(
        dataset, modalities_to_use, main_modality, num_topics, bcg_topics, model_params
):
    """
    Creates simple artm model with standard scores.

    Parameters
    ----------
    dataset : Dataset
    modalities_to_use : list of str
    main_modality : str
    num_topics : int

    Returns
    -------
    model: artm.ARTM() instance
    """

    model = init_bcg_sparse_model(
        dataset, modalities_to_use, main_modality, num_topics, bcg_topics, model_params
    )
    specific_topic_names = model.topic_names[:-bcg_topics]

    model.regularizers.add(
        artm.DecorrelatorPhiRegularizer(
            gamma=0,
            tau=model_params.get('decorrelation_tau', 0.01),
            name='decorrelation',
            topic_names=specific_topic_names,
            class_ids=modalities_to_use,
        )
    )

    return model
