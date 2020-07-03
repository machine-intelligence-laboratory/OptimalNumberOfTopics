import numpy as np

import artm

from enum import Enum
from typing import List

from topicnet.cooking_machine import Dataset
from topicnet.cooking_machine.rel_toolbox_lite import (
    count_vocab_size,
    transform_regularizer,
)
from topicnet.cooking_machine.model_constructor import (
    add_standard_scores,
    create_default_topics,
    init_model,
)
from topicnet.cooking_machine.models import TopicModel, ThetalessRegularizer


class KnownModel(Enum):
    LDA = 'LDA'
    PLSA = 'PLSA'
    SPARSE = 'sparse'
    TLESS = 'TARTM'
    DECORRELATION = 'decorrelation'
    ARTM = 'ARTM'


# TODO: maybe this shouldn't be here (let it be in Jupyter notebook with experiments)
PARAMS_EXPLORED = {
    KnownModel.LDA: {
        'prior': [
            'symmetric',
            'asymmetric',
            'heuristic',
        ]
    },
    KnownModel.PLSA: {},
    KnownModel.TLESS: {},
    KnownModel.SPARSE: {
        'smooth_bcg_tau': [0.05, 0.1],
        'sparse_sp_tau':  [-0.05, -0.1]
    },
    KnownModel.DECORRELATION: {
        'decorrelation_tau': [0.02, 0.05, 0.1]
    },
    KnownModel.ARTM: {
        'smooth_bcg_tau':    [0.05, 0.1],
        'sparse_sp_tau':     [-0.05, -0.1],
        'decorrelation_tau': [0.02, 0.05, 0.1]
    }
}


def init_model_from_family(
            family: str or KnownModel,
            dataset: Dataset,
            main_modality: str,
            num_topics: int,
            seed: int,
            all_mods: List[str] = None,
            num_processors: int = 3,
            model_params: dict = None
):
    """
    Returns
    -------
    model: TopicModel() instance
    """
    if isinstance(family, KnownModel):
        family = family.value

    if all_mods is None:
        all_mods = [main_modality]

    custom_regs = {}

    if family == "LDA":
        model = init_lda(dataset, all_mods, main_modality, num_topics, model_params)
    elif family == "PLSA":
        model = init_plsa(dataset, all_mods, main_modality, num_topics)
    elif family == "TARTM":
        result = init_thetaless(dataset, all_mods, main_modality, num_topics, 1, model_params)
        model, custom_regs = result
    elif family == "sparse":
        model = init_bcg_sparse_model(dataset, all_mods, main_modality, num_topics, 1, model_params)
    elif family == "decorrelation":
        model = init_decorrelated_plsa(dataset, all_mods, main_modality, num_topics, model_params)
    elif family == "ARTM":
        model = init_baseline_artm(dataset, all_mods, main_modality, num_topics, 1, model_params)
    else:
        raise ValueError(f'family: {family}')

    model.num_processors = num_processors

    if seed is not None:
        model.seed = seed

    dictionary = dataset.get_dictionary()
    model.initialize(dictionary)
    add_standard_scores(model, dictionary, main_modality=main_modality,
                        all_modalities=all_mods)

    model = TopicModel(
        artm_model=model,
        custom_regularizers=custom_regs
    )

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

    model = init_model(
        topic_names=specific_topic_names + background_topic_names,
        class_ids=baseline_class_ids,
    )

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
        dataset: Dataset,
        modalities_to_use: List[str],
        main_modality: str,
        num_topics: int,
        model_params: dict,
):
    """
    Creates simple artm model with standard scores.

    Parameters
    ----------
    dataset
    modalities_to_use
    main_modality
    num_topics
    model_params

    Returns
    -------
    model: artm.ARTM() instance
    """
    model = init_plsa(
        dataset, modalities_to_use, main_modality, num_topics
    )

    if model_params is None:
        model_params = dict()

    prior = model_params.get('prior', 'symmetric')

    # What GenSim returns by default (everything is 'symmetric')
    # see https://github.com/RaRe-Technologies/gensim/blob/master/gensim/models/ldamodel.py#L521
    # Note that you can specify prior shape for alpha and beta separately,
    # but we do not do that here
    if prior == "symmetric":
        alpha = 1.0 / num_topics
        eta = 1.0 / num_topics
    elif prior == "asymmetric":
        # following the recommendation from
        # http://papers.nips.cc/paper/3854-rethinking-lda-why-priors-matter
        # we will use symmetric prior over Phi and asymmetric over Theta

        # this stuff is needed for asymmetric Phi initialization:
        if False:
            artm_dict = dataset.get_dictionary()
            temp_df = artm_dict2df(artm_dict)  # noqa: F821
            num_terms = temp_df.query("class_id in @modalities_to_use").shape[0]
            eta = _init_dirichlet_prior("eta", num_topics, num_terms)

        eta = 0
        alpha = _init_dirichlet_prior("alpha", num_topics, num_terms=0)
    elif prior == "heuristic":
        # Found in doi.org/10.1007/s10664-015-9379-3 (2016)
        #  "We use the defacto standard heuristics of α=50/K and β=0.01
        #  (Biggers et al. 2014) for our hyperparameter values"
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
    if isinstance(alpha, list):
        for i, topic in enumerate(model.topic_names):
            model.regularizers.add(
                artm.SmoothSparseThetaRegularizer(
                     name=f'smooth_theta_{i}', tau=alpha[i], topic_names=topic
                )
            )
    else:
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


def init_thetaless(
        dataset, modalities_to_use, main_modality, num_topics, model_params
):
    """
    Creates a base thetaless artm model and custom regularizer to be attached later

    Parameters
    ----------
    dataset : Dataset
    modalities_to_use : list of str
    main_modality : str
    num_topics : int
    model_params : dict

    Returns
    -------
    tuple:
        model: artm.ARTM() instance
        custom_regularizers: dict
            contains a single element, custom regularizer
    """

    model = init_plsa(
        dataset, modalities_to_use, main_modality, num_topics
    )
    model.num_document_passes = 1

    thetaless_reg = ThetalessRegularizer(
        name='thetaless',
        tau=1,
        dataset=dataset, modality=main_modality,
    )
    custom_regularizers = {
        thetaless_reg.name: thetaless_reg
    }

    return model, custom_regularizers
