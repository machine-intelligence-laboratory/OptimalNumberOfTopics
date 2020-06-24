import artm

from enum import Enum
from typing import List

from topicnet.cooking_machine import Dataset
from topicnet.cooking_machine.rel_toolbox_lite import count_vocab_size, transform_regularizer
from topicnet.cooking_machine.model_constructor import (
    create_default_topics, add_standard_scores, init_model
)


class KnownModel(Enum):
    LDA = 'LDA'
    PLSA = 'PLSA'
    SPARSE = 'sparse'
    DECORRELATION = 'decorrelation'
    ARTM = 'ARTM'


def init_model_from_family(
            family: str or KnownModel,
            dataset: Dataset,
            main_modality: str,
            num_topics: int,
            seed: int,
            modalities_to_use: List[str] = None,
            num_processors: int = 3
):
    """
    """
    if isinstance(family, KnownModel):
        family = family.value

    if modalities_to_use is None:
        modalities_to_use = [main_modality]

    if family == "LDA":
        model = init_lda(dataset, modalities_to_use, main_modality, num_topics)
    elif family == "PLSA":
        model = init_plsa(dataset, modalities_to_use, main_modality, num_topics)
    elif family == "sparse":
        # TODO: TARTM
        model = init_bcg_sparse_model(dataset, modalities_to_use, main_modality, num_topics, 1)
    elif family == "decorrelation":
        model = init_decorrelated_plsa(dataset, modalities_to_use, main_modality, num_topics)
    elif family == "ARTM":
        model = init_baseline_artm(dataset, modalities_to_use, main_modality, num_topics, 1)
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
        dataset, modalities_to_use, main_modality, num_topics
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

    model = init_plsa(
        dataset, modalities_to_use, main_modality, num_topics
    )

    specific_topic_names = model.topic_names  # let's decorrelate everything
    model.regularizers.add(
        artm.DecorrelatorPhiRegularizer(
            gamma=0,
            tau=0.01,
            name='decorrelation',
            topic_names=specific_topic_names,
            class_ids=modalities_to_use,
        )
    )

    return model


def init_lda(
        dataset, modalities_to_use, main_modality,
        num_topics, prior="symmetric"
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

    # TODO: implement this LDA also
    # Found in doi.org/10.1007/s10664-015-9379-3
    # Rosen, C., Shihab, E. 2016
    # What are mobile developers asking about? A large scale study using stack overflow.
    #
    # "We use the defacto standard heuristics of α=50/K and β=0.01
    # (Biggers et al. 2014) for our hyperparameter values"

    # what GenSim returns by default (everything is 'symmetric')
    # see https://github.com/RaRe-Technologies/gensim/blob/master/gensim/models/ldamodel.py#L521
    if prior == "symmetric":
        alpha = 1.0 / num_topics
        eta = 1.0 / num_topics
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
        specific_topics, bcg_topics
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
             tau=0.1,
             class_ids=[main_modality],
        ),
        artm.SmoothSparseThetaRegularizer(
             name='smooth_theta_bcg',
             topic_names=background_topic_names,
             tau=0.1,
        ),
        artm.SmoothSparsePhiRegularizer(
             name='sparse_phi_sp',
             topic_names=specific_topic_names,
             tau=-0.05,
             class_ids=[main_modality],
            ),
        artm.SmoothSparseThetaRegularizer(
             name='sparse_theta_sp',
             topic_names=specific_topic_names,
             tau=-0.05,
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
        dataset, modalities_to_use, main_modality, num_topics, bcg_topics
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
        dataset, modalities_to_use, main_modality, num_topics, bcg_topics
    )
    specific_topic_names = model.topic_names[:-bcg_topics]

    model.regularizers.add(
        artm.DecorrelatorPhiRegularizer(
            gamma=0,
            tau=0.01,
            name='decorrelation',
            topic_names=specific_topic_names,
            class_ids=modalities_to_use,
        )
    )

    return model
