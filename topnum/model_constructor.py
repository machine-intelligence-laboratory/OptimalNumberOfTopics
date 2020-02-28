from .rel_toolbox_lite import count_vocab_size, modality_weight_rel2abs
import artm

# change log style
lc = artm.messages.ConfigureLoggingArgs()
lc.minloglevel = 3
lib = artm.wrapper.LibArtm(logging_config=lc)



def init_model(topic_names, seed=None, class_ids=None):
    """
    Creates basic artm model

    """
    model = artm.ARTM(
        topic_names=topic_names,
        # Commented for performance uncomment if has zombie issues
        # num_processors=3,
        theta_columns_naming='title',
        show_progress_bars=False,
        class_ids=class_ids,
        seed=seed
    )

    return model


def create_default_topics(specific_topics, background_topics):
    """
    Creates list of background topics and specific topics

    Parameters
    ----------
    specific_topics : list or int
    background_topics : list or int

    Returns
    -------
    (list, list)
    """
    # TODO: what if specific_topics = 4
    # and background_topics = ["topic_0"] ?
    if isinstance(specific_topics, list):
        specific_topic_names = list(specific_topics)
    else:
        specific_topics = int(specific_topics)
        specific_topic_names = [
            f'topic_{i}'
            for i in range(specific_topics)
        ]
    n_specific_topics = len(specific_topic_names)
    if isinstance(background_topics, list):
        background_topic_names = list(background_topics)
    else:
        background_topics = int(background_topics)
        background_topic_names = [
            f'background_{n_specific_topics + i}'
            for i in range(background_topics)
        ]
    if set(specific_topic_names) & set(background_topic_names):
        raise ValueError(
            "Specific topic names and background topic names should be distinct from each other!"
        )

    return specific_topic_names, background_topic_names


def init_decorrelated_PLSA(
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
    baseline_class_ids = {class_id: 1 for class_id in modalities_to_use}

    specific_topic_names, background_topic_names = create_default_topics(
        num_topics, 0
    )
    dictionary = dataset.get_dictionary()

    model = init_model(
        topic_names=specific_topic_names + background_topic_names,
        class_ids=baseline_class_ids,
    )

    model.regularizers.add(
            artm.DecorrelatorPhiRegularizer(
            gamma=0,
            tau=0.01,
            name='decorrelation',
            topic_names=specific_topic_names,
            class_ids=words_class_ids,
            dictionary=dictionary
        ),

    model.initialize(dictionary)
    add_standard_scores(model, dictionary, main_modality=main_modality,
                        all_modalities=modalities_to_use)

    return model

def init_PLSA(
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
    baseline_class_ids = {class_id: 1 for class_id in modalities_to_use}

    specific_topic_names, background_topic_names = create_default_topics(
        num_topics, 0
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

def init_LDA(
        dataset, modalities_to_use, main_modality,
        num_topics
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
    baseline_class_ids = {class_id: 1 for class_id in modalities_to_use}

    specific_topic_names, background_topic_names = create_default_topics(
        num_topics, 0
    )

    model = init_model(
        topic_names=specific_topic_names + background_topic_names,
        class_ids=baseline_class_ids,
    )

    # what GenSim returns by default (everything is 'symmetric')
    # see https://github.com/RaRe-Technologies/gensim/blob/master/gensim/models/ldamodel.py#L521
    alpha = 1.0 / num_topics
    eta = 1.0 / num_topics
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

    dictionary = dataset.get_dictionary()
    model.initialize(dictionary)
    add_standard_scores(model, dictionary, main_modality=main_modality,
                        all_modalities=modalities_to_use)

    return model

def init_bcg_sparse_model(
        dataset, modalities_to_use, main_modality,
        specific_topics, background_topics,
):
    """
    Creates simple artm model with standard scores.

    Parameters
    ----------
    dataset : Dataset
    modalities_to_use : list of str
    main_modality : str
    specific_topics : list or int
    background_topics : list or int
    modalities_weights : dict or None

    Returns
    -------
    model: artm.ARTM() instance
    """
    baseline_class_ids = {class_id: 1 for class_id in modalities_to_use}

    specific_topic_names, background_topic_names = create_default_topics(
        specific_topics, background_topics
    )
    dictionary = dataset.get_dictionary()

    model = init_model(
        topic_names=specific_topic_names + background_topic_names,
        class_ids=baseline_class_ids,
    )

    if len(background_topic_names) > 0:
        model.regularizers.add(
            artm.SmoothSparsePhiRegularizer(
                 name='smooth_phi_bcg',
                 topic_names=background_topic_names,
                 tau=0.0,
                 class_ids=[main_modality],
            ),
        )
        model.regularizers.add(
            artm.SmoothSparseThetaRegularizer(
                 name='smooth_theta_bcg',
                 topic_names=background_topic_names,
                 tau=0.0,
            ),
        )
        model.regularizers.add(
            artm.SmoothSparsePhiRegularizer(
                 name='sparse_phi_sp',
                 topic_names=background_topic_names,
                 tau=0.0,
                 class_ids=[main_modality],
            ),
        )
        model.regularizers.add(
            artm.SmoothSparseThetaRegularizer(
                 name='sparse_theta_sp',
                 topic_names=background_topic_names,
                 tau=0.0,
            ),
        )
    # TODO: TRANSFORM REGULARIZER

    model.initialize(dictionary)
    add_standard_scores(model, dictionary, main_modality=main_modality,
                        all_modalities=modalities_to_use)

    return model
