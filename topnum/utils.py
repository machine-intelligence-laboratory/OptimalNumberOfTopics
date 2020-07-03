from topnum.model_constructor import KnownModel
from topnum.search_methods.optimize_scores_method import load_models_from_disk

from topnum.scores import (
    SpectralDivergenceScore, CalinskiHarabaszScore, DiversityScore, EntropyScore,
    HoldoutPerplexityScore, IntratextCoherenceScore,
    LikelihoodBasedScore, PerplexityScore, SilhouetteScore,
    SparsityPhiScore, SparsityThetaScore,

    # Unused:
    # SimpleTopTokensCoherenceScore,
    SophisticatedTopTokensCoherenceScore
)

from inspect import signature

from collections import defaultdict

import strictyaml
from strictyaml import Map, Str, Optional, Int

import pandas as pd

from topicnet.cooking_machine.dataset import Dataset
import numpy as np
import os, glob


def split_into_train_test(dataset: Dataset, config: dict):
    documents = list(dataset._data.index)
    dn = config['batches_prefix']

    random = np.random.RandomState(seed=123)

    random.shuffle(documents)

    test_size = 0.2

    train_documents = documents[:int(1.0 - test_size * len(documents))]
    test_documents = documents[len(train_documents):]

    assert len(train_documents) + len(test_documents) == len(documents)

    # TODO: test with keep_in_memory = False just in case
    train_data = dataset._data.loc[train_documents]
    test_data = dataset._data.loc[test_documents]
    train_data['id'] = train_data.index
    test_data['id'] = test_data.index

    train_data.to_csv(f'{dn}_train.csv', index=False)
    test_data.to_csv(f'{dn}_test.csv', index=False)

    train_dataset = Dataset(
        f'{dn}_train.csv',
        batch_vectorizer_path=f'{dn}_train_internals',
        keep_in_memory=dataset._small_data,
    )
    test_dataset = Dataset(
        f'{dn}_test.csv',
        batch_vectorizer_path=f'{dn}_test_internals',
        keep_in_memory=dataset._small_data,
    )

    # TODO: quick hack, i'm not sure what for
    test_dataset._to_dataset = lambda: test_dataset
    train_dataset._to_dataset = lambda: train_dataset

    return train_dataset, test_dataset


# TODO: it needs a dummy load
# like this:
# _ = build_every_score(dataset, dataset)


def build_every_score(dataset, test_dataset, config):
    scores = [
        SpectralDivergenceScore("arun", dataset, [config['word']]),
        PerplexityScore("perp"),
        SparsityPhiScore("sparsity_phi"), SparsityThetaScore("sparsity_theta"),
        HoldoutPerplexityScore('holdout_perp', test_dataset=test_dataset)
    ]

    max_coherence_text_length = 25000
    coherence_documents = list(test_dataset._data.index)
    coherence_text_length = sum(
        len(t.split())
        for t in test_dataset._data.loc[coherence_documents, 'vw_text'].values
    )

    while coherence_text_length > max_coherence_text_length:
        d = coherence_documents.pop()
        coherence_text_length -= len(test_dataset._data.loc[d, 'vw_text'].split())

    assert len(coherence_documents) > 0

    print(
        f'Num documents for coherence: {len(coherence_documents)}, {coherence_text_length} words'
    )

    coherences = [
        IntratextCoherenceScore(
            'intra', data=test_dataset, documents=coherence_documents
        ),
        SophisticatedTopTokensCoherenceScore(
            'toptok1', data=test_dataset, documents=coherence_documents
        ),

        # TODO: and this
        #   SimpleTopTokensCoherenceScore(),
    ]

    likelihoods = [
        LikelihoodBasedScore(
            f"{mode}_sparsity_{flag}", validation_dataset=dataset, modality=config['word'],
            mode=mode, consider_sparsity=flag
        )
        for mode in ["AIC", "BIC", "MDL"] for flag in [True, False]
    ]

    renyi_variations = [
        EntropyScore(f"renyi_{threshold_factor}", threshold_factor=threshold_factor)
        for threshold_factor in [0.5, 1, 2]
    ]
    clustering = [
        CalinskiHarabaszScore("calhar", dataset), SilhouetteScore("silh", dataset)
    ]
    diversity = [
        DiversityScore(f"diversity_{metric}_{is_closest}", metric=metric, closest=is_closest)
        for metric in ["euclidean", 'jensenshannon', 'cosine', 'hellinger']
        for is_closest in [True, False]
    ]

    return scores + diversity + clustering + renyi_variations + likelihoods + coherences


def check_if_monotonous(score_result):
    signs = np.sign(score_result.diff().iloc[1:, :])
    # convert all nans to a single value
    different_signs = set(signs.values.flatten().astype(str))
    if different_signs == {'nan', '0.0'}:
        return True
    return len(different_signs) == 1


def monotonity_and_std_analysis(experiment_directory, experiment_name_template):
    informative_df = pd.DataFrame()

    all_subexperems_mask = os.path.join(experiment_directory, experiment_name_template.format("*", "*"))

    for entry in glob.glob(all_subexperems_mask):

        experiment_name = entry.split("/")[-1]
        try:
            result, detailed_result = load_models_from_disk(
                experiment_directory, experiment_name
            )
            for score in detailed_result.keys():
                max_std = detailed_result[score].std().max()
                avg_val = detailed_result[score].median().median()
                rel_error = max_std / avg_val

                if rel_error > 0.01:
                    print(score, rel_error, detailed_result[score].std().min(), max_std)

                is_monotonous = check_if_monotonous(detailed_result[score].T)
                informative_df.loc[score, experiment_name] = is_monotonous
        except IndexError as e:
            print(f"Error reading data from {entry};\nThe exception raised is\n{e}")
            pass
    return informative_df


def read_corpus_config(filename='corpus.yml'):

    schema = Map({
        'dataset_path': Str(),
        'batches_prefix': Str(),
        'word': Str(),
        'name': Str(),
        Optional("num_topics_interval"): Int(),
        'min_num_topics': Int(),
        'max_num_topics': Int(),
        'num_fit_iterations': Int(),
        'num_restarts': Int(),
    })

    with open(filename, 'r') as f:
        string = f.read()
    data = strictyaml.load(string, schema=schema).data
    return data


def trim_config(config, method):
    return {
        elem: config[elem]
        for elem in signature(method.__init__).parameters
        if elem in config
    }


def estimate_num_iterations_for_convergence(tm):
    score = tm.scores["PerplexityScore@all"]
    normalized_score = np.array(score) / np.median(score)
    contributions = abs(np.diff(normalized_score))

    return (contributions > 2e-3).sum()


def plot_everything_informative(
    experiment_directory, experiment_name_template,
    true_criteria=[], false_criteria=[]
):
    import matplotlib.pyplot as plt

    details = defaultdict(dict)

    all_subexperems_mask = os.path.join(experiment_directory, experiment_name_template.format("*", "*"))

    for entry in glob.glob(all_subexperems_mask):

        experiment_name = entry.split("/")[-1]

        result, detailed_result = load_models_from_disk(
            experiment_directory, experiment_name
        )

        for score in detailed_result.keys():
            should_plot = (
                all(t_criterion in score for t_criterion in true_criteria)
                and
                all(f_criterion not in score for f_criterion in false_criteria)
            )
            if should_plot:
                details[score][experiment_name] = detailed_result[score].T
    for score in details.keys():
        fig, axes = plt.subplots(1, 1, figsize=(10, 10))
        for experiment_name, data in details[score].items():
            is_monotonous = check_if_monotonous(data)

            # I can make a grid of plots if I do something like this:
            # my_ax = axes[index // 3][index % 3]
            my_ax = axes
            if is_monotonous:
                style = ':'
            else:
                style = '-'
            my_ax.plot(data.T.mean(axis=0), linestyle=style, label=experiment_name)

        my_ax.set_title(f"{score}")
        my_ax.legend()
        fig.show()
