import argparse
import json
import os
import traceback

from typing import (
    Dict,
    List,
    Tuple
)

from topnum.data.vowpal_wabbit_text_collection import VowpalWabbitTextCollection
from topnum.scores import (
    DiversityScore,
    EntropyScore,
    HoldoutPerplexityScore,
    IntratextCoherenceScore,
    PerplexityScore,
    SimpleTopTokensCoherenceScore,
    SophisticatedTopTokensCoherenceScore,
    SilhouetteScore,
    CalinskiHarabaszScore
)
from topnum.model_constructor import KNOWN_MODELS
from topnum.scores.diversity_score import L2
from topnum.scores.entropy_score import RENYI as RENYI_ENTROPY_NAME
from topnum.scores.base_score import BaseScore
from topnum.search_methods.constants import (
    DEFAULT_MAX_NUM_TOPICS,
    DEFAULT_MIN_NUM_TOPICS
)
from topnum.search_methods.optimize_scores_method import OptimizeScoresMethod
from topnum.search_methods.renormalization_method import (
    RenormalizationMethod,
    PHI_RENORMALIZATION_MATRIX,
    THETA_RENORMALIZATION_MATRIX
)

MESSAGE_MODALITY_FORMAT = (
    'Format: <modality name>, or <modality name>:<modality weight>.'
    ' Weight assumed to be 1.0 if not specified'
)


def _main():
    parser = argparse.ArgumentParser(prog='run_search')
    parser.add_argument(
        'vw_file_path',
        help='Path to the file with text collection in vowpal wabbit format'
    )
    parser.add_argument(
        '--mf', '--model_family',
        help=f'The family of models to optimize the number of topics for',
        default="PLSA", choices=KNOWN_MODELS
    )
    parser.add_argument(
        'main_modality',
        help=f'Main modality in text. {MESSAGE_MODALITY_FORMAT}'
    )
    parser.add_argument(
        'output_file_path',
        help='File to write the result of the search in'
    )
    # TODO: extract modalities from text if none specified
    parser.add_argument(
        '-m', '--modality',
        help=f'Other modality to take into account. {MESSAGE_MODALITY_FORMAT}',
        action='append',
        dest='modalities'
    )

    subparsers = parser.add_subparsers(
        help='Method for searching an appropriate number of topics',
        dest='search_method'
    )

    parser_optimize_scores = subparsers.add_parser(
        'optimize_scores',
        help='Find the number of topics which optimizes the score'
             ' (gives it max or min depending on the score)'
    )
    parser_renormalize = subparsers.add_parser(
        'renormalize',
        help='Fulfil topic matrix renormalization'
             ' to find the best number of topics relative to Renyi entropy'
    )

    parser_optimize_scores.add_argument(
        '--max-num-topics',
        help='Maximum number of topics',
        type=int,
        default=DEFAULT_MAX_NUM_TOPICS
    )
    parser_optimize_scores.add_argument(
        '--min-num-topics',
        help='Minimum number of topics',
        type=int,
        default=DEFAULT_MIN_NUM_TOPICS
    )
    parser_optimize_scores.add_argument(
        '--num-topics-interval',
        help='The number of topics the next model is bigger than the previous one',
        type=int,
        default=10
    )
    parser_optimize_scores.add_argument(
        '--num-fit-iterations',
        help='Number of fit iterations for model training',
        type=int,
        default=100
    )
    parser_optimize_scores.add_argument(
        '--num-restarts',
        help='Number of models to train,'
             ' each of which differs from the others by random seed'
             ' used for initialization.'
             ' Search results will be averaged over models '
             ' (suffix _std means standard deviation for restarts).',  # TODO: check English
        type=int,
        default=3
    )
    subparsers_optimize_scores = parser_optimize_scores.add_subparsers(
        help='Method for searching an appropriate number of topics',
        dest='score_name'
    )

    # TODO: try to run with several identical scores (for testing purposes)
    parser_optimize_perplexity = subparsers_optimize_scores.add_parser(
        'perplexity',
        help='Perplexity -> min'
    )
    parser_optimize_holdout_perplexity = subparsers_optimize_scores.add_parser(
        'holdout_perplexity',
        help='As usual perplexity, but on holdout sample'
    )
    parser_optimize_renyi_entropy = subparsers_optimize_scores.add_parser(
        'renyi_entropy',
        help='Renyi entropy -> min'
    )
    subparsers_optimize_scores.add_parser(
        'calinski_harabasz',
        help='CH -> max'
    )
    subparsers_optimize_scores.add_parser(
        'silhouette',
        help='SilhouetteScore -> max'
    )
    subparsers_optimize_scores.add_parser(
        'diversity',
        help='Diversity -> max'
    )
    parser_optimize_intratext = subparsers_optimize_scores.add_parser(
        'intratext_coherence',
        help='Intratext coherence -> max'
    )
    parser_optimize_toptokens = subparsers_optimize_scores.add_parser(
        'top_tokens_coherence',
        help='Top tokens coherence -> max'
    )

    # TODO: check this score using command line
    parser_optimize_holdout_perplexity.add_argument(
        '--test-vw-file-path',
        help='Path to the holdout data as vw file',
        type=str,
        required=True
    )
    parser_optimize_renyi_entropy.add_argument(
        '-f', '--threshold-factor',
        help='A greater than zero factor'
             ' by which the default 1/|W| threshold should be multiplied by',
        type=float,
        default=1.0
    )
    # TODO: add args for parser_optimize_intratext
    # TODO: add args for parser_optimize_toptokens
    parser_optimize_toptokens.add_argument(
        '--cooc-file',
        help='File with word cooccurrence values in the format'
             ' [[["word_1", "word_2"], 6.27], [["word_1", "word_3"], 1.32], ...],'
             ' i.e. there should be a list, where each item is another list:'
             ' word pair as yet another list and a numeric value corresponding to this word pair',
        type=str,
        default=None
    )

    parser_renormalize.add_argument(
        '--matrix',
        help='Matrix to be used for renormalization',
        type=str,
        default='phi',
        choices=['phi', 'theta']
    )
    # TODO: think about it: maybe these args better make general for all methods?
    parser_renormalize.add_argument(
        '--max-num-topics',
        help='Maximum number of topics',
        type=int,
        default=DEFAULT_MAX_NUM_TOPICS
    )
    parser_renormalize.add_argument(
        '--min-num-topics',
        help='Minimum number of topics',
        type=int,
        default=DEFAULT_MIN_NUM_TOPICS
    )
    parser_renormalize.add_argument(
        '--num-fit-iterations',
        help='Number of fit iterations for model training',
        type=int,
        default=100
    )
    parser_renormalize.add_argument(
        '--num-restarts',
        help='Number of models to train,'
             ' each of which differs from the others by random seed'
             ' used for initialization.'
             ' Search results will be averaged over models '
             ' (suffix _std means standard deviation for restarts).',  # TODO: check English
        type=int,
        default=3
    )

    # parser_some_other = subparsers.add_parser('other', help='some help')

    args, unparsed_args = parser.parse_known_args()

    main_modality_name, modalities = _parse_modalities(args.main_modality, args.modalities)
    modality_names = list(modalities.keys())
    vw_file_path = args.vw_file_path
    output_file_path = args.output_file_path

    text_collection = VowpalWabbitTextCollection(
        vw_file_path,
        main_modality=main_modality_name,
        modalities=modalities
    )

    if not os.path.isfile(vw_file_path):
        raise ValueError(
            f'File not found on path vw_file_path: "{vw_file_path}"!'
        )

    if not os.path.isdir(os.path.dirname(output_file_path))\
            and len(os.path.dirname(output_file_path)) > 0:

        raise ValueError(
            f'Directory not found for output file output_file_path: "{output_file_path}"!'
        )

    if args.search_method == 'optimize_scores':
        min_num_topics = args.min_num_topics
        max_num_topics = args.max_num_topics
        num_topics_interval = args.num_topics_interval
        num_fit_iterations = args.num_fit_iterations
        num_restarts = args.num_restarts
        model_family = args.model_family

        scores = list()
        scores.append(_build_score(args, text_collection, modality_names))

        while len(unparsed_args) > 0:
            current_args, unparsed_args = parser_optimize_scores.parse_known_args(
                unparsed_args
            )
            scores.append(
                _build_score(current_args, text_collection, modality_names, main_modality_name)
            )

        _optimize_scores(
            scores,
            model_family,
            text_collection,
            output_file_path,
            min_num_topics=min_num_topics,
            max_num_topics=max_num_topics,
            num_topics_interval=num_topics_interval,
            num_fit_iterations=num_fit_iterations,
            num_restarts=num_restarts
        )
    elif args.search_method == 'renormalize':
        min_num_topics = args.min_num_topics
        max_num_topics = args.max_num_topics
        num_fit_iterations = args.num_fit_iterations
        num_restarts = args.num_restarts

        if args.matrix == 'phi':
            matrix = PHI_RENORMALIZATION_MATRIX
        elif args.matrix == 'theta':
            matrix = THETA_RENORMALIZATION_MATRIX
        else:
            raise ValueError(f'matrix: {args.matrix}')  # ideally never happens

        _renormalize(
            text_collection,
            output_file_path,
            matrix=matrix,
            min_num_topics=min_num_topics,
            max_num_topics=max_num_topics,
            num_fit_iterations=num_fit_iterations,
            num_restarts=num_restarts
        )
    else:
        raise ValueError(args.search_method)

    text_collection._remove_dataset()


# TODO: test
def _extract_modality_name_and_weight(
        modality: str,
        default_weight: float = 1.0) -> Tuple[str, float]:

    if ':' not in modality:
        modality_name = modality
        modality_weight = default_weight
    else:
        components = modality.split(':')

        assert len(components) == 2

        modality_name = modality.split(':')[0]
        modality_weight = float(modality.split(':')[1])

    return modality_name, modality_weight


def _parse_modalities(
        main_modality: str, raw_modalities: List[str]) -> Tuple[str, Dict[str, float]]:

    modalities = dict()

    main_modality_name, main_modality_weight = _extract_modality_name_and_weight(
        main_modality
    )
    modalities[main_modality_name] = main_modality_weight

    # TODO: test weights: main @modality:5, -m @modality:2, -m @modality
    if raw_modalities is not None:
        for modality in raw_modalities:
            modality_name, modality_weight = _extract_modality_name_and_weight(
                modality
            )
            modalities[modality_name] = modality_weight

    return main_modality_name, modalities


def _build_score(
        args: argparse.Namespace,
        text_collection: VowpalWabbitTextCollection,
        modality_names: List[str],
        main_modality_name: str) -> BaseScore:

    # TODO: modality_names should be available via text_collection
    if args.score_name == 'perplexity':
        return PerplexityScore(
            'perplexity_score',
            class_ids=modality_names
        )
    elif args.score_name == 'holdout_perplexity':
        test_text_collection = VowpalWabbitTextCollection(
            args.test_vw_file_path,
            main_modality=main_modality_name,
            modalities=modality_names
        )

        return HoldoutPerplexityScore(
            name='holdout_perplexity_score',
            test_dataset=test_text_collection._to_dataset()
        )
    elif args.score_name == 'renyi_entropy':
        return EntropyScore(
            'renyi_entropy_score',
            entropy=RENYI_ENTROPY_NAME,
            threshold_factor=args.threshold_factor,
            class_ids=modality_names
        )
    elif args.score_name == 'calinski_harabasz':
        return CalinskiHarabaszScore(
            'calinski_harabasz_score',
            validation_dataset=text_collection._to_dataset()
        )
    elif args.score_name == 'silhouette':
        return SilhouetteScore(
            'silhouette_score',
            validation_dataset=text_collection._to_dataset()
        )
    elif args.score_name == 'diversity':
        return DiversityScore(
            'l2_diversity_score',
            metric=L2,
            class_ids=modality_names
        )
    elif args.score_name == 'intratext_coherence':
        return IntratextCoherenceScore(
            'intratext_coherence_score',
            data=text_collection
        )
    elif args.score_name == 'top_tokens_coherence' and args.cooc_file is None:
        # Actually, this one also can tame custom coocs, but in a bit different format:
        # with modalities, like ((@m, w1), (@m, w2)): 17.5, ...
        return SophisticatedTopTokensCoherenceScore(
            'top_tokens_coherence_score',
            data=text_collection
        )
    elif args.score_name == 'top_tokens_coherence' and args.cooc_file is not None:
        cooc_file = args.cooc_file

        if not os.path.isfile(cooc_file):
            raise ValueError(f'Coocs file not fould on path "{cooc_file}"!')

        try:
            raw_coocs_values = json.loads(open(cooc_file, 'r').read())
        except json.JSONDecodeError:
            raise ValueError(
                f'Coocs file "{cooc_file}" doesn\'t seem like valid JSON!'
                f' Error: {traceback.format_exc()}'
            )

        cooc_values = {
            tuple(d[0]): d[1] for d in raw_coocs_values
        }

        return SimpleTopTokensCoherenceScore(
            'top_tokens_coherence_score',
            cooccurrence_values=cooc_values,
            data=text_collection,
        )
    else:
        raise ValueError(f'Unknown score name "{args.score_name}"!')


def _optimize_scores(
        scores: List[BaseScore],
        model_family: str,
        text_collection: VowpalWabbitTextCollection,
        output_file_path: str,
        min_num_topics: int,
        max_num_topics: int,
        num_topics_interval: int,
        num_fit_iterations: int,
        num_restarts: int) -> None:

    optimizer = OptimizeScoresMethod(
        scores=scores,
        model_family=model_family,
        min_num_topics=min_num_topics,
        max_num_topics=max_num_topics,
        num_topics_interval=num_topics_interval,
        num_fit_iterations=num_fit_iterations,
        num_restarts=num_restarts
    )

    optimizer.search_for_optimum(text_collection)

    with open(output_file_path, 'w') as f:
        f.write(json.dumps(optimizer._result))


def _renormalize(
        text_collection: VowpalWabbitTextCollection,
        output_file_path: str,
        matrix: str,
        min_num_topics: int,
        max_num_topics: int,
        num_fit_iterations: int,
        num_restarts: int) -> None:

    optimizer = RenormalizationMethod(
        matrix_for_renormalization=matrix,
        min_num_topics=min_num_topics,
        max_num_topics=max_num_topics,
        num_fit_iterations=num_fit_iterations,
        num_restarts=num_restarts
    )

    optimizer.search_for_optimum(text_collection)

    with open(output_file_path, 'w') as f:
        f.write(json.dumps(optimizer._result))


if __name__ == '__main__':
    _main()
