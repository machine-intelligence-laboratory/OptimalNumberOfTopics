import argparse
import json
from typing import (
    Dict,
    List,
    Tuple
)

from topnum.data.vowpal_wabbit_text_collection import VowpalWabbitTextCollection
from topnum.scores import (
    PerplexityScore,
    EntropyScore
)
from topnum.scores.entropy_score import RENYI as RENYI_ENTROPY_NAME
from topnum.scores.base_score import BaseScore
from topnum.search_methods.constants import (
    DEFAULT_MAX_NUM_TOPICS,
    DEFAULT_MIN_NUM_TOPICS
)
from topnum.search_methods.optimize_scores_method import OptimizeScoresMethod


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

    parser_optimize_perplexity = subparsers_optimize_scores.add_parser(
        'perplexity',
        help='Perplexity -> min'
    )

    parser_optimize_renyi_entropy = subparsers_optimize_scores.add_parser(
        'renyi_entropy',
        help='Renyi_entropy -> min'
    )
    parser_optimize_renyi_entropy.add_argument(
        '-f', '--threshold-factor',
        help='A greater than zero factor'
             ' by which the default 1/|W| threshold should be multiplied by',
        type=float,
        default=1.0
    )

    # parser_some_other = subparsers.add_parser('other', help='some help')

    args, unparsed_args = parser.parse_known_args()

    if args.search_method == 'optimize_scores':
        main_modality_name, modalities = _parse_modalities(args.main_modality, args.modalities)
        modality_names = list(modalities.keys())
        vw_file_path = args.vw_file_path
        output_file_path = args.output_file_path
        min_num_topics = args.min_num_topics
        max_num_topics = args.max_num_topics
        num_topics_interval = args.num_topics_interval
        num_fit_iterations = args.num_fit_iterations
        num_restarts = args.num_restarts

        scores = list()

        scores.append(_build_score(args, modality_names))

        while len(unparsed_args) > 0:
            current_args, unparsed_args = parser_optimize_scores.parse_known_args(
                unparsed_args
            )
            scores.append(_build_score(current_args, modality_names))

        _optimize_scores(
            scores,
            vw_file_path,
            main_modality_name,
            modalities,
            output_file_path,
            min_num_topics=min_num_topics,
            max_num_topics=max_num_topics,
            num_topics_interval=num_topics_interval,
            num_fit_iterations=num_fit_iterations,
            num_restarts=num_restarts
        )
    else:
        raise ValueError(args.search_method)


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


def _build_score(args: argparse.Namespace, modality_names: List[str]) -> BaseScore:
    if args.score_name == 'perplexity':
        return _build_perplexity_score(modality_names)
    elif args.score_name == 'renyi_entropy':
        return _build_renyi_entropy_score(args.threshold_factor, modality_names)
    else:
        raise ValueError(f'Unknown score name "{args.score_name}"!')


def _build_perplexity_score(modalities: List[str]) -> PerplexityScore:
    return PerplexityScore(
        'perplexity_score',
        class_ids=modalities
    )


def _build_renyi_entropy_score(threshold_factor: float, modalities: List[str]) -> EntropyScore:
    return EntropyScore(
        'renyi_entropy_score',
        entropy=RENYI_ENTROPY_NAME,
        threshold_factor=threshold_factor,
        class_ids=modalities
    )


def _optimize_scores(
        scores: List[BaseScore],
        vw_file_path: str,
        main_modality_name: str,
        modalities: Dict[str, float],
        output_file_path: str,
        min_num_topics: int,
        max_num_topics: int,
        num_topics_interval: int,
        num_fit_iterations: int,
        num_restarts: int) -> None:

    text_collection = VowpalWabbitTextCollection(
        vw_file_path,
        main_modality=main_modality_name,
        modalities=modalities
    )

    optimizer = OptimizeScoresMethod(
        scores=scores,
        min_num_topics=min_num_topics,
        max_num_topics=max_num_topics,
        num_topics_interval=num_topics_interval,
        num_fit_iterations=num_fit_iterations,
        num_restarts=num_restarts
    )

    optimizer.search_for_optimum(text_collection)

    # TODO: check if folder exists
    with open(output_file_path, 'w') as f:
        f.write(json.dumps(optimizer._result))


if __name__ == '__main__':
    _main()
