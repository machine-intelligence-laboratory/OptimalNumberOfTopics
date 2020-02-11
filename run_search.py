import argparse
import json
from typing import Tuple

from topnum.data.vowpal_wabbit_text_collection import VowpalWabbitTextCollection
from topnum.scores.perplexity_score import PerplexityScore
from topnum.search_methods.constants import (
    DEFAULT_MAX_NUM_TOPICS,
    DEFAULT_MIN_NUM_TOPICS
)
from topnum.search_methods.optimize_score_method import OptimizeScoreMethod


MESSAGE_MODALITY_FORMAT = (
    'Format: <modality name>, or <modality name>:<modality weight>.'
    ' Weight assumed to be 1.0 if not specified'
)


# TODO: test
def extract_modality_name_and_weight(
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


def _optimize_perplexity(args: argparse.Namespace) -> None:
    modalities = dict()

    main_modality_name, main_modality_weight = extract_modality_name_and_weight(
        args.main_modality
    )
    modalities[main_modality_name] = main_modality_weight

    # TODO: test weights: main @modality:5, -m @modality:2, -m @modality
    if args.modalities is not None:
        for modality in args.modalities:
            modality_name, modality_weight = extract_modality_name_and_weight(
                modality
            )
            modalities[modality_name] = modality_weight

    text_collection = VowpalWabbitTextCollection(
        args.vw_file_path,
        main_modality=main_modality_name,
        modalities=modalities
    )

    score = PerplexityScore(
        'perplexity_score',
        class_ids=args.modalities
    )

    optimizer = OptimizeScoreMethod(
        score=score,
        min_num_topics=args.min_num_topics,
        max_num_topics=args.max_num_topics,
        num_topics_interval=args.num_topics_interval,
        num_collection_passes=10,
        num_restarts=3
    )

    optimizer.search_for_optimum(text_collection)

    # TODO: check if exists
    with open(args.output_file_path, 'w') as f:
        f.write(json.dumps(optimizer._result))


def _main():
    parser = argparse.ArgumentParser(prog='run_search')
    subparsers = parser.add_subparsers(
        help='Method for searching an appropriate number of topics',
        dest='search_method'
    )

    parser_optimize = subparsers.add_parser(
        'optimize_score',
        help='Find the number of topics which optimizes the score'
             ' (gives it max or min depending on the score)'
    )
    parser_optimize.add_argument(
        'score_name',
        help='Name of the score to optimize:'
             ' perplexity -> min',
        choices=['perplexity']
    )
    parser_optimize.add_argument(
        'vw_file_path',
        help='Path to the file with text collection in vowpal wabbit format'
    )
    parser_optimize.add_argument(
        'main_modality',
        help=f'Main modality in text. {MESSAGE_MODALITY_FORMAT}'
    )
    parser_optimize.add_argument(
        'output_file_path',
        help='File to write the result of the search in'
    )
    # TODO: extract modalities from text if no specified
    parser_optimize.add_argument(
        '-m', '--modality',
        help=f'Other modality to take into account. {MESSAGE_MODALITY_FORMAT}',
        action='append',
        dest='modalities'
    )
    parser_optimize.add_argument(
        '--max-num-topics',
        help='Maximum number of topics',
        type=int,
        default=DEFAULT_MAX_NUM_TOPICS
    )
    parser_optimize.add_argument(
        '--min-num-topics',
        help='Minimum number of topics',
        type=int,
        default=DEFAULT_MIN_NUM_TOPICS
    )
    parser_optimize.add_argument(
        '--num-topics-interval',
        help='The number of topics the next model is bigger than the previous one',
        type=int,
        default=10
    )

    # parser_some_other = subparsers.add_parser('other', help='some help')

    args = parser.parse_args()

    if args.search_method == 'optimize_score':
        if args.score_name != 'perplexity':
            raise ValueError(args.score_name)

        _optimize_perplexity(args)
    else:
        raise ValueError(args.search_method)


if __name__ == '__main__':
    _main()
