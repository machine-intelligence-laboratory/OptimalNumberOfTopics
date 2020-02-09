import argparse
import json

from topnum.data.vowpal_wabbit_text_collection import VowpalWabbitTextCollection
from topnum.scores.perplexity_score import PerplexityScore
from topnum.search_methods.constants import (
    DEFAULT_MAX_NUM_TOPICS,
    DEFAULT_MIN_NUM_TOPICS
)
from topnum.search_methods.optimize_score_method import OptimizeScoreMethod


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
        help='Main modality in text'
    )
    parser_optimize.add_argument(
        'output_file_path',
        help='File to write the result of search in'
    )
    # TODO: extract modalities from text if no specified
    # TODO: weights
    parser_optimize.add_argument(
        '-m', '--modality',
        help='Modality to take into account',
        action='append',  # TODO: extend
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

        if args.modalities is not None:
            modalities = args.modalities
        else:
            modalities = [args.main_modality]

        text_collection = VowpalWabbitTextCollection(
            args.vw_file_path,
            main_modality=args.main_modality,
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
    else:
        raise ValueError(args.search_method)


if __name__ == '__main__':
    _main()
