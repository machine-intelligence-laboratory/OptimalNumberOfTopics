import json
from topnum.data.vowpal_wabbit_text_collection import VowpalWabbitTextCollection
from topnum.scores.perplexity_score import PerplexityScore
from topnum.search_methods.optimize_score_method import OptimizeScoreMethod


def _main():
    file_path = 'data/test_vw_data.txt'
    text_collection = VowpalWabbitTextCollection(
        file_path,
        main_modality='@lemmatized',
        modalities=['@lemmatized']
    )

    score = PerplexityScore(
        'perplexity_score',
        class_ids=['@lemmatized']
    )
    optimizer = OptimizeScoreMethod(
        score=score,
        min_num_topics=2,
        max_num_topics=10,
        num_topics_interval=2,
        num_collection_passes=10,
        num_restarts=3
    )

    optimizer.search_for_optimum(text_collection)

    with open('result.json', 'w') as f:
        f.write(json.dumps(optimizer._result))

    text_collection._remove_dataset()

if __name__ == '__main__':
    _main()
