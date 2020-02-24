#!/bin/bash

general_args=(
    ./vw.txt
    @text:1
    result.json
    -m @title:3
    --modality @publisher:2
)

search_method_args=(
    optimize_scores
    --max-num-topics 10
    --min-num-topics 1
    --num-topics-interval 2
    --num-fit-iterations 2
    --num-restarts 3
    perplexity
    renyi_entropy
    intratext_coherence
    top_tokens_coherence
    --cooc-file ./cooc_values.json
)

echo "Arguments: ${general_args[@]} ${search_method_args[@]}"

python ../run_search.py "${general_args[@]}" "${search_method_args[@]}"
