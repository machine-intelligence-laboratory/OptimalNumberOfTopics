# OptimalNumberOfTopics

To begin with, searching for an optimal number of topics in a text collection seems a very poorly stated task, because this number of topics heavily depends on the task at hand.
One can take 10 topics and it might be enough, or 100 topics, or 1000.
What's more, the whole notion of a *topic* is a bit obscure: people think of topics just as of some meaningful stories, concepts or ideas.
And there is a parent-child relationship between such topics, eg. topic "Coconut juice" is a child of topic "Beverages".
This means that for one dataset one can train a good topic model with, let's say 10 big parent topics, or another good topic model with, for example 100 more concrete, smaller topics.

So, what is this repository about then?
It gives an opportunity to try different method to find an *appropriate*, *approximate* number of topics, the number which in order of magnitude is close to the number of not-so-small topics.


## Optimize Scores

The first method is just about optimizing something for the number of topics.
That is, train several models with different number of topics, calculate some quality function for those models, and find the one which is the best.

Scores, available for optimizing:
* [*Perplexity*](https://en.wikipedia.org/wiki/Perplexity) (which is minimized).
Definitely, this is not the best choice (TODO: links, why).
* [*Rényi entropy*](https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy).
This one was shown to be a good indicator of some kind of model stability: the more stable the model, the less its entropy.
    * [Sergei Koltcov. "Application of Rényi and Tsallis entropies to topic modeling optimization.", 2018](https://www.sciencedirect.com/science/article/pii/S0378437118309907)
    * [Sergei Koltcov, Vera Ignatenko, and Olessia Koltsova. "Estimating Topic Modeling Performance with Sharma–Mittal Entropy.", 2019](https://www.mdpi.com/1099-4300/21/7/660)

Let's say, one have her text collection as a vowpal wabbit file [*vw.txt*](./sample/vw.txt):
```text
doc_1 |@publisher mann_ivanov_ferber |@title atlas obscura |@text earth:8 travel:10 baobab:1 ...
doc_2 |@publisher chook_and_geek |@title black hammer |@text hero:10 whiskey:2 barbalien:4 ...
doc_3 |@publisher eksmo |@title dune |@text sand:7 arrakis:6 spice:12 destiny:2 ...
...
```

Then it is possible to find an optimal number of topics for this collection by looking at some topic model's characteristics (*scores*) and choosing the number of topics which corresponds to the best model.

The searching process can be started like this:
```bash
python run_search.py \
    vw.txt \                    # path to vowpal wabbit file
    @text:1 \                   # main modality and its weight
    result.json \               # output file path (the file may not exist)
    -m @publisher:5 \           # other modality and its weight
    --modality @title:2 \       # other modality and its weight
    optimize_scores \           # search method
    --min-num-topics 1 \        # minimum number of topics in the text collection
    --max-num-topics 10 \       # maximum number of topics in the text collection
    --num-topics-interval 2 \   # search step in number of topics
    --num-fit-iterations 100 \  # number of fit iterations for each model training
    --num-restarts 10 \         # number of training restarts that differ in seed
    perplexity \                # what score to optimize
    renyi_entropy \             # another score to optimize
    --threshold-factor 2.0 \    # previous score parameter
    intratext_coherence \       # one more score
    top_tokens_coherence \      # and yet another one
```

And the *result.json* file will look like this: (TODO: try on real data to get meaningful values)
```json
{
    "score_results":
    {
        "perplexity_score":
        {
            "optimum": 9.0,
            "optimum_std": 0.0,
            "num_topics_values": [1.0, 3.0, 5.0, 7.0, 9.0],
            "score_values": [1374.69, 685.37, 494.05, 377.24, 313.09],
            "num_topics_values_std": [0.0, 0.0, 0.0, 0.0, 0.0],
            "score_values_std": [0.0, 0.0, 0.0, 0.0, 0.0]
        },
        "renyi_entropy_score":
        {
            "optimum": 3.0,
            "optimum_std": 0.0,
            "num_topics_values": [1.0, 3.0, 5.0, 7.0, 9.0],
            "score_values": [1983797813.52, 1.37, 1.63, 1.84, 2.00],
            "num_topics_values_std": [0.0, 0.0, 0.0, 0.0, 0.0],
            "score_values_std": [9.87e-07, 2.30e-16, 2.30e-16, 4.60e-16, 6.90e-16]
        },
        "intratext_coherence_score":
        {
            "optimum": 1.0,
            "optimum_std": 0.0,
            "num_topics_values": [1.0, 3.0, 5.0, 7.0, 9.0],
            "score_values": [72.90, 21.92, 12.73, 9.21, 6.88],
            "num_topics_values_std": [0.0, 0.0, 0.0, 0.0, 0.0],
            "score_values_std": [1.47e-14, 0.0, 3.68e-15, 1.84e-15, 2.76e-15]
        },
        "top_tokens_coherence_score":
        {
            "optimum": 1.0,
            "optimum_std": 0.0,
            "num_topics_values": [1.0, 3.0, 5.0, 7.0, 9.0],
            "score_values": [0.834, 0.42, 0.76, 0.79, 0.53],
            "num_topics_values_std": [0.0, 0.0, 0.0, 0.0, 0.0],
            "score_values_std": [3.45e-16, 1.15e-16, 1.15e-16, 2.30e-16, 1.15e-16]
        }
    }
}
```

Here *optimum* means the optimal number of topics according to the score, *score_values* are the values of the score, each value corresponds to the number of topics in *num_topics_values* by the same index.

Another way to run the process may be via [bash script](./sample/optimize_scores.sh)
```bash
#!/bin/bash

general_args=(
    ./sample/vw.txt
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
    --cooc-file ./sample/cooc_values.json
)

python run_search.py "${general_args[@]}" "${search_method_args[@]}"
```

Or sitting in a .py file or a Jupyter Notebook:
```python
from topnum.data.vowpal_wabbit_text_collection import VowpalWabbitTextCollection
from topnum.scores import (
    DiversityScore,
    EntropyScore,
    IntratextCoherenceScore,
    PerplexityScore,
    SophisticatedTopTokensCoherenceScore
)
from topnum.scores.diversity_score import L2
from topnum.scores.entropy_score import RENYI as RENYI_ENTROPY_NAME
from topnum.search_methods.optimize_scores_method import OptimizeScoresMethod


text_collection = VowpalWabbitTextCollection(
    'sample/vw.txt',
    main_modality='@text',
    modalities={'@text': 1, '@title': 3, '@publisher': 2}
)
modality_names = list(modalities.keys())

scores = [
    PerplexityScore(
        'perplexity_score',
        class_ids=modality_names
    ),
    EntropyScore(
        'renyi_entropy_score',
        entropy=RENYI_ENTROPY_NAME,
        class_ids=modality_names
    ),
    DiversityScore(
        'l2_diversity_score',
        metric=L2,
        class_ids=modality_names
    ),
    IntratextCoherenceScore(
        'intratext_coherence_score',
        data=text_collection
    ),
    SophisticatedTopTokensCoherenceScore(
        'top_tokens_coherence_score',
        data=text_collection
    )
]

optimizer = OptimizeScoresMethod(
    scores=scores,
    min_num_topics=1,
    max_num_topics=10,
    num_topics_interval=2,
    num_fit_iterations=2,
    num_restarts=3
)

optimizer.search_for_optimum(text_collection)

with open('result.json', 'w') as f:
    f.write(json.dumps(optimizer._result))
```

Scores currenly available are:
* perplexity
* renyi_entropy
* diversity_score
* intratext_coherence
* top_tokens_coherence

More about scores one can find [here](https://github.com/machine-intelligence-laboratory/OptimalNumberOfTopics/tree/feature/coherences/topnum/scores) in the module.


## Renormalization

The approach is described in the following paper:  
[Sergei Koltcov, Vera Ignatenko, and Sergei Pashakhin. "Fast tuning of topic models: an application of Rényi entropy and renormalization theory.", 2019.](https://www.researchgate.net/profile/Sergei_Koltsov2/publication/337427975_5th_International_Electronic_Conference_on_Entropy_and_Its_Applications_Fast_tuning_of_topic_models_an_application_of_Renyi_entropy_and_renormalization_theory/links/5dd6d6bf458515dc2f41e248/5th-International-Electronic-Conference-on-Entropy-and-Its-Applications-Fast-tuning-of-topic-models-an-application-of-Renyi-entropy-and-renormalization-theory.pdf).

Briefly, one model with a big number of topics is trained.
Then, the number of topics is gradually reduced to one single topic: on each iteration two topics are selected by some criterion and merged into one.
Minimum value of entropy is supposed to show the best, optimal, number of topics, when the model is most stable.

The method can be invoked like this:
```bash
python run_search.py \
    vw.txt \                    # path to vowpal wabbit file
    @text:1 \                   # main modality and its weight
    result.json \               # output file path (the file may not exist)
    -m @publisher:5 \           # other modality and its weight
    --modality @title:2 \       # other modality and its weight
    renormalize \               # search method
    --max-num-topics 100 \      # maximum number of topics in the text collection
    --num-fit-iterations 100 \  # number of fit iterations for each model training
    --num-restarts 10 \         # number of training restarts that differ in seed
    --matrix phi                # matrix to use for renormalization
```

## Structure

    .
    ├── run_search.py       # Main script which handles all the methods and their parameters and provides a way to run the process through the command line
    └── topnum              # Core
        ├── data            # Train data handling (eg. Vowpal Wabbit files)
        ├── scores          # Scores that are available for optimizing or tracking
        └── search_methods  # Some techniques and ideas that can be used for finding an appropriate number of topics
