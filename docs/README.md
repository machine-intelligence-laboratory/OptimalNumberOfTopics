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
    * [Koltcov, Sergei. "Application of Rényi and Tsallis entropies to topic modeling optimization." Physica A: Statistical Mechanics and its Applications 512 (2018): 1192-1204.](https://www.sciencedirect.com/science/article/pii/S0378437118309907)
    * [Koltcov, Sergei, Vera Ignatenko, and Olessia Koltsova. "Estimating Topic Modeling Performance with Sharma–Mittal Entropy." Entropy 21.7 (2019): 660.](https://www.mdpi.com/1099-4300/21/7/660)

The method can be invoked like this:
```bash
python run_search.py \
    vw.txt \                    # path to vowpal wabbit file
    @text:1 \                   # main modality and its weight
    result.json \               # output file path (the file may not exist)
    -m @publisher:5 \           # other modality and its weight
    --modality @author:10 \     # other modality and its weight
    optimize_scores \           # search method
    --max-num-topics 1000 \     # maximum number of topics in the text collection
    --min-num-topics 100 \      # minimum number of topics in the text collection
    --num-topics-interval 50    # search step in number of topics
    perplexity \                # what score to optimize
    renyi_entropy \             # another score to optimize
    --threshold-factor 2.0      # previous score parameter
```

And the result will look like this: (TODO: try on real data to get meaningful values)
```json
{
    "score_results":
    {
        "perplexity_score":
        {
            "optimum": 9.0,
            "optimum_std": 0.0,
            "score_values": [1374.69, 654.64, 437.03, 343.46, 286.09],
            "num_topics_values": [1.0, 3.0, 5.0, 7.0, 9.0],
            "score_values_std": [0.0, 0.0, 0.0, 0.0, 0.0],
            "num_topics_values_std": [0.0, 0.0, 0.0, 0.0, 0.0]
        },
        "renyi_entropy_score":
        {
            "optimum": 7.0,
            "optimum_std": 0.0,
            "score_values": [3202736109.71, 3.18, 2.09, 1.87, 2.03],
            "num_topics_values": [1.0, 3.0, 5.0, 7.0, 9.0],
            "score_values_std": [4.94e-07, 4.60e-16, 0.0, 0.0, 4.60e-16],
            "num_topics_values_std": [0.0, 0.0, 0.0, 0.0, 0.0]
        }
    }
}
```

Here *optimum* means the optimal number of topics according to the score, *score_values* are the values of the score, each value corresponds to the number of topics in *num_topics_values* by the same index.

## Renormalization

The approach is described in the following paper:  
[Koltcov, Sergei, Vera Ignatenko, and Sergei Pashakhin. "Fast tuning of topic models: an application of Rényi entropy and renormalization theory." Conference Proceedings Paper. Vol. 18. No. 30. 2019.](https://www.researchgate.net/profile/Sergei_Koltsov2/publication/337427975_5th_International_Electronic_Conference_on_Entropy_and_Its_Applications_Fast_tuning_of_topic_models_an_application_of_Renyi_entropy_and_renormalization_theory/links/5dd6d6bf458515dc2f41e248/5th-International-Electronic-Conference-on-Entropy-and-Its-Applications-Fast-tuning-of-topic-models-an-application-of-Renyi-entropy-and-renormalization-theory.pdf).

Briefly, one model with a big number of topics is trained.
Then, the number of topics is gradually reduced to one single topic: on each iteration two topics are selected by some criterion and merged into one.
Minimum value of entropy is supposed to show the best, optimal, number of topics, when the model is most stable.


## Structure

    .
    ├── run_search.py       # Main script which handles all the methods and their parameters and provides a way to run the process through the command line
    └── topnum              # Core
        ├── data            # Train data handling (eg. Vowpal Wabbit files)
        ├── scores          # Scores that are available for optimizing or tracking
        └── search_methods  # Some techniques and ideas that can be used for finding an appropriate number of topics
