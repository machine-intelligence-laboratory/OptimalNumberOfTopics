# OptimalNumberOfTopics

To begin with, searching for an optimal number of topics in a text collection seems a very poorly stated task, because this number of topics heavily depends on the task at hand.
One can take $10$ topics and it might be enough, or $100$ topics, or $1000$.
What's more, the whole notion of a *topic* is a bit obscure: people think of topics just as of some meaningful stories, concepts or ideas.
And there is a parent-child relationship between such topics, eg. topic "Coconut juice" is a child of topic "Beverages".
This means that for one dataset one can train a good topic model with, let's say $10$ big parent topics, or another good topic model with, for example $100$ more concrete, smaller topics.

So, what is this repository about then?
It gives an opportunity to try different method to find an *appropriate*, *approximate* number of topics, the number which in order of magnitude is close to the number of not-so-small topics.


## Optimize Score

The first method is just about optimizing something for the number of topics.
That is, train several models with different number of topics, calculate some quality function for those models, and find the one which is the best.

Currently, only [*perplexity*](https://en.wikipedia.org/wiki/Perplexity) score in supported (which is minimized).

The method can be invoked like this
```bash
python run_search.py optimize_score perlexity \
    <path-to-vowpal-wabbit-file> <main-modality> <output-file-path>
```
