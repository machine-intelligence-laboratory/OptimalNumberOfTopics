# TopicBank

The idea is to search for *new* *interpretable* topics as long as possible, training many topic models.
Each model may be not very good as a whole, but if it has at least some interpretable topics, the model is already valuable.
In the end, all gathered interpretable topics supposedly give a topic model which is the best for the given text collection.
As the searching for an appropriate number of topics in a document collection is a task at hand, when all the interpretable topics are collected in the bank, their number may serve as this appropriate number of topics.

![The idea behind TopicBank](docs/images/topic_bank_concept.png)

A couple of things that seem worth noting:
* Bank collects *interpretable* topics.
Automatically, using some topic quality measure.
* Bank collects *new* topics.
It means, that there should be no linearly dependent topics in the bank.
* All the collected topics by assumption comprise a *good topic model* for the given text document collection.
It means that the model should provide high likelihood, i.e. it should describe the collection well.
Although maybe not as well as one ordinary model specially fitted for the task.


## References

* [hARTM_tutorial.ipynb](https://github.com/bigartm/bigartm-book/blob/master/hARTM_tutorial.ipynb) — some details and examples about hierarchical topic models in ARTM library. The mechanism is used in TopicBank to provide non-linearity between topics that are collected.
