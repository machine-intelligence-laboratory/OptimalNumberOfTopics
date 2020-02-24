# Scores

Here are the scores available for using for finding an appropriate number of topics in a text collection:

    .
    └── base_score.py                                       # Base score class. If a score is used anywhere in the module, it inherits from BaseScore
        ├── base_default_score.py                           # ARTM score wrapped in BaseScore
            └── perplexity_score.py
        └── base_custom_score.py                            # TopicNet score wrapped in BaseScore
            ├── diversity_score.py
            ├── entropy_score.py
            ├── intratext_coherence_score.py                # Coherence score (intratext)
            ├── simple_toptok_coherence_score.py            # Coherence score (top tokens, #1)
            └── sophisticated_toptok_coherence_score.py     # Coherence score (top tokens, #2)


## Coherence

Worth noting, that there are several types of topic coherence scores available: intratext and top tokens based ones.
Each has various parameters which define the exact way of coherence computation.
What's more, things happened that way, that there are even two different classes for top tokens based coherence (so called [sophisticated](sophisticated_toptok_coherence_score.py) and [simple](simple_toptok_coherence_score.py)).
They differ a bit from one another and it seems for now that both are worth to be kept in the module.
The main differences currently seem the following: the simple implementation requires token *cooccurrence values* for initialization, can take modalities as input to look for tokens only from specified modalities, plus the simple coherence provides an ability to take into account only *active topics* and only tokens from *topic kernels*.
On the other hand, the sophisticated implementation provides more ways for computing coherence, and estimating word-to-topic relatedness values.
The sophisticated top tokens coherence can deal without the precomputed word cooccurence values, however, if one still wants to use custom cooccurrence values (for faster computing for example), she needs to provide it in a different format than for the simple top tokens coherence (examples are in the classes' docstrings).

ARTM library can help in gathering information about word cooccurrences.
There are several links one can visit to get the knowledge about the process:

* About topic coherence: [coherence.html](https://bigartm.readthedocs.io/en/stable/tutorials/python_userguide/coherence.html)
* About the command line utility for gathering cooc information: [bigartm_cli.html](https://bigartm.readthedocs.io/en/stable/tutorials/bigartm_cli.html)
* Example of gathering coocs statistics (in Russian): [example_of_gathering.ipynb](https://nbviewer.jupyter.org/github/bigartm/bigartm-book/blob/master/junk/cooc_dictionary/example_of_gathering.ipynb)

So, unfortunately, it is impossible currently (with artm version '0.10.0' at least) to do everything just sitting in Python code.
If one wants to use BigARTM functionality for cooccurrences, she should use the command line utility.
Here is an example of bash command line instruction which helps to run the utility (it might be more convenient to place it into a .sh script)

```bash
cd <working_directory> && <path to the folder where bigartm resides>/bigartm/build/bin/bigartm \
    -c vw_natural_order.txt \
    -v vocab.txt \
    --cooc-window 10 \
    --cooc-min-tf 1 \
    --write-cooc-tf cooc_tf_ \
    --cooc-min-df 1 \
    --write-cooc-df cooc_df_ \
    --write-ppmi-tf ppmi_tf_ \
    --write-ppmi-df ppmi_df_
```

More details about a possible scenario of using the utility can be found in the notebook [Making-Decorrelation-and-Topic-Selection-Friends.ipynb](https://github.com/machine-intelligence-laboratory/TopicNet/blob/master/topicnet/demos/Making-Decorrelation-and-Topic-Selection-Friends.ipynb) in the section *Cooccurrences*.
